#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-stream causal transfer analysis for Qwen-3B / COCO_two.

Research question
-----------------
Instead of asking "where can left/right/above/below be decoded?", this script
asks a causal carrier question:

    By layer L, how much image-derived information that matters for the final
    spatial decision has already been written into TEXT token states?

The experiment uses two runs with the SAME prompt and SAME image dimensions:

    REAL : the real image
    GRAY : a constant gray image

At every selected decoder block output, we save the hidden states from both
runs.  Then we perform activation patching at TEXT-token roles.

Why this design
---------------
A naive Real -> Gray text patch can underestimate necessity, because later
layers might simply read the REAL visual tokens again.  Rather than relying on
architecture-specific attention-mask surgery, this script creates a robust
"text-only continuation" control at layer L:

    REAL base, but replace ALL visual-token hidden states at L with the GRAY
    visual-token hidden states from the matched run.

After this intervention, downstream layers no longer receive the real visual
carrier states at those sequence positions.  Any real-image information that
survives must already have been copied/mixed into the remaining REAL text
states by layer L.

For each layer L:

1) TEXT-ONLY CONTROL
       REAL base
       visual positions <- GRAY visual states

   This measures how much spatial behavior survives using the REAL text stream
   available at L while cutting the direct real visual carrier at L.

2) NECESSITY of a text role R
       REAL base
       visual positions <- GRAY visual states
       role R positions <- GRAY text states

   Compare with (1).  If performance/margin drops, image-derived information
   currently stored in role R was causally useful once future direct real
   visual carriers were removed.

3) SUFFICIENCY of a text role R
       GRAY base
       role R positions <- REAL text states

   The visual stream is gray.  If the GT spatial margin/accuracy recovers,
   role R alone carries useful real-image information at layer L.

Default roles
-------------
    subject     : subject phrase tokens
    reference   : reference phrase tokens
    subj_ref    : subject + reference tokens
    last        : final prompt token (the token whose next-token logits answer)
    other_text  : all non-visual text positions except subject/reference/last
    all_text    : every non-visual position

Important interpretation
------------------------
* This does NOT assume a Direction vector, centroid, PCA subspace, or any
  hand-designed spatial representation.
* The dependent variable is the model's own four-answer next-token decision.
* "Text-only control" is not claiming visual tokens are unimportant earlier;
  it asks whether their useful information has already been transferred into
  text states BY layer L.
* Necessity is conditional on the text-only control at that layer.
* Sufficiency is especially clean: the base image is gray, and only selected
  REAL text states are transplanted.
* Patching can create off-manifold states.  Use the all_text/full-state sanity
  checks and interpret small effects cautiously.

Computational cost
------------------
For each sample:
    2 baseline forwards (REAL, GRAY)
    per selected layer:
        1 text-only-control forward
        + one necessity forward per role
        + one sufficiency forward per role

The default quick grid is 8 layers x 3 roles = 56 intervention forwards/sample.
Start with --max-samples 20.  Once the pattern is clear, expand samples/layers.

Expected repository dependencies
--------------------------------
Run from AdaptVis/llava16 repository root.  This script reuses helpers from:
    extract_two_object_relation_states.py
    analyze_coco_head_object_residual_direction_probe_v1.py

Example quick run
-----------------
CUDA_VISIBLE_DEVICES=0 python analyze_text_stream_visual_causal_transfer_v1.py \
  --dataset coco_two \
  --data-root data \
  --model qwen-3b \
  --device cuda:0 \
  --layers 4,8,12,16,20,24,28,31 \
  --roles subj_ref,last,all_text \
  --max-samples 20 \
  --output-dir output/qwen3b_text_stream_causal_transfer_smoke \
  --overwrite

Then full test split:
CUDA_VISIBLE_DEVICES=0 python analyze_text_stream_visual_causal_transfer_v1.py \
  --dataset coco_two \
  --data-root data \
  --model qwen-3b \
  --device cuda:0 \
  --layers 4,8,12,16,20,24,28,31 \
  --roles subj_ref,last,all_text \
  --output-dir output/qwen3b_text_stream_causal_transfer_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import analyze_coco_head_object_residual_direction_probe_v1 as direction_base


RELATIONS = ("left", "right", "above", "below")
EPS = 1e-8
SCRIPT_VERSION = "text-stream-visual-causal-transfer-v1"


# -----------------------------------------------------------------------------
# CLI / small utilities
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
        help="Keep identical to the protocol you want to compare against.",
    )
    p.add_argument(
        "--split-csv",
        default="output/qwen3b_coco_grounded_consensus_v1/split.csv",
        help="Optional existing train/dev/test split. If present, only --split rows are used.",
    )
    p.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    p.add_argument("--ignore-split", action="store_true")
    p.add_argument(
        "--layers",
        default="4,8,12,16,20,24,28,31",
        help="Comma-separated decoder block indices, or 'all'.",
    )
    p.add_argument(
        "--roles",
        default="subj_ref,last,all_text",
        help=(
            "Comma-separated subset of subject,reference,subj_ref,last,other_text,all_text. "
            "Use all six only after the coarse scan; runtime grows linearly with role count."
        ),
    )
    p.add_argument("--gray-value", type=int, default=127, help="RGB value used for the matched gray image [0,255].")
    p.add_argument("--max-samples", type=int, default=None, help="Applied after split filtering; use 20 for a smoke test.")
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def norm_relation(x: Any) -> str:
    return direction_base.norm_relation(x)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def std(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.std(vals)) if vals else float("nan")


def fraction(xs: Iterable[bool]) -> float:
    vals = [bool(x) for x in xs]
    return float(np.mean(vals)) if vals else float("nan")


def parse_int_list(text: str, n_layers: int) -> List[int]:
    s = str(text).strip().lower()
    if s == "all":
        return list(range(n_layers))
    out: List[int] = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        li = int(piece)
        if not 0 <= li < n_layers:
            raise ValueError(f"Layer {li} outside [0,{n_layers - 1}]")
        out.append(li)
    out = list(dict.fromkeys(out))
    if not out:
        raise ValueError("No layers selected")
    return out


VALID_ROLES = ("subject", "reference", "subj_ref", "last", "other_text", "all_text")


def parse_roles(text: str) -> List[str]:
    out = [x.strip() for x in str(text).split(",") if x.strip()]
    bad = [x for x in out if x not in VALID_ROLES]
    if bad:
        raise ValueError(f"Unknown roles {bad}; valid={VALID_ROLES}")
    out = list(dict.fromkeys(out))
    if not out:
        raise ValueError("No roles selected")
    return out


# -----------------------------------------------------------------------------
# Token/logit helpers
# -----------------------------------------------------------------------------

def tokenizer_ids(tokenizer: Any, text: str) -> List[int]:
    try:
        return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
    except Exception:
        obj = tokenizer(text, add_special_tokens=False)
        ids = obj["input_ids"] if isinstance(obj, dict) else getattr(obj, "input_ids", [])
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(x) for x in ids]


def relation_token_variants(tokenizer: Any) -> Dict[str, List[int]]:
    """Find one-token surface variants for the four canonical answers."""
    out: Dict[str, List[int]] = {}
    unk = getattr(tokenizer, "unk_token_id", None)
    surface = {
        "left": ["left", " left", "\nleft", "Left", " Left"],
        "right": ["right", " right", "\nright", "Right", " Right"],
        "above": ["above", " above", "\nabove", "Above", " Above"],
        "below": ["below", " below", "\nbelow", "Below", " Below"],
    }
    for rel in RELATIONS:
        ids: List[int] = []
        for s in surface[rel]:
            xx = tokenizer_ids(tokenizer, s)
            if len(xx) != 1:
                continue
            tid = int(xx[0])
            if unk is not None and tid == int(unk):
                continue
            ids.append(tid)
        ids = list(dict.fromkeys(ids))
        if not ids:
            raise RuntimeError(f"No one-token variants found for relation {rel!r}")
        out[rel] = ids
    return out


def extract_logits(outputs: Any) -> torch.Tensor:
    candidates = [
        getattr(outputs, "logits", None),
        getattr(getattr(outputs, "language_model_outputs", None), "logits", None),
        getattr(getattr(outputs, "text_model_output", None), "logits", None),
    ]
    for x in candidates:
        if torch.is_tensor(x):
            return x
    if isinstance(outputs, (tuple, list)):
        for x in outputs:
            if torch.is_tensor(x) and x.ndim == 3:
                return x
    raise RuntimeError("Could not locate logits in model output")


def relation_scores(score_vector: torch.Tensor, token_map: Mapping[str, Sequence[int]]) -> torch.Tensor:
    vals: List[torch.Tensor] = []
    for rel in RELATIONS:
        ids = torch.as_tensor(list(token_map[rel]), device=score_vector.device, dtype=torch.long)
        vals.append(score_vector.index_select(0, ids).max())
    return torch.stack(vals, dim=0)


def score_forward(model: Any, batch: Mapping[str, Any], token_map: Mapping[str, Sequence[int]], gt: str) -> Dict[str, Any]:
    with torch.inference_mode():
        outputs = model(
            **batch,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    logits = extract_logits(outputs)
    scores_t = relation_scores(logits[0, -1], token_map)
    scores = [float(x) for x in scores_t.detach().float().cpu().tolist()]
    gi = RELATIONS.index(gt)
    wrong = [scores[i] for i in range(4) if i != gi]
    pred_i = int(np.argmax(np.asarray(scores)))
    pred = RELATIONS[pred_i]
    margin = float(scores[gi] - max(wrong))
    return {
        "pred": pred,
        "correct": bool(pred == gt),
        "margin": margin,
        **{f"logit_{r}": scores[i] for i, r in enumerate(RELATIONS)},
    }


def build_batch(processor: Any, rec: Any, question: str, image: Image.Image, device: torch.device):
    rendered = direction_base.build_chat_prompt(processor, question, True)
    batch = direction_base.process_inputs(processor, rendered, image, device)
    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    apos = direction_base.locate_phrase_positions(processor.tokenizer, ids, str(rec.subject))
    bpos = direction_base.locate_phrase_positions(processor.tokenizer, ids, str(rec.reference))
    return batch, ids, apos, bpos


# -----------------------------------------------------------------------------
# Layer-output capture / patch
# -----------------------------------------------------------------------------

def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x):
                return x
    raise TypeError(f"Cannot locate hidden-state tensor in output type {type(output)}")


def replace_first_tensor(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        items = list(output)
        for i, x in enumerate(items):
            if torch.is_tensor(x):
                items[i] = replacement
                return tuple(items)
    if isinstance(output, list):
        items = list(output)
        for i, x in enumerate(items):
            if torch.is_tensor(x):
                items[i] = replacement
                return items
    raise TypeError(f"Cannot replace hidden-state tensor in output type {type(output)}")


class LayerOutputCollector:
    """Collect full decoder-block output states for selected layers to CPU."""

    def __init__(self, layers: Sequence[torch.nn.Module], selected_layers: Sequence[int]):
        self.states: Dict[int, torch.Tensor] = {}
        self.handles: List[Any] = []
        self.selected_layers = list(map(int, selected_layers))
        for li in self.selected_layers:
            def make_hook(layer_id: int):
                def hook(_module: Any, _args: Any, output: Any) -> Any:
                    x = first_tensor(output)
                    if x.ndim != 3 or int(x.shape[0]) != 1:
                        raise RuntimeError(f"Unexpected L{layer_id} hidden shape {tuple(x.shape)}")
                    # Keep original dtype. One sample is released before moving to the next.
                    self.states[layer_id] = x[0].detach().cpu().clone()
                    return output
                return hook
            self.handles.append(layers[li].register_forward_hook(make_hook(li)))

    def validate(self) -> None:
        missing = [li for li in self.selected_layers if li not in self.states]
        if missing:
            raise RuntimeError(f"Collector missing layers {missing}")

    def close(self) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SingleLayerStatePatch:
    """Replace selected sequence positions at one decoder block output."""

    def __init__(
        self,
        module: torch.nn.Module,
        layer: int,
        donor_state_cpu: torch.Tensor,
        positions: Sequence[int],
    ) -> None:
        self.layer = int(layer)
        self.donor_state_cpu = donor_state_cpu
        self.positions = sorted(set(int(x) for x in positions))
        self.applied = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _args: Any, output: Any) -> Any:
        x = first_tensor(output)
        if x.ndim != 3 or int(x.shape[0]) != 1:
            raise RuntimeError(f"Unexpected L{self.layer} hidden shape {tuple(x.shape)}")
        if self.applied > 0:
            return output
        S = int(x.shape[1])
        pos = [p for p in self.positions if 0 <= p < S]
        if not pos:
            raise RuntimeError(f"L{self.layer}: patch has no valid positions")
        donor = self.donor_state_cpu
        if donor.ndim != 2 or int(donor.shape[0]) != S or int(donor.shape[1]) != int(x.shape[2]):
            raise RuntimeError(
                f"L{self.layer}: donor shape {tuple(donor.shape)} incompatible with current {tuple(x.shape)}"
            )
        idx = torch.as_tensor(pos, device=x.device, dtype=torch.long)
        donor_sel = donor.index_select(0, idx.cpu()).to(device=x.device, dtype=x.dtype)
        y = x.clone()
        y[0].index_copy_(0, idx, donor_sel)
        self.applied += 1
        return replace_first_tensor(output, y)

    def validate(self) -> None:
        if self.applied != 1:
            raise RuntimeError(f"L{self.layer}: expected exactly one patch, got {self.applied}")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def forward_with_patch(
    *,
    model: Any,
    batch: Mapping[str, Any],
    token_map: Mapping[str, Sequence[int]],
    gt: str,
    layers: Sequence[torch.nn.Module],
    layer: int,
    donor_state_cpu: torch.Tensor,
    positions: Sequence[int],
) -> Dict[str, Any]:
    patch = SingleLayerStatePatch(layers[layer], layer, donor_state_cpu, positions)
    try:
        with patch:
            result = score_forward(model, batch, token_map, gt)
        patch.validate()
        return result
    finally:
        patch.close()


# -----------------------------------------------------------------------------
# Position bookkeeping
# -----------------------------------------------------------------------------

def resolve_image_token_ids(model: Any, processor: Any) -> List[int]:
    ids: List[int] = []
    cfg = getattr(model, "config", None)
    for obj in [cfg, getattr(cfg, "text_config", None)]:
        if obj is None:
            continue
        for name in ["image_token_id", "image_token_index"]:
            v = getattr(obj, name, None)
            if isinstance(v, (int, np.integer)) and int(v) >= 0:
                ids.append(int(v))

    tok = processor.tokenizer
    # Qwen2/2.5-VL normally uses <|image_pad|> at the positions replaced by visual embeddings.
    for token_text in ["<|image_pad|>", "<image>", "<image_pad>"]:
        with contextlib.suppress(Exception):
            tid = int(tok.convert_tokens_to_ids(token_text))
            unk = getattr(tok, "unk_token_id", None)
            if tid >= 0 and (unk is None or tid != int(unk)):
                ids.append(tid)

    # Search declared additional special tokens for an image-pad-like symbol.
    special_tokens = getattr(tok, "additional_special_tokens", None) or []
    special_ids = getattr(tok, "additional_special_tokens_ids", None) or []
    for s, tid in zip(special_tokens, special_ids):
        ss = str(s).lower()
        if "image" in ss and ("pad" in ss or ss in {"<image>", "image"}):
            ids.append(int(tid))

    return list(dict.fromkeys(ids))


def role_positions(
    *,
    seq_len: int,
    visual_positions: Sequence[int],
    subject_positions: Sequence[int],
    reference_positions: Sequence[int],
) -> Dict[str, List[int]]:
    visual = set(int(x) for x in visual_positions)
    text = [i for i in range(seq_len) if i not in visual]
    text_set = set(text)
    subj = [int(x) for x in subject_positions if int(x) in text_set]
    ref = [int(x) for x in reference_positions if int(x) in text_set]
    last = [seq_len - 1] if (seq_len - 1) in text_set else [text[-1]]
    excluded = set(subj) | set(ref) | set(last)
    other = [i for i in text if i not in excluded]
    return {
        "subject": sorted(set(subj)),
        "reference": sorted(set(ref)),
        "subj_ref": sorted(set(subj) | set(ref)),
        "last": last,
        "other_text": other,
        "all_text": text,
    }


def filter_records_by_split(records: Sequence[Any], split_csv: Path, split: str) -> List[Any]:
    if split == "all":
        return list(records)
    rows = read_csv(split_csv)
    keep = {int(r["sid"]) for r in rows if str(r.get("split", "")).strip() == split}
    return [r for r in records if int(r.sid) in keep]


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def safe_recovery(value: float, gray: float, real: float) -> float:
    denom = float(real - gray)
    # Only make a normalized recovery claim where REAL actually improves the GT margin over GRAY.
    if denom <= EPS:
        return float("nan")
    return float((value - gray) / denom)


def summarize(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(int(r["layer"]), str(r["condition"]), str(r.get("role", "-")))].append(r)

    out: List[Dict[str, Any]] = []
    for (layer, condition, role), rs in sorted(groups.items()):
        item: Dict[str, Any] = {
            "layer": layer,
            "condition": condition,
            "role": role,
            "n": len(rs),
            "accuracy": fraction(bool(r["correct"]) for r in rs),
            "mean_margin": mean(float(r["margin"]) for r in rs),
            "mean_real_margin": mean(float(r["real_margin"]) for r in rs),
            "mean_gray_margin": mean(float(r["gray_margin"]) for r in rs),
            "mean_recovery_from_gray": mean(float(r.get("recovery_from_gray", float("nan"))) for r in rs),
            "std_recovery_from_gray": std(float(r.get("recovery_from_gray", float("nan"))) for r in rs),
            "frac_pred_changed_from_real": fraction(bool(r.get("pred_changed_from_real", False)) for r in rs),
            "frac_gt_recovered_from_gray": fraction(bool(r.get("gt_recovered_from_gray", False)) for r in rs),
        }
        if condition == "necessity":
            item.update({
                "mean_text_control_margin": mean(float(r["text_control_margin"]) for r in rs),
                "mean_margin_loss_vs_text_control": mean(float(r["margin_loss_vs_text_control"]) for r in rs),
                "frac_harms_text_control_correctness": fraction(bool(r["harms_text_control_correctness"]) for r in rs),
            })
        if condition == "sufficiency":
            item.update({
                "mean_margin_gain_vs_gray": mean(float(r["margin_gain_vs_gray"]) for r in rs),
            })
        out.append(item)
    return out


def print_summary(summary_rows: Sequence[Mapping[str, Any]], baseline_summary: Mapping[str, Any]) -> None:
    print("\n" + "=" * 132)
    print("TEXT-STREAM VISUAL CAUSAL TRANSFER")
    print("=" * 132)
    print(f"REAL  restricted first-step ACC : {baseline_summary['real_accuracy']:.4f}  mean margin={baseline_summary['real_mean_margin']:.5f}")
    print(f"GRAY  restricted first-step ACC : {baseline_summary['gray_accuracy']:.4f}  mean margin={baseline_summary['gray_mean_margin']:.5f}")
    print("-")
    print("Text-only control = REAL run with visual-token states replaced by matched GRAY states at that layer.")
    print("Necessity = text-only control + selected text role also replaced by GRAY.")
    print("Sufficiency = GRAY run + selected text role replaced by REAL.")
    print("-")
    print(f"{'layer':>5}  {'condition':<17} {'role':<12} {'n':>5} {'acc':>8} {'margin':>10} {'recovery':>10} {'necess.loss':>12} {'suff.gain':>11}")
    for r in summary_rows:
        loss = float(r.get("mean_margin_loss_vs_text_control", float("nan")))
        gain = float(r.get("mean_margin_gain_vs_gray", float("nan")))
        rec = float(r.get("mean_recovery_from_gray", float("nan")))
        def f(x: float) -> str:
            return f"{x:.5f}" if math.isfinite(x) else "nan"
        print(
            f"L{int(r['layer']):02d}   {str(r['condition']):<17} {str(r['role']):<12} {int(r['n']):5d} "
            f"{float(r['accuracy']):8.4f} {float(r['mean_margin']):10.5f} {f(rec):>10} {f(loss):>12} {f(gain):>11}"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if not 0 <= int(args.gray_value) <= 255:
        raise ValueError("--gray-value must be in [0,255]")

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / "per_sample_interventions.csv"
    errors_path = out_dir / "errors.jsonl"

    # ---------------- data ----------------
    records, _audit = base.load_records(args.dataset, Path(args.data_root), None)
    records = [r for r in records if norm_relation(r.relation) in RELATIONS]
    split_csv = Path(args.split_csv) if args.split_csv else None
    if not args.ignore_split and split_csv is not None and split_csv.exists():
        records = filter_records_by_split(records, split_csv, args.split)
        print(f"Using split={args.split!r} from {split_csv}; N={len(records)}")
    else:
        print(f"Using available records without split filtering; N={len(records)}")
    if args.max_samples is not None:
        records = records[: int(args.max_samples)]
    if not records:
        raise RuntimeError("No records selected")

    # ---------------- model ----------------
    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)
    kw: Dict[str, Any] = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl
    model = cls.from_pretrained(spec.repo_id, **kw)
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, layer_path = direction_base.resolve_decoder_layers(model)
    selected_layers = parse_int_list(args.layers, len(decoder_layers))
    selected_roles = parse_roles(args.roles)
    token_map = relation_token_variants(processor.tokenizer)
    image_token_ids = resolve_image_token_ids(model, processor)

    print(f"decoder={layer_path}  n_layers={len(decoder_layers)}")
    print("selected layers:", selected_layers)
    print("selected roles :", selected_roles)
    print("candidate image token ids:", image_token_ids)
    print("relation token variants:", token_map)
    if not image_token_ids:
        raise RuntimeError(
            "Could not resolve image token id. For Qwen2/2.5-VL this should normally include <|image_pad|>. "
            "Add the correct token to resolve_image_token_ids() for your checkpoint."
        )

    baseline_rows: List[Dict[str, Any]] = []
    intervention_rows: List[Dict[str, Any]] = []

    # ---------------- samples ----------------
    for sample_idx, rec in enumerate(tqdm(records, desc="text-stream causal")):
        real_img: Optional[Image.Image] = None
        gray_img: Optional[Image.Image] = None
        real_batch = None
        gray_batch = None
        try:
            sid = int(rec.sid)
            gt = norm_relation(rec.relation)
            question = args.prompt_template.format(subject=rec.subject, reference=rec.reference)
            real_img = Image.open(rec.image_path).convert("RGB")
            gray_img = Image.new("RGB", real_img.size, color=(args.gray_value,) * 3)

            real_batch, real_ids, subj_pos, ref_pos = build_batch(processor, rec, question, real_img, device)
            gray_batch, gray_ids, gray_subj_pos, gray_ref_pos = build_batch(processor, rec, question, gray_img, device)
            if real_ids != gray_ids:
                raise RuntimeError("REAL and GRAY input_ids differ; matched-position patching would be invalid")
            if list(subj_pos) != list(gray_subj_pos) or list(ref_pos) != list(gray_ref_pos):
                raise RuntimeError("REAL and GRAY object token locations differ")

            seq_len = len(real_ids)
            visual_positions = [i for i, tid in enumerate(real_ids) if int(tid) in set(image_token_ids)]
            if not visual_positions:
                # Diagnostic information helps adapt the token-id resolver if a checkpoint differs.
                special = [(i, tid) for i, tid in enumerate(real_ids) if tid in set(getattr(processor.tokenizer, "all_special_ids", []))]
                raise RuntimeError(
                    f"No visual positions found in input_ids. candidate image ids={image_token_ids}; "
                    f"special positions sample={special[:30]}"
                )
            roles = role_positions(
                seq_len=seq_len,
                visual_positions=visual_positions,
                subject_positions=subj_pos,
                reference_positions=ref_pos,
            )
            for role in selected_roles:
                if not roles[role]:
                    raise RuntimeError(f"Role {role!r} has zero positions for sid={sid}")

            # Collect matched REAL/GRAY states at every selected layer during the two baseline forwards.
            with LayerOutputCollector(decoder_layers, selected_layers) as real_col:
                real_result = score_forward(model, real_batch, token_map, gt)
                real_col.validate()
            with LayerOutputCollector(decoder_layers, selected_layers) as gray_col:
                gray_result = score_forward(model, gray_batch, token_map, gt)
                gray_col.validate()

            baseline_rows.append({
                "sid": sid,
                "gt": gt,
                "subject": str(rec.subject),
                "reference": str(rec.reference),
                "seq_len": seq_len,
                "n_visual_positions": len(visual_positions),
                "n_text_positions": seq_len - len(visual_positions),
                "real_pred": real_result["pred"],
                "real_correct": real_result["correct"],
                "real_margin": real_result["margin"],
                "gray_pred": gray_result["pred"],
                "gray_correct": gray_result["correct"],
                "gray_margin": gray_result["margin"],
            })

            visual_set = set(visual_positions)

            for li in selected_layers:
                real_state = real_col.states[li]
                gray_state = gray_col.states[li]
                if tuple(real_state.shape) != tuple(gray_state.shape):
                    raise RuntimeError(f"L{li} REAL/GRAY state shapes differ")
                if int(real_state.shape[0]) != seq_len:
                    raise RuntimeError(
                        f"L{li} hidden seq_len={real_state.shape[0]} != input_ids seq_len={seq_len}; "
                        "visual-token position mapping requires adaptation for this checkpoint"
                    )

                # ---------------------------------------------------------
                # Text-only continuation control:
                # REAL everything, except direct visual carrier states at L
                # are replaced by their matched GRAY states.
                # ---------------------------------------------------------
                text_control = forward_with_patch(
                    model=model,
                    batch=real_batch,
                    token_map=token_map,
                    gt=gt,
                    layers=decoder_layers,
                    layer=li,
                    donor_state_cpu=gray_state,
                    positions=visual_positions,
                )
                text_control_recovery = safe_recovery(
                    float(text_control["margin"]), float(gray_result["margin"]), float(real_result["margin"])
                )
                intervention_rows.append({
                    "sid": sid,
                    "gt": gt,
                    "layer": li,
                    "condition": "text_only_control",
                    "role": "-",
                    "pred": text_control["pred"],
                    "correct": text_control["correct"],
                    "margin": text_control["margin"],
                    "real_pred": real_result["pred"],
                    "real_correct": real_result["correct"],
                    "real_margin": real_result["margin"],
                    "gray_pred": gray_result["pred"],
                    "gray_correct": gray_result["correct"],
                    "gray_margin": gray_result["margin"],
                    "recovery_from_gray": text_control_recovery,
                    "pred_changed_from_real": bool(text_control["pred"] != real_result["pred"]),
                    "gt_recovered_from_gray": bool((not gray_result["correct"]) and text_control["correct"]),
                    "n_patched_positions": len(visual_positions),
                })

                for role in selected_roles:
                    rpos = roles[role]

                    # -----------------------------------------------------
                    # Necessity conditional on text-only continuation:
                    # additionally remove image-derived content at role R.
                    # -----------------------------------------------------
                    necessity_positions = sorted(visual_set | set(rpos))
                    nec = forward_with_patch(
                        model=model,
                        batch=real_batch,
                        token_map=token_map,
                        gt=gt,
                        layers=decoder_layers,
                        layer=li,
                        donor_state_cpu=gray_state,
                        positions=necessity_positions,
                    )
                    intervention_rows.append({
                        "sid": sid,
                        "gt": gt,
                        "layer": li,
                        "condition": "necessity",
                        "role": role,
                        "pred": nec["pred"],
                        "correct": nec["correct"],
                        "margin": nec["margin"],
                        "real_pred": real_result["pred"],
                        "real_correct": real_result["correct"],
                        "real_margin": real_result["margin"],
                        "gray_pred": gray_result["pred"],
                        "gray_correct": gray_result["correct"],
                        "gray_margin": gray_result["margin"],
                        "text_control_pred": text_control["pred"],
                        "text_control_correct": text_control["correct"],
                        "text_control_margin": text_control["margin"],
                        "margin_loss_vs_text_control": float(text_control["margin"] - nec["margin"]),
                        "harms_text_control_correctness": bool(text_control["correct"] and not nec["correct"]),
                        "recovery_from_gray": safe_recovery(
                            float(nec["margin"]), float(gray_result["margin"]), float(real_result["margin"])
                        ),
                        "pred_changed_from_real": bool(nec["pred"] != real_result["pred"]),
                        "gt_recovered_from_gray": bool((not gray_result["correct"]) and nec["correct"]),
                        "n_patched_positions": len(necessity_positions),
                    })

                    # -----------------------------------------------------
                    # Sufficiency:
                    # GRAY base + only REAL role-R text states.
                    # -----------------------------------------------------
                    suff = forward_with_patch(
                        model=model,
                        batch=gray_batch,
                        token_map=token_map,
                        gt=gt,
                        layers=decoder_layers,
                        layer=li,
                        donor_state_cpu=real_state,
                        positions=rpos,
                    )
                    intervention_rows.append({
                        "sid": sid,
                        "gt": gt,
                        "layer": li,
                        "condition": "sufficiency",
                        "role": role,
                        "pred": suff["pred"],
                        "correct": suff["correct"],
                        "margin": suff["margin"],
                        "real_pred": real_result["pred"],
                        "real_correct": real_result["correct"],
                        "real_margin": real_result["margin"],
                        "gray_pred": gray_result["pred"],
                        "gray_correct": gray_result["correct"],
                        "gray_margin": gray_result["margin"],
                        "margin_gain_vs_gray": float(suff["margin"] - gray_result["margin"]),
                        "recovery_from_gray": safe_recovery(
                            float(suff["margin"]), float(gray_result["margin"]), float(real_result["margin"])
                        ),
                        "pred_changed_from_real": bool(suff["pred"] != real_result["pred"]),
                        "gt_recovered_from_gray": bool((not gray_result["correct"]) and suff["correct"]),
                        "n_patched_positions": len(rpos),
                    })

            if args.print_every > 0 and (sample_idx + 1) % int(args.print_every) == 0:
                print(
                    f"[{sample_idx + 1}/{len(records)}] sid={sid} gt={gt} "
                    f"real={real_result['pred']}({real_result['margin']:+.3f}) "
                    f"gray={gray_result['pred']}({gray_result['margin']:+.3f})"
                )

        except Exception as exc:
            append_jsonl(errors_path, {
                "sid": int(getattr(rec, "sid", -1)),
                "error": repr(exc),
            })
            print(f"[WARN] sid={getattr(rec, 'sid', '?')}: {exc}")
        finally:
            with contextlib.suppress(Exception):
                if real_img is not None:
                    real_img.close()
            with contextlib.suppress(Exception):
                if gray_img is not None:
                    gray_img.close()
            del real_batch, gray_batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------- output ----------------
    if not baseline_rows:
        raise RuntimeError("No samples completed successfully; inspect errors.jsonl")

    write_csv(out_dir / "baseline_samples.csv", baseline_rows)
    write_csv(per_sample_path, intervention_rows)
    summary_rows = summarize(intervention_rows)
    write_csv(out_dir / "summary.csv", summary_rows)

    baseline_summary = {
        "n": len(baseline_rows),
        "real_accuracy": fraction(bool(r["real_correct"]) for r in baseline_rows),
        "gray_accuracy": fraction(bool(r["gray_correct"]) for r in baseline_rows),
        "real_mean_margin": mean(float(r["real_margin"]) for r in baseline_rows),
        "gray_mean_margin": mean(float(r["gray_margin"]) for r in baseline_rows),
    }

    # Useful per-layer compact view: text-only recovery + each role's necessity/sufficiency.
    compact: List[Dict[str, Any]] = []
    by_key = {(int(r["layer"]), str(r["condition"]), str(r["role"])): r for r in summary_rows}
    for li in selected_layers:
        control = by_key.get((li, "text_only_control", "-"), {})
        item: Dict[str, Any] = {
            "layer": li,
            "text_only_acc": control.get("accuracy", float("nan")),
            "text_only_margin": control.get("mean_margin", float("nan")),
            "text_only_recovery": control.get("mean_recovery_from_gray", float("nan")),
        }
        for role in selected_roles:
            nec = by_key.get((li, "necessity", role), {})
            suff = by_key.get((li, "sufficiency", role), {})
            item[f"{role}__necessity_acc"] = nec.get("accuracy", float("nan"))
            item[f"{role}__necessity_margin_loss"] = nec.get("mean_margin_loss_vs_text_control", float("nan"))
            item[f"{role}__sufficiency_acc"] = suff.get("accuracy", float("nan"))
            item[f"{role}__sufficiency_margin_gain"] = suff.get("mean_margin_gain_vs_gray", float("nan"))
            item[f"{role}__sufficiency_recovery"] = suff.get("mean_recovery_from_gray", float("nan"))
        compact.append(item)
    write_csv(out_dir / "layer_compact.csv", compact)

    metadata = {
        "script_version": SCRIPT_VERSION,
        "args": vars(args),
        "model": args.model,
        "dataset": args.dataset,
        "decoder_path": layer_path,
        "selected_layers": selected_layers,
        "selected_roles": selected_roles,
        "image_token_ids": image_token_ids,
        "relation_token_variants": token_map,
        "baseline": baseline_summary,
        "completed_samples": len(baseline_rows),
        "intervention_rows": len(intervention_rows),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print_summary(summary_rows, baseline_summary)
    print("\nSaved:")
    print(" ", out_dir / "baseline_samples.csv")
    print(" ", out_dir / "per_sample_interventions.csv")
    print(" ", out_dir / "summary.csv")
    print(" ", out_dir / "layer_compact.csv")
    print(" ", out_dir / "summary.json")
    if errors_path.exists():
        print(" ", errors_path)


if __name__ == "__main__":
    main()
