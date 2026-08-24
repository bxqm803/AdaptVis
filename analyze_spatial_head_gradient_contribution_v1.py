#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient contribution analysis for high-ACC Direction and Centroid heads.

Goal
----
For selected attention heads, measure whether the model's FINAL spatial decision
is sensitive to the head activations that were previously found to contain strong
spatial information.

The per-head activation analyzed here is exactly the pre-W_O attention output
(the input to o_proj), split into [num_heads, head_dim].  For one selected head
h at layer l, Z_lh has shape [seq_len, head_dim].

For ALL selected Direction and Centroid heads we report whole-head metrics:
    grad_norm_all
        || dM / dZ_lh ||_F

    grad_x_act_signed_all
        sum_t <dM/dZ_lh,t, Z_lh,t>
        First-order signed contribution of the current whole-head activation
        to the final spatial margin M.

    grad_x_act_abs_all
        sum |gradient * activation|
        Magnitude-only version, useful when positive/negative token
        contributions cancel.

For Direction heads we additionally report object-token metrics:
    grad_norm_subject / grad_norm_reference
    grad_x_act_subject / grad_x_act_reference

    relation_grad_norm
        || g_subject - g_reference ||

    current_relation_contribution
        0.5 * (g_subject - g_reference)^T (z_subject - z_reference)

If --direction-vectors-npz is supplied (recommended), we also load the existing
Image-NoImage residual relation vector for the SAME sample/head:
    r_res = (z_img_sub-z_img_ref) - (z_noimg_sub-z_noimg_ref)

and report:
    residual_relation_contribution
        0.5 * (g_subject - g_reference)^T r_res

    residual_relation_alignment
        cosine(g_subject-g_reference, r_res)

This is particularly useful because it asks whether the image-conditioned
spatial relation component found by the Direction probe is locally read out by
the downstream model.

Important interpretation
------------------------
* High probe ACC + high contribution/alignment:
    spatially decodable AND downstream-sensitive.
* High probe ACC + near-zero contribution/alignment:
    spatially decodable but weakly utilized (locally).
* High probe ACC + negative signed contribution:
    spatial feature is present, but the current downstream mapping pushes the
    final GT margin in the wrong direction.

Gradient is a local first-order sensitivity, NOT proof of unique causality;
redundancy and nonlinear effects remain possible.

This script is designed to live in the AdaptVis/llava16 repository root and
reuses:
    extract_two_object_relation_states.py
    analyze_coco_head_object_residual_direction_probe_v1.py

Default Qwen-3B centroid heads come from the previously observed high-centroid
set. Override with --centroid-heads for another model/experiment.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import shutil
import traceback
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

SCRIPT_VERSION = "spatial-head-gradient-contribution-v1"
RELATIONS = ("left", "right", "above", "below")
EPS = 1e-12

# Known high-centroid Qwen2.5-VL-3B heads from the current project results.
DEFAULT_CENTROID_HEADS = {
    "qwen-3b": [
        "L27H10", "L24H05", "L28H08", "L20H05",
        "L31H07", "L22H13", "L22H00", "L21H01",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl", default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
        help="eager is safest for consistency with existing analyses; attention probabilities are not required here.",
    )
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
        help="Use the same prompt as the Direction-head experiment whose ranking/vectors are supplied.",
    )
    p.add_argument(
        "--direction-results",
        default="output/qwen3b_coco_head_direction_residual/head_results.csv",
        help="Existing Direction head_results.csv; top heads are selected by residual_accuracy_mean.",
    )
    p.add_argument(
        "--direction-vectors-npz",
        default="output/qwen3b_coco_head_direction_residual/relation_vectors.npz",
        help="Existing relation_vectors.npz containing per-sample residual vectors. Set '' to disable residual alignment metrics.",
    )
    p.add_argument("--direction-top-k", type=int, default=10)
    p.add_argument("--random-control-k", type=int, default=10, help="Random non-selected heads used as a contribution baseline; 0 disables.")
    p.add_argument(
        "--direction-heads",
        default="auto",
        help="Comma-separated LxHy list, or 'auto' to use top-K from --direction-results.",
    )
    p.add_argument(
        "--centroid-heads",
        default="auto",
        help="Comma-separated LxHy list. 'auto' uses the built-in known high-centroid list for qwen-3b.",
    )
    p.add_argument(
        "--centroid-acc-json",
        default=None,
        help="Optional JSON mapping head name -> centroid ACC; used only to annotate the output.",
    )
    p.add_argument(
        "--generation-jsonl",
        default=None,
        help="Optional prior generation.jsonl; if supplied, joins generation_correct/prediction by sid for correct-vs-wrong summaries.",
    )
    p.add_argument(
        "--margin",
        choices=["gt_vs_best_wrong", "gt_vs_opposite"],
        default="gt_vs_best_wrong",
        help="Scalar final spatial margin differentiated back to selected head activations.",
    )
    p.add_argument("--pool", choices=["mean", "last"], default="mean", help="How to pool multi-token object names.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def norm_relation(x: Any) -> str:
    return direction_base.norm_relation(x)


def parse_head_name(text: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*[Ll](\d+)[Hh](\d+)\s*", str(text))
    if not m:
        raise ValueError(f"Bad head name {text!r}; expected e.g. L26H03")
    return int(m.group(1)), int(m.group(2))


def canonical_head_name(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head):02d}"


def parse_head_list(text: str) -> List[str]:
    if not text or str(text).strip().lower() == "auto":
        return []
    out = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        l, h = parse_head_name(piece)
        out.append(canonical_head_name(l, h))
    return list(dict.fromkeys(out))


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
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def safe_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.std(vals)) if vals else float("nan")


def load_direction_heads(path: Path, top_k: int) -> Tuple[List[str], Dict[str, float]]:
    rows = read_csv(path)
    scored = []
    for row in rows:
        try:
            acc = float(row.get("residual_accuracy_mean", "nan"))
            name = row.get("head_name") or canonical_head_name(int(row["layer"]), int(row["head"]))
            l, h = parse_head_name(name)
            scored.append((acc, canonical_head_name(l, h)))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [name for _, name in scored[: int(top_k)]]
    accs = {name: float(acc) for acc, name in scored}
    return selected, accs


def load_centroid_acc_json(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                l, h = parse_head_name(k)
                out[canonical_head_name(l, h)] = float(v)
            except Exception:
                pass
        return out
    raise ValueError("--centroid-acc-json must be a JSON object mapping LxHy -> accuracy")


def load_generation_jsonl(path: Optional[str]) -> Dict[int, Dict[str, Any]]:
    if not path:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "sid" in row:
                out[int(row["sid"])] = row
    return out


def load_direction_vectors(path: Optional[str]) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[int, int]]:
    if not path:
        return None, {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=False)
    if "residual" not in z or "sample_index" not in z:
        raise RuntimeError(f"{p} must contain residual and sample_index")
    arrays = {k: np.asarray(z[k]) for k in z.files}
    sid_to_row = {int(sid): i for i, sid in enumerate(arrays["sample_index"].tolist())}
    return arrays, sid_to_row


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
    result: Dict[str, List[int]] = {}
    unk = getattr(tokenizer, "unk_token_id", None)
    for relation in RELATIONS:
        surfaces = [
            relation, " " + relation, "\n" + relation,
            relation.capitalize(), " " + relation.capitalize(),
        ]
        token_ids: List[int] = []
        for surface in surfaces:
            ids = tokenizer_ids(tokenizer, surface)
            if len(ids) != 1:
                continue
            tid = int(ids[0])
            if unk is not None and tid == int(unk):
                continue
            token_ids.append(tid)
        token_ids = list(dict.fromkeys(token_ids))
        if not token_ids:
            raise RuntimeError(f"No one-token generation variant found for {relation!r}")
        result[relation] = token_ids
    return result


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
    vals = []
    for relation in RELATIONS:
        ids = torch.as_tensor(list(token_map[relation]), device=score_vector.device, dtype=torch.long)
        # max over valid one-token surface variants; gradient flows through the winning token.
        vals.append(score_vector.index_select(0, ids).max())
    return torch.stack(vals, dim=0)


OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}


def spatial_margin(scores: torch.Tensor, gt: str, mode: str) -> Tuple[torch.Tensor, str]:
    gi = RELATIONS.index(gt)
    if mode == "gt_vs_opposite":
        wrong = OPPOSITE[gt]
        wi = RELATIONS.index(wrong)
        return scores[gi] - scores[wi], wrong
    wrong_indices = [i for i in range(len(RELATIONS)) if i != gi]
    wrong_tensor = scores[torch.as_tensor(wrong_indices, device=scores.device)]
    local_idx = int(torch.argmax(wrong_tensor.detach()).item())
    wi = wrong_indices[local_idx]
    return scores[gi] - scores[wi], RELATIONS[wi]


def build_prompt_and_batch(processor: Any, rec: Any, question: str, image: Image.Image, device: torch.device):
    rendered = direction_base.build_chat_prompt(processor, question, True)
    batch = direction_base.process_inputs(processor, rendered, image, device)
    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    apos = direction_base.locate_phrase_positions(processor.tokenizer, ids, str(rec.subject))
    bpos = direction_base.locate_phrase_positions(processor.tokenizer, ids, str(rec.reference))
    return batch, ids, apos, bpos


def valid_positions(positions: Sequence[int], seq_len: int) -> List[int]:
    return [int(x) for x in positions if 0 <= int(x) < int(seq_len)]


def pool_tensor(t: torch.Tensor, positions: Sequence[int], mode: str) -> torch.Tensor:
    # t: [S,D]
    pos = valid_positions(positions, int(t.shape[0]))
    if not pos:
        raise RuntimeError("No valid object token positions inside captured sequence")
    if mode == "last":
        return t[pos[-1]]
    idx = torch.as_tensor(pos, device=t.device, dtype=torch.long)
    return t.index_select(0, idx).mean(dim=0)


class PreWOHeadCapture:
    """Capture o_proj inputs for selected layers without detaching the graph."""

    def __init__(self, layers: Sequence[Any], selected_layers: Sequence[int]):
        self.layers = layers
        self.selected_layers = sorted(set(map(int, selected_layers)))
        self.handles: List[Any] = []
        self.tensors: Dict[int, torch.Tensor] = {}
        for li in self.selected_layers:
            op = direction_base.resolve_o_proj(direction_base.resolve_self_attention(layers[li]))

            def make_hook(layer_id: int):
                def hook(_module: Any, inputs: Tuple[Any, ...]):
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError(f"L{layer_id} o_proj input unavailable")
                    self.tensors[layer_id] = inputs[0]
                    return None
                return hook

            self.handles.append(op.register_forward_pre_hook(make_hook(li)))

    def validate(self) -> None:
        missing = [li for li in self.selected_layers if li not in self.tensors]
        if missing:
            raise RuntimeError(f"Capture did not fire for layers {missing}")

    def close(self) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def head_slice(x: torch.Tensor, head: int, head_dim: int) -> torch.Tensor:
    start = int(head) * int(head_dim)
    stop = start + int(head_dim)
    if x.ndim != 3 or stop > int(x.shape[-1]):
        raise RuntimeError(f"Cannot slice H{head} dim={head_dim} from tensor {tuple(x.shape)}")
    return x[0, :, start:stop]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = float(af.norm().item() * bf.norm().item())
    if denom <= EPS:
        return float("nan")
    return float(torch.dot(af, bf).item() / denom)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["family"]), str(row["head_name"]))].append(row)

    metrics = [
        "grad_norm_all", "grad_abs_mean_all",
        "grad_x_act_signed_all", "grad_x_act_abs_all", "grad_x_act_signed_per_token",
        "grad_norm_subject", "grad_norm_reference", "grad_norm_object_pair",
        "grad_x_act_subject", "grad_x_act_reference", "grad_x_act_object_pair",
        "relation_grad_norm", "current_relation_contribution",
        "residual_relation_contribution", "residual_relation_alignment",
    ]
    out: List[Dict[str, Any]] = []
    for (family, name), rs in groups.items():
        item: Dict[str, Any] = {
            "family": family,
            "head_name": name,
            "layer": int(rs[0]["layer"]),
            "head": int(rs[0]["head"]),
            "probe_accuracy": rs[0].get("probe_accuracy", float("nan")),
            "n": len(rs),
        }
        for m in metrics:
            vals = [float(r.get(m, float("nan"))) for r in rs]
            item[f"mean_{m}"] = safe_mean(vals)
            item[f"mean_abs_{m}"] = safe_mean([abs(v) for v in vals])
            item[f"std_{m}"] = safe_std(vals)
        # split by final first-token spatial correctness
        for flag_name in ("native_correct", "generation_correct"):
            if flag_name not in rs[0]:
                continue
            for flag, suffix in ((True, "correct"), (False, "wrong")):
                sub = [r for r in rs if r.get(flag_name) is flag]
                item[f"n_{flag_name}_{suffix}"] = len(sub)
                for m in ("grad_x_act_signed_all", "residual_relation_contribution", "residual_relation_alignment"):
                    item[f"mean_{m}_{flag_name}_{suffix}"] = safe_mean(
                        [float(r.get(m, float("nan"))) for r in sub]
                    )
        out.append(item)

    # Keep family/head readability; not claiming one metric is the unique ranking.
    out.sort(key=lambda r: (r["family"], -float(r.get("probe_accuracy", -1) if math.isfinite(float(r.get("probe_accuracy", float("nan")))) else -1)))
    return out


def family_summary(head_summary: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for family in sorted(set(str(r["family"]) for r in head_summary)):
        rs = [r for r in head_summary if str(r["family"]) == family]
        result[family] = {
            "heads": len(rs),
            "mean_probe_accuracy": safe_mean([float(r.get("probe_accuracy", float("nan"))) for r in rs]),
            "mean_head_grad_norm_all": safe_mean([float(r["mean_grad_norm_all"]) for r in rs]),
            "mean_head_abs_grad_x_act_all": safe_mean([float(r["mean_abs_grad_x_act_signed_all"]) for r in rs]),
            "mean_head_signed_grad_x_act_all": safe_mean([float(r["mean_grad_x_act_signed_all"]) for r in rs]),
            "mean_direction_residual_abs_contribution": safe_mean([
                float(r.get("mean_abs_residual_relation_contribution", float("nan"))) for r in rs
            ]),
            "mean_direction_residual_alignment": safe_mean([
                float(r.get("mean_residual_relation_alignment", float("nan"))) for r in rs
            ]),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = out_dir / "per_sample_head_metrics.csv"
    error_jsonl = out_dir / "errors.jsonl"

    # -------------------------- select heads --------------------------
    direction_acc: Dict[str, float] = {}
    direction_heads = parse_head_list(args.direction_heads)
    if not direction_heads:
        path = Path(args.direction_results)
        if not path.exists():
            raise FileNotFoundError(f"Direction ranking not found: {path}")
        direction_heads, direction_acc = load_direction_heads(path, args.direction_top_k)
    else:
        if Path(args.direction_results).exists():
            _, direction_acc = load_direction_heads(Path(args.direction_results), 10**9)

    centroid_heads = parse_head_list(args.centroid_heads)
    if not centroid_heads:
        centroid_heads = list(DEFAULT_CENTROID_HEADS.get(args.model, []))
        if not centroid_heads:
            raise ValueError(
                f"No built-in centroid heads for model={args.model!r}; pass --centroid-heads LxHy,..."
            )
    centroid_acc = load_centroid_acc_json(args.centroid_acc_json)

    direction_arrays, sid_to_direction_row = load_direction_vectors(args.direction_vectors_npz)
    generation_rows = load_generation_jsonl(args.generation_jsonl)

    print("Selected Direction heads:", ", ".join(direction_heads))
    print("Selected Centroid heads :", ", ".join(centroid_heads))

    # -------------------------- data/model --------------------------
    records, _audit = base.load_records(args.dataset, Path(args.data_root), args.max_samples)
    records = [r for r in records if norm_relation(r.relation) in RELATIONS]
    print(f"[{args.dataset}] N={len(records)}")

    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)
    kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
    )
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl
    model = cls.from_pretrained(spec.repo_id, **kw)
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    layers, layer_path = direction_base.resolve_decoder_layers(model)
    n_heads, head_dim = direction_base.scan_shape(model, layers)
    print(f"decoder={layer_path} layers={len(layers)} heads={n_heads} head_dim={head_dim}")

    # Random-head control is essential if we want to say a spatial head has a
    # "large" gradient/contribution rather than merely a nonzero one.
    already = set(direction_heads) | set(centroid_heads)
    candidates = [
        canonical_head_name(l, h)
        for l in range(len(layers)) for h in range(n_heads)
        if canonical_head_name(l, h) not in already
    ]
    rng = random.Random(args.seed + 991)
    rng.shuffle(candidates)
    control_heads = candidates[: max(0, int(args.random_control_k))]

    selected_by_family = {
        "direction": direction_heads,
        "centroid": centroid_heads,
        "control": control_heads,
    }
    union_heads = list(dict.fromkeys(direction_heads + centroid_heads + control_heads))
    selected_layers = sorted(set(parse_head_name(x)[0] for x in union_heads))
    print("Random control heads    :", ", ".join(control_heads) if control_heads else "(disabled)")
    print("Captured layers         :", selected_layers)

    for name in union_heads:
        l, h = parse_head_name(name)
        if not (0 <= l < len(layers) and 0 <= h < n_heads):
            raise ValueError(f"Selected {name} outside model shape L={len(layers)}, H={n_heads}")

    token_map = relation_token_variants(processor.tokenizer)
    print("relation token variants:", token_map)

    # Pre-compute head metadata for fast loop.
    head_meta = []
    for family, names in selected_by_family.items():
        for name in names:
            l, h = parse_head_name(name)
            probe_acc = direction_acc.get(name, float("nan")) if family == "direction" else centroid_acc.get(name, float("nan"))
            head_meta.append((family, name, l, h, probe_acc))

    all_rows: List[Dict[str, Any]] = []

    # -------------------------- samples --------------------------
    for idx, rec in enumerate(tqdm(records, desc="gradient")):
        image = None
        batch = None
        capture = None
        try:
            gt = norm_relation(rec.relation)
            q = args.prompt_template.format(subject=rec.subject, reference=rec.reference)
            image = Image.open(rec.image_path).convert("RGB")
            batch, input_ids, a_pos, b_pos = build_prompt_and_batch(processor, rec, q, image, device)

            capture = PreWOHeadCapture(layers, selected_layers)
            with capture:
                # IMPORTANT: no torch.no_grad/inference_mode here; we need the autograd graph.
                outputs = model(
                    **batch,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
                capture.validate()
                logits = extract_logits(outputs)
                if logits.ndim != 3 or logits.shape[0] != 1:
                    raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
                score_vec = logits[0, -1]
                rel_scores = relation_scores(score_vec, token_map)
                margin, strongest_wrong = spatial_margin(rel_scores, gt, args.margin)
                native_idx = int(torch.argmax(rel_scores.detach()).item())
                native_pred = RELATIONS[native_idx]
                native_correct = native_pred == gt

                # autograd.grad can return gradients for all captured layer tensors in ONE traversal.
                grad_inputs = [capture.tensors[l] for l in selected_layers]
                grads = torch.autograd.grad(
                    margin,
                    grad_inputs,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_by_layer = {l: g for l, g in zip(selected_layers, grads)}

            # previous generation correctness is optional and evaluation-only
            gen_info = generation_rows.get(int(rec.sid), {})
            generation_correct = None
            if gen_info:
                if "correct" in gen_info:
                    generation_correct = bool(gen_info["correct"])
                elif "generation_correct" in gen_info:
                    generation_correct = bool(gen_info["generation_correct"])

            for family, name, l, h, probe_acc in head_meta:
                x_full = capture.tensors[l]
                g_full = grad_by_layer.get(l)
                if g_full is None:
                    continue
                z = head_slice(x_full, h, head_dim).float()   # [S,Dh]
                g = head_slice(g_full, h, head_dim).float()  # [S,Dh]
                S = int(z.shape[0])

                gx = g * z
                row: Dict[str, Any] = {
                    "sid": int(rec.sid),
                    "gt": gt,
                    "strongest_wrong": strongest_wrong,
                    "native_pred": native_pred,
                    "native_correct": bool(native_correct),
                    "native_margin": float(margin.detach().item()),
                    "family": family,
                    "head_name": name,
                    "layer": l,
                    "head": h,
                    "probe_accuracy": probe_acc,
                    "seq_len": S,
                    "grad_norm_all": float(g.norm().item()),
                    "grad_abs_mean_all": float(g.abs().mean().item()),
                    "activation_norm_all": float(z.norm().item()),
                    "grad_x_act_signed_all": float(gx.sum().item()),
                    "grad_x_act_abs_all": float(gx.abs().sum().item()),
                    "grad_x_act_signed_per_token": float(gx.sum().item() / max(S, 1)),
                }
                if generation_correct is not None:
                    row["generation_correct"] = bool(generation_correct)
                    row["generation_pred"] = gen_info.get("pred") or gen_info.get("prediction") or gen_info.get("parsed")

                if family == "direction":
                    z_sub = pool_tensor(z, a_pos, args.pool)
                    z_ref = pool_tensor(z, b_pos, args.pool)
                    g_sub = pool_tensor(g, a_pos, args.pool)
                    g_ref = pool_tensor(g, b_pos, args.pool)
                    relation_grad = g_sub - g_ref
                    current_relation = z_sub - z_ref

                    row.update({
                        "grad_norm_subject": float(g_sub.norm().item()),
                        "grad_norm_reference": float(g_ref.norm().item()),
                        "grad_norm_object_pair": float(torch.sqrt(g_sub.pow(2).sum() + g_ref.pow(2).sum()).item()),
                        "grad_x_act_subject": float(torch.dot(g_sub, z_sub).item()),
                        "grad_x_act_reference": float(torch.dot(g_ref, z_ref).item()),
                        "grad_x_act_object_pair": float((torch.dot(g_sub, z_sub) + torch.dot(g_ref, z_ref)).item()),
                        "relation_grad_norm": float(relation_grad.norm().item()),
                        "current_relation_norm": float(current_relation.norm().item()),
                        "current_relation_contribution": float(0.5 * torch.dot(relation_grad, current_relation).item()),
                        "current_relation_alignment": cosine(relation_grad, current_relation),
                    })

                    # Image-NoImage residual relation vector from the earlier Direction experiment.
                    if direction_arrays is not None and int(rec.sid) in sid_to_direction_row:
                        rr = sid_to_direction_row[int(rec.sid)]
                        residual = direction_arrays["residual"]
                        if l < residual.shape[1] and h < residual.shape[2]:
                            r_res_np = np.asarray(residual[rr, l, h], dtype=np.float32)
                            r_res = torch.from_numpy(r_res_np).to(device=relation_grad.device, dtype=torch.float32)
                            row.update({
                                "residual_relation_norm": float(r_res.norm().item()),
                                "residual_relation_contribution": float(0.5 * torch.dot(relation_grad, r_res).item()),
                                "residual_relation_alignment": cosine(relation_grad, r_res),
                            })

                all_rows.append(row)

            if args.print_every > 0 and (idx + 1) % args.print_every == 0:
                tqdm.write(
                    f"[{idx+1}/{len(records)}] sid={rec.sid} gt={gt} native={native_pred} "
                    f"margin={float(margin.detach().item()):+.4f}"
                )

            # release graph references aggressively
            del outputs, logits, score_vec, rel_scores, margin, grad_inputs, grads, grad_by_layer
            del capture.tensors
            capture = None

        except Exception as exc:
            append_jsonl(error_jsonl, {
                "sid": int(getattr(rec, "sid", -1)),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-16:],
            })
            tqdm.write(f"[ERROR] sid={getattr(rec, 'sid', '?')}: {type(exc).__name__}: {exc}")
        finally:
            if capture is not None:
                capture.close()
            if image is not None:
                image.close()
            del batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -------------------------- save/summary --------------------------
    write_csv(sample_csv, all_rows)
    head_rows = summarize_rows(all_rows)
    write_csv(out_dir / "head_summary.csv", head_rows)

    fam = family_summary(head_rows)
    summary = {
        "script_version": SCRIPT_VERSION,
        "model": args.model,
        "dataset": args.dataset,
        "n_records_requested": len(records),
        "n_sample_head_rows": len(all_rows),
        "margin": args.margin,
        "prompt_template": args.prompt_template,
        "head_activation_definition": "pre-W_O o_proj input split by head, all token positions",
        "direction_heads": direction_heads,
        "centroid_heads": centroid_heads,
        "control_heads": control_heads,
        "direction_vectors_npz": args.direction_vectors_npz,
        "family_summary": fam,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "selected_heads.json").write_text(json.dumps({
        "direction": [{"head": h, "probe_accuracy": direction_acc.get(h)} for h in direction_heads],
        "centroid": [{"head": h, "probe_accuracy": centroid_acc.get(h)} for h in centroid_heads],
        "control": [{"head": h} for h in control_heads],
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 108)
    print("SPATIAL HEAD GRADIENT CONTRIBUTION")
    print("=" * 108)
    print(f"model / dataset   : {args.model} / {args.dataset}")
    print(f"rows              : {len(all_rows)}")
    print(f"Direction heads   : {', '.join(direction_heads)}")
    print(f"Centroid heads    : {', '.join(centroid_heads)}")
    print("-")
    print("family    head      probeACC   |grad|all    |grad*x|all    signed(g*x)   obj|grad|   residual g·r   residual cos")
    for r in head_rows:
        print(
            f"{r['family']:<10s} {r['head_name']:<8s} "
            f"{float(r.get('probe_accuracy', float('nan'))):8.4f} "
            f"{float(r.get('mean_grad_norm_all', float('nan'))):11.5g} "
            f"{float(r.get('mean_abs_grad_x_act_signed_all', float('nan'))):13.5g} "
            f"{float(r.get('mean_grad_x_act_signed_all', float('nan'))):13.5g} "
            f"{float(r.get('mean_grad_norm_object_pair', float('nan'))):11.5g} "
            f"{float(r.get('mean_residual_relation_contribution', float('nan'))):14.5g} "
            f"{float(r.get('mean_residual_relation_alignment', float('nan'))):12.5g}"
        )

    print("\nSaved:")
    for name in ("per_sample_head_metrics.csv", "head_summary.csv", "summary.json", "selected_heads.json", "errors.jsonl"):
        p = out_dir / name
        if p.exists():
            print(" ", p)


if __name__ == "__main__":
    main()
