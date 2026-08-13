#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
L26 natural receiver-message scaling repair.

This script reuses the CLEAN vectors produced by:
    analyze_coco_l26_block_decomposition_v1_1.py

It does NOT construct a LEFT/RIGHT steering vector and does NOT touch direction
heads.  For every eval sample, it reuses the sample's own naturally computed L26
prompt-last attention contributions:

    c_obj_all       = all-head object-text -> prompt-last post-W_O write
    c_obj_selected  = selected receiver-head object-text -> prompt-last write
    c_nonobj        = all remaining L26 attention contribution

and modifies only the REAL L26 attention module output at prompt-last during the
generation PREFILL pass.

Modes
=====
1) obj_all
       attn_out'[last] = attn_out[last] + scale * c_obj_all

2) obj_selected
       attn_out'[last] = attn_out[last] + scale * c_obj_selected

3) nonobj_down
       attn_out'[last] = attn_out[last] - scale * c_nonobj

4) combo
       attn_out'[last] = attn_out[last]
                       + scale * c_obj_all
                       - scale * c_nonobj

The first two ask whether the model already computed the correct object relation
message but underweighted it at prompt-last.  The third asks whether non-object
attention is the main interference.  The fourth tests evidence competition.

GT-free gates
=============
all:
    patch every eval sample.

disagree:
    patch only if the TRAIN-only c_obj_all codebook prediction differs from the
    baseline greedy generation prediction.  This gate does NOT use eval GT.

For obj_selected, --gate disagree can optionally use c_obj_selected prediction
via --disagree-predictor auto (default).  For the other modes it uses c_obj_all.

No eval GT is used to decide WHETHER or HOW to patch.  GT is used only for final
ACC/W->C/C->W diagnostics.

Why this is cleaner than previous steering
==========================================
The delta is not a learned class vector.  It is the actual sample-specific
message naturally emitted by L26 attention from object tokens to prompt-last:

    c_obj = sum_h W_O^h sum_{s in objects} A_h[last,s] V_h[s]

The patch merely changes its gain.

Required decomposition directory
================================
Must contain:
    config.json
    baseline_eval.csv
    sample_vectors/<sid>.npz

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_coco_l26_receiver_message_scaling_v1.py \
  --decomposition-dir output/qwen3b_l26_block_decomposition_v1_1 \
  --model qwen-3b \
  --modes obj_all,obj_selected \
  --gates all,disagree \
  --scales 0.25,0.5,1.0,2.0 \
  --device cuda:0 \
  --output-dir output/qwen3b_l26_receiver_message_scaling_v1 \
  --overwrite

If object scaling is promising, then test interference:
=======================================================
CUDA_VISIBLE_DEVICES=0 python -u \
  eval_coco_l26_receiver_message_scaling_v1.py \
  --decomposition-dir output/qwen3b_l26_block_decomposition_v1_1 \
  --model qwen-3b \
  --modes nonobj_down,combo \
  --gates all,disagree \
  --scales 0.25,0.5,1.0 \
  --device cuda:0 \
  --output-dir output/qwen3b_l26_nonobj_scaling_v1 \
  --overwrite

Outputs
=======
config.json
patch_results.csv
patch_results.jsonl
summary.csv
relation_summary.csv
gate_diagnostics.csv
report.txt
errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import math
import random
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")

SCRIPT_VERSION = "coco-l26-receiver-message-scaling-v1"
RELATIONS = ("left", "right", "above", "below")
VALID_MODES = ("obj_all", "obj_selected", "nonobj_down", "combo")
VALID_GATES = ("all", "disagree")
EPS = 1e-12


# =============================================================================
# CLI / generic
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--decomposition-dir",
        required=True,
        help="Output directory from analyze_coco_l26_block_decomposition_v1_1.py",
    )
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument("--layer", type=int, default=26)

    p.add_argument(
        "--modes",
        default="obj_all,obj_selected",
        help="Comma-separated subset of obj_all,obj_selected,nonobj_down,combo.",
    )
    p.add_argument(
        "--gates",
        default="all,disagree",
        help="Comma-separated subset of all,disagree.",
    )
    p.add_argument(
        "--scales",
        default="0.25,0.5,1.0,2.0",
        help="Gain delta. scale=1 doubles object write in obj modes.",
    )
    p.add_argument(
        "--disagree-predictor",
        default="auto",
        choices=("auto", "c_obj_all", "c_obj_selected"),
        help=(
            "Codebook prediction used for disagree gate. auto uses selected for "
            "obj_selected mode, otherwise all-head object write."
        ),
    )
    p.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="0 = all rows in decomposition baseline_eval.csv.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
    )
    p.add_argument(
        "--probe-module",
        default="analyze_coco_head_object_residual_direction_probe_v1",
    )
    p.add_argument(
        "--attention-helper-module",
        default="analyze_coco_flip_attention_spatial_vectors_v1",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--empty-cache-every", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def parse_csv_list(text: str) -> List[str]:
    out = []
    seen = set()
    for raw in str(text).split(","):
        value = raw.strip()
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def parse_scales(text: str) -> List[float]:
    out = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = float(raw)
        if value < 0:
            raise ValueError("--scales must be non-negative")
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("No scales")
    return out


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "1", "true", "yes", "y", "t"
    }


def normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in RELATIONS:
        return text
    aliases = {
        "under": "below",
        "underneath": "below",
        "beneath": "below",
        "over": "above",
        "on": "above",
        "on top": "above",
    }
    if text in aliases:
        return aliases[text]
    # generated continuations are usually very short; first exact word wins.
    for rel in RELATIONS:
        if rel in text.split():
            return rel
    if "under" in text or "beneath" in text:
        return "below"
    if "over" in text:
        return "above"
    return None


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def safe_mean(values: Iterable[Any]) -> float:
    xs = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)
    return float(np.mean(xs)) if xs else float("nan")


def clear_sampling_defaults(model: Any) -> None:
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return
    for name in ("temperature", "top_p", "top_k"):
        if hasattr(cfg, name):
            setattr(cfg, name, None)


def first_3d(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        if output.ndim != 3:
            raise RuntimeError(
                f"Expected 3D attention output, got {tuple(output.shape)}"
            )
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    raise RuntimeError("No 3D attention tensor in module output")


def replace_first_3d(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        items = list(output)
        for index, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[index] = replacement
                return tuple(items)
    if isinstance(output, list):
        items = list(output)
        for index, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[index] = replacement
                return items
    raise RuntimeError("No 3D attention tensor to replace")


# =============================================================================
# Fixed natural-message patch
# =============================================================================

class PromptLastAttentionDelta:
    """
    Add one fixed CLEAN sample-specific residual-space vector to the REAL L26
    attention output at prompt-last, PREFILL only.
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        layer: int,
        prompt_last: int,
        prompt_length: int,
        delta: np.ndarray,
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.layer = int(layer)
        self.prompt_last = int(prompt_last)
        self.prompt_length = int(prompt_length)
        self.delta = np.asarray(delta, dtype=np.float32)
        self.handle = None
        self.applications = 0

    def __enter__(self) -> "PromptLastAttentionDelta":
        attention = self.attention_helper.resolve_self_attention(
            self.decoder_layers[self.layer]
        )

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = first_3d(output)

            # Full prompt prefill only. Cached decode steps normally have S=1.
            if int(hidden.shape[1]) != self.prompt_length:
                return None
            if not 0 <= self.prompt_last < int(hidden.shape[1]):
                return None
            if int(hidden.shape[-1]) != int(self.delta.shape[0]):
                raise RuntimeError(
                    f"L{self.layer} delta dim={self.delta.shape[0]} "
                    f"but attention hidden dim={hidden.shape[-1]}"
                )

            modified = hidden.clone()
            vector = torch.as_tensor(
                self.delta,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            modified[0, self.prompt_last] += vector
            self.applications += 1
            return replace_first_3d(output, modified)

        self.handle = attention.register_forward_hook(hook)
        return self

    def validate(self) -> None:
        if self.applications != 1:
            raise RuntimeError(
                f"Expected exactly one prefill patch at L{self.layer}, "
                f"got {self.applications}"
            )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
        self.handle = None


# =============================================================================
# Model batch / generation
# =============================================================================

def build_batch(
    *,
    probe: Any,
    processor: Any,
    question: str,
    image: Image.Image,
    device: torch.device,
) -> Any:
    rendered = probe.build_chat_prompt(processor, question, True)
    return probe.process_inputs(
        processor,
        rendered,
        image,
        device,
    )


@torch.inference_mode()
def patched_generation(
    *,
    model: Any,
    processor: Any,
    batch: Any,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    layer: int,
    prompt_last: int,
    delta: np.ndarray,
    max_new_tokens: int,
) -> Tuple[Optional[str], str]:
    prompt_length = int(batch["input_ids"].shape[1])

    with PromptLastAttentionDelta(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        layer=layer,
        prompt_last=prompt_last,
        prompt_length=prompt_length,
        delta=delta,
    ) as patch:
        output_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        patch.validate()

    text = processor.tokenizer.decode(
        output_ids[0, prompt_length:],
        skip_special_tokens=True,
    ).strip()
    del output_ids
    return normalize_relation(text), text


# =============================================================================
# Delta / gate
# =============================================================================

def vector_for_mode(
    *,
    vectors: Mapping[str, np.ndarray],
    mode: str,
    scale: float,
) -> np.ndarray:
    if mode == "obj_all":
        base = np.asarray(vectors["c_obj_all"], dtype=np.float32)
        return float(scale) * base

    if mode == "obj_selected":
        base = np.asarray(vectors["c_obj_selected"], dtype=np.float32)
        return float(scale) * base

    if mode == "nonobj_down":
        base = np.asarray(vectors["c_nonobj"], dtype=np.float32)
        return -float(scale) * base

    if mode == "combo":
        obj = np.asarray(vectors["c_obj_all"], dtype=np.float32)
        nonobj = np.asarray(vectors["c_nonobj"], dtype=np.float32)
        return float(scale) * (obj - nonobj)

    raise ValueError(mode)


def disagree_prediction_column(
    *,
    mode: str,
    predictor: str,
) -> str:
    if predictor == "c_obj_all":
        return "c_obj_all_pred"
    if predictor == "c_obj_selected":
        return "c_obj_selected_pred"
    if predictor != "auto":
        raise ValueError(predictor)
    if mode == "obj_selected":
        return "c_obj_selected_pred"
    return "c_obj_all_pred"


def gate_fires(
    *,
    row: Mapping[str, Any],
    mode: str,
    gate: str,
    predictor: str,
) -> Tuple[bool, Optional[str]]:
    if gate == "all":
        return True, None

    if gate == "disagree":
        column = disagree_prediction_column(
            mode=mode,
            predictor=predictor,
        )
        internal = normalize_relation(row.get(column))
        baseline = normalize_relation(
            row.get("generation_prediction")
        )
        if internal not in RELATIONS or baseline not in RELATIONS:
            return False, internal
        return internal != baseline, internal

    raise ValueError(gate)


# =============================================================================
# Summary
# =============================================================================

def summarize(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    baseline_by_sid = {
        int(row["sid"]): row for row in baseline_rows
    }
    baseline_acc = safe_mean(
        float(parse_bool(row["generation_correct"]))
        for row in baseline_rows
    )

    grouped: Dict[
        Tuple[str, str, float],
        List[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in patch_rows:
        grouped[
            (
                str(row["mode"]),
                str(row["gate"]),
                float(row["scale"]),
            )
        ].append(row)

    summary = []
    for (mode, gate, scale), rows in sorted(grouped.items()):
        patched_pred = {
            sid: normalize_relation(
                baseline_by_sid[sid]["generation_prediction"]
            )
            for sid in baseline_by_sid
        }

        for row in rows:
            sid = int(row["sid"])
            patched_pred[sid] = normalize_relation(
                row["patched_generation_prediction"]
            )

        patched_correct = {
            sid: patched_pred[sid]
            == normalize_relation(baseline_by_sid[sid]["gt"])
            for sid in baseline_by_sid
        }

        patched_acc = safe_mean(
            float(value) for value in patched_correct.values()
        )
        w2c = 0
        c2w = 0
        changed = 0
        for sid, baseline in baseline_by_sid.items():
            base_correct = parse_bool(
                baseline["generation_correct"]
            )
            new_correct = patched_correct[sid]
            if (not base_correct) and new_correct:
                w2c += 1
            if base_correct and (not new_correct):
                c2w += 1
            if patched_pred[sid] != normalize_relation(
                baseline["generation_prediction"]
            ):
                changed += 1

        summary.append({
            "mode": mode,
            "gate": gate,
            "scale": scale,
            "N_eval": len(baseline_rows),
            "N_patched": len(rows),
            "patch_rate": len(rows) / max(len(baseline_rows), 1),
            "baseline_acc": baseline_acc,
            "patched_acc": patched_acc,
            "delta_acc": patched_acc - baseline_acc,
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net_repairs": w2c - c2w,
            "generation_changed": changed,
            "generation_changed_rate": changed / max(len(baseline_rows), 1),
            "patched_rows_follow_internal_rate": safe_mean(
                float(
                    normalize_relation(
                        row["patched_generation_prediction"]
                    )
                    == normalize_relation(
                        row.get("gate_internal_prediction")
                    )
                )
                for row in rows
                if normalize_relation(
                    row.get("gate_internal_prediction")
                ) in RELATIONS
            ),
            "mean_delta_norm": safe_mean(
                row["delta_norm"] for row in rows
            ),
            "mean_delta_relative_to_attn_norm": safe_mean(
                row["delta_relative_to_a"]
                for row in rows
            ),
        })

    return summary


def relation_summary(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    baseline_by_sid = {
        int(row["sid"]): row for row in baseline_rows
    }
    groups: Dict[
        Tuple[str, str, float],
        List[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in patch_rows:
        groups[
            (
                str(row["mode"]),
                str(row["gate"]),
                float(row["scale"]),
            )
        ].append(row)

    out = []
    for condition, rows in sorted(groups.items()):
        mode, gate, scale = condition
        patched = {
            sid: normalize_relation(
                baseline_by_sid[sid]["generation_prediction"]
            )
            for sid in baseline_by_sid
        }
        for row in rows:
            patched[int(row["sid"])] = normalize_relation(
                row["patched_generation_prediction"]
            )

        for relation in RELATIONS:
            sids = [
                sid
                for sid, row in baseline_by_sid.items()
                if normalize_relation(row["gt"]) == relation
            ]
            if not sids:
                continue
            base_acc = safe_mean(
                float(
                    normalize_relation(
                        baseline_by_sid[sid]["generation_prediction"]
                    ) == relation
                )
                for sid in sids
            )
            patch_acc = safe_mean(
                float(patched[sid] == relation)
                for sid in sids
            )
            out.append({
                "mode": mode,
                "gate": gate,
                "scale": scale,
                "relation": relation,
                "N": len(sids),
                "baseline_acc": base_acc,
                "patched_acc": patch_acc,
                "delta_acc": patch_acc - base_acc,
            })
    return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    modes = parse_csv_list(args.modes)
    gates = parse_csv_list(args.gates)
    scales = parse_scales(args.scales)

    bad_modes = [mode for mode in modes if mode not in VALID_MODES]
    bad_gates = [gate for gate in gates if gate not in VALID_GATES]
    if bad_modes:
        raise ValueError(
            f"Bad modes {bad_modes}; valid={VALID_MODES}"
        )
    if bad_gates:
        raise ValueError(
            f"Bad gates {bad_gates}; valid={VALID_GATES}"
        )

    decomposition_dir = Path(args.decomposition_dir)
    config_path = decomposition_dir / "config.json"
    baseline_path = decomposition_dir / "baseline_eval.csv"
    vectors_dir = decomposition_dir / "sample_vectors"

    for path in (config_path, baseline_path, vectors_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    decomposition_config = json.loads(
        config_path.read_text(encoding="utf-8")
    )
    baseline_rows = read_csv(baseline_path)

    # Keep stable CSV order; optional quick cap.
    if args.max_eval_samples > 0:
        baseline_rows = baseline_rows[: args.max_eval_samples]

    if not baseline_rows:
        raise RuntimeError("No baseline eval rows")

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output dir is not empty: {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"
    patch_jsonl = output_dir / "patch_results.jsonl"

    probe = importlib.import_module(args.probe_module)
    attention_helper = importlib.import_module(
        args.attention_helper_module
    )
    base = probe.base

    records, audit = base.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    record_by_sid = {
        int(record.sid): record
        for record in records
    }

    # Sanity: the prompt/layer should match the clean vectors.
    prompt_template = str(
        decomposition_config.get("prompt_template")
        or (
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        )
    )
    clean_layer = int(
        decomposition_config.get("layer", args.layer)
    )
    if clean_layer != args.layer:
        raise RuntimeError(
            f"Decomposition vectors are L{clean_layer}, "
            f"but --layer={args.layer}"
        )

    spec = base.SPECS[args.model]
    model_class = getattr(transformers, spec.model_class)
    load_kwargs = {
        "dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
        "attn_implementation": args.attn_impl,
    }

    model = None
    processor = None

    try:
        print(
            f"Loading {args.model}: {spec.repo_id}",
            flush=True,
        )
        model = model_class.from_pretrained(
            spec.repo_id,
            **load_kwargs,
        )
        model.eval()
        clear_sampling_defaults(model)

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)

        decoder_layers, decoder_path = probe.resolve_decoder_layers(
            model
        )
        if not 0 <= args.layer < len(decoder_layers):
            raise ValueError(
                f"L{args.layer} outside decoder range "
                f"0..{len(decoder_layers)-1}"
            )

        device = torch.device(args.device)

        print("\n" + "=" * 140)
        print("L26 NATURAL RECEIVER-MESSAGE SCALING")
        print("=" * 140)
        print("decomposition :", decomposition_dir)
        print("layer         :", args.layer)
        print("modes         :", modes)
        print("gates         :", gates)
        print("scales        :", scales)
        print("N eval        :", len(baseline_rows))
        print("prompt        :", prompt_template)
        print(
            "baseline ACC  :",
            f"{100*safe_mean(float(parse_bool(r['generation_correct'])) for r in baseline_rows):.2f}%",
        )
        print("=" * 140)

        # ------------------------------------------------------------------
        # Gate diagnostics before any intervention.
        # ------------------------------------------------------------------
        gate_diag = []
        for predictor_column in (
            "c_obj_all_pred",
            "c_obj_selected_pred",
        ):
            disagreement = []
            for row in baseline_rows:
                internal = normalize_relation(
                    row.get(predictor_column)
                )
                generation = normalize_relation(
                    row.get("generation_prediction")
                )
                gt = normalize_relation(row.get("gt"))
                if (
                    internal in RELATIONS
                    and generation in RELATIONS
                    and internal != generation
                ):
                    disagreement.append({
                        "internal_correct": internal == gt,
                        "generation_correct": generation == gt,
                    })

            gate_diag.append({
                "predictor": predictor_column,
                "N_eval": len(baseline_rows),
                "N_disagree": len(disagreement),
                "disagree_rate": (
                    len(disagreement) / max(len(baseline_rows), 1)
                ),
                "internal_precision_on_disagreement_GT_diagnostic": safe_mean(
                    float(x["internal_correct"])
                    for x in disagreement
                ),
                "generation_precision_on_disagreement_GT_diagnostic": safe_mean(
                    float(x["generation_correct"])
                    for x in disagreement
                ),
            })
        write_csv(
            output_dir / "gate_diagnostics.csv",
            gate_diag,
        )

        conditions = [
            (mode, gate, scale)
            for mode in modes
            for gate in gates
            for scale in scales
            if scale > 0
        ]

        patch_rows: List[Dict[str, Any]] = []

        for condition_index, (mode, gate, scale) in enumerate(
            conditions,
            start=1,
        ):
            predictor_column = disagree_prediction_column(
                mode=mode,
                predictor=args.disagree_predictor,
            )

            eligible = []
            for row in baseline_rows:
                fires, internal = gate_fires(
                    row=row,
                    mode=mode,
                    gate=gate,
                    predictor=args.disagree_predictor,
                )
                if fires:
                    eligible.append((row, internal))

            print(
                f"\n[{condition_index}/{len(conditions)}] "
                f"mode={mode} gate={gate} scale={scale:g} "
                f"Npatch={len(eligible)}/{len(baseline_rows)}",
                flush=True,
            )

            for sample_index, (baseline_row, internal_prediction) in enumerate(
                tqdm(
                    eligible,
                    desc=f"{mode}:{gate}:s{scale:g}",
                ),
                start=1,
            ):
                sid = int(baseline_row["sid"])
                image = None
                batch = None
                try:
                    if sid not in record_by_sid:
                        raise RuntimeError(
                            f"SID {sid} missing from dataset"
                        )
                    vector_path = vectors_dir / f"{sid}.npz"
                    if not vector_path.exists():
                        raise FileNotFoundError(vector_path)

                    with np.load(vector_path) as data:
                        vectors = {
                            key: np.asarray(data[key])
                            for key in data.files
                        }

                    required = {
                        "c_obj_all",
                        "c_obj_selected",
                        "c_nonobj",
                        "a",
                    }
                    missing = required - set(vectors)
                    if missing:
                        raise RuntimeError(
                            f"{vector_path} missing {sorted(missing)}"
                        )

                    delta = vector_for_mode(
                        vectors=vectors,
                        mode=mode,
                        scale=scale,
                    ).astype(np.float32)

                    record = record_by_sid[sid]
                    question = prompt_template.format(
                        subject=record.subject,
                        reference=record.reference,
                    )
                    image = Image.open(record.image_path).convert("RGB")
                    batch = build_batch(
                        probe=probe,
                        processor=processor,
                        question=question,
                        image=image,
                        device=device,
                    )
                    prompt_last = (
                        int(batch["input_ids"].shape[1]) - 1
                    )

                    patched_prediction, patched_text = patched_generation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        decoder_layers=decoder_layers,
                        attention_helper=attention_helper,
                        layer=args.layer,
                        prompt_last=prompt_last,
                        delta=delta,
                        max_new_tokens=args.max_new_tokens,
                    )

                    gt = normalize_relation(
                        baseline_row["gt"]
                    )
                    baseline_prediction = normalize_relation(
                        baseline_row["generation_prediction"]
                    )
                    baseline_correct = parse_bool(
                        baseline_row["generation_correct"]
                    )

                    a_norm = float(
                        np.linalg.norm(
                            np.asarray(vectors["a"], dtype=np.float32)
                        )
                    )
                    delta_norm = float(np.linalg.norm(delta))

                    result = {
                        "sid": sid,
                        "mode": mode,
                        "gate": gate,
                        "scale": float(scale),
                        "gate_predictor_column": predictor_column,
                        "gate_internal_prediction": internal_prediction,
                        "gt": gt,
                        "baseline_generation_prediction": baseline_prediction,
                        "baseline_generation_correct": baseline_correct,
                        "patched_generation_prediction": patched_prediction,
                        "patched_generation_text": patched_text,
                        "patched_generation_correct": (
                            patched_prediction == gt
                        ),
                        "wrong_to_correct": (
                            (not baseline_correct)
                            and patched_prediction == gt
                        ),
                        "correct_to_wrong": (
                            baseline_correct
                            and patched_prediction != gt
                        ),
                        "generation_changed": (
                            patched_prediction != baseline_prediction
                        ),
                        "delta_norm": delta_norm,
                        "attention_output_norm_clean": a_norm,
                        "delta_relative_to_a": (
                            delta_norm / max(a_norm, EPS)
                        ),
                        "c_obj_all_norm": float(
                            np.linalg.norm(
                                np.asarray(
                                    vectors["c_obj_all"],
                                    dtype=np.float32,
                                )
                            )
                        ),
                        "c_obj_selected_norm": float(
                            np.linalg.norm(
                                np.asarray(
                                    vectors["c_obj_selected"],
                                    dtype=np.float32,
                                )
                            )
                        ),
                        "c_nonobj_norm": float(
                            np.linalg.norm(
                                np.asarray(
                                    vectors["c_nonobj"],
                                    dtype=np.float32,
                                )
                            )
                        ),
                    }
                    patch_rows.append(result)
                    append_jsonl(
                        patch_jsonl,
                        result,
                    )

                except Exception as exc:
                    append_jsonl(
                        errors_path,
                        {
                            "phase": "patch",
                            "sid": sid,
                            "mode": mode,
                            "gate": gate,
                            "scale": scale,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    if args.fail_fast:
                        raise
                finally:
                    if image is not None:
                        with contextlib.suppress(Exception):
                            image.close()
                    del batch
                    gc.collect()
                    if (
                        torch.cuda.is_available()
                        and args.empty_cache_every > 0
                        and sample_index % args.empty_cache_every == 0
                    ):
                        torch.cuda.empty_cache()

        write_csv(
            output_dir / "patch_results.csv",
            patch_rows,
        )

        summary_rows = summarize(
            baseline_rows=baseline_rows,
            patch_rows=patch_rows,
        )
        write_csv(
            output_dir / "summary.csv",
            summary_rows,
        )

        relation_rows = relation_summary(
            baseline_rows=baseline_rows,
            patch_rows=patch_rows,
        )
        write_csv(
            output_dir / "relation_summary.csv",
            relation_rows,
        )

        print("\n" + "=" * 160)
        print("L26 RECEIVER-MESSAGE SCALING SUMMARY")
        print("=" * 160)
        print(
            f"{'mode':<14s} {'gate':<9s} {'scale':>6s} "
            f"{'Npatch':>7s} {'baseACC':>9s} {'patchACC':>9s} "
            f"{'delta':>8s} {'W->C':>5s} {'C->W':>5s} {'net':>5s} "
            f"{'changed':>8s} {'d/a':>8s}"
        )
        print("-" * 160)
        for row in summary_rows:
            print(
                f"{str(row['mode']):<14s} "
                f"{str(row['gate']):<9s} "
                f"{float(row['scale']):>6.2f} "
                f"{int(row['N_patched']):>7d} "
                f"{100*float(row['baseline_acc']):>8.2f}% "
                f"{100*float(row['patched_acc']):>8.2f}% "
                f"{100*float(row['delta_acc']):>+7.2f} "
                f"{int(row['wrong_to_correct']):>5d} "
                f"{int(row['correct_to_wrong']):>5d} "
                f"{int(row['net_repairs']):>5d} "
                f"{100*float(row['generation_changed_rate']):>7.2f}% "
                f"{float(row['mean_delta_relative_to_attn_norm']):>8.3f}"
            )
        print("=" * 160)

        print("\nGT-free gate diagnostics (GT shown only to evaluate gate quality):")
        for row in gate_diag:
            print(
                f"  {row['predictor']:<20s} "
                f"disagree={int(row['N_disagree']):3d}/"
                f"{int(row['N_eval']):3d} "
                f"({100*float(row['disagree_rate']):5.2f}%) "
                f"internal precision="
                f"{100*float(row['internal_precision_on_disagreement_GT_diagnostic']):6.2f}% "
                f"generation precision="
                f"{100*float(row['generation_precision_on_disagreement_GT_diagnostic']):6.2f}%"
            )

        # Best exploratory condition by full-eval ACC.
        best = sorted(
            summary_rows,
            key=lambda row: (
                -float(row["patched_acc"]),
                -int(row["net_repairs"]),
            ),
        )

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "data_root": args.data_root,
            "layer": args.layer,
            "decomposition_dir": str(decomposition_dir),
            "decomposition_config": decomposition_config,
            "modes": modes,
            "gates": gates,
            "scales": scales,
            "disagree_predictor": args.disagree_predictor,
            "prompt_template": prompt_template,
            "N_eval": len(baseline_rows),
            "baseline_acc": safe_mean(
                float(parse_bool(row["generation_correct"]))
                for row in baseline_rows
            ),
            "uses_eval_gt_for_patch_decision": False,
            "patch_site": (
                f"L{args.layer} self-attention module output at prompt-last, prefill only"
            ),
            "mode_formulas": {
                "obj_all": "delta = scale * c_obj_all",
                "obj_selected": "delta = scale * c_obj_selected",
                "nonobj_down": "delta = -scale * c_nonobj",
                "combo": "delta = scale * (c_obj_all - c_nonobj)",
            },
            "dataset_audit": audit,
        }
        (
            output_dir / "config.json"
        ).write_text(
            json.dumps(
                config,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        lines = [
            f"script_version: {SCRIPT_VERSION}",
            f"baseline ACC: {100*float(config['baseline_acc']):.2f}%",
            "",
            "BEST EXPLORATORY CONDITIONS",
        ]
        for row in best[:10]:
            lines.append(
                f"{row['mode']} gate={row['gate']} scale={float(row['scale']):.2f}: "
                f"ACC {100*float(row['baseline_acc']):.2f}% -> "
                f"{100*float(row['patched_acc']):.2f}% "
                f"(delta={100*float(row['delta_acc']):+.2f} pp, "
                f"W->C={int(row['wrong_to_correct'])}, "
                f"C->W={int(row['correct_to_wrong'])}, "
                f"net={int(row['net_repairs'])})"
            )

        lines += [
            "",
            "READING THE RESULT",
            (
                "obj_all/obj_selected positive net gain supports the hypothesis that "
                "natural object->last receiver evidence is useful but underweighted."
            ),
            (
                "nonobj_down positive net gain supports non-object attention interference."
            ),
            (
                "combo outperforming either alone supports evidence competition at L26."
            ),
            (
                "If receiver-message scaling moves the expected internal direction but "
                "does not improve generation, c_obj is decodable but not sufficient as "
                "the downstream causal answer coordinate."
            ),
            "",
            "CAUTION",
            (
                "Choosing the best scale on this same eval set is exploratory tuning. "
                "After a promising setting is found, freeze mode/gate/scale and evaluate "
                "on a fresh held-out split for an unbiased ACC claim."
            ),
        ]
        (
            output_dir / "report.txt"
        ).write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:")
        for name in (
            "config.json",
            "gate_diagnostics.csv",
            "patch_results.csv",
            "patch_results.jsonl",
            "summary.csv",
            "relation_summary.csv",
            "report.txt",
        ):
            print(" ", output_dir / name)

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
