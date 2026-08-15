#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL-7B: trace what happens AFTER the learned L23 MLP repair.

Purpose
=======
The previous script

    eval_qwen7b_l23_mlp_good_direction_repair_v1.py

learned a TRAIN-only, relation-conditioned L23 MLP update direction and showed
that a held-out oracle-gated repair improves restricted 4-way first-step ACC,
but much less full generation ACC.

This script asks the next causal question:

    If L23 is repaired, does the repaired relation state survive L24-L27,
    or is it damaged again by a downstream attention/MLP module?

It REUSES the frozen split, learned direction and selected lambda from an
existing repair output directory.  It does NOT refit or retune anything.

For every selected TEST sample it runs two trajectories:

    clean:   normal model
    patched: apply the same frozen L23 MLP repair used previously

and traces, for every layer L in --layers (default 23,24,25,26,27):

    x_L        = block input / previous block output
    r_L        = x_L + attention_output_L
    y_L        = r_L + mlp_output_L

All x/r/y states are read with the same TRUE final-norm + LM-head relation
Logit Lens used in the earlier Qwen-7B trajectory analysis.

Primary diagnostics
===================
1. Local L23 repair:
       clean y23 wrong -> patched y23 correct

2. Repair survival:
       among locally repaired samples, what fraction remain correct at
       y24/y25/y26/y27?

3. First downstream loss site:
       if a patched-correct trajectory becomes wrong again, identify the first
       layer and whether the loss occurs in Attention (x correct -> r wrong)
       or MLP (r correct -> y wrong).

4. Final restricted first-step repair vs full generation repair:
       this separates downstream relation-state loss from a possible
       relation-ranking / vocabulary-emission mismatch.

Important interpretation
========================
- If many clean y23 wrong -> patched y23 correct samples later become wrong at
  the same downstream module, this supports sequential multi-stage failure and
  motivates a second conditional repair site.

- If patched y23 stays correct through y27, while full generation is still
  wrong, then the missing gain is NOT explained by L24-L27 relation-state
  overwrite.  The next diagnostic should be full-vocabulary rank / emission,
  not another relation-space patch.

This is a diagnostic script.  The default mode still uses an oracle failure
GATE (clean final 4-way first-step wrong) because we are testing mechanism, not
claiming a deployable GT-free controller.

Expected previous repair directory files
========================================
    natural_l23_states.csv
    split_sids.json
    learned_directions.npz
    selected_lambdas.json
    config.json

Example
=======
# Fast: only TEST samples that the oracle gate would patch (usually ~30% TEST)
CUDA_VISIBLE_DEVICES=0 python -u \
  analyze_qwen7b_l23_repair_downstream_survival_v1.py \
  --repair-dir output/qwen7b_l23_mlp_good_direction_repair_v1 \
  --mode final_oracle_predrel \
  --scope patched \
  --device cuda:0 \
  --output-dir output/qwen7b_l23_repair_downstream_survival_v1 \
  --overwrite

# Trace the whole held-out TEST split
CUDA_VISIBLE_DEVICES=0 python -u \
  analyze_qwen7b_l23_repair_downstream_survival_v1.py \
  --repair-dir output/qwen7b_l23_mlp_good_direction_repair_v1 \
  --mode final_oracle_predrel \
  --scope test \
  --device cuda:0 \
  --output-dir output/qwen7b_l23_repair_downstream_survival_testall_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import math
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


SCRIPT_VERSION = "qwen7b-l23-repair-downstream-survival-v1"
RELATIONS = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(RELATIONS)}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
DEFAULT_PROMPT = (
    "Determine the spatial relation of the {subject} to the {reference} "
    "in the image. Answer with left, right, above, or below."
)
EPS = 1e-12


# =============================================================================
# CLI / IO
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repair-dir",
        required=True,
        help="Output directory from eval_qwen7b_l23_mlp_good_direction_repair_v1.py",
    )
    p.add_argument(
        "--mode",
        default="final_oracle_predrel",
        choices=("final_oracle_predrel", "final_oracle_gt"),
        help=(
            "Frozen L23 repair mode. Both use the clean-final-wrong oracle gate; "
            "predrel selects the learned direction from clean r23's prediction."
        ),
    )
    p.add_argument(
        "--scope",
        default="patched",
        choices=("patched", "test"),
        help="patched=only TEST samples that receive the oracle repair; test=whole TEST split.",
    )
    p.add_argument("--layers", default="23,24,25,26,27")
    p.add_argument("--data-root", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument(
        "--run-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run clean + patched model.generate() for the selected cohort.",
    )
    p.add_argument("--prompt-template", default=DEFAULT_PROMPT)

    p.add_argument(
        "--helper-module",
        default="analyze_qwen7b_l26_l27_attention_overwrite_v1",
    )
    p.add_argument(
        "--probe-module",
        default="analyze_coco_head_object_residual_direction_probe_v1",
    )
    p.add_argument(
        "--attention-helper-module",
        default="analyze_coco_flip_attention_spatial_vectors_v1",
    )

    p.add_argument("--empty-cache-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def parse_layers(text: str) -> List[int]:
    values: List[int] = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if piece:
            values.append(int(piece))
    values = sorted(set(values))
    if not values:
        raise ValueError("Empty --layers")
    if 23 not in values:
        raise ValueError("L23 must be included because the intervention is at L23.")
    return values


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "none", "nan"}:
        return False
    raise ValueError(f"Cannot parse bool from {value!r}")


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


# =============================================================================
# Relation metrics
# =============================================================================

def relation_metrics(scores: np.ndarray, gt: str) -> Dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.shape[0] != len(RELATIONS):
        raise RuntimeError(f"Expected 4 relation scores, got {s.shape}")
    gt_id = RID[gt]
    pred_id = int(np.argmax(s))
    competitor = max((i for i in range(4) if i != gt_id), key=lambda i: s[i])
    return {
        "pred": RELATIONS[pred_id],
        "correct": pred_id == gt_id,
        "decision_margin": float(s[gt_id] - s[competitor]),
        "opposite_margin": float(s[gt_id] - s[RID[OPPOSITE[gt]]]),
    }


# =============================================================================
# Frozen repair loading
# =============================================================================

def load_frozen_repair(repair_dir: Path) -> Dict[str, Any]:
    required = [
        "natural_l23_states.csv",
        "split_sids.json",
        "learned_directions.npz",
        "selected_lambdas.json",
        "config.json",
    ]
    missing = [name for name in required if not (repair_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {repair_dir}: {missing}"
        )

    config = json.loads((repair_dir / "config.json").read_text(encoding="utf-8"))
    split = json.loads((repair_dir / "split_sids.json").read_text(encoding="utf-8"))
    lambdas = json.loads(
        (repair_dir / "selected_lambdas.json").read_text(encoding="utf-8")
    )

    natural_raw = read_csv_rows(repair_dir / "natural_l23_states.csv")
    natural: Dict[int, Dict[str, Any]] = {}
    for row in natural_raw:
        sid = int(row["sid"])
        natural[sid] = {
            **row,
            "sid": sid,
            "gt": str(row["gt"]),
            "firststep_correct": as_bool(row["firststep_correct"]),
            "r_correct": as_bool(row["r_correct"]),
            "y_correct": as_bool(row["y_correct"]),
            "r_pred": str(row["r_pred"]),
            "y_pred": str(row["y_pred"]),
            "firststep_pred": str(row["firststep_pred"]),
        }

    npz = np.load(repair_dir / "learned_directions.npz")
    directions: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for fit_mode in ("final", "transition"):
        for relation in RELATIONS:
            dk = f"{fit_mode}__{relation}__direction"
            tk = f"{fit_mode}__{relation}__target"
            if dk in npz and tk in npz:
                directions[fit_mode][relation] = {
                    "direction": np.asarray(npz[dk], dtype=np.float32),
                    "target_projection": float(np.asarray(npz[tk]).reshape(-1)[0]),
                }

    if "final" not in directions or len(directions["final"]) != 4:
        raise RuntimeError("Frozen repair directory lacks final relation directions.")
    if "final" not in lambdas:
        raise RuntimeError("Frozen repair directory lacks selected lambda for final mode.")

    return {
        "config": config,
        "split": split,
        "lambdas": lambdas,
        "natural": natural,
        "directions": directions,
    }


# =============================================================================
# Module-output patch
# =============================================================================

def first_3d(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        if output.ndim != 3:
            raise RuntimeError(f"Expected [B,S,D], got {tuple(output.shape)}")
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    raise RuntimeError("Could not find [B,S,D] tensor in module output.")


def replace_first_3d(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[i] = replacement
                return tuple(items)
    if isinstance(output, list):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item) and item.ndim == 3:
                items[i] = replacement
                return items
    raise RuntimeError("Could not replace [B,S,D] tensor in module output.")


class PromptLastProjectionRepairHook:
    """Apply the exact frozen 1D projection repair to L23 MLP prefill output."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        prompt_length: int,
        prompt_last: int,
        direction: np.ndarray,
        target_projection: float,
        lambda_value: float,
        label: str,
    ) -> None:
        self.module = module
        self.prompt_length = int(prompt_length)
        self.prompt_last = int(prompt_last)
        self.direction_np = np.asarray(direction, dtype=np.float32)
        self.target_projection = float(target_projection)
        self.lambda_value = float(lambda_value)
        self.label = str(label)
        self.handle = None
        self.applications = 0
        self.last_natural_projection = float("nan")
        self.last_scalar = float("nan")
        self.last_delta_norm = float("nan")

    def __enter__(self) -> "PromptLastProjectionRepairHook":
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = first_3d(output)
            # Prefill only; model.generate() later has q_len=1 decode calls.
            if int(hidden.shape[1]) != self.prompt_length:
                return None
            current = hidden[0, self.prompt_last]
            u = torch.as_tensor(
                self.direction_np,
                device=current.device,
                dtype=torch.float32,
            )
            current32 = current.float()
            projection = torch.dot(current32, u)
            scalar = self.lambda_value * (self.target_projection - projection)
            delta32 = scalar * u

            modified = hidden.clone()
            modified[0, self.prompt_last] = (current32 + delta32).to(current.dtype)

            self.applications += 1
            self.last_natural_projection = float(projection.item())
            self.last_scalar = float(scalar.item())
            self.last_delta_norm = float(delta32.norm().item())
            return replace_first_3d(output, modified)

        self.handle = self.module.register_forward_hook(hook)
        return self

    def validate(self, expected: int = 1) -> None:
        if self.applications != int(expected):
            raise RuntimeError(
                f"{self.label}: expected {expected} patch applications, got {self.applications}"
            )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            with contextlib.suppress(Exception):
                self.handle.remove()
        self.handle = None


# =============================================================================
# Model input / repair manager
# =============================================================================

def build_batch_for_record(
    *,
    record: Any,
    prompt_template: str,
    helper: Any,
    probe: Any,
    processor: Any,
    device: torch.device,
) -> Tuple[Image.Image, Mapping[str, torch.Tensor]]:
    question = prompt_template.format(
        subject=record.subject,
        reference=record.reference,
    )
    image = Image.open(record.image_path).convert("RGB")
    batch = helper.build_batch(
        probe=probe,
        processor=processor,
        question=question,
        image=image,
        device=device,
    )
    return image, batch


def make_patch_manager(
    *,
    clean_row: Mapping[str, Any],
    mode: str,
    mlp_module: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    directions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    lambda_value: float,
) -> Tuple[contextlib.AbstractContextManager, Dict[str, Any]]:
    # Both supported modes use the same oracle gate as the previous script:
    # patch only clean native restricted-first-step WRONG samples.
    if bool(clean_row["firststep_correct"]):
        return contextlib.nullcontext(), {
            "patched": False,
            "selected_relation": None,
            "lambda": 0.0,
        }

    if mode == "final_oracle_gt":
        relation = str(clean_row["gt"])
    elif mode == "final_oracle_predrel":
        relation = str(clean_row["r_pred"])
    else:
        raise ValueError(mode)

    if relation not in RELATIONS:
        raise RuntimeError(f"Bad selected relation: {relation!r}")

    spec = directions["final"][relation]
    prompt_length = int(batch["input_ids"].shape[1])
    prompt_last = prompt_length - 1

    manager = PromptLastProjectionRepairHook(
        module=mlp_module,
        prompt_length=prompt_length,
        prompt_last=prompt_last,
        direction=np.asarray(spec["direction"], dtype=np.float32),
        target_projection=float(spec["target_projection"]),
        lambda_value=float(lambda_value),
        label=mode,
    )
    return manager, {
        "patched": True,
        "selected_relation": relation,
        "lambda": float(lambda_value),
    }


# =============================================================================
# Trajectory tracing
# =============================================================================

def trace_trajectory(
    *,
    ah: Any,
    helper: Any,
    model: Any,
    batch: Mapping[str, torch.Tensor],
    token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[torch.nn.Module],
    layers: Sequence[int],
    lens: Any,
    gt: str,
    manager: contextlib.AbstractContextManager,
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    prompt_last = int(batch["input_ids"].shape[1]) - 1
    trace_layers = sorted(set([l - 1 for l in layers] + list(layers)))
    patch_stats: Dict[str, Any] = {}

    with manager as active:
        baseline, traces = ah.run_and_trace(
            model=model,
            batch=batch,
            token_map=token_map,
            decoder_layers=decoder_layers,
            layer_indices=trace_layers,
            target_positions=[prompt_last],
        )
        if isinstance(active, PromptLastProjectionRepairHook):
            active.validate(expected=1)
            patch_stats = {
                "natural_projection": active.last_natural_projection,
                "correction_scalar": active.last_scalar,
                "delta_norm": active.last_delta_norm,
            }

    final_pred = helper.normalize_relation(baseline["prediction"])
    if final_pred not in RELATIONS:
        raise RuntimeError(f"Bad final restricted relation pred: {baseline['prediction']!r}")

    per_layer: Dict[int, Dict[str, Any]] = {}
    for layer in layers:
        x = helper.trace_block_state(traces[layer - 1], prompt_last).astype(np.float32)
        attn = helper.trace_attention_state(traces[layer], prompt_last).astype(np.float32)
        r = (x + attn).astype(np.float32)
        y = helper.trace_block_state(traces[layer], prompt_last).astype(np.float32)

        scores = lens.scores(np.stack([x, r, y], axis=0))
        xm = relation_metrics(scores[0], gt)
        rm = relation_metrics(scores[1], gt)
        ym = relation_metrics(scores[2], gt)

        per_layer[int(layer)] = {
            "x": xm,
            "r": rm,
            "y": ym,
            "attention_gain": float(rm["decision_margin"] - xm["decision_margin"]),
            "mlp_gain": float(ym["decision_margin"] - rm["decision_margin"]),
        }

    del traces
    return {
        "final_pred": final_pred,
        "final_correct": final_pred == gt,
    }, per_layer, patch_stats


@torch.inference_mode()
def greedy_generation(
    *,
    model: Any,
    processor: Any,
    batch: Mapping[str, torch.Tensor],
    helper: Any,
    max_new_tokens: int,
    manager: contextlib.AbstractContextManager,
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    prompt_length = int(batch["input_ids"].shape[1])
    patch_stats: Dict[str, Any] = {}
    with manager as active:
        generated = model.generate(
            **batch,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
        )
        if isinstance(active, PromptLastProjectionRepairHook):
            active.validate(expected=1)
            patch_stats = {
                "natural_projection": active.last_natural_projection,
                "correction_scalar": active.last_scalar,
                "delta_norm": active.last_delta_norm,
            }

    text = processor.tokenizer.decode(
        generated[0, prompt_length:],
        skip_special_tokens=True,
    ).strip()
    pred = helper.normalize_relation(text)
    del generated
    return pred, text, patch_stats


# =============================================================================
# Downstream loss / summaries
# =============================================================================

def first_downstream_loss(
    patched_layers: Mapping[int, Mapping[str, Any]],
    layers: Sequence[int],
) -> Tuple[str, str]:
    """
    Only meaningful if patched y23 is correct.
    Return (first_loss_layer, first_loss_module).
    """
    if not bool(patched_layers[23]["y"]["correct"]):
        return "not_correct_at_L23", "not_applicable"

    for layer in sorted(l for l in layers if l > 23):
        info = patched_layers[layer]
        x_ok = bool(info["x"]["correct"])
        r_ok = bool(info["r"]["correct"])
        y_ok = bool(info["y"]["correct"])

        # We are looking for the first block that takes a previously correct
        # trajectory to a wrong y state.  x_L should match y_{L-1} in prediction.
        if not y_ok:
            if x_ok and not r_ok:
                return f"L{layer}", "attention"
            if r_ok and not y_ok:
                return f"L{layer}", "mlp"
            if x_ok and (not r_ok) and (not y_ok):
                return f"L{layer}", "attention_then_unrepaired"
            if not x_ok:
                # The state was already lost at the boundary; this should only
                # happen if a non-consecutive layer list is requested.
                return f"L{layer}", "already_wrong_at_input"
            return f"L{layer}", "mixed_or_unknown"

    return "none_through_last", "none"


def summarize_group_layers(
    sample_rows: Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    sample_by_sid = {int(r["sid"]): r for r in sample_rows}
    groups: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)

    for row in layer_rows:
        s = sample_by_sid[int(row["sid"])]
        names = ["all_selected"]
        if bool(s["local_l23_repair"]):
            names.append("local_l23_repair")
        if bool(s["patched_y23_correct"]):
            names.append("patched_y23_correct")
        if bool(s["final_firststep_repair"]):
            names.append("final_firststep_repair")
        if bool(s.get("generation_repair", False)):
            names.append("generation_repair")
        for name in names:
            groups[(name, int(row["layer"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (group, layer), rows in sorted(groups.items()):
        out.append({
            "group": group,
            "layer": layer,
            "N": len(rows),
            "clean_x_acc": safe_mean(float(as_bool(r["clean_x_correct"])) for r in rows),
            "patched_x_acc": safe_mean(float(as_bool(r["patched_x_correct"])) for r in rows),
            "clean_r_acc": safe_mean(float(as_bool(r["clean_r_correct"])) for r in rows),
            "patched_r_acc": safe_mean(float(as_bool(r["patched_r_correct"])) for r in rows),
            "clean_y_acc": safe_mean(float(as_bool(r["clean_y_correct"])) for r in rows),
            "patched_y_acc": safe_mean(float(as_bool(r["patched_y_correct"])) for r in rows),
            "clean_y_margin": safe_mean(float(r["clean_y_margin"]) for r in rows),
            "patched_y_margin": safe_mean(float(r["patched_y_margin"]) for r in rows),
            "delta_y_margin": safe_mean(float(r["delta_y_margin"]) for r in rows),
            "patched_attention_gain": safe_mean(float(r["patched_attention_gain"]) for r in rows),
            "patched_mlp_gain": safe_mean(float(r["patched_mlp_gain"]) for r in rows),
            "patched_attention_C_to_W": sum(
                int(as_bool(r["patched_x_correct"]) and not as_bool(r["patched_r_correct"]))
                for r in rows
            ),
            "patched_attention_W_to_C": sum(
                int((not as_bool(r["patched_x_correct"])) and as_bool(r["patched_r_correct"]))
                for r in rows
            ),
            "patched_mlp_C_to_W": sum(
                int(as_bool(r["patched_r_correct"]) and not as_bool(r["patched_y_correct"]))
                for r in rows
            ),
            "patched_mlp_W_to_C": sum(
                int((not as_bool(r["patched_r_correct"])) and as_bool(r["patched_y_correct"]))
                for r in rows
            ),
        })
    return out


def build_survival_rows(
    sample_rows: Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
) -> List[Dict[str, Any]]:
    by_sid_layer = {
        (int(r["sid"]), int(r["layer"])): r
        for r in layer_rows
    }
    cohorts = {
        "local_l23_repair": [r for r in sample_rows if bool(r["local_l23_repair"])],
        "patched_y23_correct": [r for r in sample_rows if bool(r["patched_y23_correct"])],
        "final_firststep_repair": [r for r in sample_rows if bool(r["final_firststep_repair"])],
    }
    out: List[Dict[str, Any]] = []
    for cohort, samples in cohorts.items():
        for layer in layers:
            rows = [by_sid_layer[(int(s["sid"]), int(layer))] for s in samples]
            out.append({
                "cohort": cohort,
                "layer": int(layer),
                "N": len(rows),
                "patched_x_survival": safe_mean(
                    float(as_bool(r["patched_x_correct"])) for r in rows
                ),
                "patched_r_survival": safe_mean(
                    float(as_bool(r["patched_r_correct"])) for r in rows
                ),
                "patched_y_survival": safe_mean(
                    float(as_bool(r["patched_y_correct"])) for r in rows
                ),
                "patched_y_margin_mean": safe_mean(float(r["patched_y_margin"]) for r in rows),
            })
    return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    repair_dir = Path(args.repair_dir)
    frozen = load_frozen_repair(repair_dir)
    old_config = frozen["config"]

    model_name = args.model or str(old_config.get("model", "qwen-7b"))
    dataset = args.dataset or str(old_config.get("dataset", "coco_two"))
    data_root = Path(args.data_root or "data")

    if model_name != "qwen-7b":
        raise ValueError("v1 intentionally supports the prior qwen-7b repair only.")
    if 23 != int(old_config.get("layer", 23)):
        raise ValueError("The supplied repair directory was not fit at L23.")

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    helper = importlib.import_module(args.helper_module)
    probe = importlib.import_module(args.probe_module)
    ah = importlib.import_module(args.attention_helper_module)
    base = probe.base

    # Reconstruct exactly the records referenced by the frozen TEST split.
    records, audit = base.load_records(dataset, data_root, None)
    record_by_sid = {int(r.sid): r for r in records}
    natural = frozen["natural"]
    test_sids = [int(x) for x in frozen["split"]["test"]]
    test_sids = [sid for sid in test_sids if sid in natural and sid in record_by_sid]

    if args.scope == "patched":
        selected_sids = [sid for sid in test_sids if not bool(natural[sid]["firststep_correct"])]
    else:
        selected_sids = list(test_sids)

    if not selected_sids:
        raise RuntimeError("No selected TEST samples.")

    directions = frozen["directions"]
    lambda_value = float(frozen["lambdas"]["final"])

    spec = base.SPECS[model_name]
    model_class = getattr(transformers, spec.model_class)
    model = processor = None

    try:
        print(f"Loading {model_name}: {spec.repo_id}", flush=True)
        model = model_class.from_pretrained(
            spec.repo_id,
            dtype=base.resolve_dtype(spec.dtype_name),
            low_cpu_mem_usage=True,
            trust_remote_code=spec.trust_remote_code,
            device_map={"": args.device},
            attn_implementation=args.attn_impl,
        )
        model.eval()
        helper.clear_sampling_defaults(model)

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)
        device = torch.device(args.device)

        decoder_layers, decoder_path = probe.resolve_decoder_layers(model)
        if len(decoder_layers) != 28:
            raise RuntimeError(f"Expected Qwen-7B 28 decoder layers, got {len(decoder_layers)}")
        for layer in layers:
            if not 1 <= layer < len(decoder_layers):
                raise ValueError(f"Bad trace layer L{layer}")

        block23 = decoder_layers[23]
        mlp23 = getattr(block23, "mlp", None)
        if not isinstance(mlp23, torch.nn.Module):
            raise RuntimeError("Could not resolve L23 block.mlp")

        final_norm, final_norm_path = helper.resolve_final_norm(model, decoder_path)
        if final_norm is None:
            raise RuntimeError("Could not resolve final norm")
        token_map = helper.relation_token_variants(processor.tokenizer)
        lens = helper.RelationLogitLens(
            model=model,
            final_norm=final_norm,
            token_map=token_map,
        )

        print("\n" + "=" * 200)
        print("QWEN-7B L23 REPAIR -> DOWNSTREAM SURVIVAL")
        print("=" * 200)
        print("repair dir       :", repair_dir)
        print("mode             :", args.mode)
        print("scope            :", args.scope)
        print("TEST total       :", len(test_sids))
        print("selected N       :", len(selected_sids))
        print("frozen lambda    :", lambda_value)
        print("layers           :", layers)
        print("final norm       :", final_norm_path)
        print("run generation   :", args.run_generation)
        print("=" * 200, flush=True)

        sample_rows: List[Dict[str, Any]] = []
        layer_rows: List[Dict[str, Any]] = []

        for index, sid in enumerate(
            tqdm(selected_sids, desc="L23-repair-downstream-survival"),
            start=1,
        ):
            record = record_by_sid[sid]
            clean_row = natural[sid]
            gt = helper.normalize_relation(record.relation)
            image = batch = None

            try:
                image, batch = build_batch_for_record(
                    record=record,
                    prompt_template=args.prompt_template,
                    helper=helper,
                    probe=probe,
                    processor=processor,
                    device=device,
                )

                # Clean trajectory.
                clean_final, clean_layers, _ = trace_trajectory(
                    ah=ah,
                    helper=helper,
                    model=model,
                    batch=batch,
                    token_map=token_map,
                    decoder_layers=decoder_layers,
                    layers=layers,
                    lens=lens,
                    gt=gt,
                    manager=contextlib.nullcontext(),
                )

                # Patched trajectory with the exact frozen intervention.
                patch_manager, patch_meta = make_patch_manager(
                    clean_row=clean_row,
                    mode=args.mode,
                    mlp_module=mlp23,
                    batch=batch,
                    directions=directions,
                    lambda_value=lambda_value,
                )
                patched_final, patched_layers, patch_stats = trace_trajectory(
                    ah=ah,
                    helper=helper,
                    model=model,
                    batch=batch,
                    token_map=token_map,
                    decoder_layers=decoder_layers,
                    layers=layers,
                    lens=lens,
                    gt=gt,
                    manager=patch_manager,
                )

                # Verify clean trajectory reproduces the frozen natural grouping.
                frozen_clean_pred = str(clean_row["firststep_pred"])
                if clean_final["final_pred"] != frozen_clean_pred:
                    raise RuntimeError(
                        f"sid={sid}: clean final pred changed relative to frozen repair run: "
                        f"now={clean_final['final_pred']} frozen={frozen_clean_pred}"
                    )

                local_l23_repair = (
                    (not bool(clean_layers[23]["y"]["correct"]))
                    and bool(patched_layers[23]["y"]["correct"])
                )
                patched_y23_correct = bool(patched_layers[23]["y"]["correct"])
                final_firststep_repair = (
                    (not bool(clean_final["final_correct"]))
                    and bool(patched_final["final_correct"])
                )
                first_loss_layer, first_loss_module = first_downstream_loss(
                    patched_layers,
                    layers,
                )

                clean_gen_pred = clean_gen_text = None
                patched_gen_pred = patched_gen_text = None
                clean_gen_correct = patched_gen_correct = None

                if args.run_generation:
                    clean_gen_pred, clean_gen_text, _ = greedy_generation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        helper=helper,
                        max_new_tokens=args.max_new_tokens,
                        manager=contextlib.nullcontext(),
                    )
                    gen_manager, _ = make_patch_manager(
                        clean_row=clean_row,
                        mode=args.mode,
                        mlp_module=mlp23,
                        batch=batch,
                        directions=directions,
                        lambda_value=lambda_value,
                    )
                    patched_gen_pred, patched_gen_text, _ = greedy_generation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        helper=helper,
                        max_new_tokens=args.max_new_tokens,
                        manager=gen_manager,
                    )
                    clean_gen_correct = clean_gen_pred == gt
                    patched_gen_correct = patched_gen_pred == gt

                generation_repair = bool(
                    args.run_generation
                    and (not bool(clean_gen_correct))
                    and bool(patched_gen_correct)
                )

                srow = {
                    "sid": sid,
                    "gt": gt,
                    "subject": record.subject,
                    "reference": record.reference,
                    "patched": bool(patch_meta["patched"]),
                    "selected_relation": patch_meta["selected_relation"],
                    "lambda": patch_meta["lambda"],
                    "patch_delta_norm": patch_stats.get("delta_norm", float("nan")),
                    "patch_scalar": patch_stats.get("correction_scalar", float("nan")),
                    "clean_final_firststep_pred": clean_final["final_pred"],
                    "clean_final_firststep_correct": clean_final["final_correct"],
                    "patched_final_firststep_pred": patched_final["final_pred"],
                    "patched_final_firststep_correct": patched_final["final_correct"],
                    "final_firststep_repair": final_firststep_repair,
                    "clean_y23_pred": clean_layers[23]["y"]["pred"],
                    "clean_y23_correct": clean_layers[23]["y"]["correct"],
                    "patched_y23_pred": patched_layers[23]["y"]["pred"],
                    "patched_y23_correct": patched_y23_correct,
                    "local_l23_repair": local_l23_repair,
                    "first_downstream_loss_layer": first_loss_layer,
                    "first_downstream_loss_module": first_loss_module,
                    "clean_generation_pred": clean_gen_pred,
                    "clean_generation_correct": clean_gen_correct,
                    "clean_generation_text": clean_gen_text,
                    "patched_generation_pred": patched_gen_pred,
                    "patched_generation_correct": patched_gen_correct,
                    "patched_generation_text": patched_gen_text,
                    "generation_repair": generation_repair,
                }
                sample_rows.append(srow)

                for layer in layers:
                    c = clean_layers[layer]
                    p = patched_layers[layer]
                    layer_rows.append({
                        "sid": sid,
                        "gt": gt,
                        "layer": int(layer),
                        "local_l23_repair": local_l23_repair,
                        "final_firststep_repair": final_firststep_repair,
                        "generation_repair": generation_repair,
                        "clean_x_pred": c["x"]["pred"],
                        "clean_x_correct": c["x"]["correct"],
                        "clean_x_margin": c["x"]["decision_margin"],
                        "clean_r_pred": c["r"]["pred"],
                        "clean_r_correct": c["r"]["correct"],
                        "clean_r_margin": c["r"]["decision_margin"],
                        "clean_y_pred": c["y"]["pred"],
                        "clean_y_correct": c["y"]["correct"],
                        "clean_y_margin": c["y"]["decision_margin"],
                        "clean_attention_gain": c["attention_gain"],
                        "clean_mlp_gain": c["mlp_gain"],
                        "patched_x_pred": p["x"]["pred"],
                        "patched_x_correct": p["x"]["correct"],
                        "patched_x_margin": p["x"]["decision_margin"],
                        "patched_r_pred": p["r"]["pred"],
                        "patched_r_correct": p["r"]["correct"],
                        "patched_r_margin": p["r"]["decision_margin"],
                        "patched_y_pred": p["y"]["pred"],
                        "patched_y_correct": p["y"]["correct"],
                        "patched_y_margin": p["y"]["decision_margin"],
                        "patched_attention_gain": p["attention_gain"],
                        "patched_mlp_gain": p["mlp_gain"],
                        "delta_x_margin": p["x"]["decision_margin"] - c["x"]["decision_margin"],
                        "delta_r_margin": p["r"]["decision_margin"] - c["r"]["decision_margin"],
                        "delta_y_margin": p["y"]["decision_margin"] - c["y"]["decision_margin"],
                    })

                append_jsonl(output_dir / "sample_results.jsonl", srow)

            except Exception as exc:
                append_jsonl(errors_path, {
                    "phase": "sample",
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                if args.fail_fast:
                    raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                if batch is not None:
                    del batch
                gc.collect()
                if (
                    torch.cuda.is_available()
                    and args.empty_cache_every > 0
                    and index % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        write_csv(output_dir / "sample_summary.csv", sample_rows)
        write_csv(output_dir / "layer_trajectory.csv", layer_rows)

        group_layer_summary = summarize_group_layers(sample_rows, layer_rows)
        write_csv(output_dir / "group_layer_summary.csv", group_layer_summary)

        survival_rows = build_survival_rows(sample_rows, layer_rows, layers)
        write_csv(output_dir / "repair_survival.csv", survival_rows)

        # First downstream loss histogram for trajectories that are correct at patched L23.
        loss_counter: Counter = Counter()
        local_loss_counter: Counter = Counter()
        for row in sample_rows:
            if bool(row["patched_y23_correct"]):
                loss_counter[(
                    str(row["first_downstream_loss_layer"]),
                    str(row["first_downstream_loss_module"]),
                )] += 1
            if bool(row["local_l23_repair"]):
                local_loss_counter[(
                    str(row["first_downstream_loss_layer"]),
                    str(row["first_downstream_loss_module"]),
                )] += 1

        loss_rows: List[Dict[str, Any]] = []
        for cohort_name, counter in (
            ("patched_y23_correct", loss_counter),
            ("local_l23_repair", local_loss_counter),
        ):
            total = sum(counter.values())
            for (layer_name, module_name), count in sorted(counter.items()):
                loss_rows.append({
                    "cohort": cohort_name,
                    "first_loss_layer": layer_name,
                    "first_loss_module": module_name,
                    "N": count,
                    "fraction": count / max(total, 1),
                })
        write_csv(output_dir / "downstream_loss_summary.csv", loss_rows)

        N = len(sample_rows)
        N_patched = sum(int(bool(r["patched"])) for r in sample_rows)
        N_local = sum(int(bool(r["local_l23_repair"])) for r in sample_rows)
        N_final_repair = sum(int(bool(r["final_firststep_repair"])) for r in sample_rows)
        N_gen_repair = sum(int(bool(r.get("generation_repair", False))) for r in sample_rows)
        N_y23_correct = sum(int(bool(r["patched_y23_correct"])) for r in sample_rows)
        N_local_survive = sum(
            int(
                bool(r["local_l23_repair"])
                and str(r["first_downstream_loss_layer"]) == "none_through_last"
            )
            for r in sample_rows
        )

        print("\n" + "=" * 200)
        print("DOWNSTREAM SURVIVAL SUMMARY")
        print("=" * 200)
        print(f"N analyzed                         = {N}")
        print(f"N actually patched                = {N_patched}")
        print(f"patched y23 correct               = {N_y23_correct}")
        print(f"clean y23 W -> patched y23 C      = {N_local}")
        print(f"local L23 repairs survive to L{max(layers)} = {N_local_survive}/{N_local} "
              f"({100*N_local_survive/max(N_local,1):.2f}%)")
        print(f"final restricted first-step W->C  = {N_final_repair}")
        if args.run_generation:
            print(f"full generation W->C              = {N_gen_repair}")

        print("\nREPAIR SURVIVAL")
        for cohort in ("local_l23_repair", "patched_y23_correct", "final_firststep_repair"):
            rows = [r for r in survival_rows if r["cohort"] == cohort]
            if not rows or int(rows[0]["N"]) == 0:
                print(f"  {cohort:<24s} N=0")
                continue
            parts = [
                f"L{int(r['layer'])}:{100*float(r['patched_y_survival']):.1f}%"
                for r in rows
            ]
            print(f"  {cohort:<24s} N={int(rows[0]['N']):3d} | " + " ".join(parts))

        print("\nFIRST DOWNSTREAM LOSS AFTER PATCHED-CORRECT L23")
        for cohort in ("local_l23_repair", "patched_y23_correct"):
            rows = [r for r in loss_rows if r["cohort"] == cohort]
            print(f"  {cohort}:")
            if not rows:
                print("    N=0")
            for r in rows:
                print(
                    f"    {r['first_loss_layer']:<18s} {r['first_loss_module']:<28s} "
                    f"N={int(r['N']):3d} ({100*float(r['fraction']):6.2f}%)"
                )

        if args.run_generation:
            print("\nFINAL-FIRSTSTEP REPAIRS: GENERATION OUTCOME")
            ff = [r for r in sample_rows if bool(r["final_firststep_repair"])]
            both = sum(int(bool(r["patched_generation_correct"])) for r in ff)
            print(
                f"  restricted final W->C samples = {len(ff)}, "
                f"generation correct among them = {both}/{len(ff)}"
            )
            for r in ff:
                print(
                    f"    sid={int(r['sid']):4d} gt={r['gt']:<5s} "
                    f"L23 clean/patched={r['clean_y23_pred']}/{r['patched_y23_pred']} "
                    f"loss={r['first_downstream_loss_layer']}:{r['first_downstream_loss_module']} "
                    f"gen={r['patched_generation_pred']} text={r['patched_generation_text']!r}"
                )

        print("=" * 200)

        # Compact report.
        report = [
            f"script_version: {SCRIPT_VERSION}",
            f"repair_dir: {repair_dir}",
            f"mode: {args.mode}",
            f"scope: {args.scope}",
            f"N={N}",
            f"N_patched={N_patched}",
            f"N_local_l23_repair={N_local}",
            f"N_local_survive_to_L{max(layers)}={N_local_survive}",
            f"N_final_restricted_firststep_repair={N_final_repair}",
            f"N_generation_repair={N_gen_repair if args.run_generation else 'N/A'}",
            "",
            "INTERPRETATION RULE:",
            (
                "If local_l23_repair survival drops sharply at one later layer/module, "
                "that is evidence for a second sequential failure site."
            ),
            (
                "If local repairs survive through L27 but generation remains wrong, "
                "do not add another relation-space repair yet; inspect full-vocabulary "
                "GT rank / top-1 emission and autoregressive decoding instead."
            ),
        ]
        (output_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

        config = {
            "script_version": SCRIPT_VERSION,
            "repair_dir": str(repair_dir),
            "frozen_repair_script_version": old_config.get("script_version"),
            "model": model_name,
            "dataset": dataset,
            "mode": args.mode,
            "scope": args.scope,
            "layers": layers,
            "lambda_final": lambda_value,
            "N_test_original": len(test_sids),
            "N_selected": len(selected_sids),
            "N_analyzed": N,
            "run_generation": args.run_generation,
            "final_norm_path": final_norm_path,
            "audit": audit,
        }
        (output_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\nSaved:")
        for name in (
            "sample_summary.csv",
            "layer_trajectory.csv",
            "group_layer_summary.csv",
            "repair_survival.csv",
            "downstream_loss_summary.csv",
            "sample_results.jsonl",
            "config.json",
            "report.txt",
            "errors.jsonl",
        ):
            path = output_dir / name
            if path.exists():
                print(" ", path)

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
