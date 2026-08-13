#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cumulative prefix-bundle test for the top 6 receiver heads from the single-head scan.

Prefixes:
  K2 = L26H04 + L26H06
  K3 = K2 + L26H07
  K4 = K3 + L26H11
  K5 = K4 + L24H00
  K6 = K5 + L25H13

Each head uses its CLEAN sample-specific natural object->prompt-last message:

    c_Lh = W_O^h sum_{s in object} A_h[last,s] V_h[s]

For each prefix and scale:

    attention_out'_L[last]
      = attention_out_L[last]
      + scale * sum_{h in prefix at layer L} c_Lh(clean)

Then the script runs full greedy model.generate() and reports true generation ACC,
W->C, C->W, net repair, and changed rate.

For K5/K6 this is a FIXED-CLEAN cross-layer test:
L24/L25 messages are added upstream, while the L26 messages are still the clean
cached messages from the unmodified forward. This keeps the intervention definition
directly comparable with the prior single-head scan. It is not yet an online
recomputed-message experiment.

This v1.1 file is standalone and does NOT require the previous
single-head scan script as an import.

Recommended:
CUDA_VISIBLE_DEVICES=0 python -u eval_coco_receiver_prefix_bundles_v1.py \
  --decomposition-dir output/qwen3b_l26_block_decomposition_v1_1 \
  --model qwen-3b \
  --scale 6 \
  --device cuda:0 \
  --output-dir output/qwen3b_receiver_prefix_bundles_top6_s6_v1 \
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

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


# =============================================================================
# Standalone helpers formerly imported from the single-head scan
# =============================================================================

RELATIONS = ("left", "right", "above", "below")


def _hname(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head):02d}"


def _normalize_relation(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in RELATIONS:
        return text

    hits = []
    for relation in RELATIONS:
        match = re.search(rf"\b{re.escape(relation)}\b", text)
        if match:
            hits.append((match.start(), relation))

    for pattern, relation in (
        (r"\bunder(?:neath)?\b|\bbeneath\b", "below"),
        (r"\bover\b|\bon top\b", "above"),
    ):
        match = re.search(pattern, text)
        if match:
            hits.append((match.start(), relation))

    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "1", "true", "yes", "y", "t"
    }


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
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


def _append_jsonl(
    path: Path,
    row: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(row), ensure_ascii=False) + "\n"
        )
        handle.flush()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_mean(values: Iterable[Any]) -> float:
    xs = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)
    return float(np.mean(xs)) if xs else float("nan")


def _clear_sampling_defaults(model: Any) -> None:
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return

    for name in ("temperature", "top_p", "top_k"):
        if hasattr(cfg, name):
            setattr(cfg, name, None)


def _relation_token_variants(
    tokenizer: Any,
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}

    for relation in RELATIONS:
        ids = set()
        for candidate in (
            relation,
            " " + relation,
            relation.capitalize(),
            " " + relation.capitalize(),
        ):
            token_ids = tokenizer.encode(
                candidate,
                add_special_tokens=False,
            )
            if len(token_ids) == 1:
                ids.add(int(token_ids[0]))

        if not ids:
            token_ids = tokenizer.encode(
                " " + relation,
                add_special_tokens=False,
            )
            if not token_ids:
                raise RuntimeError(
                    f"No token ID for relation {relation!r}"
                )
            ids.add(int(token_ids[-1]))

        out[relation] = sorted(ids)

    return out


def _deterministic_stratified_subset(
    rows: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in rows]

    if limit <= 0 or len(rows) <= limit:
        return rows

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gt = _normalize_relation(row.get("gt"))
        if gt in RELATIONS:
            grouped[gt].append(row)

    rng = random.Random(seed)
    for relation in RELATIONS:
        rng.shuffle(grouped[relation])

    cursors = {relation: 0 for relation in RELATIONS}
    selected: List[Dict[str, Any]] = []

    while len(selected) < limit:
        moved = False
        for relation in RELATIONS:
            group = grouped[relation]
            cursor = cursors[relation]
            if cursor < len(group) and len(selected) < limit:
                selected.append(group[cursor])
                cursors[relation] += 1
                moved = True
        if not moved:
            break

    selected.sort(key=lambda row: int(row["sid"]))
    return selected


def _first_3d(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        if output.ndim != 3:
            raise RuntimeError(
                "Expected attention output [B,S,D], "
                f"got {tuple(output.shape)}"
            )
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item

    raise RuntimeError(
        "Could not find 3D attention output"
    )


def _replace_first_3d(
    output: Any,
    replacement: torch.Tensor,
) -> Any:
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

    raise RuntimeError(
        "Could not replace 3D attention output"
    )


def _trace_target_index(
    trace: Any,
    prompt_last: int,
) -> int:
    lookup = {
        int(global_position): local
        for local, global_position
        in enumerate(trace.target_positions)
    }

    if int(prompt_last) not in lookup:
        raise RuntimeError(
            f"prompt_last={prompt_last} missing from "
            f"trace targets {trace.target_positions}"
        )

    return int(lookup[int(prompt_last)])


def _all_head_object_writes(
    *,
    trace: Any,
    prompt_last: int,
    object_positions: Sequence[int],
) -> np.ndarray:
    """
    Clean post-W_O object-text -> prompt-last write for every query head.

    Returns:
        [n_query_heads, hidden_size]
    """
    object_positions = sorted(
        set(map(int, object_positions))
    )
    if not object_positions:
        raise RuntimeError(
            "No object source positions"
        )

    local = _trace_target_index(
        trace,
        prompt_last,
    )

    source = torch.as_tensor(
        object_positions,
        dtype=torch.long,
    )

    if int(source.max()) >= int(trace.value_states.shape[1]):
        raise RuntimeError(
            f"Object source position {int(source.max())} "
            f">= source length {trace.value_states.shape[1]}"
        )

    # Query-head-resolved attention probabilities.
    weights = (
        trace.attention_weights[:, local, :]
        .index_select(1, source)
        .float()
    )  # [Hq, Sobj]

    # Trace helper expands/shared GQA values in the same head convention
    # used by the previous working single-head scan.
    values = (
        trace.value_states
        .index_select(1, source)
        .float()
    )  # [Hq, Sobj, Dh]

    pre = torch.einsum(
        "hs,hsd->hd",
        weights,
        values,
    )  # [Hq, Dh]

    # [Dmodel, Hq, Dh]
    post = torch.einsum(
        "hd,ohd->ho",
        pre,
        trace.o_proj_weight.float(),
    )  # [Hq, Dmodel]

    return (
        post.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def _extract_clean_messages(
    *,
    attention_helper: Any,
    model: Any,
    batch: Mapping[str, Any],
    relation_token_map: Mapping[str, Sequence[int]],
    decoder_layers: Sequence[Any],
    layers: Sequence[int],
    prompt_last: int,
    object_positions: Sequence[int],
    chunk_size: int,
) -> Tuple[
    Dict[int, np.ndarray],
    Dict[int, float],
]:
    """
    One clean trace -> per-layer [H,D] object->last natural messages.
    """
    all_messages: Dict[int, np.ndarray] = {}
    replay_errors: Dict[int, float] = {}
    layers = list(map(int, layers))

    if chunk_size <= 0:
        chunks = [layers]
    else:
        chunks = [
            layers[start : start + int(chunk_size)]
            for start in range(
                0,
                len(layers),
                int(chunk_size),
            )
        ]

    for chunk in chunks:
        _, traces = attention_helper.run_and_trace(
            model=model,
            batch=batch,
            token_map=relation_token_map,
            decoder_layers=decoder_layers,
            layer_indices=chunk,
            target_positions=[prompt_last],
        )

        for layer in chunk:
            trace = traces[int(layer)]

            all_messages[int(layer)] = (
                _all_head_object_writes(
                    trace=trace,
                    prompt_last=prompt_last,
                    object_positions=object_positions,
                )
            )

            replay_errors[int(layer)] = float(
                trace.replay_relative_error
            )

        del traces

    return all_messages, replay_errors


def _build_batch(
    *,
    probe: Any,
    processor: Any,
    question: str,
    image: Image.Image,
    device: torch.device,
) -> Any:
    rendered = probe.build_chat_prompt(
        processor,
        question,
        True,
    )

    return probe.process_inputs(
        processor,
        rendered,
        image,
        device,
    )


class _StandaloneScanCompat:
    # constants
    RELATIONS = RELATIONS

    # generic helpers
    hname = staticmethod(_hname)
    normalize_relation = staticmethod(_normalize_relation)
    parse_bool = staticmethod(_parse_bool)
    read_csv = staticmethod(_read_csv)
    write_csv = staticmethod(_write_csv)
    append_jsonl = staticmethod(_append_jsonl)
    load_jsonl = staticmethod(_load_jsonl)
    safe_mean = staticmethod(_safe_mean)
    clear_sampling_defaults = staticmethod(
        _clear_sampling_defaults
    )
    relation_token_variants = staticmethod(
        _relation_token_variants
    )
    deterministic_stratified_subset = staticmethod(
        _deterministic_stratified_subset
    )

    # attention / tracing helpers
    first_3d = staticmethod(_first_3d)
    replace_first_3d = staticmethod(_replace_first_3d)
    extract_clean_messages = staticmethod(
        _extract_clean_messages
    )
    build_batch = staticmethod(_build_batch)


scan = _StandaloneScanCompat()


SCRIPT_VERSION = "coco-receiver-prefix-bundles-v1.1"
RANKED_HEADS: List[Tuple[int, int]] = [
    (26, 4),
    (26, 6),
    (26, 7),
    (26, 11),
    (24, 0),
    (25, 13),
]
PREFIX_KS = (2, 3, 4, 5, 6)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--decomposition-dir", required=True)
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument("--scale", type=float, default=6.0)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--trace-chunk-size", type=int, default=0)
    p.add_argument(
        "--probe-module",
        default="analyze_coco_head_object_residual_direction_probe_v1",
    )
    p.add_argument(
        "--attention-helper-module",
        default="analyze_coco_flip_attention_spatial_vectors_v1",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--empty-cache-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def bundle_name(k: int) -> str:
    return "+".join(scan.hname(l, h) for l, h in RANKED_HEADS[:k])


def completion_key(sid: int, k: int, scale: float) -> Tuple[int, int, float]:
    return int(sid), int(k), round(float(scale), 9)


def first_3d(output: Any) -> torch.Tensor:
    return scan.first_3d(output)


def replace_first_3d(output: Any, replacement: torch.Tensor) -> Any:
    return scan.replace_first_3d(output, replacement)


class MultiLayerPromptLastDelta:
    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        prompt_length: int,
        prompt_last: int,
        deltas_by_layer: Mapping[int, np.ndarray],
    ) -> None:
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.prompt_length = int(prompt_length)
        self.prompt_last = int(prompt_last)
        self.deltas_by_layer = {
            int(layer): np.asarray(delta, dtype=np.float32)
            for layer, delta in deltas_by_layer.items()
        }
        self.handles = []
        self.applications = {layer: 0 for layer in self.deltas_by_layer}

    def __enter__(self) -> "MultiLayerPromptLastDelta":
        for layer, delta in self.deltas_by_layer.items():
            attention = self.attention_helper.resolve_self_attention(
                self.decoder_layers[layer]
            )

            def make_hook(layer_index: int, vector_np: np.ndarray):
                def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                    hidden = first_3d(output)

                    # Prefill only. Cached decode calls normally have q_len=1.
                    if int(hidden.shape[1]) != self.prompt_length:
                        return None

                    if self.prompt_last >= int(hidden.shape[1]):
                        raise RuntimeError(
                            f"L{layer_index}: prompt_last outside sequence"
                        )
                    if int(hidden.shape[-1]) != int(vector_np.shape[0]):
                        raise RuntimeError(
                            f"L{layer_index}: delta dim {vector_np.shape[0]} "
                            f"!= attention output dim {hidden.shape[-1]}"
                        )

                    modified = hidden.clone()
                    modified[0, self.prompt_last] += torch.as_tensor(
                        vector_np,
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                    self.applications[layer_index] += 1
                    return replace_first_3d(output, modified)

                return hook

            self.handles.append(
                attention.register_forward_hook(make_hook(layer, delta))
            )
        return self

    def validate(self) -> None:
        bad = {layer: n for layer, n in self.applications.items() if n != 1}
        if bad:
            raise RuntimeError(f"Expected one prefill patch per layer; got {bad}")

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()


@torch.inference_mode()
def patched_generation(
    *,
    model: Any,
    processor: Any,
    batch: Any,
    decoder_layers: Sequence[Any],
    attention_helper: Any,
    prompt_last: int,
    deltas_by_layer: Mapping[int, np.ndarray],
    max_new_tokens: int,
) -> Tuple[Optional[str], str]:
    prompt_length = int(batch["input_ids"].shape[1])

    with MultiLayerPromptLastDelta(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        prompt_length=prompt_length,
        prompt_last=prompt_last,
        deltas_by_layer=deltas_by_layer,
    ) as patch:
        output_ids = model.generate(
            **batch,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )
        patch.validate()

    text = processor.tokenizer.decode(
        output_ids[0, prompt_length:],
        skip_special_tokens=True,
    ).strip()
    del output_ids
    return scan.normalize_relation(text), text


def build_deltas(
    clean_messages: Mapping[int, np.ndarray],
    k: int,
    scale: float,
) -> Dict[int, np.ndarray]:
    grouped: Dict[int, List[np.ndarray]] = defaultdict(list)
    for layer, head in RANKED_HEADS[:k]:
        grouped[layer].append(
            np.asarray(clean_messages[layer][head], dtype=np.float32)
        )

    return {
        layer: float(scale) * np.stack(vectors, axis=0).sum(axis=0)
        for layer, vectors in grouped.items()
    }


def summarize(
    baseline_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    base_by_sid = {int(row["sid"]): row for row in baseline_rows}
    out = []

    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in patch_rows:
        grouped[int(row["k"])].append(row)

    for k in PREFIX_KS:
        rows = grouped.get(k, [])
        by_sid = {int(row["sid"]): row for row in rows}
        covered = [sid for sid in base_by_sid if sid in by_sid]
        if not covered:
            continue

        w2c = c2w = changed = 0
        patch_correct = []

        for sid in covered:
            base = base_by_sid[sid]
            gt = scan.normalize_relation(base["gt"])
            base_pred = scan.normalize_relation(base["generation_prediction"])
            new_pred = scan.normalize_relation(
                by_sid[sid]["patched_generation_prediction"]
            )
            base_correct = scan.parse_bool(base["generation_correct"])
            new_correct = new_pred == gt

            patch_correct.append(float(new_correct))
            w2c += int((not base_correct) and new_correct)
            c2w += int(base_correct and (not new_correct))
            changed += int(new_pred != base_pred)

        covered_base_acc = scan.safe_mean(
            float(scan.parse_bool(base_by_sid[sid]["generation_correct"]))
            for sid in covered
        )
        patched_acc = scan.safe_mean(patch_correct)

        out.append({
            "k": k,
            "bundle": bundle_name(k),
            "scale": float(rows[0]["scale"]),
            "N_expected": len(base_by_sid),
            "N_completed": len(covered),
            "complete": len(covered) == len(base_by_sid),
            "baseline_acc": covered_base_acc,
            "patched_acc": patched_acc,
            "delta_acc": patched_acc - covered_base_acc,
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net_repairs": w2c - c2w,
            "generation_changed": changed,
            "generation_changed_rate": changed / max(len(covered), 1),
            "mean_total_delta_norm": scan.safe_mean(
                row["total_delta_norm"] for row in rows
            ),
        })

    return out


def relation_summary(
    baseline_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    base_by_sid = {int(row["sid"]): row for row in baseline_rows}
    grouped: Dict[int, Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in patch_rows:
        grouped[int(row["k"])][int(row["sid"])] = row

    out = []
    for k in PREFIX_KS:
        by_sid = grouped.get(k, {})
        for relation in scan.RELATIONS:
            sids = [
                sid
                for sid, base in base_by_sid.items()
                if scan.normalize_relation(base["gt"]) == relation
                and sid in by_sid
            ]
            if not sids:
                continue

            base_acc = scan.safe_mean(
                float(
                    scan.normalize_relation(
                        base_by_sid[sid]["generation_prediction"]
                    ) == relation
                )
                for sid in sids
            )
            patch_acc = scan.safe_mean(
                float(
                    scan.normalize_relation(
                        by_sid[sid]["patched_generation_prediction"]
                    ) == relation
                )
                for sid in sids
            )
            out.append({
                "k": k,
                "bundle": bundle_name(k),
                "relation": relation,
                "N": len(sids),
                "baseline_acc": base_acc,
                "patched_acc": patch_acc,
                "delta_acc": patch_acc - base_acc,
            })

    return out


def main() -> None:
    args = parse_args()

    if args.scale < 0:
        raise ValueError("--scale must be >= 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    decomposition_dir = Path(args.decomposition_dir)
    config_path = decomposition_dir / "config.json"
    baseline_path = decomposition_dir / "baseline_eval.csv"

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)

    decomp_config = json.loads(config_path.read_text(encoding="utf-8"))
    baseline_rows = scan.read_csv(baseline_path)
    baseline_rows = scan.deterministic_stratified_subset(
        baseline_rows,
        args.max_eval_samples,
        args.seed,
    )
    baseline_rows.sort(key=lambda row: int(row["sid"]))

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_jsonl = output_dir / "patch_results.jsonl"
    errors_path = output_dir / "errors.jsonl"

    if (
        not args.overwrite
        and not args.resume
        and any(output_dir.iterdir())
    ):
        raise RuntimeError(
            f"{output_dir} is non-empty; use --overwrite or --resume."
        )

    prior_rows = scan.load_jsonl(patch_jsonl) if args.resume else []
    completed = {
        completion_key(
            int(row["sid"]),
            int(row["k"]),
            float(row["scale"]),
        )
        for row in prior_rows
    }

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
    record_by_sid = {int(record.sid): record for record in records}

    prompt_template = str(
        decomp_config.get("prompt_template")
        or (
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        )
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
        print(f"Loading {args.model}: {spec.repo_id}", flush=True)
        model = model_class.from_pretrained(spec.repo_id, **load_kwargs)
        model.eval()
        scan.clear_sampling_defaults(model)

        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        base.configure_processor(model, processor)

        device = torch.device(args.device)
        decoder_layers, _ = probe.resolve_decoder_layers(model)
        selected_layers = sorted({layer for layer, _ in RANKED_HEADS})

        relation_token_map = scan.relation_token_variants(
            processor.tokenizer
        )

        baseline_acc = scan.safe_mean(
            float(scan.parse_bool(row["generation_correct"]))
            for row in baseline_rows
        )

        print("\n" + "=" * 170)
        print("TOP-6 PREFIX RECEIVER BUNDLES")
        print("=" * 170)
        for rank, (layer, head) in enumerate(RANKED_HEADS, start=1):
            print(f"rank {rank}: {scan.hname(layer, head)}")
        print("prefixes :", PREFIX_KS)
        print("scale    :", args.scale)
        print("N eval   :", len(baseline_rows))
        print("baseline :", f"{100*baseline_acc:.2f}%")
        print("mode     : fixed-clean cross-layer messages")
        print("=" * 170)

        for sample_i, base_row in enumerate(
            tqdm(baseline_rows, desc="prefix-bundles"),
            start=1,
        ):
            sid = int(base_row["sid"])
            image = None
            batch = None

            try:
                needed_ks = [
                    k
                    for k in PREFIX_KS
                    if completion_key(sid, k, args.scale) not in completed
                ]
                if not needed_ks:
                    continue

                if sid not in record_by_sid:
                    raise RuntimeError(f"SID {sid} missing from dataset")

                record = record_by_sid[sid]
                gt = scan.normalize_relation(base_row["gt"])
                base_pred = scan.normalize_relation(
                    base_row["generation_prediction"]
                )
                base_correct = scan.parse_bool(
                    base_row["generation_correct"]
                )

                question = prompt_template.format(
                    subject=record.subject,
                    reference=record.reference,
                )
                image = Image.open(record.image_path).convert("RGB")
                batch = scan.build_batch(
                    probe=probe,
                    processor=processor,
                    question=question,
                    image=image,
                    device=device,
                )

                input_ids = [
                    int(x)
                    for x in batch["input_ids"][0].detach().cpu().tolist()
                ]
                prompt_last = len(input_ids) - 1

                sub_pos = probe.locate_phrase_positions(
                    processor.tokenizer,
                    input_ids,
                    str(record.subject),
                )
                ref_pos = probe.locate_phrase_positions(
                    processor.tokenizer,
                    input_ids,
                    str(record.reference),
                )
                object_positions = sorted(set(map(int, sub_pos + ref_pos)))
                if not object_positions:
                    raise RuntimeError(f"SID {sid}: no object positions")

                clean_messages, replay_errors = scan.extract_clean_messages(
                    attention_helper=attention_helper,
                    model=model,
                    batch=batch,
                    relation_token_map=relation_token_map,
                    decoder_layers=decoder_layers,
                    layers=selected_layers,
                    prompt_last=prompt_last,
                    object_positions=object_positions,
                    chunk_size=args.trace_chunk_size,
                )

                for k in needed_ks:
                    deltas = build_deltas(
                        clean_messages,
                        k,
                        args.scale,
                    )

                    patched_pred, patched_text = patched_generation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        decoder_layers=decoder_layers,
                        attention_helper=attention_helper,
                        prompt_last=prompt_last,
                        deltas_by_layer=deltas,
                        max_new_tokens=args.max_new_tokens,
                    )

                    patched_correct = patched_pred == gt
                    total_delta_norm = math.sqrt(
                        sum(
                            float(np.linalg.norm(delta)) ** 2
                            for delta in deltas.values()
                        )
                    )

                    row = {
                        "sid": sid,
                        "k": k,
                        "bundle": bundle_name(k),
                        "scale": float(args.scale),
                        "gt": gt,
                        "baseline_generation_prediction": base_pred,
                        "baseline_generation_correct": base_correct,
                        "patched_generation_prediction": patched_pred,
                        "patched_generation_text": patched_text,
                        "patched_generation_correct": patched_correct,
                        "wrong_to_correct": (
                            (not base_correct) and patched_correct
                        ),
                        "correct_to_wrong": (
                            base_correct and (not patched_correct)
                        ),
                        "generation_changed": patched_pred != base_pred,
                        "patched_layers": ",".join(
                            f"L{layer}" for layer in sorted(deltas)
                        ),
                        "n_patched_layers": len(deltas),
                        "total_delta_norm": float(total_delta_norm),
                        "mean_replay_error": scan.safe_mean(
                            replay_errors[layer] for layer in deltas
                        ),
                    }
                    scan.append_jsonl(patch_jsonl, row)
                    prior_rows.append(row)
                    completed.add(
                        completion_key(sid, k, args.scale)
                    )

                del clean_messages

            except Exception as exc:
                scan.append_jsonl(
                    errors_path,
                    {
                        "sid": sid,
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
                    and sample_i % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

        patch_rows = scan.load_jsonl(patch_jsonl)
        scan.write_csv(output_dir / "patch_results.csv", patch_rows)

        summary_rows = summarize(baseline_rows, patch_rows)
        scan.write_csv(output_dir / "summary.csv", summary_rows)

        rel_rows = relation_summary(baseline_rows, patch_rows)
        scan.write_csv(output_dir / "relation_summary.csv", rel_rows)

        print("\n" + "=" * 180)
        print("PREFIX BUNDLE SUMMARY")
        print("=" * 180)
        print(
            f"{'K':>3s} {'bundle':<70s} {'N':>7s} "
            f"{'baseACC':>9s} {'patchACC':>9s} {'delta':>8s} "
            f"{'W->C':>5s} {'C->W':>5s} {'net':>5s} {'changed':>8s}"
        )
        print("-" * 180)
        for row in summary_rows:
            print(
                f"{int(row['k']):>3d} "
                f"{str(row['bundle']):<70s} "
                f"{int(row['N_completed']):>3d}/{int(row['N_expected']):<3d} "
                f"{100*float(row['baseline_acc']):>8.2f}% "
                f"{100*float(row['patched_acc']):>8.2f}% "
                f"{100*float(row['delta_acc']):>+7.2f} "
                f"{int(row['wrong_to_correct']):>5d} "
                f"{int(row['correct_to_wrong']):>5d} "
                f"{int(row['net_repairs']):>+5d} "
                f"{100*float(row['generation_changed_rate']):>7.2f}%"
            )
        print("=" * 180)

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "decomposition_dir": str(decomposition_dir),
            "ranked_heads": [
                scan.hname(layer, head)
                for layer, head in RANKED_HEADS
            ],
            "prefix_ks": list(PREFIX_KS),
            "scale": args.scale,
            "N_eval": len(baseline_rows),
            "baseline_acc": baseline_acc,
            "uses_eval_gt_for_patch": False,
            "generation_metric": "full greedy model.generate()",
            "cross_layer_mode": "fixed clean messages",
            "dataset_audit": audit,
        }
        (output_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        lines = [
            f"baseline ACC: {100*baseline_acc:.2f}%",
            f"scale: {args.scale}",
            "",
        ]
        for row in summary_rows:
            lines.append(
                f"K{row['k']} {row['bundle']}: "
                f"{100*row['baseline_acc']:.2f}% -> "
                f"{100*row['patched_acc']:.2f}% "
                f"(delta={100*row['delta_acc']:+.2f}pp, "
                f"W->C={row['wrong_to_correct']}, "
                f"C->W={row['correct_to_wrong']}, "
                f"net={row['net_repairs']:+d})"
            )
        lines += [
            "",
            "K2 tests whether H4+H6 already captures most of the L26 bundle gain.",
            "K3/K4 add L26H7/H11 one at a time.",
            "K5/K6 then add L24H0 and L25H13 using fixed clean cross-layer messages.",
        ]
        (output_dir / "report.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        print("\nSaved:", output_dir)

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
