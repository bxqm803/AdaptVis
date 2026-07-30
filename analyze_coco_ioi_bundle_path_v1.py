#!/usr/bin/env python3
"""
Joint IOI-style sender-bundle -> receiver Q/K/V path patching for COCO two-object
spatial-relation query-swap experiments.

This script complements analyze_coco_ioi_backward_circuit_v1.py. The upstream
phase in that script patches one sender at a time. Here, every attention head in
one named bundle is patched simultaneously at the selected sender positions;
all non-bundle attention outputs at those positions are frozen to the clean
original activations. Token-wise residual/MLP computation is recomputed. The
resulting receiver Q/K/V projection state is captured and patched into a normal
clean run, so the final effect isolates the joint sender-bundle -> receiver
channel path through residual and MLPs only.

Bundle JSON format:
{
  "bundles": {
    "P_OLD5": ["L19H13", "L21H1", "L22H14", "L23H1", "L23H5"],
    "P_POS7": ["19:8", "19:13", "21:1", "21:14", "22:14", "23:1", "23:5"]
  }
}

Outputs:
  bundle_path_effect.jsonl
  bundle_path_summary.csv
  bundle_top.json
  config.json
  tokenization.json
  errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import random
import shutil
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers: {exc}")


SCRIPT_VERSION = "coco-ioi-joint-bundle-path-v1"
SCOPES = ("prompt_last", "objects_identity", "objects_role", "all")
CHANNELS = ("q", "k", "v")
KV_SCOPES = ("objects", "all")
STATUSES = ("all", "both_correct", "original_only", "swapped_only", "both_wrong")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data/model arguments expected by the imported base pipeline.
    p.add_argument("--model", required=True)
    p.add_argument("--source-output-dir", required=True)
    p.add_argument("--dataset", default="coco_two", choices=("coco_two",))
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=("eager",))
    p.add_argument("--object-state", choices=("last", "mean"), default="last")
    p.add_argument(
        "--require-single-token-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Joint bundle and receiver specification.
    p.add_argument("--bundle-json", required=True)
    p.add_argument(
        "--bundle-names",
        default="all",
        help="Comma-separated bundle names or 'all'.",
    )
    p.add_argument("--receiver-layer", type=int, default=26)
    p.add_argument("--receiver-query-head", type=int, default=0)
    p.add_argument("--receiver-channel", choices=CHANNELS, default="v")
    p.add_argument(
        "--sender-position-scopes",
        default="objects_role,objects_identity",
        help="Comma-separated sender scopes.",
    )
    p.add_argument(
        "--receiver-kv-scope",
        choices=KV_SCOPES,
        default="objects",
    )

    # Sample selection; kept compatible with the existing IOI script.
    p.add_argument("--causal-status", choices=STATUSES, default="both_correct")
    p.add_argument("--causal-max-samples", type=int, default=70)
    p.add_argument("--min-margin-denominator", type=float, default=1e-4)
    p.add_argument(
        "--causal-require-margin-sign",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--empty-cache-every", type=int, default=5)

    # Imported scripts.
    p.add_argument(
        "--ioi-script",
        default="analyze_coco_ioi_backward_circuit_v1.py",
    )
    p.add_argument(
        "--producer-script",
        default="analyze_coco_producer_qk_ov_v1.py",
    )
    p.add_argument(
        "--receiver-script",
        default="analyze_coco_receiver_qkv_v1.py",
    )
    p.add_argument(
        "--v3-script",
        default="analyze_spatial_storage_transport_utilization_v3.py",
    )
    p.add_argument(
        "--base-script",
        default="analyze_coco_centroid_generation_step1_v4.py",
    )
    p.add_argument(
        "--attention-helper",
        default="analyze_coco_flip_attention_spatial_vectors_v1.py",
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def import_file(path: Path, module_name: str) -> Any:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    sys.modules.pop(module_name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def parse_subset(value: str, allowed: Sequence[str], label: str) -> List[str]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    invalid = [item for item in items if item not in allowed]
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}; allowed={list(allowed)}")
    if not items:
        raise ValueError(f"No {label} selected")
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(items))


def parse_head(value: Any) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    text = str(value).strip()
    if text.startswith("L") and "H" in text:
        layer_text, head_text = text[1:].split("H", 1)
        return int(layer_text), int(head_text)
    if ":" in text:
        layer_text, head_text = text.split(":", 1)
        return int(layer_text), int(head_text)
    raise ValueError(f"Invalid attention head specification: {value!r}")


@dataclass(frozen=True)
class Bundle:
    name: str
    heads: Tuple[Any, ...]  # Imported module's SenderNode objects.

    @property
    def head_names(self) -> Tuple[str, ...]:
        return tuple(head.node for head in self.heads)


def load_bundles(path: Path, names_text: str, ioi: Any) -> List[Bundle]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("bundles", payload)
    if not isinstance(source, Mapping):
        raise ValueError("Bundle JSON must be an object or contain a 'bundles' object")

    if str(names_text).strip().lower() == "all":
        selected_names = list(source.keys())
    else:
        selected_names = [
            item.strip() for item in str(names_text).split(",") if item.strip()
        ]
    missing = [name for name in selected_names if name not in source]
    if missing:
        raise KeyError(f"Unknown bundles {missing}; available={list(source.keys())}")

    bundles: List[Bundle] = []
    for name in selected_names:
        raw_heads = source[name]
        if not isinstance(raw_heads, Sequence) or isinstance(raw_heads, (str, bytes)):
            raise ValueError(f"Bundle {name!r} must be a list of heads")
        parsed = [parse_head(item) for item in raw_heads]
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"Bundle {name!r} contains duplicate heads")
        if not parsed:
            raise ValueError(f"Bundle {name!r} is empty")
        nodes = tuple(
            ioi.SenderNode("attention", int(layer), int(head))
            for layer, head in parsed
        )
        bundles.append(Bundle(str(name), nodes))
    return bundles


class FreezeAttentionBundleAtPositions:
    """
    Freeze pre-WO attention vectors to original values at selected positions,
    then replace every selected bundle head with its swapped activation.
    """

    def __init__(
        self,
        *,
        decoder_layers: Sequence[Any],
        attention_helper: Any,
        receiver_module: Any,
        output_projection_fn: Any,
        original: Mapping[int, Mapping[int, torch.Tensor]],
        positions_by_layer: Mapping[int, Sequence[int]],
        senders: Sequence[Any],
        sender_mapping: Mapping[int, int],
        swapped: Mapping[int, Mapping[int, torch.Tensor]],
    ) -> None:
        self.handles: List[Any] = []
        self.layer_events: Dict[int, int] = defaultdict(int)
        self.sender_events: Dict[str, int] = defaultdict(int)
        mapping = {int(k): int(v) for k, v in sender_mapping.items()}
        by_layer: Dict[int, List[Any]] = defaultdict(list)
        for sender in senders:
            if sender.kind != "attention":
                raise ValueError("Joint bundle script currently supports attention senders only")
            by_layer[int(sender.layer)].append(sender)

        for layer_index, positions in sorted(positions_by_layer.items()):
            layer_index = int(layer_index)
            layer = decoder_layers[layer_index]
            attention = attention_helper.resolve_self_attention(layer)
            shape = receiver_module.resolve_attention_shape(attention)
            module = output_projection_fn(attention)
            selected_positions = sorted(set(map(int, positions)))
            layer_senders = tuple(by_layer.get(layer_index, ()))

            def make_hook(
                index: int,
                layer_shape: Any,
                layer_positions: Sequence[int],
                selected_senders: Sequence[Any],
            ):
                def hook(_module: Any, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
                    if not inputs:
                        raise RuntimeError("W_O pre-hook received no input")
                    tensor = inputs[0]
                    if not torch.is_tensor(tensor) or tensor.ndim != 3:
                        raise RuntimeError("W_O input must be [B,S,D]")
                    if int(tensor.shape[0]) != 1:
                        raise RuntimeError("Bundle freeze patch expects batch size 1")
                    modified = tensor.clone()

                    # Freeze all attention outputs at these positions to clean.
                    for target_position in layer_positions:
                        clean = original[index][int(target_position)].to(
                            device=tensor.device,
                            dtype=tensor.dtype,
                        )
                        if clean.numel() != int(tensor.shape[-1]):
                            raise RuntimeError(
                                f"L{index} clean W_O input dim {clean.numel()} != "
                                f"current {int(tensor.shape[-1])}"
                            )
                        modified[0, int(target_position)] = clean
                    self.layer_events[index] += 1

                    # Reinsert every selected sender head from the swapped run.
                    for sender in selected_senders:
                        head = int(sender.head)
                        start = head * int(layer_shape.query_head_dim)
                        stop = start + int(layer_shape.query_head_dim)
                        for target_position, source_position in mapping.items():
                            if int(target_position) not in layer_positions:
                                continue
                            source = swapped[index][int(source_position)][start:stop].to(
                                device=tensor.device,
                                dtype=tensor.dtype,
                            )
                            modified[0, int(target_position), start:stop] = source
                            self.sender_events[sender.node] += 1
                    return (modified, *inputs[1:])

                return hook

            self.handles.append(
                module.register_forward_pre_hook(
                    make_hook(
                        layer_index,
                        shape,
                        selected_positions,
                        layer_senders,
                    )
                )
            )

    def validate(self, senders: Sequence[Any]) -> None:
        if not self.layer_events:
            raise RuntimeError("Attention freeze hooks did not fire")
        missing = [sender.node for sender in senders if self.sender_events[sender.node] < 1]
        if missing:
            raise RuntimeError(f"Bundle sender patches did not fire: {missing}")

    def close(self) -> None:
        for handle in reversed(self.handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()


@torch.inference_mode()
def run_bundle_c_pass(
    *,
    bundle: Bundle,
    sender_mapping: Mapping[int, int],
    receiver_units: Sequence[Any],
    pair: Any,
    original_capture: Any,
    swapped_capture: Any,
    model: Any,
    decoder_layers: Sequence[Any],
    relation_token_map: Mapping[str, Sequence[int]],
    base: Any,
    receiver_module: Any,
    attention_helper: Any,
    kv_scope: str,
    ioi: Any,
) -> Dict[int, Dict[str, Dict[int, torch.Tensor]]]:
    if not receiver_units:
        return {}
    min_receiver_layer = min(int(unit.layer) for unit in receiver_units)
    invalid = [head.node for head in bundle.heads if int(head.layer) >= min_receiver_layer]
    if invalid:
        raise ValueError(
            f"Bundle {bundle.name} has senders not earlier than receiver layer "
            f"{min_receiver_layer}: {invalid}"
        )

    positions_by_layer_channel: Dict[Tuple[int, str], List[int]] = {}
    freeze_positions = set(map(int, sender_mapping.keys()))
    for unit in receiver_units:
        positions = ioi.receiver_channel_positions(pair, unit.channel, kv_scope)
        positions_by_layer_channel.setdefault((int(unit.layer), str(unit.channel)), []).extend(
            positions
        )
        freeze_positions.update(map(int, positions))

    positions_by_layer = {
        int(layer): sorted(freeze_positions)
        for layer in original_capture.layers
    }
    freeze = FreezeAttentionBundleAtPositions(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        output_projection_fn=ioi.output_projection_module,
        original=original_capture.attention,
        positions_by_layer=positions_by_layer,
        senders=bundle.heads,
        sender_mapping=sender_mapping,
        swapped=swapped_capture.attention,
    )
    projection_capture = ioi.CaptureProjectionAtPositions(
        decoder_layers=decoder_layers,
        attention_helper=attention_helper,
        receiver_module=receiver_module,
        positions_by_layer_channel=positions_by_layer_channel,
    )
    try:
        receiver_module.run_scores(
            model=model,
            batch=pair.original_batch,
            base=base,
            relation_token_map=relation_token_map,
        )
        freeze.validate(bundle.heads)
        projection_capture.validate()
        return {
            int(layer): {
                str(channel): dict(position_map)
                for channel, position_map in channel_map.items()
            }
            for layer, channel_map in projection_capture.states.items()
        }
    finally:
        projection_capture.close()
        freeze.close()


def summarize(rows: Sequence[Mapping[str, Any]], ioi: Any) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["bundle"]),
                str(row["receiver_unit"]),
                str(row["channel"]),
                str(row["sender_position_scope"]),
            )
        ].append(row)

    result: List[Dict[str, Any]] = []
    for (bundle, receiver_unit, channel, scope), values in groups.items():
        effects = np.asarray([float(v["normalized_effect"]) for v in values])
        result.append(
            {
                "bundle": bundle,
                "bundle_size": int(values[0]["bundle_size"]),
                "bundle_heads": json.dumps(values[0]["bundle_heads"]),
                "receiver_unit": receiver_unit,
                "receiver_layer": int(values[0]["receiver_layer"]),
                "receiver_query_head": int(values[0]["receiver_query_head"]),
                "receiver_unit_head": int(values[0]["receiver_unit_head"]),
                "receiver_kv_head": int(values[0]["receiver_kv_head"]),
                "shared_query_heads": json.dumps(values[0]["shared_query_heads"]),
                "channel": channel,
                "sender_position_scope": scope,
                "receiver_kv_scope": values[0]["receiver_kv_scope"],
                "N": len(values),
                "mean_raw_effect": ioi.safe_mean(v["raw_effect"] for v in values),
                "median_raw_effect": ioi.safe_median(v["raw_effect"] for v in values),
                "std_raw_effect": ioi.safe_std(v["raw_effect"] for v in values),
                "mean_normalized_effect": float(np.mean(effects)),
                "median_normalized_effect": float(np.median(effects)),
                "std_normalized_effect": float(np.std(effects)),
                "positive_effect_rate": float(np.mean(effects > 0)),
                "negative_effect_rate": float(np.mean(effects < 0)),
                "crossed_decision_boundary_rate": ioi.safe_mean(
                    int(bool(v["crossed_decision_boundary"])) for v in values
                ),
            }
        )
    result.sort(
        key=lambda row: (
            str(row["sender_position_scope"]),
            -abs(float(row["mean_normalized_effect"])),
            str(row["bundle"]),
        )
    )
    return result


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ioi = import_file(Path(args.ioi_script), "ioi_joint_base")
    source_config, source_rows = ioi.load_source_rows(args)
    producer_module = import_file(Path(args.producer_script), "ioi_joint_producer")
    receiver_module = import_file(Path(args.receiver_script), "ioi_joint_receiver")
    v3 = import_file(Path(args.v3_script), "ioi_joint_v3")
    base = import_file(Path(args.base_script), "ioi_joint_base_data")
    attention_helper = import_file(Path(args.attention_helper), "ioi_joint_attention")

    scopes = parse_subset(args.sender_position_scopes, SCOPES, "sender scopes")
    bundles = load_bundles(Path(args.bundle_json), args.bundle_names, ioi)

    model = None
    processor = None
    try:
        (
            model,
            processor,
            spec,
            decoder_layers,
            decoder_path,
            relation_token_map,
        ) = producer_module.load_model_bundle(args=args, base=base)

        token_report = ioi.tokenization_report(processor.tokenizer, relation_token_map)
        ioi.write_json(output_dir / "tokenization.json", token_report)
        if (
            args.require_single_token_labels
            and not token_report["all_relations_have_single_token_continuation"]
        ):
            raise RuntimeError("At least one relation label lacks a one-token continuation")

        receiver_layer = int(args.receiver_layer)
        if not 0 <= receiver_layer < len(decoder_layers):
            raise ValueError(f"Invalid receiver layer {receiver_layer}")
        writer = ioi.WriterNode(
            "attention",
            receiver_layer,
            int(args.receiver_query_head),
        )
        receiver_units = ioi.build_receiver_units(
            writers=[writer],
            channels=[str(args.receiver_channel)],
            decoder_layers=decoder_layers,
            attention_helper=attention_helper,
            receiver_module=receiver_module,
        )
        if len(receiver_units) != 1:
            raise RuntimeError(f"Expected one receiver unit, got {len(receiver_units)}")
        unit = receiver_units[0]

        # Validate head indices against the actual model.
        for bundle in bundles:
            for sender in bundle.heads:
                if int(sender.layer) >= receiver_layer:
                    raise ValueError(
                        f"{bundle.name}: {sender.node} is not earlier than receiver L{receiver_layer}"
                    )
                attention = attention_helper.resolve_self_attention(
                    decoder_layers[int(sender.layer)]
                )
                shape = receiver_module.resolve_attention_shape(attention)
                if not 0 <= int(sender.head) < int(shape.n_query_heads):
                    raise ValueError(
                        f"{bundle.name}: {sender.node} outside n_query_heads={shape.n_query_heads}"
                    )

        config = {
            "script_version": SCRIPT_VERSION,
            "model": args.model,
            "repo_id": spec.repo_id,
            "dataset": args.dataset,
            "source_output_dir": args.source_output_dir,
            "source_script_version": source_config.get("script_version"),
            "decoder_path": decoder_path,
            "n_layers": len(decoder_layers),
            "bundles": {
                bundle.name: list(bundle.head_names) for bundle in bundles
            },
            "receiver": {
                "unit": unit.unit,
                "layer": int(unit.layer),
                "query_head": int(unit.query_head),
                "unit_head": int(unit.unit_head),
                "kv_head": int(unit.kv_head),
                "shared_query_heads": list(unit.shared_query_heads),
                "channel": unit.channel,
                "kv_scope": args.receiver_kv_scope,
            },
            "sender_scopes": scopes,
            "score_mode": "next_token_relation_variants_at_prompt_last",
            "path_definition": "joint_sender_bundle_to_receiver_through_residual_and_mlps_only",
            "intermediate_attention": "frozen_to_original",
            "mlps": "recomputed",
            "transformers_version": transformers.__version__,
        }
        ioi.write_json(output_dir / "config.json", config)

        records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)
        config["audit"] = audit
        ioi.write_json(output_dir / "config.json", config)

        rows_to_run = ioi.eligible_rows(args, source_rows)
        if not rows_to_run:
            raise RuntimeError("No eligible samples")

        output_path = output_dir / "bundle_path_effect.jsonl"
        errors_path = output_dir / "errors.jsonl"
        existing = ioi.read_jsonl(output_path) if args.resume else []
        completed = {
            (
                int(row["sid"]),
                str(row["bundle"]),
                str(row["receiver_unit"]),
                str(row["sender_position_scope"]),
            )
            for row in existing
        }
        all_rows = list(existing)

        pending_rows = []
        for source_row in rows_to_run:
            sid = int(source_row["sid"])
            has_pending = any(
                (sid, bundle.name, unit.unit, scope) not in completed
                for bundle in bundles
                for scope in scopes
            )
            if has_pending:
                pending_rows.append(source_row)

        all_capture_layers = list(range(len(decoder_layers)))
        print(
            "Joint bundle path scan: "
            f"requested_N={len(rows_to_run)}, pending_N={len(pending_rows)}, "
            f"existing_rows={len(existing)}, bundles={len(bundles)}, "
            f"receiver={unit.unit}, scopes={scopes}",
            flush=True,
        )

        for sample_index, source_row in enumerate(
            tqdm(pending_rows, desc=f"bundle-path:{args.model}"),
            start=1,
        ):
            pair = None
            try:
                pair = receiver_module.prepare_pair(
                    args=args,
                    row=source_row,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    processor=processor,
                    device=torch.device(args.device),
                )

                original_positions = set()
                swapped_positions = set()
                for scope in scopes:
                    mapping = ioi.sender_position_mapping(pair, scope)
                    original_positions.update(map(int, mapping.keys()))
                    swapped_positions.update(map(int, mapping.values()))
                original_positions.update(
                    map(
                        int,
                        ioi.receiver_channel_positions(
                            pair,
                            unit.channel,
                            args.receiver_kv_scope,
                        ),
                    )
                )

                (
                    original_result,
                    swapped_result,
                    original_capture,
                    swapped_capture,
                ) = ioi.capture_upstream_pair(
                    pair=pair,
                    layers=all_capture_layers,
                    original_positions=sorted(original_positions),
                    swapped_positions=sorted(swapped_positions),
                    sender_mlps_present=False,
                    model=model,
                    decoder_layers=decoder_layers,
                    relation_token_map=relation_token_map,
                    base=base,
                    receiver_module=receiver_module,
                    attention_helper=attention_helper,
                )

                gt = str(pair.gt)
                original_margin = ioi.relation_margin(original_result["logits"], gt)
                swapped_margin = ioi.relation_margin(swapped_result["logits"], gt)
                denominator = float(original_margin - swapped_margin)
                if abs(denominator) < args.min_margin_denominator:
                    continue
                if args.causal_require_margin_sign and not (
                    original_margin > 0 and swapped_margin < 0
                ):
                    continue

                for scope in scopes:
                    mapping = ioi.sender_position_mapping(pair, scope)
                    for bundle in bundles:
                        key = (int(pair.sid), bundle.name, unit.unit, scope)
                        if key in completed:
                            continue
                        c_states = run_bundle_c_pass(
                            bundle=bundle,
                            sender_mapping=mapping,
                            receiver_units=[unit],
                            pair=pair,
                            original_capture=original_capture,
                            swapped_capture=swapped_capture,
                            model=model,
                            decoder_layers=decoder_layers,
                            relation_token_map=relation_token_map,
                            base=base,
                            receiver_module=receiver_module,
                            attention_helper=attention_helper,
                            kv_scope=args.receiver_kv_scope,
                            ioi=ioi,
                        )
                        intervention = ioi.run_d_receiver_patch(
                            unit=unit,
                            c_states=c_states,
                            pair=pair,
                            model=model,
                            decoder_layers=decoder_layers,
                            relation_token_map=relation_token_map,
                            base=base,
                            receiver_module=receiver_module,
                            attention_helper=attention_helper,
                            kv_scope=args.receiver_kv_scope,
                        )
                        intervention_margin = ioi.relation_margin(
                            intervention["logits"], gt
                        )
                        raw_effect = float(original_margin - intervention_margin)
                        normalized_effect = float(raw_effect / denominator)
                        row = {
                            "script_version": SCRIPT_VERSION,
                            "phase": "bundle_path",
                            "model": args.model,
                            "sid": int(pair.sid),
                            "gt": gt,
                            "generation_pair_status": source_row[
                                "generation_pair_status"
                            ],
                            "bundle": bundle.name,
                            "bundle_size": len(bundle.heads),
                            "bundle_heads": list(bundle.head_names),
                            "sender_position_scope": scope,
                            "receiver_unit": unit.unit,
                            "receiver_layer": int(unit.layer),
                            "receiver_query_head": int(unit.query_head),
                            "receiver_unit_head": int(unit.unit_head),
                            "receiver_kv_head": int(unit.kv_head),
                            "shared_query_heads": list(unit.shared_query_heads),
                            "channel": unit.channel,
                            "receiver_kv_scope": args.receiver_kv_scope,
                            "path_definition": "joint_sender_bundle_to_receiver_through_residual_and_mlps_only",
                            "intermediate_attention": "frozen_to_original",
                            "mlps": "recomputed",
                            "score_mode": "next_token_relation_variants_at_prompt_last",
                            "original_margin": original_margin,
                            "swapped_margin_fixed_axis": swapped_margin,
                            "intervention_margin_fixed_axis": intervention_margin,
                            "margin_denominator": denominator,
                            "raw_effect": raw_effect,
                            "normalized_effect": normalized_effect,
                            "expected_positive": bool(raw_effect > 0),
                            "crossed_decision_boundary": bool(
                                original_margin > 0 >= intervention_margin
                            ),
                            "original_prediction": original_result["prediction"],
                            "swapped_prediction": swapped_result["prediction"],
                            "intervention_prediction": intervention["prediction"],
                        }
                        ioi.append_jsonl(output_path, row)
                        all_rows.append(row)
                        completed.add(key)

            except Exception as exc:
                error = {
                    "phase": "bundle_path",
                    "sid": int(source_row["sid"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                ioi.append_jsonl(errors_path, error)
                print(
                    f"\n[ERROR sid={source_row['sid']}] "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if args.fail_fast:
                    raise
            finally:
                if pair is not None:
                    receiver_module.release_pair(pair)
                gc.collect()
                if torch.cuda.is_available() and (
                    args.empty_cache_every > 0
                    and sample_index % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

            if args.print_every > 0 and sample_index % args.print_every == 0:
                print(
                    f"[bundle {sample_index}/{len(pending_rows)}] "
                    f"rows={len(all_rows)}",
                    flush=True,
                )

        summary = summarize(all_rows, ioi)
        ioi.write_csv(output_dir / "bundle_path_summary.csv", summary)
        ioi.write_json(
            output_dir / "bundle_top.json",
            {
                "script_version": SCRIPT_VERSION,
                "metric": "absolute_mean_normalized_effect",
                "bundles": summary,
            },
        )
        print(f"\nSaved outputs to {output_dir}", flush=True)

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
