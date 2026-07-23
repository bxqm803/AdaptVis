#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import contextlib
import gc
import json
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from transformers import AutoProcessor
from PIL import Image, ImageDraw

import trace_centroid_generation_groups_v2_1 as core
import run_spatial_repair_three_experiments_v1 as repair
import analyze_object_visual_attention_layers_v1 as grounding


# ============================================================================
# Full-dataset experiment configuration
# ============================================================================
MODEL_NAME = "qwen-3b"
DEVICE = "cuda:0"

# 0.0 is an exact same-path baseline.
# alpha means: for every bbox visual token, add alpha times the
# per-head/per-object-token mean attention over all visual keys.
ADD_SCALES = [0.0, 0.0625, 0.125, 0.25, 0.5]

# Set 20 for a quick smoke run; None means all eligible samples.
MAX_SAMPLES = None

EXCLUDE_AMBIGUOUS = False
BBOX_THRESHOLD = 0.25
MAX_NEW_TOKENS = 8
FAIL_FAST = True

DATA_ROOT = Path("data")
PROMPT_JSONL = Path("prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
BBOX_JSONL = Path("output/groundingdino_coco_two_bboxes/bboxes_by_sid.jsonl")
OUTPUT_DIR = Path("output/bbox_object_token_additive_attention_full_v3/coco") / MODEL_NAME


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def selected_box(row, name):
    obj = row.get(name, {})
    selected = obj.get("selected") if isinstance(obj, dict) else None
    if not isinstance(selected, dict):
        return None
    box = selected.get("box_xyxy_original")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    return [float(x) for x in box]


def hard_mask(soft, threshold):
    x = np.asarray(soft, dtype=np.float32).reshape(-1)
    m = x >= threshold
    if not m.any():
        m[int(np.argmax(x))] = True
    return m


def box_mask(size, box_xyxy):
    """Create an original-image-space binary RGB mask for one xyxy box."""
    width, height = [int(v) for v in size]
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))

    if x2 > x1 and y2 > y1:
        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
    return image


def processor_mask_batch(processor, image, question_text):
    """
    Run the exact VLM processor used for the real image.
    Keep this on CPU: it is only used to recover processor geometry.
    """
    return core.make_question_batch(
        processor=processor,
        image=image,
        question_text=question_text,
        device=torch.device("cpu"),
    )


def find_pixel_tensor(batch):
    preferred = (
        "pixel_values",
        "pixel_values_images",
        "image_pixel_values",
    )
    for key in preferred:
        value = batch.get(key)
        if torch.is_tensor(value):
            return key, value.detach().float().cpu()

    for key, value in batch.items():
        if "pixel" in str(key).lower() and torch.is_tensor(value):
            return str(key), value.detach().float().cpu()

    raise RuntimeError(
        f"No processed pixel tensor. Batch keys={list(batch.keys())}"
    )


def occupancy_ratio(mask_tensor, black_tensor, white_tensor):
    if (
        mask_tensor.shape != black_tensor.shape
        or mask_tensor.shape != white_tensor.shape
    ):
        raise RuntimeError(
            "Processed mask/black/white shapes differ: "
            f"{tuple(mask_tensor.shape)}, "
            f"{tuple(black_tensor.shape)}, "
            f"{tuple(white_tensor.shape)}"
        )

    denominator = white_tensor - black_tensor
    valid = denominator.abs() > 1e-6
    ratio = torch.zeros_like(mask_tensor, dtype=torch.float32)
    ratio[valid] = (
        mask_tensor[valid] - black_tensor[valid]
    ) / denominator[valid]
    return ratio.clamp(0.0, 1.0)


def processed_mask_to_grid(
    mask_batch,
    black_batch,
    white_batch,
    grid_h,
    grid_w,
):
    mask_key, mask_tensor = find_pixel_tensor(mask_batch)
    black_key, black_tensor = find_pixel_tensor(black_batch)
    white_key, white_tensor = find_pixel_tensor(white_batch)

    if not (
        mask_key == black_key == white_key
    ):
        raise RuntimeError(
            "Processed pixel tensor keys differ: "
            f"{mask_key}, {black_key}, {white_key}"
        )

    ratio = occupancy_ratio(
        mask_tensor,
        black_tensor,
        white_tensor,
    )

    # LLaVA / CLIP-style [B,C,H,W].
    if ratio.ndim == 4:
        image = ratio[0].mean(dim=0)[None, None]
        pooled = F.adaptive_avg_pool2d(
            image, (grid_h, grid_w)
        )[0, 0]
        return pooled.numpy().astype(np.float32)

    # Possible video/tiled [B,T,C,H,W] or [B,C,T,H,W].
    if ratio.ndim == 5:
        x = ratio[0]
        if x.shape[0] in (1, 3, 4):
            image = x.mean(dim=(0, 1))[None, None]
        elif x.shape[1] in (1, 3, 4):
            image = x.mean(dim=(0, 1))[None, None]
        else:
            raise RuntimeError(
                f"Unsupported 5D pixel tensor: {tuple(ratio.shape)}"
            )
        pooled = F.adaptive_avg_pool2d(
            image, (grid_h, grid_w)
        )[0, 0]
        return pooled.numpy().astype(np.float32)

    # Qwen2.5-VL flattened patch rows [N,D] or [1,N,D].
    if ratio.ndim == 3 and ratio.shape[0] == 1:
        ratio = ratio[0]

    if ratio.ndim == 2:
        patch_occupancy = ratio.mean(dim=-1)
        grid = mask_batch.get("image_grid_thw")
        if not torch.is_tensor(grid) or grid.numel() < 3:
            raise RuntimeError(
                "Flattened pixel tensor requires image_grid_thw"
            )

        temporal, raw_h, raw_w = [
            int(v)
            for v in grid.detach().cpu().reshape(-1, 3)[0].tolist()
        ]
        expected = temporal * raw_h * raw_w
        if patch_occupancy.numel() != expected:
            raise RuntimeError(
                "Qwen processed patch count mismatch: "
                f"rows={patch_occupancy.numel()}, "
                f"grid={temporal}x{raw_h}x{raw_w}={expected}"
            )

        raw = patch_occupancy.reshape(
            temporal, raw_h, raw_w
        ).mean(dim=0)
        pooled = F.adaptive_avg_pool2d(
            raw[None, None],
            (grid_h, grid_w),
        )[0, 0]
        return pooled.numpy().astype(np.float32)

    raise RuntimeError(
        f"Unsupported processed pixel tensor shape: {tuple(ratio.shape)}"
    )


def processor_box_masks(
    processor,
    question_text,
    image_size,
    subject_box,
    reference_box,
    grid_h,
    grid_w,
):
    """
    Map original-image xyxy boxes to the exact VLM visual-token grid.

    This function is deliberately self-contained. It does not import the
    grounding-evaluation script, so helper names/signatures cannot drift.
    """
    width, height = [int(v) for v in image_size]

    subject_image = box_mask(
        (width, height), subject_box
    )
    reference_image = box_mask(
        (width, height), reference_box
    )
    black_image = Image.new(
        "RGB", (width, height), (0, 0, 0)
    )
    white_image = Image.new(
        "RGB", (width, height), (255, 255, 255)
    )

    batches = []
    try:
        subject_batch = processor_mask_batch(
            processor, subject_image, question_text
        )
        reference_batch = processor_mask_batch(
            processor, reference_image, question_text
        )
        black_batch = processor_mask_batch(
            processor, black_image, question_text
        )
        white_batch = processor_mask_batch(
            processor, white_image, question_text
        )
        batches = [
            subject_batch,
            reference_batch,
            black_batch,
            white_batch,
        ]

        subject_mask = processed_mask_to_grid(
            subject_batch,
            black_batch,
            white_batch,
            grid_h,
            grid_w,
        )
        reference_mask = processed_mask_to_grid(
            reference_batch,
            black_batch,
            white_batch,
            grid_h,
            grid_w,
        )

        pixel_key, pixel_value = find_pixel_tensor(
            subject_batch
        )
        metadata = {
            "pixel_key": pixel_key,
            "pixel_shape": list(pixel_value.shape),
            "image_grid_thw": (
                subject_batch["image_grid_thw"]
                .detach()
                .cpu()
                .tolist()
                if torch.is_tensor(
                    subject_batch.get("image_grid_thw")
                )
                else None
            ),
        }
        return subject_mask, reference_mask, metadata
    finally:
        for image in (
            subject_image,
            reference_image,
            black_image,
            white_image,
        ):
            image.close()
        del batches


def validate_mask_geometry(
    subject_soft,
    reference_soft,
    visual_count,
    grid_h,
    grid_w,
):
    expected_shape = (int(grid_h), int(grid_w))
    if subject_soft.shape != expected_shape:
        raise RuntimeError(
            f"Subject bbox mask shape={subject_soft.shape}, "
            f"expected={expected_shape}"
        )
    if reference_soft.shape != expected_shape:
        raise RuntimeError(
            f"Reference bbox mask shape={reference_soft.shape}, "
            f"expected={expected_shape}"
        )
    if int(grid_h) * int(grid_w) != int(visual_count):
        raise RuntimeError(
            f"Grid {grid_h}x{grid_w} != visual_count={visual_count}"
        )
    if not np.isfinite(subject_soft).all():
        raise RuntimeError("Subject bbox mask contains NaN/Inf")
    if not np.isfinite(reference_soft).all():
        raise RuntimeError("Reference bbox mask contains NaN/Inf")
    if float(subject_soft.max()) <= 0:
        raise RuntimeError("Subject bbox mask is empty after processing")
    if float(reference_soft.max()) <= 0:
        raise RuntimeError("Reference bbox mask is empty after processing")


class BBoxObjectTokenAddManager:
    """
    Prefill only, every decoder layer.

    For each object-token query q and each head h:
        mu = mean attention from q to ALL visual keys
        add alpha * mu to every visual key inside that object's bbox

    No multiplication and no renormalization.

    Instead of merely editing the returned attention map, reconstruct the causal
    added A*V contribution and add it to self-attention output at q.
    """

    def __init__(self, layers):
        self.layers = list(layers)
        self.handles = []
        self.hidden = {}
        self.alpha = 0.0
        self.prompt_spec = None
        self.subject_visual_mask = None
        self.reference_visual_mask = None
        self.stats = defaultdict(lambda: defaultdict(float))

        for layer_id, layer in enumerate(self.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                raise RuntimeError(f"L{layer_id} has no self_attn")
            self.handles.append(
                attn.register_forward_pre_hook(
                    self._pre_hook(layer_id), with_kwargs=True
                )
            )
            self.handles.append(
                attn.register_forward_hook(
                    self._post_hook(layer_id), with_kwargs=True
                )
            )

    def close(self):
        for handle in self.handles:
            with contextlib.suppress(Exception):
                handle.remove()
        self.handles.clear()
        self.reset()

    def reset(self):
        self.hidden.clear()
        self.alpha = 0.0
        self.prompt_spec = None
        self.subject_visual_mask = None
        self.reference_visual_mask = None

    def configure(
        self,
        alpha,
        prompt_spec,
        subject_visual_mask,
        reference_visual_mask,
    ):
        self.reset()
        self.alpha = float(alpha)
        self.prompt_spec = prompt_spec
        self.subject_visual_mask = np.asarray(
            subject_visual_mask, dtype=bool
        ).reshape(-1)
        self.reference_visual_mask = np.asarray(
            reference_visual_mask, dtype=bool
        ).reshape(-1)

    def _pre_hook(self, layer_id):
        def hook(module, args, kwargs):
            if self.alpha == 0.0:
                return None
            hidden = None
            if args and torch.is_tensor(args[0]):
                hidden = args[0]
            elif torch.is_tensor(kwargs.get("hidden_states")):
                hidden = kwargs["hidden_states"]
            if hidden is None:
                raise RuntimeError(
                    f"Cannot capture hidden states at L{layer_id}"
                )
            self.hidden[layer_id] = hidden
            return None
        return hook

    @staticmethod
    def _project_added_output(o_proj, delta_heads):
        # [B,H,Q,D] -> [B,Q,H*D] -> o_proj
        pre = delta_heads.permute(0, 2, 1, 3).contiguous()
        pre = pre.reshape(pre.shape[0], pre.shape[1], -1)
        return F.linear(
            pre.to(dtype=o_proj.weight.dtype),
            o_proj.weight,
            bias=None,
        )

    def _add_for_queries(
        self,
        *,
        layer_id,
        attention_weights,
        values,
        o_proj,
        modified_output,
        query_positions,
        visual_positions,
        bbox_visual_mask,
        label,
    ):
        q_len = int(attention_weights.shape[-2])
        k_len = int(attention_weights.shape[-1])

        q_positions = [
            int(x) for x in query_positions
            if 0 <= int(x) < q_len
        ]
        if not q_positions:
            return

        if len(visual_positions) != int(bbox_visual_mask.size):
            raise RuntimeError(
                f"L{layer_id} visual count mismatch: "
                f"positions={len(visual_positions)}, "
                f"mask={bbox_visual_mask.size}"
            )

        all_visual = [
            int(x) for x in visual_positions
            if 0 <= int(x) < k_len
        ]
        bbox_keys = [
            int(visual_positions[i])
            for i, flag in enumerate(bbox_visual_mask)
            if bool(flag)
            and 0 <= int(visual_positions[i]) < k_len
        ]
        if not all_visual or not bbox_keys:
            return

        q_idx = torch.as_tensor(
            q_positions,
            device=attention_weights.device,
            dtype=torch.long,
        )
        v_idx = torch.as_tensor(
            all_visual,
            device=attention_weights.device,
            dtype=torch.long,
        )
        b_idx = torch.as_tensor(
            bbox_keys,
            device=attention_weights.device,
            dtype=torch.long,
        )

        # [B,H,Q,K] -> object-token query rows [B,H,Qobj,K]
        rows = attention_weights.index_select(2, q_idx)
        visual_rows = rows.index_select(3, v_idx)

        # Per sample / head / object subtoken.
        mu = visual_rows.mean(dim=-1)  # [B,H,Qobj]

        # Each bbox key receives exactly alpha * mu extra weight.
        bbox_values = values.index_select(2, b_idx)  # [B,H,Bbox,D]
        bbox_value_sum = bbox_values.sum(dim=2)      # [B,H,D]
        delta_heads = (
            float(self.alpha)
            * mu.to(dtype=bbox_value_sum.dtype)[..., None]
            * bbox_value_sum[:, :, None, :]
        )  # [B,H,Qobj,D]

        delta_output = self._project_added_output(
            o_proj, delta_heads
        )
        modified_output.index_add_(
            1,
            q_idx.to(device=modified_output.device),
            delta_output.to(
                device=modified_output.device,
                dtype=modified_output.dtype,
            ),
        )

        visual_mass = visual_rows.sum(dim=-1)
        added_mass = float(self.alpha) * mu * len(bbox_keys)
        stat = self.stats[(float(self.alpha), int(layer_id))]
        n = int(mu.numel())
        stat[f"{label}_mu_sum"] += float(
            mu.detach().float().sum().cpu()
        )
        stat[f"{label}_added_mass_sum"] += float(
            added_mass.detach().float().sum().cpu()
        )
        stat[f"{label}_visual_mass_sum"] += float(
            visual_mass.detach().float().sum().cpu()
        )
        stat[f"{label}_n"] += n
        stat[f"{label}_bbox_tokens_sum"] += (
            len(bbox_keys) * n
        )
        stat[f"{label}_visual_tokens_sum"] += (
            len(all_visual) * n
        )

    def _post_hook(self, layer_id):
        def hook(module, args, kwargs, output):
            if self.alpha == 0.0:
                return output

            attention_output = repair.output_first_tensor(output)
            q_len = int(attention_output.shape[-2])

            # Only prompt prefill contains object-token query positions.
            if q_len <= 1:
                return output

            weights = repair.find_attention_weights(
                output, q_len
            )
            if weights is None:
                raise RuntimeError(
                    f"L{layer_id} did not return eager attention weights"
                )

            hidden = self.hidden.get(layer_id)
            if hidden is None:
                raise RuntimeError(
                    f"Missing hidden cache at L{layer_id}"
                )
            if self.prompt_spec is None:
                raise RuntimeError("Manager not configured")

            positions = repair.expand_positions(
                self.prompt_spec,
                hidden_length=q_len,
            )

            v_proj, o_proj = (
                repair.resolve_attention_projections(module)
            )
            current_values = v_proj(hidden)

            num_heads = int(weights.shape[1])
            head_dim = (
                int(o_proj.weight.shape[1]) // num_heads
            )
            if (
                head_dim <= 0
                or current_values.shape[-1] % head_dim != 0
            ):
                raise RuntimeError(
                    f"L{layer_id} invalid V/head dimensions: "
                    f"V={current_values.shape[-1]}, "
                    f"H={num_heads}, D={head_dim}"
                )

            kv_heads = (
                int(current_values.shape[-1]) // head_dim
            )
            values = current_values.view(
                current_values.shape[0],
                current_values.shape[1],
                kv_heads,
                head_dim,
            ).transpose(1, 2)
            values = repair.repeat_kv(values, num_heads)

            k_len = int(weights.shape[-1])
            if int(values.shape[2]) < k_len:
                raise RuntimeError(
                    f"L{layer_id} V length {values.shape[2]} "
                    f"< key length {k_len}"
                )
            values = values[:, :, :k_len, :]

            modified = attention_output.clone()

            self._add_for_queries(
                layer_id=layer_id,
                attention_weights=weights,
                values=values,
                o_proj=o_proj,
                modified_output=modified,
                query_positions=positions.subject_positions,
                visual_positions=positions.visual_positions,
                bbox_visual_mask=self.subject_visual_mask,
                label="subject",
            )
            self._add_for_queries(
                layer_id=layer_id,
                attention_weights=weights,
                values=values,
                o_proj=o_proj,
                modified_output=modified,
                query_positions=positions.reference_positions,
                visual_positions=positions.visual_positions,
                bbox_visual_mask=self.reference_visual_mask,
                label="reference",
            )

            return repair.replace_first_output(
                output, modified
            )
        return hook


def clean_generation_text(value, limit=120):
    text = " ".join(str(value).replace("\n", " ").split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_output = OUTPUT_DIR / "sample_results.jsonl"
    sample_csv_output = OUTPUT_DIR / "sample_results.csv"
    error_output = OUTPUT_DIR / "errors.jsonl"
    summary_output = OUTPUT_DIR / "summary.csv"
    stats_output = OUTPUT_DIR / "added_attention_stats.csv"
    eligible_output = OUTPUT_DIR / "eligible_sids.json"

    for path in (
        sample_output,
        sample_csv_output,
        error_output,
        summary_output,
        stats_output,
        eligible_output,
    ):
        if path.exists():
            path.unlink()

    alphas = [float(x) for x in ADD_SCALES]
    if not alphas or 0.0 not in alphas:
        raise ValueError("ADD_SCALES must contain 0.0 as the live baseline.")
    # Baseline first, then interventions from weak to strong.
    alphas = [0.0] + sorted(x for x in set(alphas) if x != 0.0)

    bbox_by_sid = {
        int(row["sid"]): row
        for row in load_jsonl(BBOX_JSONL)
    }

    backend = core.import_two_object_module()
    records, dataset_audit = backend.load_records(
        "coco_two",
        DATA_ROOT,
        None,
    )
    record_by_sid = {
        int(record.sid): record
        for record in records
    }
    prompt_rows = core.load_standard_prompts(PROMPT_JSONL)

    common_sids = sorted(
        set(record_by_sid)
        & set(prompt_rows)
        & set(bbox_by_sid)
    )

    missing_subject_box = 0
    missing_reference_box = 0
    excluded_ambiguous = 0
    eligible_sids = []

    for sid in common_sids:
        bbox_row = bbox_by_sid[sid]
        subject_box = selected_box(bbox_row, "subject")
        reference_box = selected_box(bbox_row, "reference")

        if subject_box is None:
            missing_subject_box += 1
            continue
        if reference_box is None:
            missing_reference_box += 1
            continue
        if (
            EXCLUDE_AMBIGUOUS
            and bool(bbox_row.get("either_ambiguous", False))
        ):
            excluded_ambiguous += 1
            continue

        eligible_sids.append(sid)

    if MAX_SAMPLES is not None:
        eligible_sids = eligible_sids[: int(MAX_SAMPLES)]

    if not eligible_sids:
        raise RuntimeError("No eligible full-dataset samples.")

    eligible_output.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "eligible_sids": eligible_sids,
                "n": len(eligible_sids),
                "exclude_ambiguous": EXCLUDE_AMBIGUOUS,
                "max_samples": MAX_SAMPLES,
                "audit": {
                    "records": len(record_by_sid),
                    "prompts": len(prompt_rows),
                    "bbox_rows": len(bbox_by_sid),
                    "common_sids": len(common_sids),
                    "missing_subject_box": missing_subject_box,
                    "missing_reference_box": missing_reference_box,
                    "excluded_ambiguous": excluded_ambiguous,
                    "dataset_audit": dataset_audit,
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    if MODEL_NAME not in backend.SPECS:
        raise ValueError(
            f"Unknown model {MODEL_NAME!r}; "
            f"available={sorted(backend.SPECS)}"
        )

    spec = backend.SPECS[MODEL_NAME]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} "
            f"has no {spec.model_class}"
        )

    print("=" * 160)
    print("FULL-DATASET BBOX-TARGETED ADDITIVE OBJECT-TOKEN ATTENTION V3")
    print("=" * 160)
    print(f"model={MODEL_NAME} -> {spec.repo_id}")
    print(f"records={len(record_by_sid)}")
    print(f"prompts={len(prompt_rows)}")
    print(f"bbox_rows={len(bbox_by_sid)}")
    print(f"common_sids={len(common_sids)}")
    print(f"eligible_samples={len(eligible_sids)}")
    print(f"exclude_ambiguous={EXCLUDE_AMBIGUOUS}")
    print(f"alphas={alphas}")
    print("No A/B/C/D filtering or grouped evaluation.")
    print("Every eligible full-dataset sample is generated live.")
    print("Every run prints: gt, generation text, pred, acc, baseline pred, change, running acc.")
    print("All decoder layers and all attention heads are intervened.")
    print(
        "A'[object_query,bbox_key] = "
        "A + alpha * mean(A[object_query,all_visual_keys])"
    )
    print("No multiplication. No renormalization.")
    print("=" * 160)

    model = model_cls.from_pretrained(
        spec.repo_id,
        dtype=core.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": DEVICE},
        attn_implementation="eager",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    core.configure_processor(model, processor)

    device = torch.device(DEVICE)
    layers, layers_path = core.resolve_decoder_layers(model)
    manager = BBoxObjectTokenAddManager(layers)

    print(f"decoder_path={layers_path}")
    print(f"decoder_layers={len(layers)}")
    print("=" * 160)

    rows = []
    running = {
        alpha: {
            "n": 0,
            "correct": 0,
            "changed": 0,
            "wrong_to_correct": 0,
            "correct_to_wrong": 0,
        }
        for alpha in alphas
    }

    progress = tqdm(
        enumerate(eligible_sids, 1),
        total=len(eligible_sids),
        desc=f"bbox-add-full:{MODEL_NAME}",
        unit="sample",
        dynamic_ncols=True,
    )

    try:
        for sample_index, sid in progress:
            image = None
            batch = None

            try:
                record = record_by_sid[sid]
                prompt = prompt_rows[sid]
                bbox_row = bbox_by_sid[sid]

                image = core.record_image(record)
                subject = str(prompt["subject"])
                reference = str(prompt["reference"])
                question = str(prompt["question_text"])
                gt = repair.normalize_relation(prompt["answer_raw"])

                if gt not in repair.RELATIONS:
                    raise RuntimeError(
                        f"Invalid GT for sid={sid}: "
                        f"{prompt['answer_raw']!r}"
                    )

                subject_box = selected_box(
                    bbox_row,
                    "subject",
                )
                reference_box = selected_box(
                    bbox_row,
                    "reference",
                )
                if subject_box is None or reference_box is None:
                    raise RuntimeError(
                        f"Missing selected bbox for sid={sid}"
                    )

                batch = core.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question,
                    device=device,
                )
                batch = repair.move_batch_to_device(
                    batch,
                    device,
                )

                prompt_spec = repair.build_prompt_position_spec(
                    model=model,
                    tokenizer=processor.tokenizer,
                    input_ids=batch["input_ids"],
                    subject=subject,
                    reference=reference,
                )

                input_length = int(batch["input_ids"].shape[1])
                expanded = repair.expand_positions(
                    prompt_spec,
                    hidden_length=input_length,
                )
                visual_count = len(expanded.visual_positions)

                grid_h, grid_w, grid_source = grounding.infer_grid(
                    visual_count,
                    batch,
                    tuple(image.size),
                    model,
                    processor,
                )

                subject_soft, reference_soft, mask_meta = (
                    processor_box_masks(
                        processor,
                        question,
                        tuple(image.size),
                        subject_box,
                        reference_box,
                        grid_h,
                        grid_w,
                    )
                )
                validate_mask_geometry(
                    subject_soft,
                    reference_soft,
                    visual_count,
                    grid_h,
                    grid_w,
                )

                subject_mask = hard_mask(
                    subject_soft,
                    BBOX_THRESHOLD,
                )
                reference_mask = hard_mask(
                    reference_soft,
                    BBOX_THRESHOLD,
                )

                baseline_text = None
                baseline_prediction = None
                baseline_correct = None

                for alpha in alphas:
                    manager.configure(
                        alpha=alpha,
                        prompt_spec=prompt_spec,
                        subject_visual_mask=subject_mask,
                        reference_visual_mask=reference_mask,
                    )

                    generated_text, prediction = repair.generate_relation(
                        model=model,
                        processor=processor,
                        batch=batch,
                        max_new_tokens=MAX_NEW_TOKENS,
                        need_attentions=(alpha != 0.0),
                    )
                    manager.reset()

                    if alpha == 0.0:
                        baseline_text = generated_text
                        baseline_prediction = prediction
                        baseline_correct = prediction == gt

                    if baseline_prediction is None or baseline_correct is None:
                        raise RuntimeError(
                            "Baseline alpha=0.0 was not executed first."
                        )

                    parsed = prediction in repair.RELATIONS
                    correct = prediction == gt
                    changed = prediction != baseline_prediction
                    wrong_to_correct = (
                        not baseline_correct and correct
                    )
                    correct_to_wrong = (
                        baseline_correct and not correct
                    )

                    state = running[alpha]
                    state["n"] += 1
                    state["correct"] += int(correct)
                    state["changed"] += int(changed)
                    state["wrong_to_correct"] += int(
                        wrong_to_correct
                    )
                    state["correct_to_wrong"] += int(
                        correct_to_wrong
                    )
                    running_accuracy = (
                        state["correct"] / state["n"]
                    )

                    row = {
                        "model": MODEL_NAME,
                        "sample_index": sample_index,
                        "sid": sid,
                        "gt": gt,
                        "subject": subject,
                        "reference": reference,
                        "either_ambiguous": bool(
                            bbox_row.get(
                                "either_ambiguous",
                                False,
                            )
                        ),
                        "grid_height": int(grid_h),
                        "grid_width": int(grid_w),
                        "grid_source": grid_source,
                        "pixel_key": mask_meta.get("pixel_key"),
                        "pixel_shape": mask_meta.get("pixel_shape"),
                        "subject_bbox_tokens": int(
                            subject_mask.sum()
                        ),
                        "reference_bbox_tokens": int(
                            reference_mask.sum()
                        ),
                        "visual_tokens": int(
                            subject_mask.size
                        ),
                        "alpha": float(alpha),
                        "baseline_generation": baseline_text,
                        "baseline_prediction": baseline_prediction,
                        "baseline_correct": bool(
                            baseline_correct
                        ),
                        "generation": generated_text,
                        "prediction": prediction,
                        "parsed": bool(parsed),
                        "correct": bool(correct),
                        "changed_prediction": bool(changed),
                        "wrong_to_correct": bool(
                            wrong_to_correct
                        ),
                        "correct_to_wrong": bool(
                            correct_to_wrong
                        ),
                        "running_accuracy": float(
                            running_accuracy
                        ),
                    }
                    append_jsonl(sample_output, row)
                    rows.append(row)

                    generation_display = clean_generation_text(
                        generated_text
                    )
                    print(
                        f"[{sample_index:03d}/{len(eligible_sids):03d}] "
                        f"sid={sid:04d} "
                        f"alpha={alpha:>6g} | "
                        f"gt={gt:<5s} | "
                        f'generation="{generation_display}" | '
                        f"pred={str(prediction):<5s} | "
                        f"acc={int(correct)} | "
                        f"baseline_pred={str(baseline_prediction):<5s} | "
                        f"baseline_acc={int(baseline_correct)} | "
                        f"changed={int(changed)} | "
                        f"W→C={int(wrong_to_correct)} | "
                        f"C→W={int(correct_to_wrong)} | "
                        f"running_acc={running_accuracy:.4f}",
                        flush=True,
                    )

            except Exception as exc:
                append_jsonl(
                    error_output,
                    {
                        "sid": sid,
                        "sample_index": sample_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": (
                            traceback.format_exc()
                            .splitlines()[-40:]
                        ),
                    },
                )
                print(
                    f"[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if FAIL_FAST:
                    raise

            finally:
                manager.reset()
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        progress.close()
        manager.close()

    if not rows:
        raise RuntimeError("No successful generation rows.")

    df = pd.DataFrame(rows)
    df.to_csv(sample_csv_output, index=False)

    summary_rows = []

    for alpha in alphas:
        part = df[df["alpha"] == alpha].copy()
        if part.empty:
            continue

        baseline_accuracy = float(
            part["baseline_correct"].mean()
        )
        accuracy = float(part["correct"].mean())

        baseline_wrong = part[
            ~part["baseline_correct"]
        ]
        baseline_correct_rows = part[
            part["baseline_correct"]
        ]

        summary = {
            "alpha": float(alpha),
            "n": int(len(part)),
            "parse_rate": float(part["parsed"].mean()),
            "baseline_accuracy": baseline_accuracy,
            "accuracy": accuracy,
            "accuracy_change": (
                accuracy - baseline_accuracy
            ),
            "prediction_change_rate": float(
                part["changed_prediction"].mean()
            ),
            "wrong_to_correct_count": int(
                part["wrong_to_correct"].sum()
            ),
            "correct_to_wrong_count": int(
                part["correct_to_wrong"].sum()
            ),
            "repair_rate_among_baseline_wrong": (
                float(
                    baseline_wrong[
                        "wrong_to_correct"
                    ].mean()
                )
                if len(baseline_wrong)
                else float("nan")
            ),
            "damage_rate_among_baseline_correct": (
                float(
                    baseline_correct_rows[
                        "correct_to_wrong"
                    ].mean()
                )
                if len(baseline_correct_rows)
                else float("nan")
            ),
        }

        for relation in repair.RELATIONS:
            relation_rows = part[
                part["gt"] == relation
            ]
            summary[f"{relation}_n"] = int(
                len(relation_rows)
            )
            summary[f"{relation}_accuracy"] = (
                float(
                    relation_rows["correct"].mean()
                )
                if len(relation_rows)
                else float("nan")
            )

        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_output, index=False)

    stat_rows = []
    for alpha_layer in sorted(manager.stats):
        alpha, layer_id = alpha_layer
        stat = manager.stats[alpha_layer]
        row = {
            "alpha": float(alpha),
            "layer": int(layer_id),
        }

        for label in ("subject", "reference"):
            n = max(
                1.0,
                stat.get(f"{label}_n", 0.0),
            )
            mean_mu = (
                stat.get(
                    f"{label}_mu_sum",
                    0.0,
                )
                / n
            )

            row[
                f"{label}_mean_visual_attention_per_patch"
            ] = mean_mu
            row[
                f"{label}_mean_added_attention_per_bbox_patch"
            ] = float(alpha) * mean_mu
            row[
                f"{label}_mean_added_total_attention_mass"
            ] = (
                stat.get(
                    f"{label}_added_mass_sum",
                    0.0,
                )
                / n
            )
            row[
                f"{label}_mean_original_visual_attention_mass"
            ] = (
                stat.get(
                    f"{label}_visual_mass_sum",
                    0.0,
                )
                / n
            )
            row[
                f"{label}_mean_bbox_tokens"
            ] = (
                stat.get(
                    f"{label}_bbox_tokens_sum",
                    0.0,
                )
                / n
            )
            row[
                f"{label}_mean_visual_tokens"
            ] = (
                stat.get(
                    f"{label}_visual_tokens_sum",
                    0.0,
                )
                / n
            )

        stat_rows.append(row)

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(stats_output, index=False)

    print("\n" + "=" * 180)
    print("FULL-DATASET RESULT — NO GROUPS")
    print("=" * 180)
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved:")
    print(" ", eligible_output)
    print(" ", sample_output)
    print(" ", sample_csv_output)
    print(" ", summary_output)
    print(" ", stats_output)
    if (
        error_output.exists()
        and error_output.stat().st_size
    ):
        print(" ", error_output)

    del model, processor, layers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
