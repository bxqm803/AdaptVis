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

import trace_centroid_generation_groups_v2_1 as core
import run_spatial_repair_three_experiments_v1 as repair
import analyze_object_visual_attention_layers_v1 as grounding
import evaluate_attention_grounding_with_gdino_bboxes_v1 as bbox_eval


# ============================================================================
# First experiment configuration
# ============================================================================
MODEL_NAME = "qwen-3b"
DEVICE = "cuda:0"

# 0.0 is an exact same-path baseline.
# 1.0 means: for every bbox visual token, add one per-head/per-query
# full-image mean visual attention probability.
ADD_SCALES = [0.0, 1.0]

# Set 20 for a quick smoke run; None means all eligible samples.
MAX_PER_GROUP = None

EXCLUDE_AMBIGUOUS = True
BBOX_THRESHOLD = 0.25
MAX_NEW_TOKENS = 8

DATA_ROOT = Path("data")
PROMPT_JSONL = Path("prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
INPUT_ROOT = Path("output/three_group_transfer_fresh/coco")
BBOX_JSONL = Path("output/groundingdino_coco_two_bboxes/bboxes_by_sid.jsonl")
OUTPUT_DIR = Path("output/bbox_object_token_additive_attention/coco") / MODEL_NAME


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
        stat = self.stats[layer_id]
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_output = OUTPUT_DIR / "sample_results.jsonl"
    error_output = OUTPUT_DIR / "errors.jsonl"
    summary_output = OUTPUT_DIR / "summary.csv"
    stats_output = OUTPUT_DIR / "added_attention_stats.csv"

    for path in [
        sample_output,
        error_output,
        summary_output,
        stats_output,
    ]:
        if path.exists():
            path.unlink()

    bbox_by_sid = {
        int(row["sid"]): row
        for row in load_jsonl(BBOX_JSONL)
    }

    metadata_path = (
        INPUT_ROOT
        / MODEL_NAME
        / "pass2_transfer_trace"
        / "sample_metadata.jsonl"
    )
    prior_rows = repair.read_jsonl(metadata_path)
    selected_groups = [
        repair.GROUP_A,
        repair.GROUP_B,
        repair.GROUP_C,
        repair.GROUP_D,
    ]
    prior_rows = repair.cap_rows(
        prior_rows,
        selected_groups=selected_groups,
        max_per_group=MAX_PER_GROUP,
        seed=0,
        sid=None,
    )

    prior_rows = [
        row
        for row in prior_rows
        if int(row["sid"]) in bbox_by_sid
        and selected_box(
            bbox_by_sid[int(row["sid"])], "subject"
        ) is not None
        and selected_box(
            bbox_by_sid[int(row["sid"])], "reference"
        ) is not None
        and (
            not EXCLUDE_AMBIGUOUS
            or not bool(
                bbox_by_sid[int(row["sid"])].get(
                    "either_ambiguous", False
                )
            )
        )
    ]

    backend = core.import_two_object_module()
    records, _ = backend.load_records(
        "coco_two", DATA_ROOT, None
    )
    record_by_sid = {
        int(record.sid): record for record in records
    }
    prompt_rows = core.load_standard_prompts(
        PROMPT_JSONL
    )

    spec = backend.SPECS[MODEL_NAME]
    model_cls = getattr(
        transformers, spec.model_class
    )

    print("=" * 130)
    print(
        "BBOX-TARGETED ADDITIVE OBJECT-TOKEN ATTENTION"
    )
    print("=" * 130)
    print(f"model={MODEL_NAME}")
    print(f"samples={len(prior_rows)}")
    print(f"alphas={ADD_SCALES}")
    print("all decoder layers; all attention heads")
    print(
        "A'[object_query,bbox_key] = "
        "A + alpha * "
        "mean(A[object_query,all_visual_keys])"
    )
    print("No multiplication. No renormalization.")
    print("=" * 130)

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
    layers, layers_path = (
        core.resolve_decoder_layers(model)
    )
    manager = BBoxObjectTokenAddManager(layers)

    rows = []
    progress = tqdm(
        prior_rows,
        desc=f"bbox-add:{MODEL_NAME}",
        unit="sample",
        dynamic_ncols=True,
    )

    try:
        for prior in progress:
            sid = int(prior["sid"])
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
                gt = repair.normalize_relation(
                    prompt["answer_raw"]
                )
                if gt not in repair.RELATIONS:
                    raise RuntimeError(
                        f"Invalid GT: "
                        f"{prompt['answer_raw']!r}"
                    )

                subject_box = selected_box(
                    bbox_row, "subject"
                )
                reference_box = selected_box(
                    bbox_row, "reference"
                )

                batch = core.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=question,
                    device=device,
                )
                batch = repair.move_batch_to_device(
                    batch, device
                )

                prompt_spec = (
                    repair.build_prompt_position_spec(
                        model=model,
                        tokenizer=processor.tokenizer,
                        input_ids=batch["input_ids"],
                        subject=subject,
                        reference=reference,
                    )
                )

                input_length = int(
                    batch["input_ids"].shape[1]
                )
                expanded = repair.expand_positions(
                    prompt_spec,
                    hidden_length=input_length,
                )
                visual_count = len(
                    expanded.visual_positions
                )
                grid_h, grid_w, grid_source = (
                    grounding.infer_grid(
                        visual_count,
                        batch,
                        tuple(image.size),
                        model,
                        processor,
                    )
                )

                subject_soft, reference_soft, _ = (
                    bbox_eval.processor_aware_box_masks(
                        core=core,
                        processor=processor,
                        question_text=question,
                        image_size=tuple(image.size),
                        subject_box=subject_box,
                        reference_box=reference_box,
                        grid_h=grid_h,
                        grid_w=grid_w,
                    )
                )
                subject_mask = hard_mask(
                    subject_soft, BBOX_THRESHOLD
                )
                reference_mask = hard_mask(
                    reference_soft, BBOX_THRESHOLD
                )

                results = {}
                for alpha in ADD_SCALES:
                    manager.configure(
                        alpha=alpha,
                        prompt_spec=prompt_spec,
                        subject_visual_mask=subject_mask,
                        reference_visual_mask=reference_mask,
                    )
                    text, prediction = (
                        repair.generate_relation(
                            model=model,
                            processor=processor,
                            batch=batch,
                            max_new_tokens=MAX_NEW_TOKENS,
                            need_attentions=(
                                float(alpha) != 0.0
                            ),
                        )
                    )
                    results[float(alpha)] = (
                        text, prediction
                    )
                    manager.reset()

                baseline_text, baseline_prediction = (
                    results[0.0]
                )
                baseline_correct = (
                    baseline_prediction == gt
                )

                for alpha in ADD_SCALES:
                    alpha = float(alpha)
                    text, prediction = results[alpha]
                    row = {
                        "model": MODEL_NAME,
                        "sid": sid,
                        "group": str(prior["group"]),
                        "group_short": repair.group_short(
                            str(prior["group"])
                        ),
                        "gt": gt,
                        "subject": subject,
                        "reference": reference,
                        "grid_height": grid_h,
                        "grid_width": grid_w,
                        "grid_source": grid_source,
                        "subject_bbox_tokens": int(
                            subject_mask.sum()
                        ),
                        "reference_bbox_tokens": int(
                            reference_mask.sum()
                        ),
                        "visual_tokens": int(
                            subject_mask.size
                        ),
                        "alpha": alpha,
                        "baseline_prediction": (
                            baseline_prediction
                        ),
                        "baseline_text": baseline_text,
                        "baseline_correct": bool(
                            baseline_correct
                        ),
                        "prediction": prediction,
                        "text": text,
                        "parsed": (
                            prediction in repair.RELATIONS
                        ),
                        "correct": prediction == gt,
                        "wrong_to_correct": (
                            not baseline_correct
                            and prediction == gt
                        ),
                        "correct_to_wrong": (
                            baseline_correct
                            and prediction != gt
                        ),
                        "changed_prediction": (
                            prediction
                            != baseline_prediction
                        ),
                    }
                    append_jsonl(
                        sample_output, row
                    )
                    rows.append(row)

            except Exception as exc:
                append_jsonl(
                    error_output,
                    {
                        "sid": sid,
                        "group": prior.get("group"),
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error": str(exc),
                        "traceback_tail": (
                            traceback.format_exc()
                            .splitlines()[-30:]
                        ),
                    },
                )
                tqdm.write(
                    f"[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )
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

    df = pd.DataFrame(rows)
    summary_rows = []

    for alpha in sorted(df["alpha"].unique()):
        part = df[df["alpha"] == alpha].copy()
        base_acc = float(
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
            "alpha": alpha,
            "n": len(part),
            "parse_rate": float(
                part["parsed"].mean()
            ),
            "baseline_accuracy": base_acc,
            "accuracy": accuracy,
            "accuracy_change": (
                accuracy - base_acc
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
            "wrong_to_correct_rate_all": float(
                part["wrong_to_correct"].mean()
            ),
            "correct_to_wrong_rate_all": float(
                part["correct_to_wrong"].mean()
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

        for group in selected_groups:
            group_rows = part[
                part["group"] == group
            ]
            short = repair.group_short(group)
            summary[f"{short}_n"] = len(
                group_rows
            )
            summary[f"{short}_accuracy"] = (
                float(group_rows["correct"].mean())
                if len(group_rows)
                else float("nan")
            )

        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        summary_output, index=False
    )

    stat_rows = []
    for layer_id in sorted(manager.stats):
        stat = manager.stats[layer_id]
        row = {"layer": layer_id}
        for label in ("subject", "reference"):
            n = max(
                1.0, stat.get(f"{label}_n", 0.0)
            )
            row[
                f"{label}_mean_visual_attention_per_patch"
            ] = (
                stat.get(
                    f"{label}_mu_sum", 0.0
                )
                / n
            )
            row[
                f"{label}_mean_added_total_attention_mass"
            ] = (
                stat.get(
                    f"{label}_added_mass_sum", 0.0
                )
                / n
            )
            row[
                f"{label}_mean_original_visual_attention_mass"
            ] = (
                stat.get(
                    f"{label}_visual_mass_sum", 0.0
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

    print("\n" + "=" * 150)
    print("RESULT")
    print("=" * 150)
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 150)
    print("MEAN ACTUAL ADDITION BY LAYER")
    print("=" * 150)
    show_layers = sorted(
        set(
            list(range(min(5, len(layers))))
            + list(
                range(
                    max(0, len(layers) - 5),
                    len(layers),
                )
            )
        )
    )
    print(
        stats_df[
            stats_df["layer"].isin(show_layers)
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nSaved:")
    print(" ", sample_output)
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
