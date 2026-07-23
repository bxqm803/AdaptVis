#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import math
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

import transformers
from transformers import AutoProcessor

import trace_centroid_generation_groups_v2_1 as core
import run_spatial_repair_three_experiments_v1 as base
import analyze_object_visual_attention_layers_v1 as attention_module


RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {name: i for i, name in enumerate(RELATIONS)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--n-per-group", type=int, default=5)
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--bbox-jsonl",
        default="output/groundingdino_coco_two_bboxes/bboxes_by_sid.jsonl",
    )
    p.add_argument(
        "--bbox-metrics-csv",
        default="",
        help="默认自动找 output/attention_bbox_grounding/coco/<model>/sample_layer_bbox_metrics.csv",
    )
    p.add_argument(
        "--group-npz",
        default="",
        help="默认自动找 output/object_visual_attention_layer_analysis/coco/<model>/sample_layer_metrics.npz",
    )
    p.add_argument(
        "--output-dir",
        default="output/attention_bbox_visual_samples",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def selected_box(row: Dict[str, Any], key: str) -> Optional[List[float]]:
    obj = row.get(key, {})
    if not isinstance(obj, dict):
        return None
    selected = obj.get("selected")
    if not isinstance(selected, dict):
        return None
    box = selected.get("box_xyxy_original", selected.get("box_xyxy"))
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    return [float(x) for x in box]


def load_bbox_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["sid"])] = row
    return rows


def generation_correct_from_old_group(value: str) -> float:
    text = str(value).strip().lower()

    # 旧固定 A/B/C 语义：
    # A: centroid-C & generation-C
    # B: centroid-C & generation-W
    # C: centroid-W & generation-C
    # 如果还有 D，也当 generation-W
    if text == "a":
        return 1.0
    if text == "b":
        return 0.0
    if text == "c":
        return 1.0
    if text == "d":
        return 0.0

    if "generation_correct" in text or "generation-correct" in text:
        return 1.0
    if (
        "generation_wrong" in text
        or "generation-wrong" in text
        or "generation_incorrect" in text
    ):
        return 0.0

    return np.nan


def find_default_bbox_metrics_csv(model: str) -> Path:
    p = Path("output/attention_bbox_grounding/coco") / model / "sample_layer_bbox_metrics.csv"
    if p.exists():
        return p
    raise FileNotFoundError(f"Missing bbox metrics csv: {p}")


def find_default_group_npz(model: str) -> Path:
    p = Path("output/object_visual_attention_layer_analysis/coco") / model / "sample_layer_metrics.npz"
    if p.exists():
        return p
    raise FileNotFoundError(f"Missing group npz: {p}")


def build_dynamic_groups(
    bbox_metrics_csv: Path,
    group_npz: Path,
    layer: int,
) -> pd.DataFrame:
    bbox = pd.read_csv(bbox_metrics_csv)
    bbox = bbox[bbox["layer"] == layer].copy()

    with np.load(group_npz, allow_pickle=False) as z:
        meta = pd.DataFrame({
            "sid": z["sid"].astype(int),
            "old_group": [str(x) for x in z["group"]],
        })

    meta["generation_correct"] = meta["old_group"].map(generation_correct_from_old_group)
    meta = meta.dropna(subset=["generation_correct"]).copy()
    meta["generation_correct"] = meta["generation_correct"].astype(int)

    merged = bbox.merge(meta, on="sid", how="inner", validate="many_to_one")

    merged["attention_relation_correct"] = pd.to_numeric(
        merged["attention_relation_correct"], errors="coerce"
    )
    merged = merged.dropna(subset=["attention_relation_correct"]).copy()
    merged["centroid_correct"] = merged["attention_relation_correct"] >= 0.5

    def dyn_group(row):
        c = bool(row["centroid_correct"])
        g = bool(row["generation_correct"])
        if c and g:
            return "A"
        if c and (not g):
            return "B"
        if (not c) and g:
            return "C"
        return "D"

    merged["dynamic_group"] = merged.apply(dyn_group, axis=1)
    return merged


def choose_samples(df: pd.DataFrame, n_per_group: int) -> pd.DataFrame:
    # 这里只选 A/B/D
    target_groups = ["A", "B", "D"]
    out = []

    for group in target_groups:
        sub = df[df["dynamic_group"] == group].copy()
        if len(sub) == 0:
            continue

        # 简单稳定排序：先看 binding / centroid-in-box / pointing / lift，再按 sid
        sort_cols = []
        for c in [
            "both_binding_correct",
            "both_centroid_inside",
            "both_pointing",
            "mean_self_box_lift",
            "mean_top_area_iou",
        ]:
            if c in sub.columns:
                sort_cols.append(c)

        if sort_cols:
            sub = sub.sort_values(sort_cols + ["sid"], ascending=[False] * len(sort_cols) + [True])
        else:
            sub = sub.sort_values(["sid"])

        out.append(sub.head(n_per_group))

    if not out:
        return pd.DataFrame()

    chosen = pd.concat(out, axis=0).copy()
    return chosen


def upsample_map_to_image(attn_map: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    x = torch.tensor(attn_map, dtype=torch.float32)[None, None]  # [1,1,H,W]
    y = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0]
    y = y.cpu().numpy()
    y = np.maximum(y, 0.0)
    if y.max() > 0:
        y = y / y.max()
    return y


def draw_box(ax, box, color, label):
    x1, y1, x2, y2 = box
    rect = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor=color,
        linewidth=2.0,
    )
    ax.add_patch(rect)
    ax.text(
        x1,
        max(0, y1 - 4),
        label,
        color=color,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5),
    )


def render_three_panel(
    *,
    save_path: Path,
    image_np: np.ndarray,
    subject_box: List[float],
    reference_box: List[float],
    subject_label: str,
    reference_label: str,
    sid: int,
    dynamic_group: str,
    layer: int,
    gt_relation: str,
    subject_heat: np.ndarray,
    reference_heat: np.ndarray,
):
    h, w = image_np.shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    panels = [
        ("Original + BBoxes", None),
        ("Subject attention", subject_heat),
        ("Reference attention", reference_heat),
    ]

    for ax, (title, heat) in zip(axes, panels):
        ax.imshow(image_np)
        if heat is not None:
            ax.imshow(
                heat,
                cmap="jet",
                alpha=0.42,
                extent=[0, w, h, 0],  # 与原图坐标一致
                interpolation="bilinear",
            )
        draw_box(ax, subject_box, "red", f"S: {subject_label}")
        draw_box(ax, reference_box, "blue", f"R: {reference_label}")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle(
        f"sid={sid} | group={dynamic_group} | layer=L{layer} | gt={gt_relation} | size={w}x{h}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    bbox_metrics_csv = (
        Path(args.bbox_metrics_csv)
        if args.bbox_metrics_csv
        else find_default_bbox_metrics_csv(args.model)
    )
    group_npz = (
        Path(args.group_npz)
        if args.group_npz
        else find_default_group_npz(args.model)
    )
    bbox_jsonl = Path(args.bbox_jsonl)

    chosen_root = Path(args.output_dir) / args.model / f"L{args.layer}"
    if chosen_root.exists() and args.overwrite:
        for p in chosen_root.rglob("*"):
            if p.is_file():
                p.unlink()

    df = build_dynamic_groups(bbox_metrics_csv, group_npz, args.layer)
    chosen = choose_samples(df, args.n_per_group)

    if chosen.empty:
        raise RuntimeError("No samples selected.")

    chosen = chosen.reset_index(drop=True)
    print("=" * 120)
    print(f"Selected {len(chosen)} samples at layer L{args.layer}")
    print(chosen[[
        "sid",
        "dynamic_group",
        "mean_self_box_lift",
        "mean_top_area_iou",
        "both_pointing",
        "both_centroid_inside",
        "both_binding_correct",
    ]].to_string(index=False))
    print("=" * 120)

    # 保存抽样清单
    chosen_csv = chosen_root / "chosen_samples.csv"
    chosen_root.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(chosen_csv, index=False)

    # 加载 bbox rows
    bbox_rows = load_bbox_rows(bbox_jsonl)

    # 加载数据 / prompts / model
    backend = core.import_two_object_module()
    records, _ = backend.load_records(args.dataset, Path(args.data_root), None)
    record_by_sid = {int(r.sid): r for r in records}
    prompt_rows = core.load_standard_prompts(Path(args.prompt_jsonl))

    if args.model not in backend.SPECS:
        raise ValueError(f"Unknown model {args.model}; available={sorted(backend.SPECS)}")

    spec = backend.SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers has no {spec.model_class}")

    print(f"Loading model: {args.model} -> {spec.repo_id}")
    model = model_cls.from_pretrained(
        spec.repo_id,
        dtype=core.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    core.configure_processor(model, processor)
    device = torch.device(args.device)

    layers, layers_path = core.resolve_decoder_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError(f"layer {args.layer} out of range [0,{len(layers)-1}]")

    if not hasattr(attention_module, "LayerGroundingCapture"):
        raise RuntimeError("analyze_object_visual_attention_layers_v1.py missing LayerGroundingCapture")

    capture = attention_module.LayerGroundingCapture(
        layers=layers,
        model=model,
        processor=processor,
        similarity_mode="softmax",
        similarity_temperature=0.07,
        top_fraction=0.10,
        save_maps=True,
    )

    summary_rows = []

    for i, row in chosen.iterrows():
        sid = int(row["sid"])
        group = str(row["dynamic_group"])

        try:
            record = record_by_sid[sid]
            prompt = prompt_rows[sid]
            bbox_row = bbox_rows[sid]

            subject_box = selected_box(bbox_row, "subject")
            reference_box = selected_box(bbox_row, "reference")
            if subject_box is None or reference_box is None:
                print(f"[SKIP] sid={sid} missing bbox")
                continue

            image = core.record_image(record)
            image_np = np.asarray(image.convert("RGB"))
            h, w = image_np.shape[:2]

            question_text = str(prompt["question_text"])
            gt_name = base.normalize_relation(prompt["answer_raw"])
            if gt_name not in REL_TO_ID:
                print(f"[SKIP] sid={sid} invalid gt relation={prompt['answer_raw']!r}")
                image.close()
                continue

            batch = core.make_question_batch(
                processor=processor,
                image=image,
                question_text=question_text,
                device=device,
            )
            batch = base.move_batch_to_device(batch, device)

            prompt_spec = base.build_prompt_position_spec(
                model=model,
                tokenizer=processor.tokenizer,
                input_ids=batch["input_ids"],
                subject=str(prompt["subject"]),
                reference=str(prompt["reference"]),
            )

            capture.configure(
                prompt_spec=prompt_spec,
                batch=batch,
                image_size=tuple(image.size),
                gt_code=REL_TO_ID[gt_name],
            )

            with torch.inference_mode():
                model(
                    **batch,
                    use_cache=False,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                )

            result = capture.results[args.layer]
            if capture.grid_shape is None:
                raise RuntimeError("grid_shape is None")

            grid_h, grid_w = capture.grid_shape
            subject_map = np.asarray(result.attention_subject_map, dtype=np.float32).reshape(grid_h, grid_w)
            reference_map = np.asarray(result.attention_reference_map, dtype=np.float32).reshape(grid_h, grid_w)

            subject_heat = upsample_map_to_image(subject_map, h, w)
            reference_heat = upsample_map_to_image(reference_map, h, w)

            save_name = f"{i+1:02d}_{group}_sid{sid}_L{args.layer}.png"
            save_path = chosen_root / group / save_name

            render_three_panel(
                save_path=save_path,
                image_np=image_np,
                subject_box=subject_box,
                reference_box=reference_box,
                subject_label=str(prompt["subject"]),
                reference_label=str(prompt["reference"]),
                sid=sid,
                dynamic_group=group,
                layer=args.layer,
                gt_relation=gt_name,
                subject_heat=subject_heat,
                reference_heat=reference_heat,
            )

            summary_rows.append({
                "sid": sid,
                "group": group,
                "save_path": str(save_path),
                "subject": str(prompt["subject"]),
                "reference": str(prompt["reference"]),
                "gt_relation": gt_name,
                "grid_h": grid_h,
                "grid_w": grid_w,
            })

            print(f"[OK] {save_path}")
            image.close()

        except Exception as e:
            print(f"[ERROR] sid={sid}: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            capture.reset()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_path = chosen_root / "rendered_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    capture.close()
    del model, processor, layers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nDone.")
    print(f"Chosen list:  {chosen_csv}")
    print(f"Rendered csv: {summary_path}")
    print(f"Figures dir:  {chosen_root}")


if __name__ == "__main__":
    main()
