#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract subject/reference pseudo bounding boxes for AdaptVis COCO-two.

Design constraints
------------------
1. GroundingDINO receives the complete original PIL image. This script never
   center-crops, square-crops, or otherwise removes image borders.
2. GroundingDINO may resize/pad internally using its own processor. Detection
   boxes are post-processed back to ORIGINAL-IMAGE pixel coordinates (xyxy).
3. Subject and reference are queried separately.
4. Candidate selection is label-free: the highest GroundingDINO score is used.
   The spatial GT relation is stored only as metadata and is never used to pick
   a box.
5. No patch-grid mapping is performed here. Later, rasterize each original-space
   box into a mask and pass that mask through the exact VLM processor used for
   the corresponding centroid run. This avoids resize/pad/crop mismatch.

Expected AdaptVis layout
------------------------
  data/coco_qa_two_obj.json
  data/val2017/000000xxxxxx.jpg
  extract_two_object_relation_states.py

The annotation rows are expected to follow the current repo format:
  [image_id, caption, opposite_caption, ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Unable to import transformers GroundingDINO classes: {exc}")


SCRIPT_VERSION = "coco-groundingdino-subject-reference-bboxes-v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="data")
    p.add_argument("--annotation-json", default=None)
    p.add_argument("--image-dir", default=None)
    p.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-base",
        help="HF GroundingDINO checkpoint. Use tiny for a fast smoke test.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="float32 is the most robust default for detector inference.",
    )
    p.add_argument("--box-threshold", type=float, default=0.25)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.50)
    p.add_argument("--max-boxes-per-object", type=int, default=5)
    p.add_argument(
        "--ambiguity-score-gap",
        type=float,
        default=0.08,
        help=(
            "Flag an object as ambiguous when the two highest retained scores "
            "differ by no more than this value."
        ),
    )
    p.add_argument(
        "--sid-npz",
        default="",
        help=(
            "Optional NPZ containing a sid array. Only these source sids are "
            "processed; useful for matching sample_layer_metrics.npz."
        ),
    )
    p.add_argument("--start-sid", type=int, default=0)
    p.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of usable records after filtering; -1 means all.",
    )
    p.add_argument(
        "--output-dir",
        default="output/groundingdino_coco_two_bboxes",
    )
    p.add_argument(
        "--vis-first",
        type=int,
        default=40,
        help="Save audit visualizations for the first N processed records.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute sids already present in the output JSONL.",
    )
    return p.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def read_jsonl_by_sid(path: Path) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rows[int(row["sid"])] = row
            except Exception as exc:
                raise ValueError(f"Bad output JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def normalize_query(text: str) -> str:
    text = " ".join(str(text).replace("\n", " ").split()).strip(" .?\t")
    if not text:
        raise ValueError("Empty GroundingDINO query")
    return text + "."


def box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """IoU between one xyxy box [4] and boxes [N,4]."""
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area_b = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-12)


def nms_indices(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> List[int]:
    """Small dependency-free NMS for original-image xyxy boxes."""
    if boxes.numel() == 0:
        return []
    order = torch.argsort(scores, descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        index = int(order[0].item())
        keep.append(index)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = box_iou_one_to_many(boxes[index], boxes[remaining])
        order = remaining[ious <= float(iou_threshold)]
    return keep


def clamp_box(box: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def get_image_processor_config(processor: Any) -> Dict[str, Any]:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return {}
    if hasattr(image_processor, "to_dict"):
        try:
            return jsonable(image_processor.to_dict())
        except Exception:
            pass
    fields = (
        "do_resize",
        "size",
        "resample",
        "do_rescale",
        "rescale_factor",
        "do_normalize",
        "image_mean",
        "image_std",
        "do_pad",
        "pad_size",
        "do_center_crop",
        "crop_size",
    )
    return {
        field: jsonable(getattr(image_processor, field))
        for field in fields
        if hasattr(image_processor, field)
    }


def move_inputs(
    inputs: Any,
    device: torch.device,
    floating_dtype: torch.dtype,
) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=floating_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


@torch.inference_mode()
def detect_candidates(
    *,
    image: Image.Image,
    query: str,
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
    model_dtype: torch.dtype,
    box_threshold: float,
    text_threshold: float,
    nms_iou: float,
    max_boxes: int,
) -> List[Dict[str, Any]]:
    """Return candidate boxes in ORIGINAL-IMAGE xyxy pixel coordinates."""
    image = image.convert("RGB")
    width, height = image.size
    prompt = normalize_query(query)

    # The complete raw image is passed here. No crop is performed by this script.
    encoded = processor(images=image, text=prompt, return_tensors="pt")
    encoded = move_inputs(encoded, device, model_dtype)
    outputs = model(**encoded)

    post_kwargs = dict(
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        target_sizes=[(height, width)],
    )
    input_ids = encoded.get("input_ids")
    try:
        processed = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=input_ids,
            **post_kwargs,
        )[0]
    except TypeError:
        # Compatibility with transformers versions where input_ids is positional.
        processed = processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            **post_kwargs,
        )[0]

    boxes = processed.get("boxes", torch.empty((0, 4))).detach().cpu().float()
    scores = processed.get("scores", torch.empty((0,))).detach().cpu().float()
    labels = processed.get("text_labels", processed.get("labels", []))
    labels = [str(label) for label in labels]

    valid_indices: List[int] = []
    clamped_boxes: List[List[float]] = []
    for index, box in enumerate(boxes.tolist()):
        clamped = clamp_box(box, width, height)
        x1, y1, x2, y2 = clamped
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue
        valid_indices.append(index)
        clamped_boxes.append(clamped)

    if not valid_indices:
        return []

    valid_boxes = torch.tensor(clamped_boxes, dtype=torch.float32)
    valid_scores = scores[torch.tensor(valid_indices, dtype=torch.long)]
    keep_local = nms_indices(valid_boxes, valid_scores, nms_iou)
    keep_local = keep_local[: max(1, int(max_boxes))]

    candidates: List[Dict[str, Any]] = []
    for rank, local_index in enumerate(keep_local):
        original_index = valid_indices[local_index]
        box = valid_boxes[local_index].tolist()
        x1, y1, x2, y2 = box
        candidates.append(
            {
                "rank": rank,
                "box_xyxy_original": [float(v) for v in box],
                "box_xyxy_normalized": [
                    float(x1 / width),
                    float(y1 / height),
                    float(x2 / width),
                    float(y2 / height),
                ],
                "score": float(scores[original_index].item()),
                "label": labels[original_index] if original_index < len(labels) else prompt,
                "area_fraction": float(((x2 - x1) * (y2 - y1)) / (width * height)),
            }
        )
    return candidates


def object_result(
    phrase: str,
    candidates: List[Dict[str, Any]],
    ambiguity_score_gap: float,
) -> Dict[str, Any]:
    selected = candidates[0] if candidates else None
    score_gap = None
    ambiguous = False
    if len(candidates) >= 2:
        score_gap = float(candidates[0]["score"] - candidates[1]["score"])
        ambiguous = score_gap <= float(ambiguity_score_gap)
    return {
        "phrase": phrase,
        "query": normalize_query(phrase),
        "num_candidates": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "selected_by": "highest_groundingdino_score_after_nms",
        "score_gap_top1_top2": score_gap,
        "missing": selected is None,
        "ambiguous": bool(ambiguous),
    }


def draw_audit(
    image: Image.Image,
    sid: int,
    subject: Dict[str, Any],
    reference: Dict[str, Any],
    output_path: Path,
) -> None:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width = max(2, int(round(min(canvas.size) / 180)))

    def draw_object(obj: Dict[str, Any], color: Tuple[int, int, int], prefix: str) -> None:
        for candidate in obj["candidates"]:
            x1, y1, x2, y2 = candidate["box_xyxy_original"]
            rank = int(candidate["rank"])
            line_width = width + 2 if rank == 0 else width
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            text = f"{prefix}{rank}:{candidate['score']:.3f}"
            draw.text((x1 + 2, max(0, y1 - 13)), text, fill=color)

    draw_object(subject, (255, 40, 40), "S")
    draw_object(reference, (40, 120, 255), "R")
    header = (
        f"sid={sid} | S={subject['phrase']} | R={reference['phrase']} | "
        f"S_amb={subject['ambiguous']} R_amb={reference['ambiguous']}"
    )
    draw.rectangle([0, 0, min(canvas.width, 12 * len(header)), 19], fill=(255, 255, 255))
    draw.text((3, 3), header, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def load_sid_filter(path: str) -> Optional[set[int]]:
    if not path:
        return None
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing --sid-npz: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        if "sid" not in data.files:
            raise KeyError(f"{npz_path} has no 'sid' array; keys={data.files}")
        return {int(value) for value in data["sid"].tolist()}


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if not 0.0 <= args.box_threshold <= 1.0:
        raise ValueError("--box-threshold must be in [0,1]")
    if not 0.0 <= args.text_threshold <= 1.0:
        raise ValueError("--text-threshold must be in [0,1]")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be in [0,1]")

    repo_root = Path.cwd()
    data_root = Path(args.data_root)
    annotation_path = (
        Path(args.annotation_json)
        if args.annotation_json
        else data_root / "coco_qa_two_obj.json"
    )
    image_dir = Path(args.image_dir) if args.image_dir else data_root / "val2017"
    relation_module_path = repo_root / "extract_two_object_relation_states.py"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotations: {annotation_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not relation_module_path.exists():
        raise FileNotFoundError(
            "Run from the AdaptVis repository root; missing "
            f"{relation_module_path}"
        )

    import importlib

    relation_module = importlib.import_module("extract_two_object_relation_states")
    if not hasattr(relation_module, "parse_relation_caption"):
        raise AttributeError(
            "extract_two_object_relation_states.py has no parse_relation_caption"
        )

    raw_rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    sid_filter = load_sid_filter(args.sid_npz)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "bboxes_by_sid.jsonl"
    error_jsonl = output_dir / "errors.jsonl"
    summary_json = output_dir / "summary.json"
    config_json = output_dir / "config.json"
    vis_dir = output_dir / "visualizations"

    existing = {} if args.overwrite else read_jsonl_by_sid(output_jsonl)
    if args.overwrite:
        for path in (output_jsonl, error_jsonl, summary_json, config_json):
            if path.exists():
                path.unlink()

    device = torch.device(args.device)
    model_dtype = resolve_dtype(args.dtype)

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Loading GroundingDINO: {args.model_id}")
    print(f"Device={device}; dtype={model_dtype}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    processor_config = get_image_processor_config(processor)
    do_center_crop = bool(processor_config.get("do_center_crop", False))
    if do_center_crop:
        raise RuntimeError(
            "GroundingDINO processor unexpectedly reports do_center_crop=True. "
            "Stop here rather than risk border truncation."
        )

    config = {
        "script_version": SCRIPT_VERSION,
        "model_id": args.model_id,
        "device": args.device,
        "dtype": args.dtype,
        "data_root": str(data_root),
        "annotation_json": str(annotation_path),
        "image_dir": str(image_dir),
        "sid_npz": args.sid_npz,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "nms_iou": args.nms_iou,
        "max_boxes_per_object": args.max_boxes_per_object,
        "ambiguity_score_gap": args.ambiguity_score_gap,
        "input_geometry": "full_original_image_no_script_crop",
        "output_box_geometry": "original_image_xyxy_pixels",
        "candidate_selection_uses_gt_relation": False,
        "processor_config": processor_config,
    }
    config_json.write_text(
        json.dumps(jsonable(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("GroundingDINO image processor geometry:")
    print(json.dumps(processor_config, ensure_ascii=False, indent=2)[:4000])
    print("No center crop will be used by this script.")
    print("Boxes will be saved in original-image xyxy pixel coordinates.")

    counts: Counter[str] = Counter()
    processed_now = 0
    vis_saved = 0

    for sid, row in enumerate(tqdm(raw_rows, desc="GroundingDINO bboxes")):
        if sid < args.start_sid:
            continue
        if sid_filter is not None and sid not in sid_filter:
            continue
        if sid in existing and not args.overwrite:
            counts["resumed_existing"] += 1
            continue
        if args.limit > 0 and processed_now >= args.limit:
            break

        try:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                raise ValueError(f"Invalid annotation row: {row!r}")
            image_id = str(row[0])
            caption = str(row[1])
            opposite_caption = str(row[2])
            parsed = relation_module.parse_relation_caption(caption)
            if parsed is None:
                raise ValueError(f"Could not parse relation caption: {caption!r}")
            subject_phrase, reference_phrase, relation = parsed

            image_path = image_dir / f"{int(image_id):012d}.jpg"
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image: {image_path}")
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            subject_candidates = detect_candidates(
                image=image,
                query=subject_phrase,
                processor=processor,
                model=model,
                device=device,
                model_dtype=model_dtype,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                nms_iou=args.nms_iou,
                max_boxes=args.max_boxes_per_object,
            )
            reference_candidates = detect_candidates(
                image=image,
                query=reference_phrase,
                processor=processor,
                model=model,
                device=device,
                model_dtype=model_dtype,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                nms_iou=args.nms_iou,
                max_boxes=args.max_boxes_per_object,
            )

            subject_result = object_result(
                subject_phrase,
                subject_candidates,
                args.ambiguity_score_gap,
            )
            reference_result = object_result(
                reference_phrase,
                reference_candidates,
                args.ambiguity_score_gap,
            )

            record = {
                "sid": sid,
                "image_id": image_id,
                "image_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "caption": caption,
                "opposite_caption": opposite_caption,
                "relation_gt_metadata_only": relation,
                "subject": subject_result,
                "reference": reference_result,
                "both_found": bool(
                    not subject_result["missing"] and not reference_result["missing"]
                ),
                "either_ambiguous": bool(
                    subject_result["ambiguous"] or reference_result["ambiguous"]
                ),
                "detector": args.model_id,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "nms_iou": args.nms_iou,
                "input_geometry": "full_original_image_no_script_crop",
                "box_coordinate_system": "original_image_xyxy_pixels",
                "selection_uses_gt_relation": False,
            }
            append_jsonl(output_jsonl, record)

            processed_now += 1
            counts["processed"] += 1
            if record["both_found"]:
                counts["both_found"] += 1
            if subject_result["missing"]:
                counts["subject_missing"] += 1
            if reference_result["missing"]:
                counts["reference_missing"] += 1
            if subject_result["ambiguous"]:
                counts["subject_ambiguous"] += 1
            if reference_result["ambiguous"]:
                counts["reference_ambiguous"] += 1
            if record["either_ambiguous"]:
                counts["either_ambiguous"] += 1

            if vis_saved < args.vis_first:
                draw_audit(
                    image,
                    sid,
                    subject_result,
                    reference_result,
                    vis_dir / f"sid_{sid:04d}.jpg",
                )
                vis_saved += 1

        except Exception as exc:
            counts["errors"] += 1
            append_jsonl(
                error_jsonl,
                {
                    "sid": sid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "row": row,
                },
            )

    # Re-read output so resumed runs receive a complete summary.
    all_output = read_jsonl_by_sid(output_jsonl)
    complete_counts: Counter[str] = Counter()
    for record in all_output.values():
        complete_counts["records"] += 1
        if record.get("both_found"):
            complete_counts["both_found"] += 1
        if record.get("subject", {}).get("missing"):
            complete_counts["subject_missing"] += 1
        if record.get("reference", {}).get("missing"):
            complete_counts["reference_missing"] += 1
        if record.get("subject", {}).get("ambiguous"):
            complete_counts["subject_ambiguous"] += 1
        if record.get("reference", {}).get("ambiguous"):
            complete_counts["reference_ambiguous"] += 1
        if record.get("either_ambiguous"):
            complete_counts["either_ambiguous"] += 1

    denominator = max(1, complete_counts["records"])
    summary = {
        "script_version": SCRIPT_VERSION,
        "output_jsonl": str(output_jsonl),
        "error_jsonl": str(error_jsonl),
        "visualization_dir": str(vis_dir),
        "current_run_counts": dict(counts),
        "complete_counts": dict(complete_counts),
        "both_found_rate": complete_counts["both_found"] / denominator,
        "either_ambiguous_rate": complete_counts["either_ambiguous"] / denominator,
        "selection_uses_gt_relation": False,
        "box_coordinate_system": "original_image_xyxy_pixels",
    }
    summary_json.write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("GROUNDINGDINO COCO-TWO BBOX EXTRACTION")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"BBoxes: {output_jsonl}")
    print(f"Errors: {error_jsonl}")
    print(f"Audit images: {vis_dir}")
    print(f"Config: {config_json}")


if __name__ == "__main__":
    main()
