#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate whether object-token attention maps localize GroundingDINO boxes.

GroundingDINO boxes are drawn as masks in the ORIGINAL image, then passed
through the exact same VLM processor as the real image. This avoids manual
resize/crop/pad assumptions. The frozen model is rerun with eager attention.
"""
from __future__ import annotations

import argparse, contextlib, csv, gc, importlib, json, math, random, shutil, traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import run_spatial_repair_three_experiments_v1 as base

SCRIPT_VERSION = "attention-bbox-grounding-v1"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {x: i for i, x in enumerate(RELATIONS)}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", default="qwen-3b,qwen-7b")
    p.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--bbox-jsonl", default="output/groundingdino_coco_two_bboxes/bboxes_by_sid.jsonl")
    p.add_argument("--metadata-root", default="output/object_visual_attention_layer_analysis/coco")
    p.add_argument("--output-root", default="output/attention_bbox_grounding/coco")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bbox-hard-threshold", type=float, default=0.25)
    p.add_argument("--include-ambiguous", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--sid", type=int, default=None)
    p.add_argument("--vis-first", type=int, default=30)
    p.add_argument("--vis-layer", default="auto")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--empty-cache-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attention-module", default="analyze_object_visual_attention_layers_v1")
    p.add_argument("--core-module", default="trace_centroid_generation_groups_v2_1")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"Bad JSONL {path}:{ln}: {e}") from e
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def selected_box(row: Mapping[str, Any], key: str) -> Optional[List[float]]:
    obj = row.get(key, {})
    sel = obj.get("selected") if isinstance(obj, Mapping) else None
    if not isinstance(sel, Mapping):
        return None
    box = sel.get("box_xyxy_original", sel.get("box_xyxy"))
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    return [float(v) for v in box]


def load_bbox_by_sid(path: Path) -> Dict[int, Dict[str, Any]]:
    out = {int(x["sid"]): x for x in read_jsonl(path)}
    if not out:
        raise RuntimeError(f"No bbox records in {path}")
    return out


def target_sids(model: str, metadata_root: Path, all_sids: Sequence[int], max_samples, only_sid):
    p = metadata_root / model / "sample_metadata.jsonl"
    sids = [int(x["sid"]) for x in read_jsonl(p)] if p.exists() else list(all_sids)
    sids = list(dict.fromkeys(sids))
    if only_sid is not None:
        sids = [x for x in sids if x == int(only_sid)]
    if max_samples is not None:
        sids = sids[:int(max_samples)]
    return sids


def resolve_vis_layer(model: str, metadata_root: Path, value: str, n_layers: int) -> int:
    if str(value).lower() != "auto":
        layer = int(value)
        if not 0 <= layer < n_layers:
            raise ValueError(f"vis layer {layer} outside [0,{n_layers-1}]")
        return layer
    p = metadata_root / model / "layer_summary.csv"
    if p.exists():
        with p.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        scored = []
        for r in rows:
            try:
                score = float(r.get("attention_macro_accuracy", r.get("attention_accuracy", "nan")))
                layer = int(r["layer"])
                if math.isfinite(score): scored.append((score, -layer, layer))
            except Exception:
                pass
        if scored:
            return max(scored)[2]
    return n_layers // 2


def box_mask(size: Tuple[int, int], box: Sequence[float]) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 > x1 and y2 > y1:
        d.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
    return img


def make_batch(core, processor, image, question_text, device):
    return core.make_question_batch(processor=processor, image=image, question_text=question_text, device=device)


def pixel_tensor(batch: Mapping[str, Any]) -> Tuple[str, torch.Tensor]:
    for key in ("pixel_values", "pixel_values_images", "image_pixel_values"):
        if torch.is_tensor(batch.get(key)):
            return key, batch[key].detach().float().cpu()
    for key, value in batch.items():
        if "pixel" in str(key).lower() and torch.is_tensor(value):
            return str(key), value.detach().float().cpu()
    raise RuntimeError(f"No pixel tensor; keys={list(batch.keys())}")


def occupancy_ratio(mask: torch.Tensor, black: torch.Tensor, white: torch.Tensor) -> torch.Tensor:
    if mask.shape != black.shape or mask.shape != white.shape:
        raise RuntimeError(f"mask/black/white shapes differ: {mask.shape}, {black.shape}, {white.shape}")
    den = white - black
    valid = den.abs() > 1e-6
    out = torch.zeros_like(mask, dtype=torch.float32)
    out[valid] = (mask[valid] - black[valid]) / den[valid]
    return out.clamp(0, 1)


def processed_mask_grid(mask_batch, black_batch, white_batch, grid_h: int, grid_w: int) -> np.ndarray:
    km, m = pixel_tensor(mask_batch); kb, b = pixel_tensor(black_batch); kw, w = pixel_tensor(white_batch)
    if km != kb or km != kw:
        raise RuntimeError(f"pixel keys differ: {km}, {kb}, {kw}")
    ratio = occupancy_ratio(m, b, w)

    if ratio.ndim == 4:  # [B,C,H,W], LLaVA/CLIP-like
        x = ratio[0].mean(0)[None, None]
        return F.adaptive_avg_pool2d(x, (grid_h, grid_w))[0, 0].numpy().astype(np.float32)

    if ratio.ndim == 5:
        x = ratio[0]
        if x.shape[0] in (1, 3, 4):
            x = x.mean((0, 1))[None, None]
        elif x.shape[1] in (1, 3, 4):
            x = x.mean((0, 1))[None, None]
        else:
            raise RuntimeError(f"Unsupported 5D pixel shape {tuple(ratio.shape)}")
        return F.adaptive_avg_pool2d(x, (grid_h, grid_w))[0, 0].numpy().astype(np.float32)

    if ratio.ndim == 3 and ratio.shape[0] == 1:
        ratio = ratio[0]
    if ratio.ndim == 2:  # Qwen2.5-VL flattened patch rows [N,D]
        occ = ratio.mean(-1)
        g = mask_batch.get("image_grid_thw")
        if not torch.is_tensor(g) or g.numel() < 3:
            raise RuntimeError("flattened pixel tensor requires image_grid_thw")
        t, raw_h, raw_w = [int(v) for v in g.detach().cpu().reshape(-1, 3)[0].tolist()]
        if occ.numel() != t * raw_h * raw_w:
            raise RuntimeError(f"patch rows={occ.numel()} != image_grid_thw={t}x{raw_h}x{raw_w}")
        raw = occ.reshape(t, raw_h, raw_w).mean(0)[None, None]
        return F.adaptive_avg_pool2d(raw, (grid_h, grid_w))[0, 0].numpy().astype(np.float32)

    raise RuntimeError(f"Unsupported pixel tensor shape {tuple(ratio.shape)}")


def processor_box_masks(core, processor, question_text, size, sbox, rbox, grid_h, grid_w):
    cpu = torch.device("cpu")
    w, h = size
    si, ri = box_mask(size, sbox), box_mask(size, rbox)
    bi, wi = Image.new("RGB", (w, h), (0, 0, 0)), Image.new("RGB", (w, h), (255, 255, 255))
    sb = make_batch(core, processor, si, question_text, cpu)
    rb = make_batch(core, processor, ri, question_text, cpu)
    bb = make_batch(core, processor, bi, question_text, cpu)
    wb = make_batch(core, processor, wi, question_text, cpu)
    sm = processed_mask_grid(sb, bb, wb, grid_h, grid_w)
    rm = processed_mask_grid(rb, bb, wb, grid_h, grid_w)
    meta = {"pixel_key": pixel_tensor(sb)[0], "pixel_shape": list(pixel_tensor(sb)[1].shape),
            "image_grid_thw": sb["image_grid_thw"].detach().cpu().tolist() if torch.is_tensor(sb.get("image_grid_thw")) else None}
    for x in (si, ri, bi, wi): x.close()
    del sb, rb, bb, wb
    return sm, rm, meta

def hard_mask(mask: np.ndarray, threshold: float) -> np.ndarray:
    x = np.asarray(mask, dtype=np.float64).reshape(-1)
    out = x >= threshold
    if not out.any(): out[int(np.argmax(x))] = True
    return out


def map_metrics(prob_map: np.ndarray, target_soft: np.ndarray, threshold: float) -> Dict[str, float]:
    p = np.clip(np.asarray(prob_map, dtype=np.float64).reshape(-1), 0, None)
    p = p / p.sum() if p.sum() > 1e-12 else np.full_like(p, 1 / len(p))
    t = np.clip(np.asarray(target_soft, dtype=np.float64).reshape(-1), 0, 1)
    if len(p) != len(t): raise RuntimeError(f"map/target mismatch {len(p)} vs {len(t)}")
    h = hard_mask(t, threshold)
    area_soft, area_hard = float(t.mean()), float(h.mean())
    mass = float(np.dot(p, t)); lift = mass / max(area_soft, 1e-12)
    peak = float(h[int(np.argmax(p))])

    gh, gw = target_soft.shape
    ys = (np.arange(gh) + .5) / gh; xs = (np.arange(gw) + .5) / gw
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    cx, cy = float(np.dot(p, xx.ravel())), float(np.dot(p, yy.ravel()))
    col = min(gw - 1, max(0, int(math.floor(cx * gw))))
    row = min(gh - 1, max(0, int(math.floor(cy * gh))))
    centroid_inside = float(h[row * gw + col])

    k = max(1, int(h.sum()))
    idx = np.argpartition(p, -k)[-k:]
    pred = np.zeros_like(h); pred[idx] = True
    inter = int(np.logical_and(pred, h).sum()); union = int(np.logical_or(pred, h).sum())
    inside_mean = float(p[h].mean())
    outside_mean = float(p[~h].mean()) if (~h).any() else float("nan")
    return {
        "box_area_soft": area_soft, "box_area_hard": area_hard,
        "box_mass": mass, "box_lift": lift, "pointing": peak,
        "centroid_inside": centroid_inside,
        "top_area_iou": inter / max(1, union),
        "inside_outside_ratio": inside_mean / max(outside_mean, 1e-12),
        "centroid_x": cx, "centroid_y": cy,
    }


def box_iou(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    x, y = hard_mask(a, threshold), hard_mask(b, threshold)
    return int(np.logical_and(x, y).sum()) / max(1, int(np.logical_or(x, y).sum()))


def safe_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64); x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def layer_summary(rows: Sequence[Mapping[str, Any]], clean: bool, include_ambiguous: bool):
    groups = defaultdict(list)
    for r in rows:
        if clean and bool(r["either_ambiguous"]) and not include_ambiguous: continue
        groups[int(r["layer"])].append(r)
    mean_fields = (
        "subject_self_box_mass", "reference_self_box_mass", "mean_self_box_mass",
        "subject_self_box_lift", "reference_self_box_lift", "mean_self_box_lift",
        "subject_cross_box_lift", "reference_cross_box_lift", "mean_binding_lift_margin",
        "subject_top_area_iou", "reference_top_area_iou", "mean_top_area_iou",
        "box_mask_iou", "attention_relation_correct",
    )
    rate_fields = (
        "subject_pointing", "reference_pointing", "both_pointing",
        "subject_centroid_inside", "reference_centroid_inside", "both_centroid_inside",
        "subject_binding_correct", "reference_binding_correct", "both_binding_correct",
        "relation_correct_and_both_pointing", "relation_correct_but_not_both_pointing",
        "relation_correct_and_both_centroid", "relation_correct_but_not_both_centroid",
    )
    out = []
    for layer in sorted(groups):
        rs = groups[layer]
        row = {"layer": layer, "n": len(rs), "scope": "clean" if clean else "all_both_found"}
        for f in mean_fields + rate_fields: row[f] = safe_mean(float(x[f]) for x in rs)
        out.append(row)
    return out


def best_layers(rows: Sequence[Mapping[str, Any]]):
    objectives = {
        "best_mean_self_box_lift": "mean_self_box_lift",
        "best_both_pointing": "both_pointing",
        "best_both_centroid_inside": "both_centroid_inside",
        "best_both_binding": "both_binding_correct",
        "best_mean_top_area_iou": "mean_top_area_iou",
    }
    out = {}
    for name, field in objectives.items():
        valid = [r for r in rows if math.isfinite(float(r.get(field, float("nan"))))]
        if valid:
            x = max(valid, key=lambda r: (float(r[field]), -int(r["layer"])))
            out[name] = {"layer": int(x["layer"]), "value": float(x[field]), "metric": field}
    return out


def save_vis(path: Path, image: Image.Image, sid: int, subject: str, reference: str,
             sbox, rbox, sattn, rattn, smask, rmask, layer: int):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0,0].imshow(np.asarray(image.convert("RGB")))
    for box, label, color in ((sbox, f"S:{subject}", "red"), (rbox, f"R:{reference}", "blue")):
        x1,y1,x2,y2 = box
        ax[0,0].add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=2))
        ax[0,0].text(x1, max(0,y1-3), label, color=color, fontsize=9)
    ax[0,0].set_title(f"sid={sid} original boxes"); ax[0,0].axis("off")
    for a, data, title in (
        (ax[0,1], sattn, f"L{layer} subject attention"),
        (ax[0,2], rattn, f"L{layer} reference attention"),
        (ax[1,1], smask, "processed subject bbox mask"),
        (ax[1,2], rmask, "processed reference bbox mask"),
    ):
        a.imshow(data, interpolation="nearest"); a.set_title(title); a.axis("off")
    ax[1,0].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)

def run_model(args, model_name: str, core, attn_mod, backend, bbox_by_sid):
    records, audit = backend.load_records(args.dataset, Path(args.data_root), None)
    record_by_sid = {int(r.sid): r for r in records}
    prompts = core.load_standard_prompts(Path(args.prompt_jsonl))
    sids = target_sids(model_name, Path(args.metadata_root), sorted(record_by_sid), args.max_samples, args.sid)
    sids = [sid for sid in sids if sid in record_by_sid and sid in prompts and sid in bbox_by_sid
            and selected_box(bbox_by_sid[sid], "subject") is not None
            and selected_box(bbox_by_sid[sid], "reference") is not None]
    if not sids: raise RuntimeError(f"No eligible sids for {model_name}")
    if model_name not in backend.SPECS: raise ValueError(f"Unknown model {model_name}")
    spec = backend.SPECS[model_name]
    model_cls = getattr(transformers, spec.model_class)

    print("\n" + "="*120 + f"\nLOADING {model_name}: {spec.repo_id}\n" + "="*120)
    model = model_cls.from_pretrained(
        spec.repo_id, dtype=core.resolve_dtype(spec.dtype_name), low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code, device_map={"": args.device},
        attn_implementation="eager")
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    core.configure_processor(model, processor)
    device = torch.device(args.device)
    layers, layers_path = core.resolve_decoder_layers(model); n_layers = len(layers)
    if not hasattr(attn_mod, "LayerGroundingCapture"):
        raise RuntimeError(
            f"{args.attention_module}.py lacks LayerGroundingCapture. Replace it with "
            "analyze_object_visual_attention_layers_v1_fixed.py and rename it to "
            "analyze_object_visual_attention_layers_v1.py")

    out = Path(args.output_root) / model_name
    if args.overwrite and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    sample_csv = out / "sample_layer_bbox_metrics.csv"
    clean_csv = out / "layer_bbox_summary_clean.csv"
    all_csv = out / "layer_bbox_summary_all.csv"
    errors = out / "errors.jsonl"
    summary_json = out / "summary.json"
    vis_dir = out / "visualizations"
    if not args.overwrite and any(x.exists() for x in (sample_csv, clean_csv, all_csv, summary_json)):
        raise FileExistsError(f"Results exist in {out}; use --overwrite")

    vis_layer = resolve_vis_layer(model_name, Path(args.metadata_root), args.vis_layer, n_layers)
    capture = attn_mod.LayerGroundingCapture(
        layers=layers, model=model, processor=processor,
        similarity_mode="softmax", similarity_temperature=0.07,
        top_fraction=0.10, save_maps=True)

    rows, map_examples = [], []
    processed = ambiguous = 0
    progress = tqdm(sids, desc=f"attention-bbox:{model_name}", unit="sample", dynamic_ncols=True)
    try:
        for index, sid in enumerate(progress, 1):
            image = batch = None
            try:
                record, prompt, brow = record_by_sid[sid], prompts[sid], bbox_by_sid[sid]
                sbox, rbox = selected_box(brow, "subject"), selected_box(brow, "reference")
                either_ambiguous = bool(brow.get("either_ambiguous", False)); ambiguous += int(either_ambiguous)
                image = core.record_image(record)
                question = str(prompt["question_text"])
                gt = base.normalize_relation(prompt["answer_raw"])
                if gt not in REL_TO_ID: raise RuntimeError(f"Invalid GT {prompt['answer_raw']!r}")
                batch = core.make_question_batch(processor=processor, image=image, question_text=question, device=device)
                batch = base.move_batch_to_device(batch, device)
                prompt_spec = base.build_prompt_position_spec(
                    model=model, tokenizer=processor.tokenizer, input_ids=batch["input_ids"],
                    subject=str(prompt["subject"]), reference=str(prompt["reference"]))
                capture.configure(prompt_spec=prompt_spec, batch=batch, image_size=tuple(image.size), gt_code=REL_TO_ID[gt])
                with torch.inference_mode():
                    model(**batch, use_cache=False, output_attentions=True,
                          output_hidden_states=False, return_dict=True)
                if capture.grid_shape is None: raise RuntimeError("Grid unresolved")
                gh, gw = capture.grid_shape
                smask, rmask, map_meta = processor_box_masks(
                    core, processor, question, tuple(image.size), sbox, rbox, gh, gw)
                if len(map_examples) < 5: map_examples.append({"sid": sid, **map_meta})
                overlap = box_iou(smask, rmask, args.bbox_hard_threshold)
                vis_s = vis_r = None

                for layer in range(n_layers):
                    result = capture.results[layer]
                    sattn = np.asarray(result.attention_subject_map, np.float32).reshape(gh, gw)
                    rattn = np.asarray(result.attention_reference_map, np.float32).reshape(gh, gw)
                    ss = map_metrics(sattn, smask, args.bbox_hard_threshold)
                    sx = map_metrics(sattn, rmask, args.bbox_hard_threshold)
                    rr = map_metrics(rattn, rmask, args.bbox_hard_threshold)
                    rx = map_metrics(rattn, smask, args.bbox_hard_threshold)
                    smargin, rmargin = ss["box_lift"]-sx["box_lift"], rr["box_lift"]-rx["box_lift"]
                    sbind, rbind = float(smargin > 0), float(rmargin > 0)
                    both_bind = float(sbind > 0 and rbind > 0)
                    both_point = float(ss["pointing"] > 0 and rr["pointing"] > 0)
                    both_cent = float(ss["centroid_inside"] > 0 and rr["centroid_inside"] > 0)
                    rel_ok = float(int(result.attention_prediction) == REL_TO_ID[gt])
                    rows.append({
                        "model": model_name, "sid": sid, "layer": layer, "gt": gt,
                        "attention_prediction": RELATIONS[int(result.attention_prediction)],
                        "attention_relation_correct": rel_ok,
                        "subject": str(prompt["subject"]), "reference": str(prompt["reference"]),
                        "either_ambiguous": either_ambiguous,
                        "grid_height": gh, "grid_width": gw, "grid_source": capture.grid_source,
                        "box_mask_iou": overlap,
                        "subject_self_box_mass": ss["box_mass"], "reference_self_box_mass": rr["box_mass"],
                        "mean_self_box_mass": .5*(ss["box_mass"]+rr["box_mass"]),
                        "subject_self_box_lift": ss["box_lift"], "reference_self_box_lift": rr["box_lift"],
                        "mean_self_box_lift": .5*(ss["box_lift"]+rr["box_lift"]),
                        "subject_cross_box_lift": sx["box_lift"], "reference_cross_box_lift": rx["box_lift"],
                        "subject_binding_lift_margin": smargin, "reference_binding_lift_margin": rmargin,
                        "mean_binding_lift_margin": .5*(smargin+rmargin),
                        "subject_binding_correct": sbind, "reference_binding_correct": rbind,
                        "both_binding_correct": both_bind,
                        "subject_pointing": ss["pointing"], "reference_pointing": rr["pointing"],
                        "both_pointing": both_point,
                        "subject_centroid_inside": ss["centroid_inside"],
                        "reference_centroid_inside": rr["centroid_inside"],
                        "both_centroid_inside": both_cent,
                        "subject_top_area_iou": ss["top_area_iou"], "reference_top_area_iou": rr["top_area_iou"],
                        "mean_top_area_iou": .5*(ss["top_area_iou"]+rr["top_area_iou"]),
                        "subject_box_area_soft": ss["box_area_soft"], "reference_box_area_soft": rr["box_area_soft"],
                        "subject_inside_outside_ratio": ss["inside_outside_ratio"],
                        "reference_inside_outside_ratio": rr["inside_outside_ratio"],
                        "relation_correct_and_both_pointing": float(rel_ok > 0 and both_point > 0),
                        "relation_correct_but_not_both_pointing": float(rel_ok > 0 and both_point == 0),
                        "relation_correct_and_both_centroid": float(rel_ok > 0 and both_cent > 0),
                        "relation_correct_but_not_both_centroid": float(rel_ok > 0 and both_cent == 0),
                    })
                    if layer == vis_layer: vis_s, vis_r = sattn, rattn

                if args.vis_first > 0 and processed < args.vis_first:
                    save_vis(vis_dir/f"sid_{sid:04d}_L{vis_layer:02d}.png", image, sid,
                             str(prompt["subject"]), str(prompt["reference"]), sbox, rbox,
                             vis_s, vis_r, smask, rmask, vis_layer)
                processed += 1
                if args.print_every > 0 and processed % args.print_every == 0:
                    progress.set_postfix_str(f"processed={processed}, ambiguous={ambiguous}", refresh=False)
            except Exception as exc:
                append_jsonl(errors, {"model": model_name, "sid": int(sid),
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-30:]})
                tqdm.write(f"\n[ERROR] {model_name} sid={sid}: {type(exc).__name__}: {exc}")
            finally:
                capture.reset()
                if image is not None:
                    with contextlib.suppress(Exception): image.close()
                del batch; gc.collect()
            if args.empty_cache_every > 0 and index % args.empty_cache_every == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        progress.close(); capture.close()

    if not rows: raise RuntimeError(f"No successful rows for {model_name}")
    write_csv(sample_csv, rows)
    clean = layer_summary(rows, True, args.include_ambiguous)
    all_rows = layer_summary(rows, False, True)
    write_csv(clean_csv, clean); write_csv(all_csv, all_rows)
    clean_sids = {int(r["sid"]) for r in rows if args.include_ambiguous or not bool(r["either_ambiguous"])}
    all_sids = {int(r["sid"]) for r in rows}
    summary = {
        "script_version": SCRIPT_VERSION, "model": model_name,
        "processed_samples": processed, "all_both_found_samples": len(all_sids),
        "clean_samples": len(clean_sids), "ambiguous_samples": ambiguous,
        "n_layers": n_layers, "vis_layer": vis_layer,
        "best_layers_clean": best_layers(clean), "best_layers_all": best_layers(all_rows),
        "processor_mapping_examples": map_examples,
        "uniform_box_lift_baseline": 1.0,
        "relation_correct_but_not_both_pointing_note":
            "suspected, not definitive, lucky-correct relation case",
        "outputs": {"sample_csv": str(sample_csv), "clean_layer_csv": str(clean_csv),
                    "all_layer_csv": str(all_csv), "errors": str(errors), "visualizations": str(vis_dir)},
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out/"config.json").write_text(json.dumps({
        "script_version": SCRIPT_VERSION, "model": model_name, "repo_id": spec.repo_id,
        "decoder_path": layers_path, "bbox_jsonl": str(args.bbox_jsonl),
        "processor_aware_bbox_mapping": True, "bbox_input_coordinates": "original_xyxy_pixels",
        "model_modified": False, "selection_uses_gt_relation": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "="*120 + f"\nATTENTION ↔ BBOX: {model_name}\n" + "="*120)
    print(json.dumps(summary["best_layers_clean"], ensure_ascii=False, indent=2))
    print(f"Processed={processed}, clean={len(clean_sids)}, vis={vis_dir}")
    del model, processor, layers; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not 0 < args.bbox_hard_threshold <= 1:
        raise ValueError("bbox hard threshold must be in (0,1]")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    core = importlib.import_module(args.core_module)
    attn_mod = importlib.import_module(args.attention_module)
    backend = core.import_two_object_module()
    bboxes = load_bbox_by_sid(Path(args.bbox_jsonl))
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    completed, failed = [], []
    print("="*120)
    print("PROCESSOR-AWARE ATTENTION ↔ GROUNDINGDINO BBOX EVALUATION")
    print("="*120)
    print("models=", models)
    print("bbox=", args.bbox_jsonl)
    for model in models:
        try:
            run_model(args, model, core, attn_mod, backend, bboxes)
            completed.append(model)
        except Exception as exc:
            failed.append({"model": model, "error_type": type(exc).__name__, "error": str(exc)})
            print(f"\n[FATAL] {model}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print("\n" + "="*120)
    print(f"COMPLETE: {len(completed)}/{len(models)}")
    print("completed:", completed)
    if failed: print("failures:", json.dumps(failed, ensure_ascii=False, indent=2))
    if not completed: raise SystemExit(1)


if __name__ == "__main__":
    main()
