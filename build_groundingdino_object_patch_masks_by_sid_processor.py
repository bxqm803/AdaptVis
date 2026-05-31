import os
import re
import sys
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from torch.utils.data import DataLoader

from model_zoo import get_model
from dataset_zoo import get_dataset

try:
    from misc import _default_collate
except Exception:
    _default_collate = None

import save_llava_hidden_similarity_features as sf


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def norm_text(x: Any) -> str:
    if isinstance(x, list):
        return str(x[0]).strip() if x else ""
    return str(x).strip()


def iter_samples(loader):
    sid = 0
    for batch in loader:
        for i_option in batch["image_options"]:
            for image in i_option:
                yield sid, image
                sid += 1


def pil_from_any_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if hasattr(image, "convert"):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image)).convert("RGB")


def tensor_to_pil(pixel_values: torch.Tensor, processor: Any) -> Image.Image:
    """Invert LLaVA/CLIP image normalization.

    Used only for processor-mapped bbox masks. This gives the image-space after
    the exact same processor call used by the VLM.
    """
    arr = pixel_values.detach().float().cpu().numpy()
    image_processor = getattr(processor, "image_processor", processor)
    mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    arr = arr * std + mean
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def process_image_to_pil(wrapper_processor: Any, prompt: str, image: Image.Image, max_length: int, device: str) -> Image.Image:
    """Return the actual processor-space image used by the VLM, inverted to RGB."""
    inp = wrapper_processor(
        text=prompt,
        images=image,
        padding="max_length",
        return_tensors="pt",
        max_length=max_length,
    )
    if hasattr(inp, "to"):
        inp = inp.to(device)
    return tensor_to_pil(inp["pixel_values"][0], wrapper_processor)


def draw_patch_overlay(img: Image.Image, patch_ids: List[int], patch_side: int) -> Image.Image:
    """Draw selected patch ids on a processor-space image."""
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    cell_w = w / patch_side
    cell_h = h / patch_side
    for pid in patch_ids:
        r = int(pid) // patch_side
        c = int(pid) % patch_side
        x0, y0 = c * cell_w, r * cell_h
        x1, y1 = (c + 1) * cell_w, (r + 1) * cell_h
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 220), fill=(255, 0, 0, 80), width=2)
    return Image.alpha_composite(img, overlay).convert("RGB")


def parse_candidate_objects(prompt: str) -> Tuple[str, str]:
    """Best-effort parser for Controlled_Images prompts.

    This is metadata and target construction only. If it fails, the script can
    still use --target-mode gold.
    """
    p = str(prompt).strip()

    quoted = re.findall(r"['\"]([^'\"]+)['\"]", p)
    if len(quoted) >= 2:
        return quoted[0].strip(), quoted[1].strip()

    patterns = [
        r"between\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+and\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:[?.]|$)",
        r"(?:a\s+|an\s+|the\s+)?(.+?)\s+or\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:[?.]|$)",
        r"(?:a\s+|an\s+|the\s+)?(.+?)\s+and\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:[?.]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, p, flags=re.I)
        if m:
            left = m.group(1).strip()
            right = m.group(2).strip()
            # Avoid swallowing the whole question on the left side.
            left = re.split(r"[,;:]", left)[-1].strip()
            for prefix in ["which is larger", "which is smaller", "which object is larger", "which object is smaller"]:
                if left.lower().startswith(prefix):
                    left = left[len(prefix):].strip()
            return left, right

    return "", ""


def build_targets(prompt: str, gold: str, target_mode: str) -> List[str]:
    obj1, obj2 = parse_candidate_objects(prompt)

    if target_mode == "gold":
        targets = [gold]
    elif target_mode == "candidates":
        targets = [x for x in [obj1, obj2] if x]
        if not targets:
            targets = [gold]
    elif target_mode == "gold_and_candidates":
        targets = [gold] + [x for x in [obj1, obj2] if x]
    elif target_mode == "prompt":
        targets = [prompt]
    else:
        raise ValueError(f"Unknown target_mode={target_mode}")

    # Clean and deduplicate.
    out = []
    seen = set()
    for t in targets:
        t = str(t).strip()
        if not t:
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def add_period(text: str) -> str:
    text = str(text).strip()
    if not text.endswith("."):
        text += "."
    return text


# ---------------------------------------------------------------------
# Official GroundingDINO backend
# ---------------------------------------------------------------------
def load_official_groundingdino(args):
    if args.groundingdino_dir:
        gdino_dir = os.path.abspath(args.groundingdino_dir)
        if gdino_dir not in sys.path:
            sys.path.insert(0, gdino_dir)

    from groundingdino.util.inference import load_model

    model = load_model(args.dino_config, args.dino_checkpoint)
    model = model.to(args.dino_device)
    model.eval()
    return model


def official_preprocess_image(image_pil: Image.Image):
    import groundingdino.datasets.transforms as T

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)
    return image_tensor


def cxcywh_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    # boxes normalized cxcywh in [0, 1]
    scale = torch.tensor([width, height, width, height], dtype=boxes.dtype, device=boxes.device)
    boxes = boxes * scale
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def detect_official(
    dino_model: Any,
    image_pil: Image.Image,
    targets: List[str],
    box_threshold: float,
    text_threshold: float,
    device: str,
    max_boxes_per_target: int,
) -> List[Dict[str, Any]]:
    from groundingdino.util.inference import predict

    width, height = image_pil.size
    image_tensor = official_preprocess_image(image_pil)
    detections = []

    for target in targets:
        caption = add_period(target).lower()
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=dino_model,
                image=image_tensor,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device,
            )

        if boxes is None or len(boxes) == 0:
            continue

        if torch.is_tensor(logits):
            order = torch.argsort(logits, descending=True)
            if max_boxes_per_target > 0:
                order = order[:max_boxes_per_target]
            boxes = boxes[order]
            logits = logits[order]
            phrases = [phrases[int(i)] for i in order.detach().cpu().tolist()]
        else:
            order = list(range(len(boxes)))[:max_boxes_per_target]
            boxes = boxes[order]
            logits = [logits[i] for i in order]
            phrases = [phrases[i] for i in order]

        xyxy = cxcywh_to_xyxy(boxes, width=width, height=height).detach().cpu().numpy()

        for i, box in enumerate(xyxy):
            score = float(logits[i].detach().cpu()) if torch.is_tensor(logits) else float(logits[i])
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = max(0.0, min(float(width), x1))
            x2 = max(0.0, min(float(width), x2))
            y1 = max(0.0, min(float(height), y1))
            y2 = max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "label": str(phrases[i]),
                "target": str(target),
                "score": score,
                "box_xyxy": [x1, y1, x2, y2],
                "backend": "official",
            })

    return detections


# ---------------------------------------------------------------------
# HuggingFace GroundingDINO backend, optional fallback
# ---------------------------------------------------------------------
def load_hf_groundingdino(args):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained(args.hf_dino_model, cache_dir=args.cache_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.hf_dino_model, cache_dir=args.cache_dir)
    model = model.to(args.dino_device)
    model.eval()
    return processor, model


def detect_hf(
    processor: Any,
    model: Any,
    image_pil: Image.Image,
    targets: List[str],
    box_threshold: float,
    text_threshold: float,
    device: str,
    max_boxes_per_target: int,
) -> List[Dict[str, Any]]:
    detections = []
    width, height = image_pil.size

    for target in targets:
        text = add_period(target).lower()
        inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image_pil.size[::-1]], device=device)
        try:
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]

        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        labels = results.get("labels", [target] * len(boxes))

        if len(boxes) == 0:
            continue
        scores_tensor = scores if torch.is_tensor(scores) else torch.tensor(scores)
        order = torch.argsort(scores_tensor, descending=True)
        if max_boxes_per_target > 0:
            order = order[:max_boxes_per_target]

        for idx in order.detach().cpu().tolist():
            box = boxes[idx]
            score = scores[idx]
            label = labels[idx]
            x1, y1, x2, y2 = [float(x) for x in box.detach().cpu().tolist()]
            x1 = max(0.0, min(float(width), x1))
            x2 = max(0.0, min(float(width), x2))
            y1 = max(0.0, min(float(height), y1))
            y2 = max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "label": str(label),
                "target": str(target),
                "score": float(score.detach().cpu() if torch.is_tensor(score) else score),
                "box_xyxy": [x1, y1, x2, y2],
                "backend": "hf",
            })

    return detections


# ---------------------------------------------------------------------
# Processor-aware bbox -> patch ids
# ---------------------------------------------------------------------
def get_processed_mask_from_boxes(
    wrapper_processor: Any,
    prompt: str,
    image_size: Tuple[int, int],
    boxes_xyxy: List[List[float]],
    max_length: int,
    device: str,
) -> Image.Image:
    """Draw bbox mask on original image, then pass mask through the same VLM processor.

    This is the key part. It avoids mismatch from resize / center-crop / pad.
    """
    w, h = image_size
    mask = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(mask)

    for box in boxes_xyxy:
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))

    inp = wrapper_processor(
        text=prompt,
        images=mask,
        padding="max_length",
        return_tensors="pt",
        max_length=max_length,
    )
    if hasattr(inp, "to"):
        inp = inp.to(device)

    proc_mask = tensor_to_pil(inp["pixel_values"][0], wrapper_processor)
    return proc_mask.convert("L")


def mask_to_patch_ids(
    proc_mask: Image.Image,
    patch_side: int,
    mask_threshold: float,
    min_patch_area_frac: float,
) -> List[int]:
    arr = np.asarray(proc_mask).astype(np.float32) / 255.0
    h, w = arr.shape
    cell_h = h / patch_side
    cell_w = w / patch_side

    patch_ids = []
    for r in range(patch_side):
        for c in range(patch_side):
            y0 = int(round(r * cell_h))
            y1 = int(round((r + 1) * cell_h))
            x0 = int(round(c * cell_w))
            x1 = int(round((c + 1) * cell_w))
            crop = arr[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            area_frac = float((crop >= mask_threshold).mean())
            if area_frac >= min_patch_area_frac:
                patch_ids.append(r * patch_side + c)

    return patch_ids


def save_debug_vis(
    vis_dir: str,
    sid: int,
    image_pil: Image.Image,
    prompt: str,
    wrapper_processor: Any,
    detections: List[Dict[str, Any]],
    proc_mask: Image.Image,
    patch_ids: List[int],
    patch_side: int,
    max_length: int,
    device: str,
) -> None:
    """Save sanity-check images.

    Outputs:
      1. raw original image + GroundingDINO bbox
      2. processor-space mask + selected patch ids
      3. processor-space real image + selected patch ids  <-- most important
    """
    os.makedirs(vis_dir, exist_ok=True)

    raw = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(raw)
    for det in detections:
        box = det["box_xyxy"]
        draw.rectangle(box, outline=(255, 0, 0), width=4)
        label = f"{det.get('target', '')}:{det.get('score', 0):.2f}"
        draw.text((box[0], max(0, box[1] - 12)), label, fill=(255, 0, 0))
    raw.save(os.path.join(vis_dir, f"sid{sid:04d}_raw_bbox.png"))

    proc_mask_rgb = Image.merge("RGB", [proc_mask, proc_mask, proc_mask])
    draw_patch_overlay(proc_mask_rgb, patch_ids, patch_side).save(
        os.path.join(vis_dir, f"sid{sid:04d}_processed_mask_patch_ids.png")
    )

    processed_img = process_image_to_pil(
        wrapper_processor=wrapper_processor,
        prompt=prompt,
        image=image_pil,
        max_length=max_length,
        device=device,
    )
    processed_img.save(os.path.join(vis_dir, f"sid{sid:04d}_processed_image.png"))
    draw_patch_overlay(processed_img, patch_ids, patch_side).save(
        os.path.join(vis_dir, f"sid{sid:04d}_processed_image_patch_ids.png")
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="Controlled_Images_B")
    parser.add_argument("--option", default="four")
    parser.add_argument("--model-name", default="llava1.5")
    parser.add_argument("--method", default="adapt_vis")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--fresh-limit", type=int, default=-1)

    parser.add_argument("--backend", choices=["official", "hf"], default="official")

    # Official GroundingDINO options.
    parser.add_argument("--groundingdino-dir", default="external/GroundingDINO")
    parser.add_argument("--dino-config", default="external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--dino-checkpoint", default="weights/groundingdino_swint_ogc.pth")

    # HuggingFace backend options.
    parser.add_argument("--hf-dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--cache-dir", default="data")

    parser.add_argument("--dino-device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-boxes-per-target", type=int, default=1)

    parser.add_argument(
        "--target-mode",
        choices=["gold", "candidates", "gold_and_candidates", "prompt"],
        default="candidates",
        help="candidates mimics the old two-object object-box setting; gold detects only the answer object.",
    )

    parser.add_argument("--patch-side", type=int, default=24)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-patch-area-frac", type=float, default=0.01)

    parser.add_argument("--out-json", default="output/groundingdino_object_patch_masks_B_by_sid.json")
    parser.add_argument("--missing-json", default="output/groundingdino_object_patch_masks_B_missing.json")
    parser.add_argument("--manifest-json", default="output/groundingdino_dataset_manifest_B.json")
    parser.add_argument("--vis-dir", default="")
    parser.add_argument("--vis-first", type=int, default=0)

    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ensure_dir_for_file(args.out_json)
    ensure_dir_for_file(args.missing_json)
    ensure_dir_for_file(args.manifest_json)

    print("[LOAD VLM PROCESSOR]", args.model_name, args.method)
    # We only need wrapper.processor, but get_model is the repo's reliable way to get it.
    wrapper, _ = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    wrapper.model.eval()

    print("[LOAD DATASET RAW]", args.dataset)
    dataset = get_dataset(args.dataset, image_preprocess=None, download=False)
    collate_fn = _default_collate
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    prompts, answers = sf.load_prompts(args.dataset, args.option)

    if args.backend == "official":
        print("[LOAD OFFICIAL GroundingDINO]")
        dino_model = load_official_groundingdino(args)
        dino_pack = (dino_model,)
    else:
        print("[LOAD HF GroundingDINO]", args.hf_dino_model)
        dino_pack = load_hf_groundingdino(args)

    by_sid: Dict[str, Dict[str, Any]] = {}
    missing: Dict[str, Dict[str, Any]] = {}
    manifest: List[Dict[str, Any]] = []

    for sid, image in tqdm(iter_samples(loader), desc="build masks"):
        if args.fresh_limit > 0 and sid >= args.fresh_limit:
            break
        if sid >= len(prompts):
            break

        image_pil = pil_from_any_image(image)
        image_path = getattr(image, "filename", "")
        prompt = prompts[sid]
        gold = norm_text(answers[sid])
        obj1, obj2 = parse_candidate_objects(prompt)
        targets = build_targets(prompt, gold, args.target_mode)

        if args.backend == "official":
            detections = detect_official(
                dino_model=dino_pack[0],
                image_pil=image_pil,
                targets=targets,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=args.dino_device,
                max_boxes_per_target=args.max_boxes_per_target,
            )
        else:
            detections = detect_hf(
                processor=dino_pack[0],
                model=dino_pack[1],
                image_pil=image_pil,
                targets=targets,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=args.dino_device,
                max_boxes_per_target=args.max_boxes_per_target,
            )

        boxes = [d["box_xyxy"] for d in detections]
        proc_mask = get_processed_mask_from_boxes(
            wrapper_processor=wrapper.processor,
            prompt=prompt,
            image_size=image_pil.size,
            boxes_xyxy=boxes,
            max_length=args.max_length,
            device=args.device,
        )
        patch_ids = mask_to_patch_ids(
            proc_mask=proc_mask,
            patch_side=args.patch_side,
            mask_threshold=args.mask_threshold,
            min_patch_area_frac=args.min_patch_area_frac,
        )

        rec = {
            "sample_id": int(sid),
            "image_path": image_path,
            "prompt": prompt,
            "gold": gold,
            "obj1": obj1,
            "obj2": obj2,
            "targets": targets,
            "target_mode": args.target_mode,
            "image_width": int(image_pil.size[0]),
            "image_height": int(image_pil.size[1]),
            "patch_side": int(args.patch_side),
            "patch_ids": [int(x) for x in patch_ids],
            "detections": detections,
            "backend": args.backend,
            "box_threshold": float(args.box_threshold),
            "text_threshold": float(args.text_threshold),
            "mask_threshold": float(args.mask_threshold),
            "min_patch_area_frac": float(args.min_patch_area_frac),
            "processor_aware_patch_mapping": True,
        }
        by_sid[str(sid)] = rec

        manifest.append({
            "sample_id": int(sid),
            "image_path": image_path,
            "prompt": prompt,
            "gold": gold,
            "targets": targets,
            "num_detections": len(detections),
            "num_patch_ids": len(patch_ids),
        })

        if len(detections) == 0 or len(patch_ids) == 0:
            missing[str(sid)] = rec

        if args.vis_dir and sid < args.vis_first:
            save_debug_vis(
                vis_dir=args.vis_dir,
                sid=sid,
                image_pil=image_pil,
                prompt=prompt,
                wrapper_processor=wrapper.processor,
                detections=detections,
                proc_mask=proc_mask,
                patch_ids=patch_ids,
                patch_side=args.patch_side,
                max_length=args.max_length,
                device=args.device,
            )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(by_sid, f, ensure_ascii=False, indent=2)
    with open(args.missing_json, "w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)
    with open(args.manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print("[OUT]", args.out_json, "num_records=", len(by_sid))
    print("[MISSING]", args.missing_json, "num_missing=", len(missing))
    print("[MANIFEST]", args.manifest_json)
    if args.vis_dir:
        print("[VIS]", args.vis_dir)


if __name__ == "__main__":
    main()
