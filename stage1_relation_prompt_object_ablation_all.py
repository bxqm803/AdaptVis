import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoProcessor

try:
    from transformers import AutoModelForZeroShotObjectDetection as GroundingDINOModel
except ImportError:
    from transformers import GroundingDinoForObjectDetection as GroundingDINOModel

try:
    from transformers import LlavaForConditionalGeneration as LlavaModel
except ImportError:
    from transformers import AutoModelForVision2Seq as LlavaModel

from dataset_zoo import get_dataset


LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LLAVA_REVISION = "a272c74"

RELATIONS = ["left", "right", "on", "under"]


# ============================================================
# Prompt / dataset helpers
# ============================================================

def load_prompt_rows(dataset_name: str, option: str) -> List[dict]:
    path = Path(f"prompts/{dataset_name}_with_answer_{option}_options.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_question(q: str) -> str:
    q = str(q)
    q = q.replace("<image>", " ")
    q = q.replace("USER:", " ")
    q = q.replace("ASSISTANT:", " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def strip_article(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"^(a|an|the)\s+", "", x, flags=re.IGNORECASE)
    x = re.sub(r"[.?!,:;]+$", "", x)
    return x.strip()


def parse_two_objects_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    q = clean_question(prompt)

    # Remove answer instruction.
    q = re.sub(r"Answer\s+with\s+.*$", "", q, flags=re.IGNORECASE).strip()

    patterns = [
        r"Where\s+is\s+the\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?",
        r"Where\s+is\s+the\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
    ]

    for p in patterns:
        m = re.search(p, q, flags=re.IGNORECASE)
        if m:
            return strip_article(m.group(1)), strip_article(m.group(2))

    return None, None


def normalize_relation(x: str) -> str:
    s = str(x).strip().lower()

    if "under" in s:
        return "under"
    if re.search(r"\bon\b", s) and "front" not in s:
        return "on"
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"

    return "unknown"


def get_gold_from_prompt_row(row: dict) -> str:
    ans = row.get("answer", "")
    if isinstance(ans, list):
        ans = ans[0] if ans else ""
    return normalize_relation(ans)


def get_raw_pil_from_dataset(dataset, idx: int) -> Image.Image:
    item = dataset[idx]

    if not isinstance(item, dict):
        raise TypeError(f"Expected dataset item to be dict, got {type(item)}")

    if "image_options" in item:
        image = item["image_options"][0]
    elif "image" in item:
        image = item["image"]
    else:
        raise KeyError(f"Cannot find image in dataset item keys: {list(item.keys())}")

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL image, got {type(image)}")

    return image.convert("RGB")


def load_dataset_any_signature(dataset_name: str, root_dir: str, download: bool):
    try:
        return get_dataset(
            dataset_name,
            root_dir=root_dir,
            image_preprocess=None,
            download=download,
        )
    except TypeError:
        return get_dataset(
            dataset_name,
            image_preprocess=None,
            download=download,
        )


# ============================================================
# LLaVA preprocess geometry
# ============================================================

def get_size_value(size_obj, key: str, default: int) -> int:
    if isinstance(size_obj, dict):
        return int(size_obj.get(key, default))
    if isinstance(size_obj, int):
        return int(size_obj)
    return int(default)


def get_resample_from_processor(image_processor):
    resample = getattr(image_processor, "resample", None)
    if resample is not None:
        return resample
    return Image.BICUBIC


def infer_llava_geometry(image_processor, fallback_size: int = 336) -> Dict:
    do_resize = bool(getattr(image_processor, "do_resize", True))
    do_center_crop = bool(getattr(image_processor, "do_center_crop", True))
    do_pad = bool(getattr(image_processor, "do_pad", False))

    size = getattr(image_processor, "size", None) or {}
    crop_size = getattr(image_processor, "crop_size", None) or {}

    shortest_edge = None
    resize_h = None
    resize_w = None

    if isinstance(size, dict):
        if "shortest_edge" in size:
            shortest_edge = int(size["shortest_edge"])
        if "height" in size and "width" in size:
            resize_h = int(size["height"])
            resize_w = int(size["width"])
    elif isinstance(size, int):
        shortest_edge = int(size)

    crop_h = get_size_value(crop_size, "height", fallback_size)
    crop_w = get_size_value(crop_size, "width", fallback_size)

    if shortest_edge is None and resize_h is None:
        shortest_edge = fallback_size

    return {
        "do_resize": do_resize,
        "do_center_crop": do_center_crop,
        "do_pad": do_pad,
        "shortest_edge": shortest_edge,
        "resize_h": resize_h,
        "resize_w": resize_w,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "resample": get_resample_from_processor(image_processor),
    }


def expand2square(img: Image.Image, background_color: Tuple[int, int, int]):
    w, h = img.size
    if w == h:
        return img, 0, 0, w

    size = max(w, h)
    out = Image.new("RGB", (size, size), background_color)

    pad_x = (size - w) // 2
    pad_y = (size - h) // 2

    out.paste(img, (pad_x, pad_y))
    return out, pad_x, pad_y, size


def make_processed_pil_like_llava(
    raw: Image.Image,
    image_processor,
    force_mode: str = "auto",
) -> Tuple[Image.Image, Dict]:
    raw = raw.convert("RGB")
    geom = infer_llava_geometry(image_processor)

    if force_mode not in ["auto", "crop", "pad"]:
        raise ValueError(f"Unknown force_mode={force_mode}")

    mode = force_mode
    if mode == "auto":
        mode = "pad" if geom["do_pad"] else "crop"

    if mode == "pad":
        mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        bg = tuple(int(float(x) * 255) for x in mean)

        square, pad_x, pad_y, square_size = expand2square(raw, bg)

        target_h = geom["crop_h"]
        target_w = geom["crop_w"]
        processed = square.resize((target_w, target_h), geom["resample"])

        meta = {
            "mode": "pad",
            "raw_size": raw.size,
            "square_size": square_size,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "processed_size": processed.size,
            "grid": target_h // 14,
        }
        return processed, meta

    # crop mode
    w, h = raw.size

    if geom["resize_h"] is not None and geom["resize_w"] is not None:
        resized = raw.resize((geom["resize_w"], geom["resize_h"]), geom["resample"])
    else:
        shortest = geom["shortest_edge"]
        scale = shortest / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = raw.resize((new_w, new_h), geom["resample"])

    rw, rh = resized.size
    crop_w = geom["crop_w"]
    crop_h = geom["crop_h"]

    left = max(0, int(round((rw - crop_w) / 2)))
    top = max(0, int(round((rh - crop_h) / 2)))

    processed = resized.crop((left, top, left + crop_w, top + crop_h))

    meta = {
        "mode": "crop",
        "raw_size": raw.size,
        "resized_size": resized.size,
        "crop_left": left,
        "crop_top": top,
        "processed_size": processed.size,
        "grid": crop_h // 14,
    }
    return processed, meta


def get_mask_color(image_processor) -> Tuple[int, int, int]:
    mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    return tuple(int(float(x) * 255) for x in mean)


# ============================================================
# GroundingDINO on processed image
# ============================================================

def detect_one(
    image: Image.Image,
    phrase: str,
    processor,
    model,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> Tuple[Optional[List[float]], Optional[float]]:
    text = phrase.strip()
    if not text.endswith("."):
        text += "."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)

    try:
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        result = processor.post_process_grounded_object_detection(
            outputs,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]

    boxes = result.get("boxes", [])
    scores = result.get("scores", [])

    if len(boxes) == 0:
        return None, None

    best = int(torch.argmax(scores).item())
    box = boxes[best].detach().cpu().tolist()
    score = float(scores[best].detach().cpu().item())

    return box, score


# ============================================================
# Patch ids / masking
# ============================================================

def box_to_patch_ids_and_grid_box(
    box: Optional[List[float]],
    image_size: int = 336,
    patch_size: int = 14,
) -> Tuple[List[int], Optional[Tuple[int, int, int, int]]]:
    if box is None:
        return [], None

    x1, y1, x2, y2 = box
    grid = image_size // patch_size

    c1 = max(0, min(grid - 1, int(math.floor(x1 / patch_size))))
    r1 = max(0, min(grid - 1, int(math.floor(y1 / patch_size))))
    c2 = max(0, min(grid - 1, int(math.ceil(x2 / patch_size)) - 1))
    r2 = max(0, min(grid - 1, int(math.ceil(y2 / patch_size)) - 1))

    ids = []
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            ids.append(r * grid + c)

    return ids, (r1, c1, r2, c2)


def grid_box_to_ids(
    grid_box: Optional[Tuple[int, int, int, int]],
    grid: int,
) -> List[int]:
    if grid_box is None:
        return []

    r1, c1, r2, c2 = grid_box
    ids = []
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            ids.append(r * grid + c)
    return ids


def mask_patch_ids(
    image: Image.Image,
    patch_ids: List[int],
    patch_size: int,
    mask_color: Tuple[int, int, int],
) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    image_size = out.size[0]
    grid = image_size // patch_size

    for pid in sorted(set(patch_ids)):
        r = pid // grid
        c = pid % grid

        x1 = c * patch_size
        y1 = r * patch_size
        x2 = x1 + patch_size
        y2 = y1 + patch_size

        draw.rectangle([x1, y1, x2, y2], fill=mask_color)

    return out


def sample_background_ids_like_objects(
    grid_box1: Optional[Tuple[int, int, int, int]],
    grid_box2: Optional[Tuple[int, int, int, int]],
    grid: int,
    seed: int,
) -> List[int]:
    rng = random.Random(seed)

    object_ids = set(grid_box_to_ids(grid_box1, grid)) | set(grid_box_to_ids(grid_box2, grid))
    bg_ids = set()

    def try_place_like(gb):
        if gb is None:
            return []

        r1, c1, r2, c2 = gb
        h = r2 - r1 + 1
        w = c2 - c1 + 1

        for _ in range(1000):
            rr = rng.randint(0, max(0, grid - h))
            cc = rng.randint(0, max(0, grid - w))

            cand = []
            for r in range(rr, rr + h):
                for c in range(cc, cc + w):
                    cand.append(r * grid + c)

            cand_set = set(cand)
            if cand_set.isdisjoint(object_ids) and cand_set.isdisjoint(bg_ids):
                return cand

        return []

    ids1 = try_place_like(grid_box1)
    bg_ids.update(ids1)

    ids2 = try_place_like(grid_box2)
    bg_ids.update(ids2)

    target_count = len(object_ids)
    if len(bg_ids) < target_count:
        all_non_obj = [i for i in range(grid * grid) if i not in object_ids and i not in bg_ids]
        rng.shuffle(all_non_obj)
        need = target_count - len(bg_ids)
        bg_ids.update(all_non_obj[:need])

    return sorted(bg_ids)


# ============================================================
# LLaVA closed-set relation scoring
# ============================================================

def load_llava(model_id: str, revision: str, cache_dir: str, device: str, dtype: str):
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
    )

    model = LlavaModel.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    # Fix HF warning:
    # "Expanding inputs for image tokens in LLaVa should be done in processing..."
    patch_size = getattr(getattr(model.config, "vision_config", None), "patch_size", 14)
    vision_feature_select_strategy = getattr(
        model.config,
        "vision_feature_select_strategy",
        "default",
    )

    processor.patch_size = patch_size
    processor.vision_feature_select_strategy = vision_feature_select_strategy

    model = model.to(device).eval()
    return processor, model


def score_candidates_batch(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    candidates: List[str],
    device: str,
) -> Dict[str, Dict[str, float]]:
    """
    Closed-set relation scoring.

    S(candidate) = average log-probability of candidate answer tokens
    under the original relation prompt.

    This scores the final candidate tokens in each full sequence.
    It is robust to padding and avoids prompt-length indexing bugs.
    """
    tokenizer = processor.tokenizer

    answer_texts = [" " + c for c in candidates]
    full_texts = [prompt + a for a in answer_texts]
    images = [image] * len(candidates)

    inputs = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # logits[:, t-1] predicts input_ids[:, t]
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    result = {}

    for b, cand in enumerate(candidates):
        cand_text = " " + cand
        cand_ids = tokenizer(
            cand_text,
            add_special_tokens=False,
        ).input_ids

        n_tok = len(cand_ids)
        if n_tok == 0:
            result[cand] = {
                "sum_logprob": float("-inf"),
                "avg_logprob": float("-inf"),
                "num_tokens": 0,
            }
            continue

        # Non-padding token positions in full input_ids.
        nonpad_positions = torch.nonzero(attention_mask[b], as_tuple=False).squeeze(-1)

        # Candidate answer is appended at the end.
        cand_positions_in_input = nonpad_positions[-n_tok:]

        # Convert input token positions t to target positions t-1.
        cand_positions_in_target = cand_positions_in_input - 1
        cand_positions_in_target = cand_positions_in_target[
            cand_positions_in_target >= 0
        ]

        vals = token_log_probs[b, cand_positions_in_target]

        if vals.numel() == 0:
            sum_lp = float("-inf")
            avg_lp = float("-inf")
            n_used = 0
        else:
            sum_lp = float(vals.sum().detach().cpu().item())
            avg_lp = float(vals.mean().detach().cpu().item())
            n_used = int(vals.numel())

        result[cand] = {
            "sum_logprob": sum_lp,
            "avg_logprob": avg_lp,
            "num_tokens": n_used,
        }

    return result


def simple_score_dict(scores_nested: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {k: v["avg_logprob"] for k, v in scores_nested.items()}


def pred_from_scores(scores: Dict[str, float]) -> str:
    return max(scores.items(), key=lambda kv: kv[1])[0]


def correct_margin(scores: Dict[str, float], gold: str) -> Optional[float]:
    if gold not in scores:
        return None

    others = [v for k, v in scores.items() if k != gold]
    if not others:
        return None

    return scores[gold] - max(others)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--option", default="four")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--llava-model-id", default=LLAVA_MODEL_ID)
    parser.add_argument("--llava-revision", default=LLAVA_REVISION)
    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-base")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--patch-size", type=int, default=14)

    parser.add_argument("--preprocess-mode", default="auto", choices=["auto", "crop", "pad"])

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        args.dtype = "float32"

    out_dir = Path(args.out_dir or f"output/stage1_relation_ablation_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "results.jsonl"
    csv_path = out_dir / "summary.csv"

    if jsonl_path.exists():
        jsonl_path.unlink()
    if csv_path.exists():
        csv_path.unlink()

    print(f"[INFO] dataset={args.dataset}")
    print(f"[INFO] device={device}, dtype={args.dtype}")
    print(f"[INFO] out_dir={out_dir}")

    print("[INFO] loading LLaVA")
    llava_processor, llava_model = load_llava(
        model_id=args.llava_model_id,
        revision=args.llava_revision,
        cache_dir=args.root_dir,
        device=device,
        dtype=args.dtype,
    )
    llava_image_processor = llava_processor.image_processor

    geom = infer_llava_geometry(llava_image_processor)
    print("[INFO] inferred LLaVA image geometry:")
    for k, v in geom.items():
        if k != "resample":
            print(f"  {k}: {v}")
    print("[INFO] preprocess mode:", args.preprocess_mode)

    mask_color = get_mask_color(llava_image_processor)

    print("[INFO] loading GroundingDINO")
    gdino_processor = AutoProcessor.from_pretrained(args.grounding_model_id)
    gdino_model = GroundingDINOModel.from_pretrained(args.grounding_model_id).to(device).eval()

    print("[INFO] loading dataset")
    dataset = load_dataset_any_signature(
        dataset_name=args.dataset,
        root_dir=args.root_dir,
        download=args.download,
    )

    prompt_rows = load_prompt_rows(args.dataset, args.option)

    n_total = min(len(dataset), len(prompt_rows))
    indices = list(range(n_total))

    if args.max_samples > 0:
        random.seed(args.seed)
        indices = random.sample(indices, min(args.max_samples, len(indices)))

    print(f"[INFO] total samples to run: {len(indices)}")

    csv_fields = [
        "sample_id",
        "gold",
        "obj1",
        "obj2",
        "obj1_found",
        "obj2_found",
        "obj1_score",
        "obj2_score",
        "obj1_grid_box",
        "obj2_grid_box",
        "obj1_patch_count",
        "obj2_patch_count",
        "orig_pred",
        "mask_obj1_pred",
        "mask_obj2_pred",
        "mask_both_pred",
        "mask_bg_pred",
        "orig_margin",
        "mask_obj1_margin",
        "mask_obj2_margin",
        "mask_both_margin",
        "mask_bg_margin",
        "drop_mask_obj1",
        "drop_mask_obj2",
        "drop_mask_both",
        "drop_mask_bg",
    ]

    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    jsonl_file = jsonl_path.open("w", encoding="utf-8")

    processed_count = 0
    skipped_count = 0

    for run_i, sample_id in enumerate(indices):
        prompt = prompt_rows[sample_id].get("question", "")
        gold = get_gold_from_prompt_row(prompt_rows[sample_id])
        obj1, obj2 = parse_two_objects_from_prompt(prompt)

        if obj1 is None or obj2 is None or gold not in RELATIONS:
            skipped_count += 1
            print(f"[SKIP] sample_id={sample_id}, cannot parse obj/gold")
            continue

        raw = get_raw_pil_from_dataset(dataset, sample_id)

        processed, meta = make_processed_pil_like_llava(
            raw=raw,
            image_processor=llava_image_processor,
            force_mode=args.preprocess_mode,
        )

        image_size = processed.size[0]
        grid = image_size // args.patch_size

        box1, score1 = detect_one(
            image=processed,
            phrase=obj1,
            processor=gdino_processor,
            model=gdino_model,
            device=device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        box2, score2 = detect_one(
            image=processed,
            phrase=obj2,
            processor=gdino_processor,
            model=gdino_model,
            device=device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        obj1_ids, obj1_grid_box = box_to_patch_ids_and_grid_box(
            box1,
            image_size=image_size,
            patch_size=args.patch_size,
        )
        obj2_ids, obj2_grid_box = box_to_patch_ids_and_grid_box(
            box2,
            image_size=image_size,
            patch_size=args.patch_size,
        )

        both_ids = sorted(set(obj1_ids) | set(obj2_ids))

        bg_ids = sample_background_ids_like_objects(
            obj1_grid_box,
            obj2_grid_box,
            grid=grid,
            seed=args.seed + sample_id,
        )

        images_by_condition = {
            "original": processed,
            "mask_obj1": mask_patch_ids(processed, obj1_ids, args.patch_size, mask_color),
            "mask_obj2": mask_patch_ids(processed, obj2_ids, args.patch_size, mask_color),
            "mask_both": mask_patch_ids(processed, both_ids, args.patch_size, mask_color),
            "mask_background": mask_patch_ids(processed, bg_ids, args.patch_size, mask_color),
        }

        scores_by_condition = {}
        pred_by_condition = {}
        margin_by_condition = {}

        for cond, img in images_by_condition.items():
            nested = score_candidates_batch(
                model=llava_model,
                processor=llava_processor,
                image=img,
                prompt=prompt,
                candidates=RELATIONS,
                device=device,
            )
            scores = simple_score_dict(nested)
            scores_by_condition[cond] = scores
            pred_by_condition[cond] = pred_from_scores(scores)
            margin_by_condition[cond] = correct_margin(scores, gold)

        orig_margin = margin_by_condition["original"]

        def drop(cond_name: str):
            if orig_margin is None or margin_by_condition[cond_name] is None:
                return None
            return orig_margin - margin_by_condition[cond_name]

        row = {
            "sample_id": sample_id,
            "gold": gold,
            "prompt": prompt,
            "clean_prompt": clean_question(prompt),
            "obj1": obj1,
            "obj2": obj2,
            "obj1_found": box1 is not None,
            "obj2_found": box2 is not None,
            "obj1_score": score1,
            "obj2_score": score2,
            "obj1_box": box1,
            "obj2_box": box2,
            "obj1_grid_box": obj1_grid_box,
            "obj2_grid_box": obj2_grid_box,
            "obj1_patch_ids": obj1_ids,
            "obj2_patch_ids": obj2_ids,
            "background_patch_ids": bg_ids,
            "processed_meta": meta,
            "scores": scores_by_condition,
            "pred": pred_by_condition,
            "margin": margin_by_condition,
            "drop": {
                "mask_obj1": drop("mask_obj1"),
                "mask_obj2": drop("mask_obj2"),
                "mask_both": drop("mask_both"),
                "mask_background": drop("mask_background"),
            },
        }

        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl_file.flush()

        csv_row = {
            "sample_id": sample_id,
            "gold": gold,
            "obj1": obj1,
            "obj2": obj2,
            "obj1_found": box1 is not None,
            "obj2_found": box2 is not None,
            "obj1_score": score1,
            "obj2_score": score2,
            "obj1_grid_box": str(obj1_grid_box),
            "obj2_grid_box": str(obj2_grid_box),
            "obj1_patch_count": len(obj1_ids),
            "obj2_patch_count": len(obj2_ids),
            "orig_pred": pred_by_condition["original"],
            "mask_obj1_pred": pred_by_condition["mask_obj1"],
            "mask_obj2_pred": pred_by_condition["mask_obj2"],
            "mask_both_pred": pred_by_condition["mask_both"],
            "mask_bg_pred": pred_by_condition["mask_background"],
            "orig_margin": margin_by_condition["original"],
            "mask_obj1_margin": margin_by_condition["mask_obj1"],
            "mask_obj2_margin": margin_by_condition["mask_obj2"],
            "mask_both_margin": margin_by_condition["mask_both"],
            "mask_bg_margin": margin_by_condition["mask_background"],
            "drop_mask_obj1": drop("mask_obj1"),
            "drop_mask_obj2": drop("mask_obj2"),
            "drop_mask_both": drop("mask_both"),
            "drop_mask_bg": drop("mask_background"),
        }

        csv_writer.writerow(csv_row)
        csv_file.flush()

        processed_count += 1

        print("\n" + "=" * 100)
        print(f"[{run_i + 1}/{len(indices)}] sample_id={sample_id}")
        print(
            f"gold={gold} | obj1={obj1} found={box1 is not None} "
            f"score={score1} grid={obj1_grid_box} patches={len(obj1_ids)}"
        )
        print(
            f"gold={gold} | obj2={obj2} found={box2 is not None} "
            f"score={score2} grid={obj2_grid_box} patches={len(obj2_ids)}"
        )
        print("question:", clean_question(prompt))

        for cond in ["original", "mask_obj1", "mask_obj2", "mask_both", "mask_background"]:
            s = scores_by_condition[cond]
            print(
                f"{cond:16s} "
                f"pred={pred_by_condition[cond]:6s} "
                f"margin={margin_by_condition[cond]} "
                f"scores={{left:{s['left']:.3f}, right:{s['right']:.3f}, "
                f"on:{s['on']:.3f}, under:{s['under']:.3f}}}"
            )

        print(
            "drops:",
            {
                "mask_obj1": drop("mask_obj1"),
                "mask_obj2": drop("mask_obj2"),
                "mask_both": drop("mask_both"),
                "mask_background": drop("mask_background"),
            },
        )

    csv_file.close()
    jsonl_file.close()

    print("\n[DONE]")
    print("processed:", processed_count)
    print("skipped:", skipped_count)
    print("jsonl:", jsonl_path)
    print("csv:", csv_path)


if __name__ == "__main__":
    main()
