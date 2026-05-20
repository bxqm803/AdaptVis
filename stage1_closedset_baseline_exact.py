import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

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


def clean_question(q: str) -> str:
    q = str(q)
    q = q.replace("<image>", " ")
    q = q.replace("USER:", " ")
    q = q.replace("ASSISTANT:", " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


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
# LLaVA-like manual processed PIL, copied from stage1 logic
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
        mean = getattr(
            image_processor,
            "image_mean",
            [0.48145466, 0.4578275, 0.40821073],
        )
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


# ============================================================
# HF LLaVA exact baseline loading
# ============================================================

def load_llava_hf(model_id: str, revision: str, cache_dir: str, device: str, dtype: str):
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

    # Important for official HF LLaVA path.
    patch_size = getattr(getattr(model.config, "vision_config", None), "patch_size", 14)
    vision_feature_select_strategy = getattr(
        model.config,
        "vision_feature_select_strategy",
        "default",
    )

    processor.patch_size = patch_size
    processor.vision_feature_select_strategy = vision_feature_select_strategy

    model = model.to(device).eval()

    print("[INFO] loaded official HF LLaVA")
    print(f"  model_id={model_id}")
    print(f"  revision={revision}")
    print(f"  model={type(model)}")
    print(f"  processor={type(processor)}")
    print(f"  patch_size={processor.patch_size}")
    print(f"  vision_feature_select_strategy={processor.vision_feature_select_strategy}")

    return processor, model


# ============================================================
# Exact closed-set scoring from ablation baseline
# ============================================================

@torch.no_grad()
def score_candidates_batch(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    candidates: List[str],
    device: str,
    debug: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Exact stage1 closed-set baseline.

    S(candidate) = average log-probability of appended candidate answer tokens.

    Constructs:
        prompt + " left"
        prompt + " right"
        prompt + " on"
        prompt + " under"

    Then scores only the final candidate answer tokens.
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

    outputs = model(**inputs)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    if debug:
        image_token_id = getattr(model.config, "image_token_index", 32000)
        if hasattr(processor.tokenizer, "convert_tokens_to_ids"):
            maybe_id = processor.tokenizer.convert_tokens_to_ids("<image>")
            if maybe_id is not None and maybe_id != processor.tokenizer.unk_token_id:
                image_token_id = maybe_id

        print("[DEBUG SHAPES]")
        print("  input_ids.shape:", tuple(input_ids.shape))
        print("  outputs.logits.shape:", tuple(outputs.logits.shape))
        print("  attention_sum:", attention_mask.sum(dim=1).detach().cpu().tolist())
        print(
            "  num_image_tokens:",
            (input_ids == image_token_id).sum(dim=1).detach().cpu().tolist(),
            "image_token_id=",
            image_token_id,
        )

    log_probs = F.log_softmax(logits.float(), dim=-1)
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
                "candidate_text": cand_text,
                "candidate_token_ids": [],
                "candidate_token_text": "",
                "token_logprobs": [],
            }
            continue

        nonpad_positions = torch.nonzero(
            attention_mask[b],
            as_tuple=False,
        ).squeeze(-1)

        cand_positions_in_input = nonpad_positions[-n_tok:]
        cand_positions_in_target = cand_positions_in_input - 1
        cand_positions_in_target = cand_positions_in_target[
            cand_positions_in_target >= 0
        ]

        vals = token_log_probs[b, cand_positions_in_target]

        if vals.numel() == 0:
            sum_lp = float("-inf")
            avg_lp = float("-inf")
            n_used = 0
            token_lps = []
        else:
            sum_lp = float(vals.sum().detach().cpu().item())
            avg_lp = float(vals.mean().detach().cpu().item())
            n_used = int(vals.numel())
            token_lps = [float(x) for x in vals.detach().cpu().tolist()]

        answer_ids = input_ids[b, cand_positions_in_input].detach().cpu().tolist()
        answer_ids = [int(x) for x in answer_ids]
        answer_text = tokenizer.decode(
            answer_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        if debug:
            print(
                f"  cand={cand:>5s}",
                "cand_text=", repr(cand_text),
                "cand_ids=", cand_ids,
                "answer_ids=", answer_ids,
                "answer_text=", repr(answer_text),
                "cand_positions_in_input=",
                cand_positions_in_input.detach().cpu().tolist(),
                "cand_positions_in_target=",
                cand_positions_in_target.detach().cpu().tolist(),
                "avg_lp=",
                avg_lp,
            )

        result[cand] = {
            "sum_logprob": sum_lp,
            "avg_logprob": avg_lp,
            "num_tokens": n_used,
            "candidate_text": cand_text,
            "candidate_token_ids": answer_ids,
            "candidate_token_text": answer_text,
            "token_logprobs": token_lps,
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
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Comma-separated sample ids, e.g. 0,1,2. If set, only these samples are processed.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--llava-model-id", default=LLAVA_MODEL_ID)
    parser.add_argument("--llava-revision", default=LLAVA_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--preprocess-mode", default="auto", choices=["auto", "crop", "pad"])
    parser.add_argument(
        "--image-source",
        default="processed",
        choices=["processed", "raw"],
        help="processed keeps original stage1 baseline behavior. raw is for checking raw dataset image path.",
    )

    parser.add_argument("--debug-first", action="store_true")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        args.dtype = "float32"

    out_dir = Path(args.out_dir or f"output/stage1_closedset_baseline_exact_{args.dataset}")
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
    print(f"[INFO] image_source={args.image_source}")

    print("[INFO] loading official HF LLaVA")
    processor, model = load_llava_hf(
        model_id=args.llava_model_id,
        revision=args.llava_revision,
        cache_dir=args.root_dir,
        device=device,
        dtype=args.dtype,
    )

    image_processor = processor.image_processor
    geom = infer_llava_geometry(image_processor)

    print("[INFO] inferred LLaVA image geometry:")
    for k, v in geom.items():
        if k != "resample":
            print(f"  {k}: {v}")
    print("[INFO] preprocess mode:", args.preprocess_mode)

    print("[INFO] loading dataset")
    dataset = load_dataset_any_signature(
        dataset_name=args.dataset,
        root_dir=args.root_dir,
        download=args.download,
    )

    prompt_rows = load_prompt_rows(args.dataset, args.option)
    n_total = min(len(dataset), len(prompt_rows))

    if args.sample_ids.strip():
        indices = []
        for x in args.sample_ids.split(","):
            x = x.strip()
            if not x:
                continue
            sid = int(x)
            if 0 <= sid < n_total:
                indices.append(sid)
            else:
                print(f"[WARN] sample_id out of range and skipped: {sid}")
    else:
        indices = list(range(n_total))
        if args.max_samples > 0:
            random.seed(args.seed)
            indices = random.sample(indices, min(args.max_samples, len(indices)))

    print(f"[INFO] total samples to run: {len(indices)}")
    print(f"[INFO] sample ids: {indices[:50]}{' ...' if len(indices) > 50 else ''}")

    csv_fields = [
        "sample_id",
        "gold",
        "pred",
        "correct",
        "margin",
        "score_left",
        "score_right",
        "score_on",
        "score_under",
        "image_source",
        "raw_size",
        "processed_size",
        "processed_meta",
    ]

    correct_count = 0
    processed_count = 0
    skipped_count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, \
         jsonl_path.open("w", encoding="utf-8") as jsonl_file:

        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        for run_i, sample_id in enumerate(tqdm(indices)):
            prompt = prompt_rows[sample_id].get("question", "")
            gold = get_gold_from_prompt_row(prompt_rows[sample_id])

            if gold not in RELATIONS:
                skipped_count += 1
                print(f"[SKIP] sample_id={sample_id}, invalid gold={gold}")
                continue

            raw = get_raw_pil_from_dataset(dataset, sample_id)

            processed, meta = make_processed_pil_like_llava(
                raw=raw,
                image_processor=image_processor,
                force_mode=args.preprocess_mode,
            )

            image = processed if args.image_source == "processed" else raw

            nested = score_candidates_batch(
                model=model,
                processor=processor,
                image=image,
                prompt=prompt,
                candidates=RELATIONS,
                device=device,
                debug=bool(args.debug_first and processed_count == 0),
            )
            scores = simple_score_dict(nested)
            pred = pred_from_scores(scores)
            margin = correct_margin(scores, gold)
            correct = bool(pred == gold)

            if correct:
                correct_count += 1
            processed_count += 1

            row = {
                "sample_id": sample_id,
                "gold": gold,
                "prompt": prompt,
                "clean_prompt": clean_question(prompt),
                "pred": pred,
                "correct": correct,
                "margin": margin,
                "scores": scores,
                "nested_scores": nested,
                "image_source": args.image_source,
                "raw_size": raw.size,
                "processed_size": processed.size,
                "processed_meta": meta,
            }

            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl_file.flush()

            csv_row = {
                "sample_id": sample_id,
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "margin": margin,
                "score_left": scores["left"],
                "score_right": scores["right"],
                "score_on": scores["on"],
                "score_under": scores["under"],
                "image_source": args.image_source,
                "raw_size": str(raw.size),
                "processed_size": str(processed.size),
                "processed_meta": json.dumps(meta, ensure_ascii=False),
            }
            writer.writerow(csv_row)
            csv_file.flush()

            print(
                f"[{run_i + 1}/{len(indices)}] "
                f"sample_id={sample_id} gold={gold} pred={pred} correct={correct} "
                f"scores={{left:{scores['left']:.4f}, right:{scores['right']:.4f}, "
                f"on:{scores['on']:.4f}, under:{scores['under']:.4f}}}"
            )

    acc = correct_count / processed_count if processed_count else 0.0

    print("\n[DONE]")
    print("processed:", processed_count)
    print("skipped:", skipped_count)
    print(f"acc: {correct_count}/{processed_count}={acc:.6f}")
    print("jsonl:", jsonl_path)
    print("csv:", csv_path)


if __name__ == "__main__":
    main()
