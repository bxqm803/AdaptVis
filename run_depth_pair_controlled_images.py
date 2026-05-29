import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List

from PIL import Image
import torch
from tqdm import tqdm

from model_zoo import get_model


CHOICES = ["left", "right", "on", "under"]


def load_jsonl_by_id(path: Path) -> Dict[int, Dict[str, Any]]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)
            out[int(x["id"])] = x
    return out


def resolve_rgb_path(p: str) -> Path:
    p = Path(p)
    candidates = [
        p,
        Path("data") / p,
        Path("data/controlled_images") / p.name,
        Path("data/controlled_clevr") / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"RGB image not found. Tried: {[str(x) for x in candidates]}")


def find_depth_path(depth_dir: Path, rgb_path: Path) -> Path:
    stem = rgb_path.stem
    candidates = [
        depth_dir / f"{stem}_depth.png",
        depth_dir / f"{stem}.png",
        depth_dir / f"{stem}_depth_gray.png",
        depth_dir / f"{stem}_depth_gray_aligned.png",
    ]

    for p in candidates:
        if p.exists():
            return p

    globbed = sorted(depth_dir.glob(f"{stem}*"))
    if globbed:
        return globbed[0]

    raise FileNotFoundError(
        f"Depth image not found for stem={stem}. Tried: {[str(x) for x in candidates]}"
    )


def build_two_image_prompt(original_question: str) -> str:
    prompt = str(original_question)

    if "<image>" in prompt:
        prompt = prompt.replace("<image>", "<image>\n<image>", 1)
    else:
        prompt = "<image>\n<image>\n" + prompt

    intro = (
        "You are given two images: the first is the original robot observation, "
        "and the second is its aligned depth map. "
    )

    if "USER:" in prompt:
        prompt = prompt.replace("USER:", "USER: " + intro, 1)
    else:
        prompt = "<image>\n<image>\nUSER: " + intro + prompt + "\nASSISTANT:"

    return prompt


def normalize_gold(answer_field) -> str:
    if isinstance(answer_field, list):
        if len(answer_field) == 0:
            return ""
        ans = str(answer_field[0])
    else:
        ans = str(answer_field)

    ans = ans.strip().lower()
    return ans


def extract_prediction(text: str) -> str:
    """
    从生成文本里抽取 left/right/on/under。
    优先匹配独立单词，找不到就做宽松匹配。
    """
    t = text.strip().lower()
    t = t.replace(",", " ").replace(".", " ").replace(";", " ").replace(":", " ")
    t = re.sub(r"\s+", " ", t)

    for c in CHOICES:
        if re.search(rf"\b{re.escape(c)}\b", t):
            return c

    # 宽松回退
    for c in CHOICES:
        if c in t:
            return c

    return ""


def run_one_sample(
    wrapper,
    device: str,
    rgb_path: Path,
    depth_path: Path,
    prompt: str,
    max_length: int,
    max_new_tokens: int,
):
    rgb = Image.open(rgb_path).convert("RGB")
    depth = Image.open(depth_path).convert("RGB")

    single_input = wrapper.processor(
        text=prompt,
        images=[rgb, depth],
        padding="max_length",
        return_tensors="pt",
        max_length=max_length,
    ).to(device)

    image_token_index = getattr(wrapper.model.config, "image_token_index", 32001)
    num_img_tokens = int((single_input["input_ids"] == image_token_index).sum().item())

    with torch.no_grad():
        output = wrapper.model.generate(
            **single_input,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

    prompt_len = len(single_input["input_ids"][-1])
    gen = wrapper.processor.decode(
        output["sequences"][0][prompt_len:],
        skip_special_tokens=True,
    )

    return {
        "generation": gen,
        "num_img_tokens": num_img_tokens,
        "input_ids_shape": tuple(single_input["input_ids"].shape),
        "pixel_values_shape": tuple(single_input["pixel_values"].shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", type=str, default="data/controlled_images_dataset.json")
    parser.add_argument(
        "--prompt-jsonl",
        type=str,
        default="prompts/Controlled_Images_A_with_answer_four_options.jsonl",
    )
    parser.add_argument("--depth-dir", type=str, default="output/depthanything3_all")
    parser.add_argument("--model-name", type=str, default="llava1.5")
    parser.add_argument("--method", type=str, default="adapt_vis")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--print-first", type=int, default=3)
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default="output/depth_pair_controlled_images_A_results.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="output/depth_pair_controlled_images_A_summary.json",
    )
    args = parser.parse_args()

    dataset_json = Path(args.dataset_json)
    prompt_jsonl = Path(args.prompt_jsonl)
    depth_dir = Path(args.depth_dir)
    out_jsonl = Path(args.out_jsonl)
    summary_json = Path(args.summary_json)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    print("[LOAD MODEL]")
    wrapper, _ = get_model(args.model_name, args.device, args.method)
    wrapper.model.eval()

    print("[LOAD DATASET JSON]", dataset_json)
    dataset = json.load(open(dataset_json, "r", encoding="utf-8"))

    print("[LOAD PROMPT JSONL]", prompt_jsonl)
    prompts = load_jsonl_by_id(prompt_jsonl)

    n_total_dataset = len(dataset)
    print("[DATASET LEN]", n_total_dataset)
    print("[PROMPT LEN]", len(prompts))

    start = max(0, args.start)
    end = n_total_dataset if args.limit < 0 else min(n_total_dataset, start + args.limit)

    total = 0
    correct = 0
    invalid_pred = 0
    missing_prompt = 0
    records: List[Dict[str, Any]] = []

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for sid in tqdm(range(start, end), desc="depth_pair_eval"):
            item = dataset[sid]

            if sid not in prompts:
                missing_prompt += 1
                rec = {
                    "sid": sid,
                    "status": "missing_prompt",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            prompt_item = prompts[sid]
            original_question = prompt_item["question"]
            gold = normalize_gold(prompt_item.get("answer", ""))

            rgb_path = resolve_rgb_path(item["image_path"])
            depth_path = find_depth_path(depth_dir, rgb_path)

            prompt = build_two_image_prompt(original_question)

            info = run_one_sample(
                wrapper=wrapper,
                device=args.device,
                rgb_path=rgb_path,
                depth_path=depth_path,
                prompt=prompt,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )

            generation = info["generation"]
            pred = extract_prediction(generation)
            is_correct = (pred == gold)

            total += 1
            correct += int(is_correct)
            if pred == "":
                invalid_pred += 1

            rec = {
                "sid": sid,
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "question": original_question,
                "actual_prompt": prompt,
                "gold": gold,
                "generation": generation,
                "pred": pred,
                "correct": bool(is_correct),
                "num_img_tokens": info["num_img_tokens"],
                "input_ids_shape": list(info["input_ids_shape"]),
                "pixel_values_shape": list(info["pixel_values_shape"]),
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)

            if total <= args.print_first:
                print("=" * 100)
                print("[SID]", sid)
                print("[RGB]", rgb_path)
                print("[DEPTH]", depth_path)
                print("[ORIGINAL QUESTION]")
                print(original_question)
                print("[ACTUAL PROMPT]")
                print(prompt)
                print("[GOLD]", gold)
                print("[GENERATION]", generation)
                print("[PRED]", pred)
                print("[CORRECT]", is_correct)
                print("[pixel_values shape]", info["pixel_values_shape"])
                print("[num image tokens]", info["num_img_tokens"])

    acc = 0.0 if total == 0 else correct / total
    summary = {
        "dataset_json": str(dataset_json),
        "prompt_jsonl": str(prompt_jsonl),
        "depth_dir": str(depth_dir),
        "model_name": args.model_name,
        "method": args.method,
        "device": args.device,
        "start": start,
        "end": end,
        "num_evaluated": total,
        "num_correct": correct,
        "accuracy": acc,
        "num_invalid_pred": invalid_pred,
        "num_missing_prompt": missing_prompt,
        "out_jsonl": str(out_jsonl),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("[DONE]")
    print("[EVALUATED]", total)
    print("[CORRECT]", correct)
    print("[ACC]", acc)
    print("[INVALID PRED]", invalid_pred)
    print("[MISSING PROMPT]", missing_prompt)
    print("[RESULTS]", out_jsonl)
    print("[SUMMARY]", summary_json)
    print("=" * 100)


if __name__ == "__main__":
    main()
