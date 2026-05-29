import os
import csv
import json
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from model_zoo import get_model
from dataset_zoo import get_dataset
from model_zoo.llava15 import change_greedy_to_add_weight

try:
    from misc import _default_collate
except Exception:
    _default_collate = None

import save_llava_hidden_similarity_features as sf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Controlled_Images_A")
    p.add_argument("--option", default="four")
    p.add_argument("--model-name", default="llava1.5")
    p.add_argument("--method", default="adapt_vis")
    p.add_argument("--root-dir", default="data")
    p.add_argument("--device", default="cuda")

    # Key setting for this experiment.
    p.add_argument(
        "--variant",
        default="neg_mul_img",
        help="Use neg_mul_img: only negative image logits are multiplied.",
    )
    p.add_argument("--layers", default="0,1,2,3,4")
    p.add_argument("--object-patch-json", required=True)
    p.add_argument(
        "--missing-mask-mode",
        default="none",
        choices=["none", "all"],
        help="If a sample has no detected object patches: none=no intervention; all=all patches active.",
    )

    # AdaptVis threshold rule:
    # base confidence < threshold -> weight1, otherwise weight2.
    # For original AdaptVis-style mixed run, use --weight1 0.5 --weight2 1.5.
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--weight1", type=float, default=0.5)
    p.add_argument("--weight2", type=float, default=1.5)

    p.add_argument("--max-length", type=int, default=77)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--fresh-limit", type=int, default=-1)
    p.add_argument("--out-csv", default="output/objectbox_negpatch_summary.csv")
    p.add_argument("--save-records", default="")
    p.add_argument("--debug-first", type=int, default=0)
    return p.parse_args()


def norm_gold(x):
    if isinstance(x, list):
        return str(x[0]).strip() if x else ""
    return str(x).strip()


def raw_generation_correct(gold, gen):
    gold = norm_gold(gold)
    gen = str(gen)
    ok = (gold in gen) or (gold.lower() in gen.lower())
    if gold.lower() == "on" and "front" in gen.strip().lower():
        ok = False
    return bool(ok)


def make_keys_from_input_ids(single_input, model=None):
    image_token_index = getattr(getattr(model, "config", None), "image_token_index", None)
    if image_token_index is None:
        image_token_index = 32001
    return [
        torch.where(input_id == int(image_token_index), 1, 0)
        for input_id in single_input["input_ids"]
    ]


def generation_scores(output):
    if hasattr(output, "scores"):
        return output.scores
    if isinstance(output, dict):
        return output.get("scores", None)
    return output["scores"]


def generation_sequences(output):
    if hasattr(output, "sequences"):
        return output.sequences
    if isinstance(output, dict):
        return output.get("sequences", None)
    return output["sequences"]


def first_step_confidence(output):
    scores = generation_scores(output)
    if scores is None or len(scores) == 0:
        return 0.0
    prob = torch.nn.functional.softmax(scores[0], dim=-1)
    return float(torch.max(prob[0]).detach().float().cpu())


def decode_generated(processor, output, prompt_len):
    seq = generation_sequences(output)
    return processor.decode(seq[0][int(prompt_len):], skip_special_tokens=True)


def iter_samples(loader):
    sid = 0
    for batch in loader:
        for i_option in batch["image_options"]:
            for image in i_option:
                yield sid, image
                sid += 1


def load_patch_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept both:
    #   {"0": {"patch_ids": [...]}, ...}
    # and:
    #   [{"sample_id": 0, "patch_ids": [...]}, ...]
    if isinstance(data, list):
        data = {str(int(x["sample_id"])): x for x in data}
    if not isinstance(data, dict):
        raise TypeError(f"Unsupported object patch json type: {type(data)}")
    return data


def get_patch_ids(mask_data, sid):
    rec = mask_data.get(str(sid), None)
    if rec is None:
        return []
    ids = rec.get("patch_ids", [])
    return [int(x) for x in ids]


def set_common_env(args):
    os.environ["ADAPTVIS_ATTENTION_VARIANT"] = args.variant
    os.environ["ADAPTVIS_LAYER_MODE"] = "list"
    os.environ["ADAPTVIS_LAYERS"] = args.layers
    os.environ["ADAPTVIS_NUM_LAYERS"] = "32"
    # Disable block selector; use exact patch-id selector per sample.
    os.environ["ADAPTVIS_PATCH_BLOCK_MODE"] = "all"


def set_patch_env_for_sample(patch_ids, missing_mode="none"):
    if patch_ids:
        os.environ["ADAPTVIS_PATCH_ID_MODE"] = "only"
        os.environ["ADAPTVIS_PATCH_IDS"] = ",".join(str(int(x)) for x in sorted(set(patch_ids)))
    else:
        if missing_mode == "all":
            os.environ["ADAPTVIS_PATCH_ID_MODE"] = "all"
            os.environ["ADAPTVIS_PATCH_IDS"] = ""
        else:
            os.environ["ADAPTVIS_PATCH_ID_MODE"] = "only"
            os.environ["ADAPTVIS_PATCH_IDS"] = ""


def set_weight_env_for_sample(selected_w):
    # Some generate() paths drop custom kwargs such as weight before reaching LLaMAAttention.
    # modeling_llama_add_attn.py should read ADAPTVIS_WEIGHT as fallback.
    os.environ["ADAPTVIS_WEIGHT"] = str(float(selected_w))


def run_pass(args, wrapper, loader, prompts, answers, mode_name, mask_data=None, base_records=None):
    records = []
    correct = 0
    selected_05 = 0
    selected_15 = 0

    print("\n" + "=" * 80)
    print("[RUN]", mode_name)
    print("[VARIANT]", os.environ.get("ADAPTVIS_ATTENTION_VARIANT"))
    print("[LAYERS]", os.environ.get("ADAPTVIS_LAYERS"))
    print("[PATCH_ID_MODE] per sample" if mask_data is not None else "[PATCH_ID_MODE] off/base")
    print("=" * 80)

    with torch.no_grad():
        pbar = tqdm(iter_samples(loader), desc=mode_name)
        for sid, image in pbar:
            if args.fresh_limit > 0 and sid >= args.fresh_limit:
                break

            prompt = prompts[sid]
            gold = norm_gold(answers[sid])

            if mask_data is not None:
                patch_ids = get_patch_ids(mask_data, sid)
                set_patch_env_for_sample(patch_ids, args.missing_mask_mode)
            else:
                patch_ids = []
                os.environ["ADAPTVIS_PATCH_ID_MODE"] = "all"
                os.environ["ADAPTVIS_PATCH_IDS"] = ""

            single_input = wrapper.processor(
                text=prompt,
                images=image,
                padding="max_length",
                return_tensors="pt",
                max_length=args.max_length,
            ).to(args.device)

            keys = make_keys_from_input_ids(single_input, wrapper.model)
            prompt_len = len(single_input["input_ids"][-1])

            if base_records is None:
                selected_w = 1.0
            else:
                conf = float(base_records[sid]["confidence"])
                selected_w = args.weight1 if conf < args.threshold else args.weight2

            set_weight_env_for_sample(selected_w)

            if abs(selected_w - 0.5) <= 1e-6:
                selected_05 += 1
            elif abs(selected_w - 1.5) <= 1e-6:
                selected_15 += 1

            # Tag optional SAVE_IMG_LOGITS dumps so base/object files are distinguishable.
            os.environ["SAVE_IMG_LOGITS_TAG"] = (
                f"{mode_name}_sid{sid:04d}_w{str(float(selected_w)).replace('.', 'p')}"
            )

            if args.debug_first > 0 and sid < args.debug_first:
                print(
                    f"[RUN DEBUG] mode={mode_name} sid={sid} selected_w={selected_w} "
                    f"patch_mode={os.environ.get('ADAPTVIS_PATCH_ID_MODE')} "
                    f"npatch={len(patch_ids)} first10={patch_ids[:10]} "
                    f"ADAPTVIS_WEIGHT={os.environ.get('ADAPTVIS_WEIGHT')}"
                )

            output = wrapper.model.generate(
                **single_input,
                keys=keys,
                weight=float(selected_w),
                max_new_tokens=args.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

            gen = decode_generated(wrapper.processor, output, prompt_len)
            corr = raw_generation_correct(gold, gen)
            conf = first_step_confidence(output)

            correct += int(corr)
            records.append({
                "sample_id": int(sid),
                "gold": gold,
                "generation": gen,
                "correct": bool(corr),
                "confidence": float(conf),
                "selected_weight": float(selected_w),
                "num_object_patch_ids": int(len(patch_ids)),
                "object_patch_ids": patch_ids,
            })

            pbar.set_postfix({
                "sid": sid,
                "acc": f"{correct / max(len(records), 1):.3f}",
                "w0.5": selected_05,
                "w1.5": selected_15,
                "npatch": len(patch_ids),
            })

    summary = {
        "mode": mode_name,
        "num_total": len(records),
        "acc": correct / max(len(records), 1),
        "num_correct": correct,
        "selected_0p5_count": selected_05,
        "selected_1p5_count": selected_15,
    }
    return summary, records


def compare_groups(base_records, mixed_records):
    base_by_sid = {int(r["sample_id"]): r for r in base_records}
    stats = {
        "wrong_to_correct": 0,
        "correct_to_wrong": 0,
        "correct_to_correct": 0,
        "wrong_to_wrong": 0,
    }
    for r in mixed_records:
        sid = int(r["sample_id"])
        b_corr = bool(base_by_sid[sid]["correct"])
        m_corr = bool(r["correct"])
        if (not b_corr) and m_corr:
            stats["wrong_to_correct"] += 1
        elif b_corr and (not m_corr):
            stats["correct_to_wrong"] += 1
        elif b_corr and m_corr:
            stats["correct_to_correct"] += 1
        else:
            stats["wrong_to_wrong"] += 1
    stats["net_gain"] = stats["wrong_to_correct"] - stats["correct_to_wrong"]
    return stats


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    mask_data = load_patch_json(args.object_patch_json)
    set_common_env(args)

    change_greedy_to_add_weight()

    print("[LOAD MODEL]", args.model_name, args.method, args.device)
    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    wrapper.model.eval()

    print("[LOAD DATASET]", args.dataset)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=False)
    collate_fn = _default_collate if image_preprocess is None else None

    def make_loader():
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    prompts, answers = sf.load_prompts(args.dataset, args.option)

    # Base once. ADAPTVIS_WEIGHT=1.0 is a no-op, but keeps debug behavior consistent.
    base_summary, base_records = run_pass(
        args, wrapper, make_loader(), prompts, answers,
        mode_name="base", mask_data=None, base_records=None,
    )

    # Object-box-only negative-logit multiply.
    obj_summary, obj_records = run_pass(
        args, wrapper, make_loader(), prompts, answers,
        mode_name="object_box_negpatch", mask_data=mask_data, base_records=base_records,
    )
    stats = compare_groups(base_records, obj_records)

    rows = [
        {
            "mode": "base",
            "num_total": base_summary["num_total"],
            "acc": base_summary["acc"],
            "num_correct": base_summary["num_correct"],
            "base_acc": base_summary["acc"],
            "mixed_acc": "",
            "base_num_correct": base_summary["num_correct"],
            "mixed_num_correct": "",
            "wrong_to_correct": "",
            "correct_to_wrong": "",
            "net_gain": "",
            "selected_0p5_count": "",
            "selected_1p5_count": "",
        },
        {
            "mode": "object_box_negpatch",
            "num_total": obj_summary["num_total"],
            "acc": obj_summary["acc"],
            "num_correct": obj_summary["num_correct"],
            "base_acc": base_summary["acc"],
            "mixed_acc": obj_summary["acc"],
            "base_num_correct": base_summary["num_correct"],
            "mixed_num_correct": obj_summary["num_correct"],
            "wrong_to_correct": stats["wrong_to_correct"],
            "correct_to_wrong": stats["correct_to_wrong"],
            "net_gain": stats["net_gain"],
            "selected_0p5_count": obj_summary["selected_0p5_count"],
            "selected_1p5_count": obj_summary["selected_1p5_count"],
        },
    ]

    fieldnames = [
        "mode", "num_total", "acc", "num_correct",
        "base_acc", "mixed_acc", "base_num_correct", "mixed_num_correct",
        "wrong_to_correct", "correct_to_wrong", "net_gain",
        "selected_0p5_count", "selected_1p5_count",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if args.save_records:
        os.makedirs(os.path.dirname(args.save_records), exist_ok=True)
        with open(args.save_records, "w", encoding="utf-8") as f:
            json.dump({"base": base_records, "object_box_negpatch": obj_records}, f, ensure_ascii=False, indent=2)

    print("\n[DONE]")
    print("[CSV SAVED]", args.out_csv)
    print("[SUMMARY]", rows[-1])


if __name__ == "__main__":
    main()
