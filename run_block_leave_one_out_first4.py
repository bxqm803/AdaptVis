import os
import csv
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

    p.add_argument("--layers", default="0,1,2,3", help="Early layers used by the intervention.")
    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument("--patch-grid", type=int, default=4, help="4 means 4x4 spatial blocks = 16 blocks.")

    p.add_argument("--variant", default="negonly_median_img")
    p.add_argument("--posneg-strength", type=float, default=1.0)
    p.add_argument("--posneg-clamp-zero", default="1")

    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--weight1", type=float, default=0.5)
    p.add_argument("--weight2", type=float, default=1.5)

    p.add_argument("--max-length", type=int, default=77)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--fresh-limit", type=int, default=-1)

    p.add_argument("--out-csv", default="output/block_leave_one_out_first4_summary.csv")
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


def set_common_env(args, block_mode="all", blocks=""):
    os.environ["ADAPTVIS_ATTENTION_VARIANT"] = str(args.variant)
    os.environ["ADAPTVIS_POSNEG_STRENGTH"] = str(args.posneg_strength)
    os.environ["ADAPTVIS_POSNEG_CLAMP_ZERO"] = str(args.posneg_clamp_zero)

    os.environ["ADAPTVIS_LAYER_MODE"] = "list"
    os.environ["ADAPTVIS_LAYERS"] = str(args.layers)
    os.environ["ADAPTVIS_NUM_LAYERS"] = str(args.num_layers)

    # Keep head intervention unrestricted; this experiment ablates spatial blocks.
    os.environ["ADAPTVIS_HEAD_MODE"] = "all"
    os.environ.pop("ADAPTVIS_HEADS", None)

    os.environ["ADAPTVIS_PATCH_GRID"] = str(args.patch_grid)
    os.environ["ADAPTVIS_PATCH_BLOCK_MODE"] = str(block_mode)
    if blocks:
        os.environ["ADAPTVIS_PATCH_BLOCKS"] = str(blocks)
    else:
        os.environ.pop("ADAPTVIS_PATCH_BLOCKS", None)


def run_pass(args, wrapper, loader, prompts, answers, mode_name, base_records=None, block_mode="all", blocks=""):
    set_common_env(args, block_mode=block_mode, blocks=blocks)

    print("\n" + "=" * 80)
    print("[RUN]", mode_name)
    print("[VARIANT]", os.environ.get("ADAPTVIS_ATTENTION_VARIANT"))
    print("[LAYERS]", os.environ.get("ADAPTVIS_LAYERS"))
    print("[PATCH_GRID]", os.environ.get("ADAPTVIS_PATCH_GRID"))
    print("[PATCH_BLOCK_MODE]", os.environ.get("ADAPTVIS_PATCH_BLOCK_MODE"))
    print("[PATCH_BLOCKS]", os.environ.get("ADAPTVIS_PATCH_BLOCKS", ""))
    print("=" * 80)

    records = []
    num_correct = 0
    selected_05 = 0
    selected_15 = 0

    with torch.no_grad():
        pbar = tqdm(iter_samples(loader), desc=mode_name)
        for sid, image in pbar:
            if args.fresh_limit > 0 and sid >= args.fresh_limit:
                break

            prompt = prompts[sid]
            gold = norm_gold(answers[sid])

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

            if abs(selected_w - 0.5) <= 1e-6:
                selected_05 += 1
            elif abs(selected_w - 1.5) <= 1e-6:
                selected_15 += 1

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
            num_correct += int(corr)

            records.append({
                "sample_id": int(sid),
                "gold": gold,
                "generation": gen,
                "correct": bool(corr),
                "confidence": float(conf),
                "selected_weight": float(selected_w),
                "mode": mode_name,
                "block_mode": block_mode,
                "blocks": blocks,
            })

            pbar.set_postfix({
                "sid": sid,
                "acc": f"{num_correct / max(len(records), 1):.3f}",
                "w0.5": selected_05,
                "w1.5": selected_15,
            })

    summary = {
        "mode": mode_name,
        "block_mode": block_mode,
        "blocks": blocks,
        "num_total": len(records),
        "acc": num_correct / max(len(records), 1),
        "num_correct": num_correct,
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
        "selected0p5_wrong_to_correct": 0,
        "selected0p5_correct_to_wrong": 0,
        "selected0p5_correct_to_correct": 0,
        "selected0p5_wrong_to_wrong": 0,
    }
    for r in mixed_records:
        b = base_by_sid[int(r["sample_id"])]
        b_corr = bool(b["correct"])
        m_corr = bool(r["correct"])
        sw = float(r["selected_weight"])
        if (not b_corr) and m_corr:
            stats["wrong_to_correct"] += 1
            if abs(sw - 0.5) <= 1e-6:
                stats["selected0p5_wrong_to_correct"] += 1
        elif b_corr and (not m_corr):
            stats["correct_to_wrong"] += 1
            if abs(sw - 0.5) <= 1e-6:
                stats["selected0p5_correct_to_wrong"] += 1
        elif b_corr and m_corr:
            stats["correct_to_correct"] += 1
            if abs(sw - 0.5) <= 1e-6:
                stats["selected0p5_correct_to_correct"] += 1
        else:
            stats["wrong_to_wrong"] += 1
            if abs(sw - 0.5) <= 1e-6:
                stats["selected0p5_wrong_to_wrong"] += 1
    stats["net_gain"] = stats["wrong_to_correct"] - stats["correct_to_wrong"]
    stats["selected0p5_net_gain"] = stats["selected0p5_wrong_to_correct"] - stats["selected0p5_correct_to_wrong"]
    return stats


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    change_greedy_to_add_weight()

    print("[LOAD MODEL]", args.model_name, args.method, args.device)
    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    wrapper.model.eval()

    print("[LOAD DATASET]", args.dataset)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=False)
    collate_fn = _default_collate if image_preprocess is None else None

    def make_loader():
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    prompts, answers = sf.load_prompts(args.dataset, args.option)

    # Base is run once. It is used for confidence selection and group comparisons.
    base_summary, base_records = run_pass(
        args, wrapper, make_loader(), prompts, answers,
        mode_name="base",
        base_records=None,
        block_mode="all",
        blocks="",
    )

    # Full early spatial intervention baseline: all 4x4 blocks active.
    full_summary, full_records = run_pass(
        args, wrapper, make_loader(), prompts, answers,
        mode_name="full_all_blocks",
        base_records=base_records,
        block_mode="all",
        blocks="",
    )
    full_stats = compare_groups(base_records, full_records)

    rows = []
    rows.append({
        "mode": "base",
        "block_id": "",
        "block_row": "",
        "block_col": "",
        "excluded_block": "",
        "num_total": base_summary["num_total"],
        "base_acc": base_summary["acc"],
        "full_acc": "",
        "except_acc": "",
        "drop_acc_from_full": "",
        "base_num_correct": base_summary["num_correct"],
        "full_num_correct": "",
        "except_num_correct": "",
        "drop_correct_from_full": "",
        "selected_0p5_count": "",
        "selected_1p5_count": "",
        "full_net_gain": "",
        "except_net_gain": "",
        "drop_net_gain_from_full": "",
        "full_selected0p5_net_gain": "",
        "except_selected0p5_net_gain": "",
        "drop_selected0p5_net_gain_from_full": "",
    })
    rows.append({
        "mode": "full_all_blocks",
        "block_id": "",
        "block_row": "",
        "block_col": "",
        "excluded_block": "",
        "num_total": full_summary["num_total"],
        "base_acc": base_summary["acc"],
        "full_acc": full_summary["acc"],
        "except_acc": "",
        "drop_acc_from_full": "",
        "base_num_correct": base_summary["num_correct"],
        "full_num_correct": full_summary["num_correct"],
        "except_num_correct": "",
        "drop_correct_from_full": "",
        "selected_0p5_count": full_summary["selected_0p5_count"],
        "selected_1p5_count": full_summary["selected_1p5_count"],
        "full_net_gain": full_stats["net_gain"],
        "except_net_gain": "",
        "drop_net_gain_from_full": "",
        "full_selected0p5_net_gain": full_stats["selected0p5_net_gain"],
        "except_selected0p5_net_gain": "",
        "drop_selected0p5_net_gain_from_full": "",
    })

    num_blocks = int(args.patch_grid) * int(args.patch_grid)
    for block_id in range(num_blocks):
        block_row = block_id // int(args.patch_grid)
        block_col = block_id % int(args.patch_grid)
        mode_name = f"except_block_{block_id:02d}_r{block_row}_c{block_col}"

        except_summary, except_records = run_pass(
            args, wrapper, make_loader(), prompts, answers,
            mode_name=mode_name,
            base_records=base_records,
            block_mode="except",
            blocks=str(block_id),
        )
        except_stats = compare_groups(base_records, except_records)

        row = {
            "mode": mode_name,
            "block_id": block_id,
            "block_row": block_row,
            "block_col": block_col,
            "excluded_block": block_id,
            "num_total": except_summary["num_total"],
            "base_acc": base_summary["acc"],
            "full_acc": full_summary["acc"],
            "except_acc": except_summary["acc"],
            "drop_acc_from_full": full_summary["acc"] - except_summary["acc"],
            "base_num_correct": base_summary["num_correct"],
            "full_num_correct": full_summary["num_correct"],
            "except_num_correct": except_summary["num_correct"],
            "drop_correct_from_full": full_summary["num_correct"] - except_summary["num_correct"],
            "selected_0p5_count": except_summary["selected_0p5_count"],
            "selected_1p5_count": except_summary["selected_1p5_count"],
            "full_net_gain": full_stats["net_gain"],
            "except_net_gain": except_stats["net_gain"],
            "drop_net_gain_from_full": full_stats["net_gain"] - except_stats["net_gain"],
            "full_selected0p5_net_gain": full_stats["selected0p5_net_gain"],
            "except_selected0p5_net_gain": except_stats["selected0p5_net_gain"],
            "drop_selected0p5_net_gain_from_full": full_stats["selected0p5_net_gain"] - except_stats["selected0p5_net_gain"],
        }
        rows.append(row)
        print("[ROW]", row)

        # Stream-save after every block so a long run can be inspected/resumed manually.
        fieldnames = [
            "mode", "block_id", "block_row", "block_col", "excluded_block", "num_total",
            "base_acc", "full_acc", "except_acc", "drop_acc_from_full",
            "base_num_correct", "full_num_correct", "except_num_correct", "drop_correct_from_full",
            "selected_0p5_count", "selected_1p5_count",
            "full_net_gain", "except_net_gain", "drop_net_gain_from_full",
            "full_selected0p5_net_gain", "except_selected0p5_net_gain", "drop_selected0p5_net_gain_from_full",
        ]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print("\n[DONE]")
    print("[CSV SAVED]", args.out_csv)


if __name__ == "__main__":
    main()
