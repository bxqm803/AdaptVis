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

    p.add_argument("--layers", default="0,1,2,3,4", help="Transformer layers used by the intervention.")
    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument("--patch-grid", type=int, default=4, help="4 means 4x4 spatial blocks = 16 blocks.")

    p.add_argument("--variant", default="negonly_mean_img")
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
    p.add_argument("--save-records", default="", help="Optional json path to save base/full/only/except records.")
    p.add_argument(
        "--run-mode",
        default="leave_one_out",
        choices=["leave_one_out", "only_blocks", "all_blocks"],
        help="leave_one_out: original 16-block ablation; only_blocks: intervene only selected blocks; all_blocks: intervene all patches only.",
    )
    p.add_argument("--blocks", default="13,14", help="Comma-separated spatial block ids for --run-mode only_blocks.")
    p.add_argument("--patch-side", type=int, default=24, help="LLaVA-1.5 uses 24x24 image patches for 336 input.")
    p.add_argument("--true-base", action="store_true", help="Run base without keys/weight intervention. Recommended for sanity checking.")
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


def parse_block_list(blocks):
    if blocks is None or str(blocks).strip() == "":
        return []
    return [int(x) for x in str(blocks).split(",") if x.strip() != ""]


def block_ids_to_patch_ids(blocks, patch_grid=4, patch_side=24):
    """
    Convert row-major spatial block ids to explicit 24x24 patch ids.

    For patch_grid=4, block layout is:
      0   1   2   3
      4   5   6   7
      8   9  10  11
      12 13  14  15

    Each block covers (24 / 4) x (24 / 4) = 6 x 6 patches.
    """
    block_ids = parse_block_list(blocks)
    patch_grid = int(patch_grid)
    patch_side = int(patch_side)
    assert patch_side % patch_grid == 0, f"patch_side={patch_side} not divisible by patch_grid={patch_grid}"

    block_h = patch_side // patch_grid
    block_w = patch_side // patch_grid

    patch_ids = []
    for bid in block_ids:
        br = int(bid) // patch_grid
        bc = int(bid) % patch_grid
        if br < 0 or br >= patch_grid or bc < 0 or bc >= patch_grid:
            raise ValueError(f"Invalid block id {bid} for patch_grid={patch_grid}")
        r0, r1 = br * block_h, (br + 1) * block_h
        c0, c1 = bc * block_w, (bc + 1) * block_w
        for r in range(r0, r1):
            for c in range(c0, c1):
                patch_ids.append(r * patch_side + c)
    return sorted(set(patch_ids))


def get_patch_ids_for_block_mode(args, block_mode="all", blocks=""):
    patch_side = int(args.patch_side)
    patch_grid = int(args.patch_grid)
    all_patch_ids = list(range(patch_side * patch_side))

    if block_mode == "all":
        return all_patch_ids

    if block_mode == "only":
        return block_ids_to_patch_ids(blocks, patch_grid=patch_grid, patch_side=patch_side)

    if block_mode == "except":
        remove = set(block_ids_to_patch_ids(blocks, patch_grid=patch_grid, patch_side=patch_side))
        return [pid for pid in all_patch_ids if pid not in remove]

    raise ValueError(f"Unknown block_mode={block_mode}")

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

    # IMPORTANT: use the verified explicit patch-id path.
    # The older ADAPTVIS_PATCH_BLOCK_MODE path is not reliably consumed by the patched attention code.
    patch_ids = get_patch_ids_for_block_mode(args, block_mode=block_mode, blocks=blocks)
    os.environ["ADAPTVIS_PATCH_ID_MODE"] = "only"
    os.environ["ADAPTVIS_PATCH_IDS"] = ",".join(str(int(x)) for x in patch_ids)

    # Keep these only for logging / compatibility. Do not rely on them for selection.
    os.environ["ADAPTVIS_PATCH_GRID"] = str(args.patch_grid)
    os.environ["ADAPTVIS_PATCH_BLOCK_MODE"] = str(block_mode)
    if blocks:
        os.environ["ADAPTVIS_PATCH_BLOCKS"] = str(blocks)
    else:
        os.environ.pop("ADAPTVIS_PATCH_BLOCKS", None)

    print("[PATCH_ID_MODE]", os.environ.get("ADAPTVIS_PATCH_ID_MODE"))
    print("[NUM_PATCH_IDS]", len(patch_ids))
    print("[PATCH_IDS_HEAD]", patch_ids[:20])


def run_pass(args, wrapper, loader, prompts, answers, mode_name, base_records=None, block_mode="all", blocks=""):
    set_common_env(args, block_mode=block_mode, blocks=blocks)

    print("\n" + "=" * 80)
    print("[RUN]", mode_name)
    print("[VARIANT]", os.environ.get("ADAPTVIS_ATTENTION_VARIANT"))
    print("[LAYERS]", os.environ.get("ADAPTVIS_LAYERS"))
    print("[PATCH_GRID]", os.environ.get("ADAPTVIS_PATCH_GRID"))
    print("[PATCH_BLOCK_MODE]", os.environ.get("ADAPTVIS_PATCH_BLOCK_MODE"))
    print("[PATCH_BLOCKS]", os.environ.get("ADAPTVIS_PATCH_BLOCKS", ""))
    print("[PATCH_ID_MODE]", os.environ.get("ADAPTVIS_PATCH_ID_MODE", ""))
    print("[NUM_PATCH_IDS]", len(os.environ.get("ADAPTVIS_PATCH_IDS", "").split(",")) if os.environ.get("ADAPTVIS_PATCH_IDS", "") else 0)
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

            if base_records is None and args.true_base:
                # Clean baseline: no keys and no weight intervention.
                output = wrapper.model.generate(
                    **single_input,
                    max_new_tokens=args.max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            else:
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

def save_records_if_needed(args, payload):
    if not args.save_records:
        return
    d = os.path.dirname(args.save_records)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.save_records, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("[RECORDS SAVED]", args.save_records)


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

    rows = []
    records_payload = {
        "args": vars(args),
        "base_summary": base_summary,
        "base_records": base_records,
    }

    # If user wants only specific blocks, do not run full or leave-one-out.
    if args.run_mode == "only_blocks":
        only_name = f"only_blocks_{str(args.blocks).replace(',', '_')}"
        only_summary, only_records = run_pass(
            args, wrapper, make_loader(), prompts, answers,
            mode_name=only_name,
            base_records=base_records,
            block_mode="only",
            blocks=args.blocks,
        )
        only_stats = compare_groups(base_records, only_records)

        rows.append({
            "mode": "base",
            "block_id": "",
            "block_row": "",
            "block_col": "",
            "excluded_block": "",
            "blocks": "",
            "block_mode": "all",
            "num_total": base_summary["num_total"],
            "base_acc": base_summary["acc"],
            "mode_acc": base_summary["acc"],
            "num_correct": base_summary["num_correct"],
            "wrong_to_correct": "",
            "correct_to_wrong": "",
            "correct_to_correct": "",
            "wrong_to_wrong": "",
            "net_gain": "",
            "selected_0p5_count": "",
            "selected_1p5_count": "",
            "selected0p5_net_gain": "",
        })
        rows.append({
            "mode": only_name,
            "block_id": "",
            "block_row": "",
            "block_col": "",
            "excluded_block": "",
            "blocks": args.blocks,
            "block_mode": "only",
            "num_total": only_summary["num_total"],
            "base_acc": base_summary["acc"],
            "mode_acc": only_summary["acc"],
            "num_correct": only_summary["num_correct"],
            "wrong_to_correct": only_stats["wrong_to_correct"],
            "correct_to_wrong": only_stats["correct_to_wrong"],
            "correct_to_correct": only_stats["correct_to_correct"],
            "wrong_to_wrong": only_stats["wrong_to_wrong"],
            "net_gain": only_stats["net_gain"],
            "selected_0p5_count": only_summary["selected_0p5_count"],
            "selected_1p5_count": only_summary["selected_1p5_count"],
            "selected0p5_net_gain": only_stats["selected0p5_net_gain"],
        })

        fieldnames = list(rows[0].keys())
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        records_payload.update({
            "only_summary": only_summary,
            "only_stats": only_stats,
            "only_records": only_records,
        })
        save_records_if_needed(args, records_payload)
        print("[ONLY SUMMARY]", only_summary)
        print("[ONLY STATS]", only_stats)
        print("[CSV SAVED]", args.out_csv)
        return

    # Full spatial intervention baseline: all patches active.
    full_summary, full_records = run_pass(
        args, wrapper, make_loader(), prompts, answers,
        mode_name="full_all_blocks",
        base_records=base_records,
        block_mode="all",
        blocks="",
    )
    full_stats = compare_groups(base_records, full_records)

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

    records_payload.update({
        "full_summary": full_summary,
        "full_stats": full_stats,
        "full_records": full_records,
    })

    if args.run_mode == "all_blocks":
        fieldnames = list(rows[0].keys())
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        save_records_if_needed(args, records_payload)
        print("[FULL SUMMARY]", full_summary)
        print("[FULL STATS]", full_stats)
        print("[CSV SAVED]", args.out_csv)
        return

    num_blocks = int(args.patch_grid) * int(args.patch_grid)
    except_payload = {}
    fieldnames = [
        "mode", "block_id", "block_row", "block_col", "excluded_block", "num_total",
        "base_acc", "full_acc", "except_acc", "drop_acc_from_full",
        "base_num_correct", "full_num_correct", "except_num_correct", "drop_correct_from_full",
        "selected_0p5_count", "selected_1p5_count",
        "full_net_gain", "except_net_gain", "drop_net_gain_from_full",
        "full_selected0p5_net_gain", "except_selected0p5_net_gain", "drop_selected0p5_net_gain_from_full",
    ]

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

        except_payload[mode_name] = {
            "summary": except_summary,
            "stats": except_stats,
            "records": except_records,
        }

        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                # only write original-compatible columns
                w.writerow({k: r.get(k, "") for k in fieldnames})

    records_payload["except"] = except_payload
    save_records_if_needed(args, records_payload)

    print("\n[DONE]")
    print("[CSV SAVED]", args.out_csv)


if __name__ == "__main__":
    main()
