import os
import csv
import json
import argparse
from decimal import Decimal
import torch
from tqdm import tqdm
from PIL import Image
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
    p.add_argument(
        "--processed-manifest-json",
        default="",
        help="Optional manifest from dump_coco_qa_two_obj_processed_images_pad.py. If set, use processed images instead of dataset_zoo images.",
    )
    p.add_argument(
        "--object-patch-json",
        default="",
        help="Optional per-sample patch-id json. Each record should contain sample_id and patch_ids.",
    )
    p.add_argument(
        "--missing-mask-mode",
        default="none",
        choices=["none", "all"],
        help="If a sample has no patch_ids: none means intervene on no patches; all means fall back to all patches.",
    )
    p.add_argument(
        "--print-each-sample",
        action="store_true",
        help="Print per-sample patch count, AdaptVis gate, selected weight, correctness, and running accuracy.",
    )

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

    p.add_argument("--threshold-start", type=float, default=0.33)
    p.add_argument("--threshold-end", type=float, default=0.38)
    p.add_argument("--threshold-step", type=float, default=0.01)
    p.add_argument(
        "--thresholds",
        default="",
        help="Optional comma-separated thresholds, e.g. 0.33,0.34,0.35. If set, overrides start/end/step.",
    )
    p.add_argument(
        "--out-by-gold-csv",
        default="",
        help="Optional CSV path for per-gold statistics. Default: derive from --out-csv.",
    )
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

def load_patch_json(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        out = {}
        for x in data:
            if isinstance(x, dict):
                sid = x.get("sample_id", x.get("sid", x.get("idx", None)))
                if sid is not None:
                    out[str(int(sid))] = x
        data = out
    if not isinstance(data, dict):
        raise TypeError(f"Unsupported patch json type: {type(data)}")
    return data


def get_sample_patch_ids(mask_data, sid):
    if not mask_data:
        return []
    rec = mask_data.get(str(int(sid)), None)
    if rec is None:
        return []
    ids = rec.get("patch_ids", [])
    out = []
    for x in ids:
        try:
            xi = int(x)
        except Exception:
            continue
        if 0 <= xi < 576:
            out.append(xi)
    return sorted(set(out))


def set_patch_env_for_sample(patch_ids, missing_mask_mode="none", patch_side=24):
    patch_ids = sorted(set(int(x) for x in patch_ids if 0 <= int(x) < int(patch_side) * int(patch_side)))

    if patch_ids:
        os.environ["ADAPTVIS_PATCH_ID_MODE"] = "only"
        os.environ["ADAPTVIS_PATCH_IDS"] = ",".join(str(int(x)) for x in patch_ids)
    else:
        if missing_mask_mode == "all":
            all_ids = list(range(int(patch_side) * int(patch_side)))
            os.environ["ADAPTVIS_PATCH_ID_MODE"] = "only"
            os.environ["ADAPTVIS_PATCH_IDS"] = ",".join(str(int(x)) for x in all_ids)
        else:
            os.environ["ADAPTVIS_PATCH_ID_MODE"] = "only"
            os.environ["ADAPTVIS_PATCH_IDS"] = ""

    return (
        len(os.environ.get("ADAPTVIS_PATCH_IDS", "").split(","))
        if os.environ.get("ADAPTVIS_PATCH_IDS", "")
        else 0
    )

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
    # If --object-patch-json is provided, patch ids are set per sample in run_pass().
    # Otherwise, use the block/all-patch selection requested by --run-mode/--blocks.
    if getattr(args, "object_patch_json", ""):
        patch_ids = []
        os.environ["ADAPTVIS_PATCH_ID_MODE"] = "only"
        os.environ["ADAPTVIS_PATCH_IDS"] = ""
    else:
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


def set_weight_env_for_sample(selected_w):
    """
    Some generate() paths drop custom kwargs such as `weight` before they reach
    LLaMAAttention. The modified attention code reads ADAPTVIS_WEIGHT as a
    fallback, same as run_objectbox_negpatch_once.py.
    """
    os.environ["ADAPTVIS_WEIGHT"] = str(float(selected_w))


def run_pass(args, wrapper, loader, prompts, answers, mode_name, base_records=None, block_mode="all", blocks="", mask_data=None):
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
    adaptvis_count = 0
    total_patch_ids_used = 0
    total_patch_ids_adapted = 0

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

            base_conf_for_gate = None
            adaptvis_gate = False

            if base_records is None:
                selected_w = 1.0
            else:
                base_conf_for_gate = float(base_records[sid]["confidence"])
                adaptvis_gate = base_conf_for_gate < float(args.threshold)
                selected_w = args.weight1 if adaptvis_gate else args.weight2

            # If using per-sample CLIP low-sim patch ids, only set patch ids for samples
            # that pass the AdaptVis confidence gate. High-confidence samples get no patches.
            if getattr(args, "object_patch_json", ""):
                if base_records is not None and adaptvis_gate:
                    sample_patch_ids = get_sample_patch_ids(mask_data or {}, sid)
                else:
                    sample_patch_ids = []
                num_patch_ids = set_patch_env_for_sample(
                    sample_patch_ids,
                    missing_mask_mode=args.missing_mask_mode,
                    patch_side=args.patch_side,
                )
            else:
                num_patch_ids = (
                    len(os.environ.get("ADAPTVIS_PATCH_IDS", "").split(","))
                    if os.environ.get("ADAPTVIS_PATCH_IDS", "")
                    else 0
                )

            did_adaptvis = bool(base_records is not None and adaptvis_gate and num_patch_ids > 0)
            total_patch_ids_used += int(num_patch_ids)
            if did_adaptvis:
                adaptvis_count += 1
                total_patch_ids_adapted += int(num_patch_ids)

            # Important: the modified attention module reads ADAPTVIS_WEIGHT
            # as a fallback when generate() drops custom kwargs.
            set_weight_env_for_sample(selected_w)

            if sid < 3:
                print(
                    f"[DEBUG] mode={mode_name} sid={sid} "
                    f"base_conf={base_conf_for_gate} "
                    f"gate={adaptvis_gate} "
                    f"selected_w={selected_w} "
                    f"ADAPTVIS_WEIGHT={os.environ.get('ADAPTVIS_WEIGHT')} "
                    f"PATCH_ID_MODE={os.environ.get('ADAPTVIS_PATCH_ID_MODE')} "
                    f"num_patch_ids={num_patch_ids} "
                    f"did_adaptvis={did_adaptvis}"
                )

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
                "base_conf_for_gate": base_conf_for_gate,
                "adaptvis_gate": bool(adaptvis_gate),
                "did_adaptvis": bool(did_adaptvis),
                "num_patch_ids": int(num_patch_ids),
                "mode": mode_name,
                "block_mode": block_mode,
                "blocks": blocks,
            })

            running_acc = num_correct / max(len(records), 1)

            if args.print_each_sample:
                print(
                    f"[SAMPLE] mode={mode_name} sid={sid} "
                    f"patches={num_patch_ids} "
                    f"gate={adaptvis_gate} "
                    f"did_adaptvis={did_adaptvis} "
                    f"selected_w={selected_w} "
                    f"correct={bool(corr)} "
                    f"running_acc={running_acc:.4f}"
                )

            pbar.set_postfix({
                "sid": sid,
                "acc": f"{running_acc:.3f}",
                "patches": num_patch_ids,
                "adapt": adaptvis_count,
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
        "adaptvis_count": adaptvis_count,
        "avg_num_patch_ids": total_patch_ids_used / max(len(records), 1),
        "avg_num_patch_ids_adapted": total_patch_ids_adapted / max(adaptvis_count, 1),
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


def parse_thresholds(args):
    if str(args.thresholds).strip():
        vals = [float(x) for x in str(args.thresholds).split(",") if x.strip()]
        return vals

    start = Decimal(str(args.threshold_start))
    end = Decimal(str(args.threshold_end))
    step = Decimal(str(args.threshold_step))
    if step <= 0:
        raise ValueError("--threshold-step must be positive")

    vals = []
    x = start
    # inclusive end
    while x <= end + Decimal("1e-12"):
        vals.append(float(x))
        x += step
    return vals


def group_stats(records):
    n = len(records)
    correct = sum(int(bool(r.get("correct", False))) for r in records)
    avg_conf = sum(float(r.get("confidence", 0.0)) for r in records) / max(n, 1)
    w05 = sum(1 for r in records if abs(float(r.get("selected_weight", -999)) - 0.5) < 1e-6)
    w15 = sum(1 for r in records if abs(float(r.get("selected_weight", -999)) - 1.5) < 1e-6)
    adaptvis_count = sum(1 for r in records if bool(r.get("did_adaptvis", False)))
    avg_num_patch_ids = sum(int(r.get("num_patch_ids", 0)) for r in records) / max(n, 1)
    avg_num_patch_ids_adapted = (
        sum(int(r.get("num_patch_ids", 0)) for r in records if bool(r.get("did_adaptvis", False)))
        / max(adaptvis_count, 1)
    )
    return {
        "n": n,
        "acc": correct / max(n, 1),
        "num_correct": correct,
        "avg_conf": avg_conf,
        "w0p5": w05,
        "w1p5": w15,
        "adaptvis_count": adaptvis_count,
        "avg_num_patch_ids": avg_num_patch_ids,
        "avg_num_patch_ids_adapted": avg_num_patch_ids_adapted,
    }


def by_gold_stats(records):
    by_gold = {}
    for r in records:
        gold = str(r.get("gold", "")).strip()
        by_gold.setdefault(gold, []).append(r)

    rows = []
    for gold in sorted(by_gold.keys()):
        s = group_stats(by_gold[gold])
        s["gold"] = gold
        rows.append(s)
    return rows


def write_csv(path, rows, fieldnames):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def default_by_gold_path(out_csv):
    root, ext = os.path.splitext(out_csv)
    return root + "_by_gold" + (ext or ".csv")


class ProcessedManifestDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for preprocessed/padded images.

    Each item mimics the repo dataset format expected by iter_samples():
      batch["image_options"] -> [[PIL.Image]]
    Use DataLoader(..., collate_fn=lambda xs: xs[0]) with batch_size=1.
    """
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["processed_image_path"]).convert("RGB")
        img.filename = r["processed_image_path"]
        return {"image_options": [[img]]}


def load_processed_manifest(path, fresh_limit=-1):
    data = json.load(open(path, "r", encoding="utf-8"))

    if isinstance(data, dict):
        records = list(data.values())
    elif isinstance(data, list):
        records = data
    else:
        raise TypeError(f"Unsupported processed manifest type: {type(data)}")

    def get_idx(r):
        return int(r.get("sample_idx", r.get("sample_id", 0)))

    records = sorted(records, key=get_idx)
    if fresh_limit > 0:
        records = records[:fresh_limit]

    prompts = []
    answers = []

    for r in records:
        prompt = str(r.get("prompt", "")).strip()
        gold = str(r.get("gold", r.get("answer", ""))).strip()

        # Fallback if manifest has obj1/obj2 but not prompt.
        if not prompt:
            obj1 = str(r.get("obj1", "")).strip()
            obj2 = str(r.get("obj2", "")).strip()
            if obj1 and obj2:
                prompt = (
                    "<image>\n"
                    f"USER: Where is the {obj1} in relation to the {obj2}? "
                    "Answer with left, right, above, or below.\n"
                    "ASSISTANT:"
                )

        prompts.append(prompt)
        answers.append(gold)

    return records, prompts, answers


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    thresholds = parse_thresholds(args)
    print("[THRESHOLDS]", thresholds)

    change_greedy_to_add_weight()

    print("[LOAD MODEL]", args.model_name, args.method, args.device)
    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    wrapper.model.eval()

    if args.processed_manifest_json:
        print("[LOAD PROCESSED MANIFEST]", args.processed_manifest_json)
        records, prompts, answers = load_processed_manifest(
            args.processed_manifest_json,
            fresh_limit=args.fresh_limit,
        )
        dataset = ProcessedManifestDataset(records)
        collate_fn = lambda xs: xs[0]

        def make_loader():
            return DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )

        # fresh_limit is already applied to manifest records.
        args.fresh_limit = -1
        print("[PROCESSED DATASET] num records:", len(records))
        if records:
            print("[PROCESSED IMAGE 0]", records[0].get("processed_image_path", ""))
            print("[PROMPT 0]", prompts[0])
            print("[GOLD 0]", answers[0])
    else:
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

    mask_data = load_patch_json(args.object_patch_json) if args.object_patch_json else {}
    if args.object_patch_json:
        print("[LOAD OBJECT PATCH JSON]", args.object_patch_json)
        print("[NUM PATCH RECORDS]", len(mask_data))

    # Run base once. Its confidence is used for all threshold sweeps.
    base_summary, base_records = run_pass(
        args, wrapper, make_loader(), prompts, answers,
        mode_name="base",
        base_records=None,
        block_mode="all",
        blocks="",
    )

    summary_rows = []
    by_gold_rows = []

    base_overall = group_stats(base_records)
    summary_rows.append({
        "threshold": "",
        "mode": "base",
        "block_mode": "all",
        "blocks": "",
        "num_total": base_overall["n"],
        "acc": base_overall["acc"],
        "num_correct": base_overall["num_correct"],
        "avg_conf": base_overall["avg_conf"],
        "w0p5": base_overall["w0p5"],
        "w1p5": base_overall["w1p5"],
        "adaptvis_count": base_overall["adaptvis_count"],
        "avg_num_patch_ids": base_overall["avg_num_patch_ids"],
        "avg_num_patch_ids_adapted": base_overall["avg_num_patch_ids_adapted"],
        "wrong_to_correct": "",
        "correct_to_wrong": "",
        "correct_to_correct": "",
        "wrong_to_wrong": "",
        "net_gain": "",
        "selected0p5_wrong_to_correct": "",
        "selected0p5_correct_to_wrong": "",
        "selected0p5_net_gain": "",
    })

    for s in by_gold_stats(base_records):
        by_gold_rows.append({
            "threshold": "",
            "mode": "base",
            "gold": s["gold"],
            "n": s["n"],
            "acc": s["acc"],
            "num_correct": s["num_correct"],
            "avg_conf": s["avg_conf"],
            "w0p5": s["w0p5"],
            "w1p5": s["w1p5"],
            "adaptvis_count": s["adaptvis_count"],
            "avg_num_patch_ids": s["avg_num_patch_ids"],
            "avg_num_patch_ids_adapted": s["avg_num_patch_ids_adapted"],
        })

    records_payload = {
        "args": vars(args),
        "thresholds": thresholds,
        "base_summary": base_summary,
        "base_records": base_records,
        "sweeps": {},
    }

    if args.run_mode == "only_blocks":
        block_mode = "only"
        blocks = args.blocks
        mode_prefix = f"only_blocks_{str(args.blocks).replace(',', '_')}"
    elif args.run_mode == "all_blocks":
        block_mode = "all"
        blocks = ""
        mode_prefix = "full_all_blocks"
    else:
        raise ValueError("This sweep script supports --run-mode all_blocks or only_blocks. Leave-one-out is too expensive for threshold sweep.")

    for th in thresholds:
        args.threshold = float(th)
        mode_name = f"{mode_prefix}_thr_{th:.3f}".replace(".", "p")

        run_summary, run_records = run_pass(
            args, wrapper, make_loader(), prompts, answers,
            mode_name=mode_name,
            base_records=base_records,
            block_mode=block_mode,
            blocks=blocks,
            mask_data=mask_data,
        )
        stats = compare_groups(base_records, run_records)
        overall = group_stats(run_records)

        row = {
            "threshold": f"{th:.6f}",
            "mode": mode_name,
            "block_mode": block_mode,
            "blocks": blocks,
            "num_total": overall["n"],
            "acc": overall["acc"],
            "num_correct": overall["num_correct"],
            "avg_conf": overall["avg_conf"],
            "w0p5": overall["w0p5"],
            "w1p5": overall["w1p5"],
            "adaptvis_count": overall["adaptvis_count"],
            "avg_num_patch_ids": overall["avg_num_patch_ids"],
            "avg_num_patch_ids_adapted": overall["avg_num_patch_ids_adapted"],
            "wrong_to_correct": stats["wrong_to_correct"],
            "correct_to_wrong": stats["correct_to_wrong"],
            "correct_to_correct": stats["correct_to_correct"],
            "wrong_to_wrong": stats["wrong_to_wrong"],
            "net_gain": stats["net_gain"],
            "selected0p5_wrong_to_correct": stats["selected0p5_wrong_to_correct"],
            "selected0p5_correct_to_wrong": stats["selected0p5_correct_to_wrong"],
            "selected0p5_net_gain": stats["selected0p5_net_gain"],
        }
        summary_rows.append(row)

        for s in by_gold_stats(run_records):
            by_gold_rows.append({
                "threshold": f"{th:.6f}",
                "mode": mode_name,
                "gold": s["gold"],
                "n": s["n"],
                "acc": s["acc"],
                "num_correct": s["num_correct"],
                "avg_conf": s["avg_conf"],
                "w0p5": s["w0p5"],
                "w1p5": s["w1p5"],
            })

        records_payload["sweeps"][f"{th:.6f}"] = {
            "summary": run_summary,
            "stats": stats,
            "records": run_records,
        }

        print("[THRESHOLD ROW]", row)

        summary_fields = [
            "threshold", "mode", "block_mode", "blocks", "num_total",
            "acc", "num_correct", "avg_conf", "w0p5", "w1p5",
            "adaptvis_count", "avg_num_patch_ids", "avg_num_patch_ids_adapted",
            "wrong_to_correct", "correct_to_wrong", "correct_to_correct", "wrong_to_wrong", "net_gain",
            "selected0p5_wrong_to_correct", "selected0p5_correct_to_wrong", "selected0p5_net_gain",
        ]
        by_gold_fields = [
            "threshold", "mode", "gold", "n", "acc", "num_correct", "avg_conf", "w0p5", "w1p5",
            "adaptvis_count", "avg_num_patch_ids", "avg_num_patch_ids_adapted",
        ]

        write_csv(args.out_csv, summary_rows, summary_fields)
        by_gold_path = args.out_by_gold_csv if args.out_by_gold_csv else default_by_gold_path(args.out_csv)
        write_csv(by_gold_path, by_gold_rows, by_gold_fields)

    save_records_if_needed(args, records_payload)

    print("\n[DONE]")
    print("[SUMMARY CSV SAVED]", args.out_csv)
    print("[BY GOLD CSV SAVED]", args.out_by_gold_csv if args.out_by_gold_csv else default_by_gold_path(args.out_csv))


if __name__ == "__main__":
    main()
