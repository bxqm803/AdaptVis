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

    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--weight1", type=float, default=0.5)
    p.add_argument("--weight2", type=float, default=1.5)

    p.add_argument("--max-length", type=int, default=77)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--fresh-limit", type=int, default=-1)

    p.add_argument("--num-layers", type=int, default=32)

    # 从后往前消融：
    # full: all layers
    # end=30: use layers 0..30, disable layer31
    # end=29: use layers 0..29, disable layer30..31
    p.add_argument("--start-end-layer", type=int, default=13)
    p.add_argument("--stop-end-layer", type=int, default=0)

    p.add_argument(
        "--out-csv",
        default="output/layer_ablation_backward_once_summary.csv",
    )
    p.add_argument(
        "--save-json",
        action="store_true",
        help="Optional: save per-mode generation json files. Default false.",
    )
    p.add_argument(
        "--json-dir",
        default="output/layer_ablation_backward_once_json",
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

    # follow current repo rule
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
    try:
        return output["scores"]
    except Exception:
        return None


def generation_sequences(output):
    if hasattr(output, "sequences"):
        return output.sequences
    if isinstance(output, dict):
        return output.get("sequences", None)
    return output["sequences"]


def first_step_confidence(output):
    scores = generation_scores(output)
    if scores is None or len(scores) == 0:
        return None

    s0 = scores[0]  # [batch, vocab]
    prob = torch.nn.functional.softmax(s0, dim=-1)
    return float(torch.max(prob[0]).detach().float().cpu())


def decode_generated(processor, output, prompt_len):
    seq = generation_sequences(output)
    return processor.decode(
        seq[0][int(prompt_len):],
        skip_special_tokens=True,
    )


def iter_samples(loader):
    sid = 0
    for batch in loader:
        for i_option in batch["image_options"]:
            for image in i_option:
                yield sid, image
                sid += 1


def run_generation_one_pass(
    *,
    args,
    wrapper,
    loader,
    prompts,
    answers,
    mode_name,
    layer_mode,
    layer_start=None,
    layer_end=None,
    use_base_conf=None,
):
    """
    If use_base_conf is None:
        run base with weight=1.0 and collect confidence.
    Else:
        run mixed; selected weight from use_base_conf[sid]["confidence"].
    """

    os.environ["ADAPTVIS_LAYER_MODE"] = str(layer_mode)
    os.environ["ADAPTVIS_NUM_LAYERS"] = str(args.num_layers)

    if layer_start is not None:
        os.environ["ADAPTVIS_LAYER_START"] = str(layer_start)
    else:
        os.environ.pop("ADAPTVIS_LAYER_START", None)

    if layer_end is not None:
        os.environ["ADAPTVIS_LAYER_END"] = str(layer_end)
    else:
        os.environ.pop("ADAPTVIS_LAYER_END", None)

    print()
    print("=" * 80)
    print(f"[RUN MODE] {mode_name}")
    print("[ADAPTVIS_LAYER_MODE]", os.environ.get("ADAPTVIS_LAYER_MODE"))
    print("[ADAPTVIS_LAYER_START]", os.environ.get("ADAPTVIS_LAYER_START"))
    print("[ADAPTVIS_LAYER_END]", os.environ.get("ADAPTVIS_LAYER_END"))
    print("=" * 80)

    records = []
    num_correct = 0
    selected_05 = 0
    selected_15 = 0

    desc = "base generation" if use_base_conf is None else f"mixed {mode_name}"
    pbar = tqdm(iter_samples(loader), desc=desc)

    with torch.no_grad():
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

            if use_base_conf is None:
                selected_w = 1.0
            else:
                conf = float(use_base_conf[sid]["confidence"])
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
            num_correct += int(corr)

            conf = first_step_confidence(output)

            rec = {
                "sample_id": int(sid),
                "gold": gold,
                "generation": gen,
                "correct": bool(corr),
                "confidence": float(conf) if conf is not None else 0.0,
                "selected_weight": float(selected_w),
                "mode": mode_name,
                "layer_mode": layer_mode,
                "layer_start": layer_start,
                "layer_end": layer_end,
            }
            records.append(rec)

            pbar.set_postfix({
                "sid": sid,
                "acc": f"{num_correct / max(len(records), 1):.3f}",
                "w0.5": selected_05,
                "w1.5": selected_15,
            })

    n = max(len(records), 1)
    summary = {
        "mode": mode_name,
        "layer_mode": layer_mode,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "num_total": len(records),
        "acc": num_correct / n,
        "num_correct": num_correct,
        "selected_0p5_count": selected_05,
        "selected_1p5_count": selected_15,
    }

    return summary, records


def compare_groups(base_records, mixed_records):
    base_by_sid = {int(r["sample_id"]): r for r in base_records}

    wrong_to_correct = 0
    correct_to_wrong = 0
    correct_to_correct = 0
    wrong_to_wrong = 0

    selected_05_wrong_to_correct = 0
    selected_05_correct_to_wrong = 0
    selected_05_correct_to_correct = 0
    selected_05_wrong_to_wrong = 0

    for r in mixed_records:
        sid = int(r["sample_id"])
        b = base_by_sid[sid]

        b_corr = bool(b["correct"])
        m_corr = bool(r["correct"])
        sw = float(r["selected_weight"])

        if (not b_corr) and m_corr:
            wrong_to_correct += 1
            if abs(sw - 0.5) <= 1e-6:
                selected_05_wrong_to_correct += 1
        elif b_corr and (not m_corr):
            correct_to_wrong += 1
            if abs(sw - 0.5) <= 1e-6:
                selected_05_correct_to_wrong += 1
        elif b_corr and m_corr:
            correct_to_correct += 1
            if abs(sw - 0.5) <= 1e-6:
                selected_05_correct_to_correct += 1
        else:
            wrong_to_wrong += 1
            if abs(sw - 0.5) <= 1e-6:
                selected_05_wrong_to_wrong += 1

    return {
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "correct_to_correct": correct_to_correct,
        "wrong_to_wrong": wrong_to_wrong,
        "net_gain": wrong_to_correct - correct_to_wrong,
        "selected0p5_wrong_to_correct": selected_05_wrong_to_correct,
        "selected0p5_correct_to_wrong": selected_05_correct_to_wrong,
        "selected0p5_correct_to_correct": selected_05_correct_to_correct,
        "selected0p5_wrong_to_wrong": selected_05_wrong_to_wrong,
        "selected0p5_net_gain": selected_05_wrong_to_correct - selected_05_correct_to_wrong,
    }


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    if args.save_json:
        os.makedirs(args.json_dir, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    change_greedy_to_add_weight()

    print("[LOAD MODEL]", args.model_name, args.method, args.device)
    wrapper, image_preprocess = get_model(
        args.model_name,
        args.device,
        args.method,
        root_dir=args.root_dir,
    )
    wrapper.model.eval()

    print("[LOAD DATASET]", args.dataset)
    dataset = get_dataset(
        args.dataset,
        image_preprocess=image_preprocess,
        download=False,
    )

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

    # 1. Base only once
    base_summary, base_records = run_generation_one_pass(
        args=args,
        wrapper=wrapper,
        loader=make_loader(),
        prompts=prompts,
        answers=answers,
        mode_name="base",
        layer_mode="all",
        layer_start=None,
        layer_end=None,
        use_base_conf=None,
    )

    print("[BASE SUMMARY]", base_summary)

    # 2. Build mode list
    modes = []

    # full-layer AdaptVis baseline
    modes.append({
        "mode_name": "all_layers",
        "layer_mode": "all",
        "layer_start": None,
        "layer_end": None,
    })

    # backward ablation:
    # 0..30, 0..29, ..., 0..0
    for end_layer in range(args.start_end_layer, args.stop_end_layer - 1, -1):
        modes.append({
            "mode_name": f"layers_0_to_{end_layer}",
            "layer_mode": "range",
            "layer_start": 0,
            "layer_end": int(end_layer),
        })

    rows = []

    # base row
    rows.append({
        **base_summary,
        "base_acc": base_summary["acc"],
        "mixed_acc": "",
        "base_num_correct": base_summary["num_correct"],
        "mixed_num_correct": "",
        "wrong_to_correct": "",
        "correct_to_wrong": "",
        "net_gain": "",
        "selected0p5_wrong_to_correct": "",
        "selected0p5_correct_to_wrong": "",
        "selected0p5_net_gain": "",
    })

    # 3. Mixed modes; base not rerun
    for m in modes:
        mixed_summary, mixed_records = run_generation_one_pass(
            args=args,
            wrapper=wrapper,
            loader=make_loader(),
            prompts=prompts,
            answers=answers,
            mode_name=m["mode_name"],
            layer_mode=m["layer_mode"],
            layer_start=m["layer_start"],
            layer_end=m["layer_end"],
            use_base_conf=base_records,
        )

        group_stats = compare_groups(base_records, mixed_records)

        row = {
            **mixed_summary,
            "base_acc": base_summary["acc"],
            "mixed_acc": mixed_summary["acc"],
            "base_num_correct": base_summary["num_correct"],
            "mixed_num_correct": mixed_summary["num_correct"],
            **group_stats,
        }
        rows.append(row)

        print("[MIXED SUMMARY]", row)

        if args.save_json:
            path = os.path.join(args.json_dir, f"{m['mode_name']}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(mixed_records, f, ensure_ascii=False, indent=2)

    # 4. Save CSV only
    fieldnames = [
        "mode",
        "layer_mode",
        "layer_start",
        "layer_end",
        "num_total",
        "base_acc",
        "mixed_acc",
        "base_num_correct",
        "mixed_num_correct",
        "acc",
        "num_correct",
        "selected_0p5_count",
        "selected_1p5_count",
        "wrong_to_correct",
        "correct_to_wrong",
        "correct_to_correct",
        "wrong_to_wrong",
        "net_gain",
        "selected0p5_wrong_to_correct",
        "selected0p5_correct_to_wrong",
        "selected0p5_correct_to_correct",
        "selected0p5_wrong_to_wrong",
        "selected0p5_net_gain",
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print()
    print("[DONE]")
    print("[CSV SAVED]", args.out_csv)


if __name__ == "__main__":
    main()
