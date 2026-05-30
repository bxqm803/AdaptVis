import os
import re
import json
import glob
import random
import shutil
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def norm_gold(x):
    if isinstance(x, list):
        return str(x[0]).strip() if x else ""
    return str(x).strip()


def get_object_key(data):
    ks = [k for k in data.keys() if k != "base"]
    if len(ks) == 1:
        return ks[0]
    for k in ks:
        if "negonly_mean" in k:
            return k
    for k in ks:
        if k.startswith("object_box"):
            return k
    raise ValueError(f"Cannot infer object intervention key from {list(data.keys())}")


def iter_samples(loader):
    sid = 0
    for batch in loader:
        for i_option in batch["image_options"]:
            for image in i_option:
                yield sid, image
                sid += 1


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


def select_sids(records_json, n_per_group=5, random_pick=False, seed=0):
    data = load_json(records_json)
    obj_key = get_object_key(data)

    base = {int(r["sample_id"]): r for r in data["base"]}
    obj = {int(r["sample_id"]): r for r in data[obj_key]}

    wrong_to_correct = []
    correct_to_wrong = []

    for sid in sorted(base.keys()):
        b = bool(base[sid]["correct"])
        o = bool(obj[sid]["correct"])
        if (not b) and o:
            wrong_to_correct.append(sid)
        elif b and (not o):
            correct_to_wrong.append(sid)

    if random_pick:
        rng = random.Random(seed)
        rng.shuffle(wrong_to_correct)
        rng.shuffle(correct_to_wrong)

    return {
        "object_key": obj_key,
        "wrong_to_correct": wrong_to_correct[:n_per_group],
        "correct_to_wrong": correct_to_wrong[:n_per_group],
        "all_data": data,
    }


def set_common_env(layers, variant):
    os.environ["ADAPTVIS_ATTENTION_VARIANT"] = variant
    os.environ["ADAPTVIS_LAYER_MODE"] = "list"
    os.environ["ADAPTVIS_LAYERS"] = layers
    os.environ["ADAPTVIS_NUM_LAYERS"] = "32"
    os.environ["ADAPTVIS_PATCH_BLOCK_MODE"] = "all"


def set_patch_env_for_sample(patch_ids):
    os.environ["ADAPTVIS_PATCH_ID_MODE"] = "only"
    os.environ["ADAPTVIS_PATCH_IDS"] = ",".join(str(int(x)) for x in sorted(set(patch_ids)))


def set_weight_env_for_sample(selected_w):
    os.environ["ADAPTVIS_WEIGHT"] = str(float(selected_w))


def clear_debug_env():
    for k in [
        "SAVE_IMG_LOGITS_PATH",
        "SAVE_IMG_LOGITS_LAYERS",
        "SAVE_IMG_LOGITS_TAG",
        "SAVE_ATTN_PATH",
        "SAVE_ATTN_LAYERS",
    ]:
        if k in os.environ:
            del os.environ[k]


def tensor_to_pil(pixel_values, processor):
    arr = pixel_values.detach().float().cpu().numpy()
    mean = getattr(getattr(processor, "image_processor", processor), "image_mean", [0.48145466, 0.4578275, 0.40821073])
    std = getattr(getattr(processor, "image_processor", processor), "image_std", [0.26862954, 0.26130258, 0.27577711])
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    arr = arr * std + mean
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def patch_id_to_rc(pid, patch_side):
    r = int(pid) // int(patch_side)
    c = int(pid) % int(patch_side)
    return r, c


def draw_patch_overlay_on_square(img, object_patch_ids, negative_patch_ids, patch_side, alpha=100):
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    cell_w = w / patch_side
    cell_h = h / patch_side

    for pid in object_patch_ids:
        r, c = patch_id_to_rc(pid, patch_side)
        x0, y0 = c * cell_w, r * cell_h
        x1, y1 = (c + 1) * cell_w, (r + 1) * cell_h
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0, 220), width=2)

    for pid in negative_patch_ids:
        r, c = patch_id_to_rc(pid, patch_side)
        x0, y0 = c * cell_w, r * cell_h
        x1, y1 = (c + 1) * cell_w, (r + 1) * cell_h
        draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, alpha), outline=(255, 0, 0, 220), width=2)

    return Image.alpha_composite(img, overlay).convert("RGB")


def load_layer_npz(logit_dir, sid, layer):
    patt = os.path.join(logit_dir, f"sid{sid:04d}_layer{layer:02d}_img_logits.npz")
    files = glob.glob(patt)
    if len(files) == 0:
        existing = sorted(os.listdir(logit_dir)) if os.path.exists(logit_dir) else []
        raise FileNotFoundError(
            f"Cannot find {patt}\nlogit_dir={logit_dir}\nexisting_files={existing[:20]}"
        )
    return np.load(files[0])


def _parse_attn_fname(path):
    base = os.path.basename(path)
    m = re.search(r"_(\d+)_start(-?\d+)_end(-?\d+)\.npy$", base)
    if not m:
        return None
    layer = int(m.group(1))
    start = int(m.group(2))
    end = int(m.group(3))
    return layer, start, end


def _pick_best_attn_file(files):
    if not files:
        return None
    parsed = []
    for f in files:
        info = _parse_attn_fname(f)
        if info is not None:
            _, start, end = info
            parsed.append((f, start, end))
    if not parsed:
        return sorted(files)[0]
    valid = [x for x in parsed if x[1] >= 0 and x[2] >= x[1]]
    if valid:
        valid = sorted(valid, key=lambda x: (x[1], x[2], x[0]))
        return valid[0][0]
    parsed = sorted(parsed, key=lambda x: (x[1], x[2], x[0]))
    return parsed[0][0]


def load_layer_attn(attn_dir, layer):
    ori_files = glob.glob(os.path.join(attn_dir, f"ori_postsoftmax_{layer}_start*_end*.npy"))
    fin_files = glob.glob(os.path.join(attn_dir, f"final_postsoftmax_{layer}_start*_end*.npy"))
    ori_file = _pick_best_attn_file(ori_files)
    fin_file = _pick_best_attn_file(fin_files)
    if ori_file is None or fin_file is None:
        existing = sorted(os.listdir(attn_dir)) if os.path.exists(attn_dir) else []
        raise FileNotFoundError(
            f"Cannot find attn files for layer={layer} in {attn_dir}\nexisting_files={existing[:20]}"
        )
    ori = np.load(ori_file)
    fin = np.load(fin_file)
    return ori, fin, ori_file, fin_file


def grid_from_vector(v, patch_side):
    return np.array(v).reshape(patch_side, patch_side)


def normalize_map(x):
    x = np.array(x, dtype=np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def make_heat_overlay(base_img, grid_map, cmap="jet", alpha=0.45):
    base = np.array(base_img.convert("RGB"))
    H, W = base.shape[:2]
    norm_map = normalize_map(grid_map)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(base)
    ax.imshow(norm_map, cmap=cmap, alpha=alpha, interpolation="bilinear", extent=[0, W, H, 0])
    ax.axis("off")
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return Image.fromarray(img)


def save_heatmap_only(grid_map, out_path, title=""):
    plt.figure(figsize=(5, 5))
    plt.imshow(grid_map, cmap="jet")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_panel(out_path, pre_img, pre_overlay, ori_prob_grid, edit_prob_grid, diff_prob_grid, title_lines):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(pre_img)
    axes[0, 0].set_title("Preprocessed model input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pre_overlay)
    axes[0, 1].set_title("Object patches + negative patches")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(make_heat_overlay(pre_img, diff_prob_grid, cmap="bwr", alpha=0.45))
    axes[0, 2].set_title("Attention diff overlay")
    axes[0, 2].axis("off")

    im1 = axes[1, 0].imshow(ori_prob_grid, cmap="jet")
    axes[1, 0].set_title("Original post-softmax attention")
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = axes[1, 1].imshow(edit_prob_grid, cmap="jet")
    axes[1, 1].set_title("Edited post-softmax attention")
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im3 = axes[1, 2].imshow(diff_prob_grid, cmap="bwr")
    axes[1, 2].set_title("Edited - original attention")
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle("\n".join(title_lines), fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path)
    plt.close(fig)


def compute_layer_maps(logit_dir, attn_dir, sid, layer):
    d = load_layer_npz(logit_dir, sid, layer)
    ori_logits = d["ori_img_logits"][0]      # [H, P]
    edited_logits = d["edited_img_logits"][0]  # [H, P]
    image_start = int(d["image_start"])
    image_end = int(d["image_end"])

    ori, fin, ori_file, fin_file = load_layer_attn(attn_dir, layer)  # [1,H,K]
    ori = ori[0]
    fin = fin[0]

    img_ori_prob = ori[:, image_start:image_end]  # [H, P]
    img_fin_prob = fin[:, image_start:image_end]  # [H, P]

    stats = {
        "mean_ori_logits": ori_logits.mean(axis=0),
        "mean_edited_logits": edited_logits.mean(axis=0),
        "neg_frac": (ori_logits < 0).astype(np.float32).mean(axis=0),
        "mean_ori_prob": img_ori_prob.mean(axis=0),
        "mean_edit_prob": img_fin_prob.mean(axis=0),
        "image_start": image_start,
        "image_end_exclusive": image_end,
        "ori_file": os.path.basename(ori_file),
        "fin_file": os.path.basename(fin_file),
    }
    return stats


def compute_mean_layers(layer_stats):
    keys = ["mean_ori_logits", "neg_frac", "mean_ori_prob", "mean_edit_prob"]
    out = {}
    for k in keys:
        stacked = np.stack([x[k] for x in layer_stats], axis=0)
        out[k] = stacked.mean(axis=0)
    out["image_start"] = int(layer_stats[0]["image_start"])
    out["image_end_exclusive"] = int(layer_stats[0]["image_end_exclusive"])
    out["ori_files"] = [x["ori_file"] for x in layer_stats]
    out["fin_files"] = [x["fin_file"] for x in layer_stats]
    return out


def save_one_visual_set(out_dir, pre_img, patch_ids, patch_side, stats, neg_frac_threshold, title_lines, keep_raw_meta=True):
    ensure_dir(out_dir)

    negative_patch_ids = [pid for pid in patch_ids if stats["neg_frac"][pid] >= float(neg_frac_threshold)]
    pre_overlay = draw_patch_overlay_on_square(
        pre_img,
        object_patch_ids=patch_ids,
        negative_patch_ids=negative_patch_ids,
        patch_side=patch_side,
    )

    ori_prob_grid = grid_from_vector(stats["mean_ori_prob"], patch_side)
    edit_prob_grid = grid_from_vector(stats["mean_edit_prob"], patch_side)
    diff_prob_grid = edit_prob_grid - ori_prob_grid

    pre_img.save(os.path.join(out_dir, "preprocessed_input.png"))
    pre_overlay.save(os.path.join(out_dir, "negative_patch_overlay_preprocessed.png"))
    make_heat_overlay(pre_img, ori_prob_grid, cmap="jet", alpha=0.45).save(os.path.join(out_dir, "attention_overlay_original.png"))
    make_heat_overlay(pre_img, edit_prob_grid, cmap="jet", alpha=0.45).save(os.path.join(out_dir, "attention_overlay_edited.png"))
    make_heat_overlay(pre_img, diff_prob_grid, cmap="bwr", alpha=0.45).save(os.path.join(out_dir, "attention_overlay_diff.png"))

    save_heatmap_only(ori_prob_grid, os.path.join(out_dir, "heatmap_original_probs.png"), title="Original post-softmax attention")
    save_heatmap_only(edit_prob_grid, os.path.join(out_dir, "heatmap_edited_probs.png"), title="Edited post-softmax attention")
    save_heatmap_only(diff_prob_grid, os.path.join(out_dir, "heatmap_diff_probs.png"), title="Edited - original attention")
    save_panel(os.path.join(out_dir, "panel.png"), pre_img, pre_overlay, ori_prob_grid, edit_prob_grid, diff_prob_grid, title_lines)

    meta = {
        "negative_patch_ids": negative_patch_ids,
        "neg_frac_threshold": float(neg_frac_threshold),
        "num_negative_patches": len(negative_patch_ids),
        "image_start": int(stats["image_start"]),
        "image_end_exclusive": int(stats["image_end_exclusive"]),
    }
    if keep_raw_meta:
        meta["negative_patch_fraction"] = {str(pid): float(stats["neg_frac"][pid]) for pid in patch_ids}
        meta["mean_ori_logits"] = {str(pid): float(stats["mean_ori_logits"][pid]) for pid in patch_ids}
    save_json(meta, os.path.join(out_dir, "meta.json"))


def run_one_sample(sid, group_name, args, wrapper, dataset_loader, prompts, answers, patch_meta, records_data):
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    obj_key = records_data["object_key"]
    all_data = records_data["all_data"]

    base_records = {int(r["sample_id"]): r for r in all_data["base"]}
    obj_records = {int(r["sample_id"]): r for r in all_data[obj_key]}

    base_rec = base_records[sid]
    obj_rec = obj_records[sid]

    meta = patch_meta[str(sid)]
    patch_ids = [int(x) for x in meta["patch_ids"]]
    patch_side = int(meta["patch_side"])

    sample_dir = os.path.join(args.out_dir, group_name, f"sid_{sid:04d}")
    ensure_dir(sample_dir)
    logit_dir = os.path.join(sample_dir, "img_logits")
    attn_dir = os.path.join(sample_dir, "attn")
    ensure_dir(logit_dir)
    ensure_dir(attn_dir)

    image = None
    for cur_sid, cur_img in iter_samples(dataset_loader):
        if cur_sid == sid:
            image = cur_img
            break
    if image is None:
        raise ValueError(f"Could not find image for sid={sid}")

    prompt = prompts[sid]
    gold = norm_gold(answers[sid])

    single_input = wrapper.processor(
        text=prompt,
        images=image,
        padding="max_length",
        return_tensors="pt",
        max_length=args.max_length,
    ).to(args.device)

    pre_img = tensor_to_pil(single_input["pixel_values"][0], wrapper.processor)
    keys = make_keys_from_input_ids(single_input, wrapper.model)
    prompt_len = len(single_input["input_ids"][-1])

    set_common_env(args.layers, args.variant)
    set_patch_env_for_sample(patch_ids)
    set_weight_env_for_sample(args.weight)

    os.environ["SAVE_IMG_LOGITS"] = "1"
    os.environ["SAVE_IMG_LOGITS_PATH"] = logit_dir
    os.environ["SAVE_IMG_LOGITS_LAYERS"] = args.layers
    os.environ["SAVE_IMG_LOGITS_TAG"] = f"sid{sid:04d}"

    os.environ["SAVE_ATTN"] = "1"
    os.environ["SAVE_ATTN_PATH"] = attn_dir
    os.environ["SAVE_ATTN_LAYERS"] = args.layers
    os.environ["SAVE_ORI"] = "1"

    with torch.no_grad():
        output = wrapper.model.generate(
            **single_input,
            keys=keys,
            weight=float(args.weight),
            max_new_tokens=args.max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

    rerun_gen = decode_generated(wrapper.processor, output, prompt_len)
    rerun_conf = first_step_confidence(output)

    layer_stats = []
    for layer in layers:
        layer_stat = compute_layer_maps(logit_dir, attn_dir, sid, layer)
        layer_stats.append(layer_stat)

        layer_dir = os.path.join(sample_dir, f"layer_{layer:02d}")
        title_lines = [
            f"{group_name} | sid={sid} | layer={layer}",
            f"gold={gold}",
            f"base_correct={base_rec['correct']} | edited_correct={obj_rec['correct']}",
            f"base_gen={base_rec['generation']}",
            f"edited_gen={obj_rec['generation']}",
        ]
        save_one_visual_set(
            out_dir=layer_dir,
            pre_img=pre_img,
            patch_ids=patch_ids,
            patch_side=patch_side,
            stats=layer_stat,
            neg_frac_threshold=args.neg_frac_threshold,
            title_lines=title_lines,
            keep_raw_meta=True,
        )

    if args.save_mean_layers:
        mean_stats = compute_mean_layers(layer_stats)
        mean_dir = os.path.join(sample_dir, "mean_layers")
        title_lines = [
            f"{group_name} | sid={sid} | mean layers={args.layers}",
            f"gold={gold}",
            f"base_correct={base_rec['correct']} | edited_correct={obj_rec['correct']}",
            f"base_gen={base_rec['generation']}",
            f"edited_gen={obj_rec['generation']}",
        ]
        save_one_visual_set(
            out_dir=mean_dir,
            pre_img=pre_img,
            patch_ids=patch_ids,
            patch_side=patch_side,
            stats=mean_stats,
            neg_frac_threshold=args.neg_frac_threshold,
            title_lines=title_lines,
            keep_raw_meta=True,
        )
        shutil.copy2(os.path.join(mean_dir, "panel.png"), os.path.join(sample_dir, "panel_mean_layers.png"))

    sample_meta = {
        "sample_id": sid,
        "group": group_name,
        "prompt": prompt,
        "gold": gold,
        "base_correct": bool(base_rec["correct"]),
        "edited_correct": bool(obj_rec["correct"]),
        "base_generation": base_rec["generation"],
        "edited_generation": obj_rec["generation"],
        "base_confidence_from_records": float(base_rec.get("confidence", 0.0)),
        "edited_selected_weight_from_records": float(obj_rec.get("selected_weight", args.weight)),
        "rerun_first_step_confidence": float(rerun_conf),
        "rerun_generation": rerun_gen,
        "variant": args.variant,
        "layers": layers,
        "num_object_patch_ids": len(patch_ids),
        "object_patch_ids": patch_ids,
        "patch_side": patch_side,
        "image_path": meta.get("image_path", ""),
        "neg_frac_threshold": float(args.neg_frac_threshold),
    }
    save_json(sample_meta, os.path.join(sample_dir, "meta.json"))

    if not args.keep_raw:
        if os.path.exists(logit_dir):
            shutil.rmtree(logit_dir)
        if os.path.exists(attn_dir):
            shutil.rmtree(attn_dir)

    clear_debug_env()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-json", default="output/objectbox_negmean_all_l0_4_records.json")
    parser.add_argument("--patch-json", default="output/groundingdino_object_patch_masks_by_sid.json")
    parser.add_argument("--dataset", default="Controlled_Images_A")
    parser.add_argument("--option", default="four")
    parser.add_argument("--model-name", default="llava1.5")
    parser.add_argument("--method", default="adapt_vis")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", default="negonly_mean_img")
    parser.add_argument("--layers", default="0,1,2,3,4")
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--num-per-group", type=int, default=5)
    parser.add_argument("--random-pick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="output/negmean_patch_visuals")
    parser.add_argument("--neg-frac-threshold", type=float, default=0.5)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--save-mean-layers", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    selection = select_sids(
        args.records_json,
        n_per_group=args.num_per_group,
        random_pick=args.random_pick,
        seed=args.seed,
    )
    patch_meta = load_json(args.patch_json)

    save_json(
        {
            "wrong_to_correct": selection["wrong_to_correct"],
            "correct_to_wrong": selection["correct_to_wrong"],
            "object_key": selection["object_key"],
        },
        os.path.join(args.out_dir, "selected_sids.json"),
    )

    os.environ["SAVE_IMG_LOGITS"] = "1"
    os.environ["SAVE_ATTN"] = "1"
    os.environ["SAVE_ORI"] = "1"

    change_greedy_to_add_weight()

    print("[LOAD MODEL]", args.model_name, args.method, args.device)
    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    wrapper.model.eval()

    print("[LOAD DATASET]", args.dataset)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=False)
    collate_fn = _default_collate if image_preprocess is None else None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    prompts, answers = sf.load_prompts(args.dataset, args.option)

    plan = []
    for sid in selection["wrong_to_correct"]:
        plan.append((sid, "wrong_to_correct"))
    for sid in selection["correct_to_wrong"]:
        plan.append((sid, "correct_to_wrong"))

    print("[SELECTED]", plan)

    for sid, group_name in tqdm(plan, desc="visualizing"):
        run_one_sample(
            sid=sid,
            group_name=group_name,
            args=args,
            wrapper=wrapper,
            dataset_loader=loader,
            prompts=prompts,
            answers=answers,
            patch_meta=patch_meta,
            records_data=selection,
        )

    print("[DONE]")
    print("[OUT DIR]", args.out_dir)


if __name__ == "__main__":
    main()
