import os
import json
import glob
import math
import random
import argparse
from pathlib import Path

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


# -----------------------------
# basic helpers
# -----------------------------
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


# -----------------------------
# selection
# -----------------------------
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


# -----------------------------
# env setup
# -----------------------------
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
        "SAVE_IMG_LOGITS",
        "SAVE_IMG_LOGITS_PATH",
        "SAVE_IMG_LOGITS_LAYERS",
        "SAVE_IMG_LOGITS_TAG",
        "SAVE_ATTN",
        "SAVE_ATTN_PATH",
        "SAVE_ORI",
    ]:
        if k in os.environ:
            del os.environ[k]


# -----------------------------
# image helpers
# -----------------------------
def tensor_to_pil(pixel_values, processor):
    """
    pixel_values: [3, H, W] normalized tensor
    """
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
    """
    img: PIL, should be the model input image (preprocessed square image)
    """
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    cell_w = w / patch_side
    cell_h = h / patch_side

    # object patch outlines
    for pid in object_patch_ids:
        r, c = patch_id_to_rc(pid, patch_side)
        x0, y0 = c * cell_w, r * cell_h
        x1, y1 = (c + 1) * cell_w, (r + 1) * cell_h
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0, 220), width=2)

    # negative patches filled red
    for pid in negative_patch_ids:
        r, c = patch_id_to_rc(pid, patch_side)
        x0, y0 = c * cell_w, r * cell_h
        x1, y1 = (c + 1) * cell_w, (r + 1) * cell_h
        draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, alpha), outline=(255, 0, 0, 220), width=2)

    return Image.alpha_composite(img, overlay).convert("RGB")


def get_clipstyle_resize_crop(orig_w, orig_h, target_w, target_h):
    """
    Approximate CLIP-style preprocess:
    resize shortest side -> target, then center crop target_w x target_h
    """
    scale = float(min(target_w, target_h)) / float(min(orig_w, orig_h))
    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))
    crop_left = max(0, (resized_w - target_w) / 2.0)
    crop_top = max(0, (resized_h - target_h) / 2.0)
    return scale, resized_w, resized_h, crop_left, crop_top


def patch_boxes_on_original_image(orig_img, object_patch_ids, negative_patch_ids, patch_side, target_w, target_h, alpha=80):
    """
    Map square crop patch grid back to original image approximately.
    """
    orig = orig_img.convert("RGBA")
    overlay = Image.new("RGBA", orig.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    orig_w, orig_h = orig.size
    scale, resized_w, resized_h, crop_left, crop_top = get_clipstyle_resize_crop(orig_w, orig_h, target_w, target_h)

    cell_w = target_w / patch_side
    cell_h = target_h / patch_side

    def crop_patch_to_orig_box(pid):
        r, c = patch_id_to_rc(pid, patch_side)
        x0c, y0c = c * cell_w, r * cell_h
        x1c, y1c = (c + 1) * cell_w, (r + 1) * cell_h

        # map from crop coords -> resized coords -> original coords
        x0r = x0c + crop_left
        x1r = x1c + crop_left
        y0r = y0c + crop_top
        y1r = y1c + crop_top

        x0o = x0r / scale
        x1o = x1r / scale
        y0o = y0r / scale
        y1o = y1r / scale

        x0o = max(0, min(orig_w, x0o))
        x1o = max(0, min(orig_w, x1o))
        y0o = max(0, min(orig_h, y0o))
        y1o = max(0, min(orig_h, y1o))
        return [x0o, y0o, x1o, y1o]

    for pid in object_patch_ids:
        draw.rectangle(crop_patch_to_orig_box(pid), outline=(255, 255, 0, 220), width=2)

    for pid in negative_patch_ids:
        draw.rectangle(crop_patch_to_orig_box(pid), fill=(255, 0, 0, alpha), outline=(255, 0, 0, 220), width=2)

    return Image.alpha_composite(orig, overlay).convert("RGB")


# -----------------------------
# npy/npz loading
# -----------------------------
def load_layer_npz(logit_dir, sid, layer):
    patt = os.path.join(logit_dir, f"sid{sid:04d}_layer{layer:02d}_img_logits.npz")
    files = glob.glob(patt)
    if len(files) == 0:
        raise FileNotFoundError(f"Cannot find {patt}")
    return np.load(files[0])


def load_layer_attn(attn_dir, layer):
    ori_files = glob.glob(os.path.join(attn_dir, f"ori_postsoftmax_{layer}_start*_end*.npy"))
    fin_files = glob.glob(os.path.join(attn_dir, f"final_postsoftmax_{layer}_start*_end*.npy"))
    if len(ori_files) == 0 or len(fin_files) == 0:
        raise FileNotFoundError(f"Cannot find attn files for layer={layer} in {attn_dir}")
    ori = np.load(ori_files[0])   # [1, H, kv_len]
    fin = np.load(fin_files[0])   # [1, H, kv_len]
    return ori, fin


def aggregate_maps(logit_dir, attn_dir, sid, layers):
    """
    Returns:
      mean_ori_logits: [P]
      neg_frac: [P]
      mean_ori_prob: [P]
      mean_edit_prob: [P]
      image_start, image_end_exclusive
    """
    ori_logits_list = []
    negmask_list = []
    ori_prob_list = []
    edit_prob_list = []
    image_start = None
    image_end = None

    for layer in layers:
        d = load_layer_npz(logit_dir, sid, layer)

        ori_logits = d["ori_img_logits"][0]       # [H, P]
        edited_logits = d["edited_img_logits"][0] # [H, P], here not directly used
        image_start = int(d["image_start"])
        image_end = int(d["image_end"])  # exclusive in npz

        ori, fin = load_layer_attn(attn_dir, layer)   # [1, H, kv_len]
        ori = ori[0]
        fin = fin[0]

        img_ori_prob = ori[:, image_start:image_end]   # [H, P]
        img_fin_prob = fin[:, image_start:image_end]   # [H, P]

        ori_logits_list.append(ori_logits)
        negmask_list.append((ori_logits < 0).astype(np.float32))
        ori_prob_list.append(img_ori_prob)
        edit_prob_list.append(img_fin_prob)

    ori_logits_stack = np.stack(ori_logits_list, axis=0)   # [L, H, P]
    negmask_stack = np.stack(negmask_list, axis=0)         # [L, H, P]
    ori_prob_stack = np.stack(ori_prob_list, axis=0)       # [L, H, P]
    edit_prob_stack = np.stack(edit_prob_list, axis=0)     # [L, H, P]

    mean_ori_logits = ori_logits_stack.mean(axis=(0, 1))
    neg_frac = negmask_stack.mean(axis=(0, 1))
    mean_ori_prob = ori_prob_stack.mean(axis=(0, 1))
    mean_edit_prob = edit_prob_stack.mean(axis=(0, 1))

    return mean_ori_logits, neg_frac, mean_ori_prob, mean_edit_prob, image_start, image_end


# -----------------------------
# plotting
# -----------------------------
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
    ax.imshow(norm_map, cmap=cmap, alpha=alpha, interpolation="bilinear",
              extent=[0, W, H, 0])
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


def save_panel(sample_dir, pre_img, orig_img_overlay, pre_overlay, ori_prob_grid, edit_prob_grid, diff_prob_grid, title_lines):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(orig_img_overlay)
    axes[0, 0].set_title("Original image + mapped object/negative patches")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pre_img)
    axes[0, 1].set_title("Model input image (preprocessed)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pre_overlay)
    axes[0, 2].set_title("Preprocessed image + negative patch overlay")
    axes[0, 2].axis("off")

    im1 = axes[1, 0].imshow(ori_prob_grid, cmap="jet")
    axes[1, 0].set_title("Original post-softmax attention (image tokens)")
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = axes[1, 1].imshow(edit_prob_grid, cmap="jet")
    axes[1, 1].set_title("Edited post-softmax attention (image tokens)")
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im3 = axes[1, 2].imshow(diff_prob_grid, cmap="bwr")
    axes[1, 2].set_title("Edited - original attention")
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        if ax not in [axes[1, 0], axes[1, 1], axes[1, 2]]:
            ax.axis("off")

    fig.suptitle("\n".join(title_lines), fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(sample_dir, "panel.png"))
    plt.close(fig)


# -----------------------------
# core
# -----------------------------
def run_one_sample(
    sid, group_name, args, wrapper, dataset_loader, prompts, answers, patch_meta, records_data
):
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

    # find image from dataset
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
    target_w, target_h = pre_img.size

    # try original image path first
    orig_path = meta.get("image_path", "")
    if orig_path and os.path.exists(orig_path):
        orig_img = Image.open(orig_path).convert("RGB")
    else:
        # fallback
        if isinstance(image, Image.Image):
            orig_img = image.convert("RGB")
        else:
            orig_img = pre_img.copy()

    keys = make_keys_from_input_ids(single_input, wrapper.model)
    prompt_len = len(single_input["input_ids"][-1])

    # env
    set_common_env(args.layers, args.variant)
    set_patch_env_for_sample(patch_ids)
    set_weight_env_for_sample(args.weight)

    os.environ["SAVE_IMG_LOGITS"] = "1"
    os.environ["SAVE_IMG_LOGITS_PATH"] = logit_dir
    os.environ["SAVE_IMG_LOGITS_LAYERS"] = args.layers
    os.environ["SAVE_IMG_LOGITS_TAG"] = f"sid{sid:04d}"

    os.environ["SAVE_ATTN"] = "1"
    os.environ["SAVE_ATTN_PATH"] = attn_dir
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

    gen = decode_generated(wrapper.processor, output, prompt_len)
    conf = first_step_confidence(output)

    mean_ori_logits, neg_frac, mean_ori_prob, mean_edit_prob, image_start, image_end = aggregate_maps(
        logit_dir, attn_dir, sid, layers
    )

    # define negative patches:
    # among object patch ids, mark negative if mean pre-softmax image logit < 0
    negative_patch_ids = [pid for pid in patch_ids if mean_ori_logits[pid] < 0.0]

    # prepare overlays
    pre_overlay = draw_patch_overlay_on_square(
        pre_img,
        object_patch_ids=patch_ids,
        negative_patch_ids=negative_patch_ids,
        patch_side=patch_side,
    )

    orig_overlay = patch_boxes_on_original_image(
        orig_img,
        object_patch_ids=patch_ids,
        negative_patch_ids=negative_patch_ids,
        patch_side=patch_side,
        target_w=target_w,
        target_h=target_h,
    )

    ori_prob_grid = grid_from_vector(mean_ori_prob, patch_side)
    edit_prob_grid = grid_from_vector(mean_edit_prob, patch_side)
    diff_prob_grid = edit_prob_grid - ori_prob_grid

    # heat overlays
    ori_heat_overlay = make_heat_overlay(pre_img, ori_prob_grid, cmap="jet", alpha=0.45)
    edit_heat_overlay = make_heat_overlay(pre_img, edit_prob_grid, cmap="jet", alpha=0.45)
    diff_heat_overlay = make_heat_overlay(pre_img, diff_prob_grid, cmap="bwr", alpha=0.45)

    pre_img.save(os.path.join(sample_dir, "preprocessed_input.png"))
    orig_img.save(os.path.join(sample_dir, "original_image.png"))
    pre_overlay.save(os.path.join(sample_dir, "negative_patch_overlay_preprocessed.png"))
    orig_overlay.save(os.path.join(sample_dir, "negative_patch_overlay_original.png"))
    ori_heat_overlay.save(os.path.join(sample_dir, "attention_overlay_original.png"))
    edit_heat_overlay.save(os.path.join(sample_dir, "attention_overlay_edited.png"))
    diff_heat_overlay.save(os.path.join(sample_dir, "attention_overlay_diff.png"))

    save_heatmap_only(ori_prob_grid, os.path.join(sample_dir, "heatmap_original_probs.png"), title="Original post-softmax attention")
    save_heatmap_only(edit_prob_grid, os.path.join(sample_dir, "heatmap_edited_probs.png"), title="Edited post-softmax attention")
    save_heatmap_only(diff_prob_grid, os.path.join(sample_dir, "heatmap_diff_probs.png"), title="Edited - original attention")

    title_lines = [
        f"{group_name} | sid={sid}",
        f"gold={gold}",
        f"base_correct={base_rec['correct']} | edited_correct={obj_rec['correct']}",
        f"base_gen={base_rec['generation']}",
        f"edited_gen={obj_rec['generation']}",
    ]
    save_panel(sample_dir, pre_img, orig_overlay, pre_overlay, ori_prob_grid, edit_prob_grid, diff_prob_grid, title_lines)

    save_json(
        {
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
            "rerun_first_step_confidence": float(conf),
            "variant": args.variant,
            "layers": layers,
            "num_object_patch_ids": len(patch_ids),
            "object_patch_ids": patch_ids,
            "negative_patch_ids": negative_patch_ids,
            "negative_patch_fraction_mean_over_layers_heads": {
                str(i): float(neg_frac[i]) for i in negative_patch_ids
            },
            "image_start": int(image_start),
            "image_end_exclusive": int(image_end),
            "patch_side": patch_side,
            "image_path": meta.get("image_path", ""),
        },
        os.path.join(sample_dir, "meta.json"),
    )

    # also save raw numpy for later analysis
    np.save(os.path.join(sample_dir, "mean_ori_logits.npy"), mean_ori_logits)
    np.save(os.path.join(sample_dir, "neg_frac.npy"), neg_frac)
    np.save(os.path.join(sample_dir, "mean_ori_prob.npy"), mean_ori_prob)
    np.save(os.path.join(sample_dir, "mean_edit_prob.npy"), mean_edit_prob)

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
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--num-per-group", type=int, default=5)
    parser.add_argument("--random-pick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="output/negmean_patch_visuals")
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
