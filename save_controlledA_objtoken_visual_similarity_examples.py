import os
import re
import csv
import json
import shutil
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from matplotlib import cm

from model_zoo import get_model
from dataset_zoo import get_dataset
from model_zoo.llava15 import change_greedy_to_add_weight
import save_llava_hidden_similarity_features as sf

try:
    from misc import _default_collate
except Exception:
    _default_collate = None


def iter_samples(loader):
    sid = 0
    for batch in loader:
        for i_option in batch["image_options"]:
            for image in i_option:
                yield sid, image
                sid += 1


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
    return [torch.where(input_id == int(image_token_index), 1, 0) for input_id in single_input["input_ids"]]


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
    prob = F.softmax(scores[0], dim=-1)
    return float(torch.max(prob[0]).detach().float().cpu())


def decode_generated(processor, output, prompt_len):
    seq = generation_sequences(output)
    return processor.decode(seq[0][int(prompt_len):], skip_special_tokens=True)


def clean_obj(x):
    x = str(x).strip().strip(".").strip("?")
    x = re.sub(r"^(a|an|the)\s+", "", x, flags=re.I)
    return x.strip()


def parse_objects_from_prompt(prompt):
    s = str(prompt)
    s = s.replace("<image>", " ")
    s = re.sub(r"USER:\s*", " ", s, flags=re.I)
    s = re.sub(r"ASSISTANT:\s*", " ", s, flags=re.I)
    patterns = [
        r"Where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+in relation to\s+(?:the\s+)?(.+?)\?",
        r"Where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+relative to\s+(?:the\s+)?(.+?)\?",
        r"Where\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+with respect to\s+(?:the\s+)?(.+?)\?",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            return clean_obj(m.group(1)), clean_obj(m.group(2))
    return "", ""


def find_subsequence(haystack, needle, banned_positions=None):
    if banned_positions is None:
        banned_positions = set()
    n = len(needle)
    if n == 0:
        return []
    hits = []
    for i in range(0, len(haystack) - n + 1):
        if any((i + j) in banned_positions for j in range(n)):
            continue
        if haystack[i:i+n] == needle:
            hits.append(list(range(i, i+n)))
    return hits


def encode_phrase_variants(tokenizer, phrase):
    phrase = str(phrase).strip()
    variants = []
    for v in [
        phrase, " " + phrase,
        phrase.lower(), " " + phrase.lower(),
        phrase.title(), " " + phrase.title(),
        phrase.upper(), " " + phrase.upper(),
    ]:
        if v not in variants:
            variants.append(v)
    encoded = []
    for v in variants:
        ids = tokenizer(v, add_special_tokens=False).input_ids
        if ids and ids not in encoded:
            encoded.append(ids)
    return encoded


def get_image_token_index(model):
    image_token_index = getattr(getattr(model, "config", None), "image_token_index", None)
    if image_token_index is None:
        image_token_index = 32001
    return int(image_token_index)


def find_object_token_span(tokenizer, input_ids, obj, image_token_index):
    ids = [int(x) for x in input_ids]
    banned = {i for i, x in enumerate(ids) if x == int(image_token_index)}
    candidates = encode_phrase_variants(tokenizer, obj)
    for cand in candidates:
        hits = find_subsequence(ids, cand, banned_positions=banned)
        if hits:
            return hits[0], cand
    return [], []


def pixel_values_to_pil(single_input, processor):
    pv = single_input.get("pixel_values", None)
    if pv is None:
        return None
    x = pv[0].detach().float().cpu()
    image_processor = getattr(processor, "image_processor", None)
    mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    x = x * std + mean
    x = x.clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_overlay(base_img, arr24, out_path, alpha=0.45):
    base = base_img.convert("RGB").resize((336, 336))
    a = np.asarray(arr24, dtype=np.float32)
    a = a - np.nanmin(a)
    if np.nanmax(a) > 0:
        a = a / np.nanmax(a)
    heat_small = Image.fromarray((a * 255).astype(np.uint8)).resize(base.size, Image.BICUBIC)
    heat = np.asarray(heat_small).astype(np.float32) / 255.0
    color = (cm.inferno(heat)[:, :, :3] * 255).astype(np.uint8)
    base_arr = np.asarray(base).astype(np.float32)
    out = (1 - alpha) * base_arr + alpha * color
    out = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^a-zA-Z0-9_\-]+", "_", x)
    return x.strip("_")[:80]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_patch_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
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
    os.environ["ADAPTVIS_NUM_LAYERS"] = str(int(args.num_layers))
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
    os.environ["ADAPTVIS_WEIGHT"] = str(float(selected_w))


def get_llava_visual_projected_features(model, pixel_values):
    vision_tower = getattr(model, "vision_tower", None)
    if isinstance(vision_tower, (list, tuple)):
        vision_tower = vision_tower[0]
    if vision_tower is None:
        raise AttributeError("model has no vision_tower")

    projector = getattr(model, "multi_modal_projector", None)
    if projector is None:
        projector = getattr(model, "mm_projector", None)
    if projector is None:
        raise AttributeError("model has no multi_modal_projector/mm_projector")

    vision_feature_layer = getattr(getattr(model, "config", None), "vision_feature_layer", -2)
    vision_feature_select_strategy = getattr(getattr(model, "config", None), "vision_feature_select_strategy", "default")

    vt_out = vision_tower(pixel_values, output_hidden_states=True)
    hs = vt_out.hidden_states[vision_feature_layer]
    if vision_feature_select_strategy == "default":
        hs = hs[:, 1:]
    projected = projector(hs)
    return projected


def get_object_embedding(model, tokenizer, input_ids_tensor, obj):
    input_ids = [int(x) for x in input_ids_tensor[0].detach().cpu().tolist()]
    image_token_index = get_image_token_index(model)
    span, matched_ids = find_object_token_span(tokenizer, input_ids, obj, image_token_index)
    if not span:
        return None, [], []
    emb_layer = model.get_input_embeddings()
    obj_ids = torch.tensor([input_ids[p] for p in span], device=input_ids_tensor.device, dtype=torch.long)
    obj_emb = emb_layer(obj_ids).mean(dim=0)
    token_texts = [tokenizer.decode([input_ids[p]], skip_special_tokens=False) for p in span]
    return obj_emb, span, token_texts


def compute_similarity_maps(wrapper, image, prompt, obj1, obj2, max_length=77):
    single_input = wrapper.processor(
        text=prompt,
        images=image,
        padding="max_length",
        return_tensors="pt",
        max_length=max_length,
    ).to(next(wrapper.model.parameters()).device)

    proc_img = pixel_values_to_pil(single_input, wrapper.processor)
    if proc_img is None:
        proc_img = image if isinstance(image, Image.Image) else Image.new("RGB", (336, 336), (0, 0, 0))

    pv = single_input.get("pixel_values", None)
    if pv is None:
        raise RuntimeError("pixel_values missing")

    with torch.no_grad():
        vis = get_llava_visual_projected_features(wrapper.model, pv)[0]   # [576, d]
        vis = F.normalize(vis, dim=-1)

        maps = {}
        meta = {}
        for obj_label, obj in [("obj1", obj1), ("obj2", obj2)]:
            obj_emb, span, token_texts = get_object_embedding(wrapper.model, wrapper.processor.tokenizer, single_input["input_ids"], obj)
            if obj_emb is None:
                continue
            obj_emb = F.normalize(obj_emb, dim=-1)
            sim = torch.matmul(vis, obj_emb)  # [576]
            arr = sim.detach().float().cpu().numpy().reshape(24, 24)
            maps[obj_label] = arr
            meta[obj_label] = {"obj": obj, "orig_span": span, "token_texts": token_texts}

    return {"proc_img": proc_img, "maps": maps, "meta": meta}


def run_generation(wrapper, image, prompt, gold, args, selected_w, patch_ids):
    set_common_env(args)
    set_patch_env_for_sample(patch_ids, args.missing_mask_mode)
    set_weight_env_for_sample(selected_w)

    single_input = wrapper.processor(
        text=prompt,
        images=image,
        padding="max_length",
        return_tensors="pt",
        max_length=args.max_length,
    ).to(args.device)

    keys = make_keys_from_input_ids(single_input, wrapper.model)
    prompt_len = len(single_input["input_ids"][-1])

    with torch.no_grad():
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
    return {"generation": gen, "correct": bool(corr), "confidence": float(conf)}


def choose_examples(records, n_each=5):
    groups = {"correct_to_wrong": [], "wrong_to_correct": [], "correct_to_correct": []}
    for r in records:
        if r["base_correct"] and (not r["adapt_correct"]):
            groups["correct_to_wrong"].append(r)
        elif (not r["base_correct"]) and r["adapt_correct"]:
            groups["wrong_to_correct"].append(r)
        elif r["base_correct"] and r["adapt_correct"]:
            groups["correct_to_correct"].append(r)
    selected = []
    for g in ["correct_to_wrong", "wrong_to_correct", "correct_to_correct"]:
        selected.extend(groups[g][:int(n_each)])
    return selected, groups


def copy_examples(selected_records, out_dir, example_dir):
    ensure_dir(example_dir)
    for r in selected_records:
        sid = int(r["sample_id"])
        src = os.path.join(out_dir, f"idx{sid:04d}")
        dst = os.path.join(example_dir, f"{r['transition']}_idx{sid:04d}")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Controlled_Images_A")
    p.add_argument("--option", default="four")
    p.add_argument("--model-name", default="llava1.5")
    p.add_argument("--method", default="adapt_vis")
    p.add_argument("--root-dir", default="data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--variant", default="negonly_mean_img")
    p.add_argument("--layers", default="0,1,2,3,4")
    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument("--object-patch-json", default="output/groundingdino_object_patch_masks_by_sid.json")
    p.add_argument("--missing-mask-mode", default="none", choices=["none", "all"])
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--weight1", type=float, default=0.5)
    p.add_argument("--weight2", type=float, default=1.5)
    p.add_argument("--max-length", type=int, default=77)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--fresh-limit", type=int, default=-1)
    p.add_argument("--idxs", default="")
    p.add_argument("--save-model-input", action="store_true")
    p.add_argument("--out-dir", default="output/controlledA_objtoken_visual_similarity")
    p.add_argument("--example-dir", default="output/controlledA_objtoken_visual_similarity_examples")
    p.add_argument("--num-examples-per-group", type=int, default=5)
    p.add_argument("--summary-csv", default="output/controlledA_objtoken_visual_similarity_summary.csv")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.summary_csv) or ".")

    idx_set = None
    if str(args.idxs).strip():
        idx_set = {int(x) for x in str(args.idxs).split(",") if x.strip()}

    mask_data = load_patch_json(args.object_patch_json)
    change_greedy_to_add_weight()

    print("[LOAD MODEL]", args.model_name, args.method, args.device)
    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    wrapper.model.eval()

    print("[LOAD DATASET]", args.dataset)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=False)
    collate_fn = _default_collate if image_preprocess is None else None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    prompts, answers = sf.load_prompts(args.dataset, args.option)

    records = []
    base_correct = 0
    adapt_correct = 0
    w05 = 0
    w15 = 0

    pbar = tqdm(iter_samples(loader), desc="objtok-sim-base-vs-adapt")
    for sid, image in pbar:
        if sid >= len(prompts):
            break
        if idx_set is not None and sid not in idx_set:
            continue
        if idx_set is None and args.fresh_limit > 0 and len(records) >= args.fresh_limit:
            break

        prompt = prompts[sid]
        gold = norm_gold(answers[sid])
        obj1, obj2 = parse_objects_from_prompt(prompt)
        if not obj1 or not obj2:
            print("[WARN] parse obj failed sid", sid)
            continue

        # Similarity maps: once only, shared by base/adaptvis.
        try:
            sim_res = compute_similarity_maps(wrapper, image, prompt, obj1, obj2, max_length=args.max_length)
        except Exception as e:
            print(f"[WARN] similarity extraction failed sid={sid}: {e}")
            continue

        base_res = run_generation(wrapper, image, prompt, gold, args, selected_w=1.0, patch_ids=[])
        selected_w = args.weight1 if float(base_res["confidence"]) < float(args.threshold) else args.weight2
        patch_ids = get_patch_ids(mask_data, sid)
        adapt_res = run_generation(wrapper, image, prompt, gold, args, selected_w=selected_w, patch_ids=patch_ids)

        base_correct += int(base_res["correct"])
        adapt_correct += int(adapt_res["correct"])
        if abs(float(selected_w) - 0.5) <= 1e-6:
            w05 += 1
        if abs(float(selected_w) - 1.5) <= 1e-6:
            w15 += 1

        sample_dir = os.path.join(args.out_dir, f"idx{sid:04d}")
        ensure_dir(sample_dir)
        if args.save_model_input:
            sim_res["proc_img"].save(os.path.join(sample_dir, "model_input_image.png"))
        for obj_label, obj_name in [("obj1", obj1), ("obj2", obj2)]:
            if obj_label not in sim_res["maps"]:
                continue
            save_overlay(sim_res["proc_img"], sim_res["maps"][obj_label], os.path.join(sample_dir, f"sim_{obj_label}_{safe_name(obj_name)}_overlay.png"))

        transition = (
            "correct_to_wrong" if base_res["correct"] and (not adapt_res["correct"]) else
            "wrong_to_correct" if (not base_res["correct"]) and adapt_res["correct"] else
            "correct_to_correct" if base_res["correct"] and adapt_res["correct"] else
            "wrong_to_wrong"
        )

        records.append({
            "sample_id": int(sid),
            "obj1": obj1,
            "obj2": obj2,
            "gold": gold,
            "base_generation": base_res["generation"],
            "base_correct": bool(base_res["correct"]),
            "base_confidence": float(base_res["confidence"]),
            "adapt_generation": adapt_res["generation"],
            "adapt_correct": bool(adapt_res["correct"]),
            "adapt_confidence": float(adapt_res["confidence"]),
            "selected_weight": float(selected_w),
            "threshold": float(args.threshold),
            "num_patch_ids": int(len(patch_ids)),
            "transition": transition,
        })

        pbar.set_postfix({
            "sid": sid,
            "base_acc": f"{base_correct / max(len(records), 1):.3f}",
            "adapt_acc": f"{adapt_correct / max(len(records), 1):.3f}",
            "w0.5": w05,
            "w1.5": w15,
        })

        if sid < 3:
            print(
                f"[DEBUG] sid={sid} gold={gold} base={base_res['generation']} ({base_res['correct']}, conf={base_res['confidence']:.4f}) "
                f"adapt={adapt_res['generation']} ({adapt_res['correct']}, conf={adapt_res['confidence']:.4f}) "
                f"w={selected_w} npatch={len(patch_ids)}"
            )

    fieldnames = [
        "sample_id", "obj1", "obj2", "gold",
        "base_generation", "base_correct", "base_confidence",
        "adapt_generation", "adapt_correct", "adapt_confidence",
        "selected_weight", "threshold", "num_patch_ids", "transition",
    ]
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)

    selected_records, groups = choose_examples(records, n_each=args.num_examples_per_group)
    copy_examples(selected_records, args.out_dir, args.example_dir)

    print("\n[DONE]")
    print("[ALL OUT DIR]", args.out_dir)
    print("[EXAMPLE DIR]", args.example_dir)
    print("[SUMMARY CSV]", args.summary_csv)
    print("num_total:", len(records))
    print("base_acc:", base_correct / max(len(records), 1))
    print("adapt_acc:", adapt_correct / max(len(records), 1))
    print("correct_to_wrong:", len(groups["correct_to_wrong"]))
    print("wrong_to_correct:", len(groups["wrong_to_correct"]))
    print("correct_to_correct:", len(groups["correct_to_correct"]))
    print("selected_examples:", len(selected_records))


if __name__ == "__main__":
    main()
