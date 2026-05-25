import os
import glob
import json
import math
import csv
import random
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="Controlled_Images_A")
    parser.add_argument("--option", default="four")
    parser.add_argument("--test-tag", default="True")

    parser.add_argument("--mixed-json", default=None)
    parser.add_argument("--base-json", default=None)

    parser.add_argument(
        "--mixed-feature-dir",
        default="output/hidden_features_mixed_w10p5_w21p5_thr0p4",
    )
    parser.add_argument(
        "--base-feature-dir",
        default="output/hidden_features_base_w1",
    )
    parser.add_argument(
        "--out-dir",
        default="output/feature_similarity_analysis_selected0p5_lastquery",
    )

    parser.add_argument("--num-vis", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    # New: choose what this script compares.
    # attention: use post-softmax attention saved by save_llava_hidden_similarity_features.py
    # hidden_similarity: original behavior, cosine(last_prompt_hidden, token_hidden)
    parser.add_argument(
        "--score-type",
        default="attention",
        choices=["attention", "hidden_similarity"],
        help="attention uses saved post-softmax attention; hidden_similarity keeps the old cosine-feature analysis.",
    )
    parser.add_argument(
        "--attn-img-key",
        default="attn_last_to_img_last_layer_mean",
        help="NPZ key for last-token -> image-token post-softmax attention.",
    )
    parser.add_argument(
        "--attn-text-key",
        default="attn_last_to_text_last_layer_mean",
        help="NPZ key for last-token -> text-token post-softmax attention.",
    )
    parser.add_argument(
        "--allow-fallback-hidden",
        action="store_true",
        help="If attention keys are missing, fall back to hidden similarity instead of raising an error.",
    )
    parser.add_argument(
        "--visual-normalize",
        action="store_true",
        help="Only affects display. If set, normalize base/low heatmaps to 0-1. Default keeps true attention values.",
    )
    parser.add_argument(
        "--save-combined",
        action="store_true",
        default=True,
        help="Save one combined figure with visual maps and text curves, similar to the previous debug figure.",
    )
    parser.add_argument(
        "--no-save-combined",
        action="store_false",
        dest="save_combined",
    )

    # Optional integrated feature extraction.
    # When enabled, this script first generates missing post-softmax attention .npz
    # for selected_weight=0.5 samples, then performs the same acc grouping/plotting.
    parser.add_argument(
        "--generate-features",
        action="store_true",
        help="Generate missing base/low attention feature npz files inside this script before analysis.",
    )
    parser.add_argument("--model-name", default="llava1.5")
    parser.add_argument("--method", default="adapt_vis")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--feature-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--compress-features", action="store_true")
    parser.add_argument("--overwrite-generated-features", action="store_true")
    parser.add_argument("--save-all-layers-attn", action="store_true")
    parser.add_argument(
        "--generate-limit",
        type=int,
        default=-1,
        help="Debug only: generate features for at most this many selected sids.",
    )

    return parser.parse_args()


def find_one(name, patterns, exclude=()):
    hits = []
    for p in patterns:
        hits.extend(glob.glob(p))

    hits = [
        h for h in sorted(set(hits))
        if h.endswith(".json")
        and not h.endswith("_scores.json")
        and "summary" not in h
        and "alpha_effect_stats" not in h
        and "attention" not in h
        and all(x not in h for x in exclude)
    ]

    if not hits:
        raise FileNotFoundError(
            f"No file found for {name}:\n" + "\n".join(patterns)
        )

    print(f"[USE {name}] {hits[0]}")
    return hits[0]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm_gold(x):
    if isinstance(x, list):
        return str(x[0]).strip() if x else ""
    return str(x).strip()


def gen_text(item):
    return str(
        item.get(
            "RawGeneration",
            item.get("Generation", item.get("generation", "")),
        )
    )


def correct(item):
    for k in ["RawGenerationCorrect", "Correct", "correct"]:
        if k in item:
            return bool(item[k])

    gold = norm_gold(item.get("Golden", item.get("gold", "")))
    gen = gen_text(item)

    ok = (gold in gen) or (gold.lower() in gen.lower())
    if gold.lower() == "on" and "front" in gen.strip().lower():
        ok = False

    return bool(ok)


def selected_weight(item):
    for k in ["selected_weight", "SelectedWeight", "FinalWeight", "final_weight"]:
        if k in item and item[k] not in [None, ""]:
            try:
                return float(item[k])
            except Exception:
                return None
    return None


def sid_of(i, item):
    return int(item.get("sample_id", item.get("SampleID", item.get("id", i))))


def feature_file(feature_dir, sid, kind):
    if kind == "base":
        patterns = [
            os.path.join(feature_dir, f"sid{sid:04d}_base*.npz"),
            os.path.join(feature_dir, f"sid{sid:04d}_*alpha1*.npz"),
        ]
    elif kind == "low":
        patterns = [
            os.path.join(feature_dir, f"sid{sid:04d}_low*.npz"),
            os.path.join(feature_dir, f"sid{sid:04d}_*alpha0p5*.npz"),
        ]
    else:
        raise ValueError(kind)

    hits = []
    for p in patterns:
        hits.extend(glob.glob(p))

    hits = sorted(set(hits))
    return hits[0] if hits else None



def selected0p5_sids_for_generation(base, mixed, max_items=-1):
    """Return unique sample ids where mixed selected_weight is 0.5."""
    sids = []
    for i in range(min(len(base), len(mixed))):
        sw = selected_weight(mixed[i])
        if sw is None or abs(sw - 0.5) > 1e-6:
            continue
        sids.append(sid_of(i, base[i]))

    sids = sorted(set(int(x) for x in sids))
    if max_items is not None and max_items > 0:
        sids = sids[:max_items]
    return sids


def expected_feature_path(feature_dir, sid, alpha):
    """Use the same naming convention as save_llava_hidden_similarity_features.py."""
    alpha = float(alpha)
    alpha_name = {1.0: "base", 0.5: "low", 1.5: "high"}.get(
        alpha,
        f"alpha{safe_float_tag_local(alpha)}",
    )
    return Path(feature_dir) / f"sid{sid:04d}_{alpha_name}_alpha{safe_float_tag_local(alpha)}.npz"


def safe_float_tag_local(x):
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


def npz_has_attention(path, args):
    if path is None or not os.path.exists(path):
        return False
    if args.score_type != "attention":
        return True
    try:
        x = np.load(path, allow_pickle=True)
        has_img = args.attn_img_key in x.files or "attn_last_to_img_last_layer_heads" in x.files
        has_txt = args.attn_text_key in x.files or "attn_last_to_text_last_layer_heads" in x.files
        try:
            x.close()
        except Exception:
            pass
        return bool(has_img and has_txt)
    except Exception:
        return False


def generate_missing_attention_features(args, selected_sids):
    """Generate base alpha=1.0 and low alpha=0.5 attention npz for selected sids.

    This keeps compare_selected0p5_lastquery_attention.py as a one-command script:
    it loads the existing eval JSONs for acc/grouping, generates missing attention
    features only for selected_weight=0.5 samples, then plots/analyzes them.

    Requires the modified save_llava_hidden_similarity_features.py to be present in
    the repo root, because we reuse its tokenizer/merge/attention extraction helpers.
    Also requires the modified modeling_llama_add_attn.py, so outputs.attentions are
    final post-softmax attentions actually used by AdaptVis forward.
    """
    if not args.generate_features:
        return

    if args.score_type != "attention":
        print("[GENERATE] --generate-features is intended for --score-type attention; skipping.")
        return

    selected_sids = sorted(set(int(s) for s in selected_sids))
    if not selected_sids:
        print("[GENERATE] no selected_weight=0.5 sids; nothing to generate.")
        return

    # Decide which sid/alpha files are missing or stale.
    plan = [
        (1.0, Path(args.base_feature_dir)),
        (0.5, Path(args.mixed_feature_dir)),
    ]
    pending = {}
    for sid in selected_sids:
        for alpha, out_dir in plan:
            out_path = expected_feature_path(out_dir, sid, alpha)
            if args.overwrite_generated_features or not npz_has_attention(out_path, args):
                pending.setdefault(sid, []).append((float(alpha), out_dir, out_path))

    if not pending:
        print("[GENERATE] all selected attention feature files already exist with attention keys.")
        return

    print(f"[GENERATE] selected sids: {len(selected_sids)}")
    print(f"[GENERATE] pending sids: {len(pending)}")
    print(f"[GENERATE] base feature dir:  {args.base_feature_dir}")
    print(f"[GENERATE] mixed feature dir: {args.mixed_feature_dir}")

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from model_zoo import get_model
    from dataset_zoo import get_dataset
    try:
        from misc import _default_collate
    except Exception:
        _default_collate = None

    import save_llava_hidden_similarity_features as sf

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("[GENERATE LOAD MODEL]", args.model_name, args.method, args.device)
    wrapper, image_preprocess = get_model(
        args.model_name,
        args.device,
        args.method,
        root_dir=args.root_dir,
    )
    wrapper.model.eval()

    print("[GENERATE LOAD DATASET]", args.dataset)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)
    collate_fn = _default_collate if image_preprocess is None else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    prompts, answers = sf.load_prompts(args.dataset, args.option)
    token_id_map = sf.closed_set_token_ids(wrapper)

    # Metadata is written per output dir for easier debugging/resume.
    meta_handles = {}
    for _, out_dir in plan:
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_dir not in meta_handles:
            meta_handles[out_dir] = (out_dir / "metadata_generated_by_compare.jsonl").open("a", encoding="utf-8")

    max_needed_sid = max(pending.keys())
    generated = 0
    global_idx = 0

    def close_meta_handles():
        for f in meta_handles.values():
            try:
                f.close()
            except Exception:
                pass

    try:
        with torch.no_grad():
            pbar = tqdm(loader, desc="generating attention features")
            for batch in pbar:
                for i_option in batch["image_options"]:
                    for image in i_option:
                        sid = global_idx
                        global_idx += 1

                        if sid > max_needed_sid and len(pending) == 0:
                            close_meta_handles()
                            print("[GENERATE DONE] written npz files:", generated)
                            return

                        if sid not in pending:
                            continue

                        prompt = prompts[sid]
                        gold = answers[sid]
                        obj1, obj2, obj1_span, obj2_span = sf.parse_objects_with_spans(prompt)

                        inputs = sf.build_inputs(wrapper, prompt, image, args.max_length, args.device)
                        input_ids_1d = inputs["input_ids"][0]
                        image_pos, image_token_id = sf.find_image_token_pos(input_ids_1d, wrapper)

                        tok_input_ids, offsets, tok_attention_mask, tokens = sf.tokenizer_offsets(
                            wrapper,
                            prompt,
                            args.max_length,
                        )
                        obj1_pre_pos = sf.token_positions_for_span(
                            offsets,
                            tok_attention_mask,
                            obj1_span,
                            exclude_pos=image_pos,
                        )
                        obj2_pre_pos = sf.token_positions_for_span(
                            offsets,
                            tok_attention_mask,
                            obj2_span,
                            exclude_pos=image_pos,
                        )

                        for alpha, out_dir, save_path in list(pending[sid]):
                            if save_path.exists() and not args.overwrite_generated_features and npz_has_attention(save_path, args):
                                continue

                            alpha_name = {1.0: "base", 0.5: "low", 1.5: "high"}.get(
                                float(alpha),
                                f"alpha{sf.safe_float_tag(alpha)}",
                            )

                            attn_tmp_dir = Path(args.out_dir) / "_tmp_attn_forward" / f"sid{sid:04d}_{alpha_name}"
                            attn_tmp_dir.mkdir(parents=True, exist_ok=True)
                            os.environ["SAVE_ATTN_PATH"] = str(attn_tmp_dir) + "/"

                            forward_kwargs = dict(inputs)
                            forward_kwargs.update({
                                "weight": float(alpha),
                                "output_hidden_states": True,
                                "output_attentions": True,
                                "return_dict": True,
                                "use_cache": False,
                            })

                            outputs = wrapper.model(**forward_kwargs)
                            hidden_states = sf.get_hidden_states_from_outputs(outputs)
                            last_hidden = hidden_states[-1][0]
                            logits_last = outputs.logits[0, -1]

                            merged_len = int(last_hidden.shape[0])
                            num_img_tokens = sf.get_num_image_tokens_from_inputs(wrapper, inputs)
                            image_start = sf.image_start_after_merge(tok_attention_mask, image_pos)
                            image_end = int(image_start + num_img_tokens)

                            if image_end > merged_len:
                                raise RuntimeError(
                                    f"Image range exceeds merged hidden length: start={image_start}, "
                                    f"end={image_end}, merged_len={merged_len}, num_img_tokens={num_img_tokens}, "
                                    f"sid={sid}, image_pos={image_pos}"
                                )

                            H_img = last_hidden[image_start:image_end]
                            text_pre_pos, text_merged_pos = sf.nonpad_text_positions(
                                tok_attention_mask,
                                image_pos,
                                num_img_tokens,
                            )
                            pairs = [
                                (pre, merged)
                                for pre, merged in zip(text_pre_pos, text_merged_pos)
                                if 0 <= int(merged) < merged_len
                            ]
                            text_pre_pos_valid = [int(p) for p, _ in pairs]
                            text_merged_pos = [int(m) for _, m in pairs]
                            H_text = (
                                last_hidden[torch.tensor(text_merged_pos, device=last_hidden.device)]
                                if text_merged_pos
                                else last_hidden[:0]
                            )

                            obj1_merged_pos = sf.map_premerge_to_merged_positions(
                                obj1_pre_pos,
                                tok_attention_mask,
                                image_pos,
                                num_img_tokens,
                            )
                            obj2_merged_pos = sf.map_premerge_to_merged_positions(
                                obj2_pre_pos,
                                tok_attention_mask,
                                image_pos,
                                num_img_tokens,
                            )
                            obj1_merged_pos = [p for p in obj1_merged_pos if 0 <= p < merged_len]
                            obj2_merged_pos = [p for p in obj2_merged_pos if 0 <= p < merged_len]

                            H_obj1 = (
                                last_hidden[torch.tensor(obj1_merged_pos, device=last_hidden.device)].mean(dim=0)
                                if obj1_merged_pos
                                else torch.full((last_hidden.shape[-1],), float("nan"), device=last_hidden.device)
                            )
                            H_obj2 = (
                                last_hidden[torch.tensor(obj2_merged_pos, device=last_hidden.device)].mean(dim=0)
                                if obj2_merged_pos
                                else torch.full((last_hidden.shape[-1],), float("nan"), device=last_hidden.device)
                            )

                            last_pre_pos = max([
                                i for i, m in enumerate(tok_attention_mask)
                                if int(m) == 1 and i != image_pos
                            ])
                            last_merged_pos = sf.map_premerge_to_merged_positions(
                                [last_pre_pos],
                                tok_attention_mask,
                                image_pos,
                                num_img_tokens,
                            )[0]
                            H_last_prompt = last_hidden[last_merged_pos]

                            arrays = {
                                "sample_id": np.array(sid, dtype=np.int32),
                                "alpha": np.array(float(alpha), dtype=np.float32),
                                "alpha_name": np.array(alpha_name),
                                "gold": np.array(gold),
                                "prompt": np.array(prompt),
                                "obj1": np.array(obj1),
                                "obj2": np.array(obj2),
                                "image_token_id": np.array(image_token_id, dtype=np.int32),
                                "image_start": np.array(image_start, dtype=np.int32),
                                "image_end": np.array(image_end, dtype=np.int32),
                                "num_img_tokens": np.array(num_img_tokens, dtype=np.int32),
                                "merged_len": np.array(merged_len, dtype=np.int32),
                                "hidden_dim": np.array(last_hidden.shape[-1], dtype=np.int32),
                                "input_ids": np.array(tok_input_ids, dtype=np.int64),
                                "attention_mask": np.array(tok_attention_mask, dtype=np.int64),
                                "tokens": np.array(tokens, dtype=object),
                                "text_pre_positions": np.array(text_pre_pos_valid, dtype=np.int32),
                                "text_merged_positions": np.array(text_merged_pos, dtype=np.int32),
                                "obj1_pre_positions": np.array(obj1_pre_pos, dtype=np.int32),
                                "obj2_pre_positions": np.array(obj2_pre_pos, dtype=np.int32),
                                "obj1_merged_positions": np.array(obj1_merged_pos, dtype=np.int32),
                                "obj2_merged_positions": np.array(obj2_merged_pos, dtype=np.int32),
                                "last_pre_position": np.array(last_pre_pos, dtype=np.int32),
                                "last_merged_position": np.array(last_merged_pos, dtype=np.int32),
                                "image_hidden": sf.to_numpy_dtype(H_img, args.feature_dtype),
                                "text_hidden": sf.to_numpy_dtype(H_text, args.feature_dtype),
                                "obj1_hidden": sf.to_numpy_dtype(H_obj1, args.feature_dtype),
                                "obj2_hidden": sf.to_numpy_dtype(H_obj2, args.feature_dtype),
                                "last_prompt_hidden": sf.to_numpy_dtype(H_last_prompt, args.feature_dtype),
                            }

                            cls_logits = sf.best_closed_set_logits(logits_last, token_id_map)
                            arrays["closed_set_words"] = np.array(list(cls_logits.keys()), dtype=object)
                            arrays["closed_set_logits"] = np.array(
                                [cls_logits[k] for k in cls_logits.keys()],
                                dtype=np.float32,
                            )

                            sf.add_attention_arrays(
                                arrays=arrays,
                                outputs=outputs,
                                wrapper=wrapper,
                                merged_len=merged_len,
                                last_merged_pos=last_merged_pos,
                                image_start=image_start,
                                image_end=image_end,
                                text_merged_pos=text_merged_pos,
                                save_all_layers_attn=args.save_all_layers_attn,
                            )

                            sf.save_npz(save_path, args.compress_features, **arrays)
                            generated += 1

                            meta = {
                                "sample_id": sid,
                                "alpha": float(alpha),
                                "alpha_name": alpha_name,
                                "path": str(save_path),
                                "gold": gold,
                                "obj1": obj1,
                                "obj2": obj2,
                                "image_start": image_start,
                                "image_end": image_end,
                                "num_img_tokens": num_img_tokens,
                                "merged_len": merged_len,
                                "hidden_dim": int(last_hidden.shape[-1]),
                                "attn_keys": True,
                            }
                            meta_f = meta_handles[out_dir]
                            meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                            meta_f.flush()

                            print(f"[GENERATED] sid={sid:04d} alpha={alpha} -> {save_path}")

                        pending.pop(sid, None)
                        pbar.set_postfix({"sid": sid, "pending": len(pending), "written": generated})

                        if len(pending) == 0:
                            close_meta_handles()
                            print("[GENERATE DONE] written npz files:", generated)
                            return
    finally:
        close_meta_handles()

    if pending:
        print("[GENERATE WARN] some selected sids were not found/generated:", sorted(pending.keys())[:50])
    print("[GENERATE DONE] written npz files:", generated)


def as_float(x):
    return np.asarray(x, dtype=np.float32)


def norm_vec(x):
    x = as_float(x).reshape(-1)
    return x / (np.linalg.norm(x) + EPS)


def norm_rows(x):
    x = as_float(x)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def cosine_dist(a, b):
    a = norm_vec(a)
    b = norm_vec(b)
    return float(1.0 - np.dot(a, b))


def sim_vec_to_tokens(vec, token_hidden):
    v = norm_vec(vec)
    H = norm_rows(token_hidden)
    return H @ v


def softmax_np(x):
    x = as_float(x)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + EPS)


def prob_dist(x):
    """Normalize nonnegative attention values into a conditional distribution."""
    x = as_float(x).reshape(-1)
    x = np.maximum(x, 0.0)
    s = float(np.sum(x))
    if s <= EPS:
        if len(x) == 0:
            return x
        return np.ones_like(x, dtype=np.float32) / float(len(x))
    return x / s


def dist_from_values(x, score_type):
    if score_type == "attention":
        return prob_dist(x)
    return softmax_np(x)


def entropy_from_values(x, score_type):
    p = dist_from_values(x, score_type)
    return float(-np.sum(p * np.log(p + EPS)))


def js_divergence_values(a, b, score_type):
    p = dist_from_values(a, score_type)
    q = dist_from_values(b, score_type)
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * (np.log(p + EPS) - np.log(m + EPS)))
        + 0.5 * np.sum(q * (np.log(q + EPS) - np.log(m + EPS)))
    )


def tv_distance_values(a, b, score_type):
    p = dist_from_values(a, score_type)
    q = dist_from_values(b, score_type)
    return float(0.5 * np.sum(np.abs(p - q)))


def topk_set(x, k):
    return set(np.argsort(-as_float(x))[:k].tolist())


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def weighted_center(scores, score_type):
    p = dist_from_values(scores, score_type)
    n = len(p)
    side = int(round(math.sqrt(n)))

    if side * side != n:
        xs = np.arange(n, dtype=np.float32)
        return float(np.sum(xs * p)), 0.0

    ys, xs = np.divmod(np.arange(n), side)
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    cx = float(np.sum(xs * p))
    cy = float(np.sum(ys * p))
    return cx, cy


def center_shift(a, b, score_type):
    ax, ay = weighted_center(a, score_type)
    bx, by = weighted_center(b, score_type)
    return float(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def score_grid(scores):
    scores = as_float(scores)
    n = scores.shape[0]
    side = int(round(math.sqrt(n)))

    if side * side != n:
        raise ValueError(f"Cannot reshape {n} to square grid.")

    return scores.reshape(side, side)


def normalize_view(x):
    x = as_float(x)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)

    if hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))

    if hi <= lo:
        return np.zeros_like(x)

    return np.clip((x - lo) / (hi - lo), 0, 1)


def fmt(x):
    return "NA" if x is None else f"{x:.6f}"


def safe_mean(xs):
    xs = [float(x) for x in xs if x is not None and not np.isnan(x)]
    return sum(xs) / len(xs) if xs else None


def safe_median(xs):
    xs = sorted([float(x) for x in xs if x is not None and not np.isnan(x)])

    if not xs:
        return None

    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def clean_token(tok):
    tok = str(tok)
    tok = tok.replace("▁", " ")
    tok = tok.replace("</s>", "<eos>")
    tok = tok.replace("<s>", "<bos>")
    return tok


def npz_has(x, key):
    return key in getattr(x, "files", [])


def read_attention_vector(x, key, heads_key):
    """Read mean attention vector. If only per-head attention exists, average heads."""
    if npz_has(x, key):
        return as_float(x[key]).reshape(-1)
    if npz_has(x, heads_key):
        arr = as_float(x[heads_key])
        # Expected [num_heads, num_tokens]
        if arr.ndim == 2:
            return arr.mean(axis=0).reshape(-1)
        # Expected [num_layers, num_heads, num_tokens]
        if arr.ndim == 3:
            return arr[-1].mean(axis=0).reshape(-1)
    return None


def get_tokens_full(x):
    if "tokens" not in x.files:
        return []
    return [clean_token(t) for t in x["tokens"].tolist()]


def get_text_tokens_from_feature(x, T, prefer_attn=True):
    """Return labels aligned to text scores.

    For attention vectors, the NPZ may contain merged indices. We map merged text
    positions back to tokenizer pre-merge positions using text_merged_positions
    and text_pre_positions when possible.
    """
    tokens_full = get_tokens_full(x)

    if not tokens_full:
        return [f"tok{i}" for i in range(T)]

    # If tokens were already saved in text-score order.
    if len(tokens_full) == T:
        return tokens_full

    # Attention path: attn_text_indices are merged-sequence indices.
    if prefer_attn and npz_has(x, "attn_text_indices") and npz_has(x, "text_merged_positions") and npz_has(x, "text_pre_positions"):
        attn_text_indices = [int(v) for v in np.asarray(x["attn_text_indices"]).reshape(-1).tolist()]
        text_merged = [int(v) for v in np.asarray(x["text_merged_positions"]).reshape(-1).tolist()]
        text_pre = [int(v) for v in np.asarray(x["text_pre_positions"]).reshape(-1).tolist()]
        merged_to_pre = {m: p for p, m in zip(text_pre, text_merged)}

        labels = []
        for m in attn_text_indices:
            p = merged_to_pre.get(m, None)
            if p is not None and 0 <= p < len(tokens_full):
                labels.append(tokens_full[p])
            else:
                labels.append(f"m{m}")

        if len(labels) == T:
            return labels

    # Hidden-sim path: text_pre_positions indexes into tokens_full.
    if npz_has(x, "text_pre_positions"):
        text_pre = [int(v) for v in np.asarray(x["text_pre_positions"]).reshape(-1).tolist()]
        if len(text_pre) == T:
            labels = []
            for p in text_pre:
                labels.append(tokens_full[p] if 0 <= p < len(tokens_full) else f"p{p}")
            return labels

    return [f"tok{i}" for i in range(T)]


def get_visual_scores(xb, xl, args):
    if args.score_type == "attention":
        sim_b = read_attention_vector(
            xb,
            args.attn_img_key,
            "attn_last_to_img_last_layer_heads",
        )
        sim_l = read_attention_vector(
            xl,
            args.attn_img_key,
            "attn_last_to_img_last_layer_heads",
        )

        if sim_b is not None and sim_l is not None:
            return sim_b, sim_l

        if not args.allow_fallback_hidden:
            raise KeyError(
                "Attention keys are missing from one or both feature files. "
                f"Expected {args.attn_img_key} or attn_last_to_img_last_layer_heads. "
                "Run save_llava_hidden_similarity_features.py with --save-attn, "
                "and use the modified modeling_llama_add_attn.py."
            )

        print("[WARN] attention image keys missing; fallback to hidden similarity.")

    H_img_b = as_float(xb["image_hidden"])
    H_img_l = as_float(xl["image_hidden"])

    sim_b = sim_vec_to_tokens(xb["last_prompt_hidden"], H_img_b)
    sim_l = sim_vec_to_tokens(xl["last_prompt_hidden"], H_img_l)

    return sim_b, sim_l


def get_text_scores(xb, xl, args):
    prefer_attn = args.score_type == "attention"

    if args.score_type == "attention":
        sim_b = read_attention_vector(
            xb,
            args.attn_text_key,
            "attn_last_to_text_last_layer_heads",
        )
        sim_l = read_attention_vector(
            xl,
            args.attn_text_key,
            "attn_last_to_text_last_layer_heads",
        )

        if sim_b is not None and sim_l is not None:
            if len(sim_b) != len(sim_l):
                raise RuntimeError(
                    f"text attention length mismatch: base={len(sim_b)} low={len(sim_l)}"
                )
            tokens_b = get_text_tokens_from_feature(xb, len(sim_b), prefer_attn=True)
            tokens_l = get_text_tokens_from_feature(xl, len(sim_l), prefer_attn=True)
            return sim_b, sim_l, tokens_b, tokens_l

        if not args.allow_fallback_hidden:
            raise KeyError(
                "Attention text keys are missing from one or both feature files. "
                f"Expected {args.attn_text_key} or attn_last_to_text_last_layer_heads. "
                "Run save_llava_hidden_similarity_features.py with --save-attn."
            )

        print("[WARN] attention text keys missing; fallback to hidden similarity.")
        prefer_attn = False

    Hb = as_float(xb["text_hidden"])
    Hl = as_float(xl["text_hidden"])

    sim_b = sim_vec_to_tokens(xb["last_prompt_hidden"], Hb)
    sim_l = sim_vec_to_tokens(xl["last_prompt_hidden"], Hl)

    if len(sim_b) != len(sim_l):
        raise RuntimeError(
            f"text sim length mismatch: base={len(sim_b)} low={len(sim_l)}"
        )

    tokens_b = get_text_tokens_from_feature(xb, len(sim_b), prefer_attn=prefer_attn)
    tokens_l = get_text_tokens_from_feature(xl, len(sim_l), prefer_attn=prefer_attn)

    return sim_b, sim_l, tokens_b, tokens_l


def score_label(args):
    return "attention prob" if args.score_type == "attention" else "cosine sim"


def visual_title_prefix(args):
    if args.score_type == "attention":
        return "last query post-softmax attention → visual tokens"
    return "last query ↔ visual tokens"


def text_title_prefix(args):
    if args.score_type == "attention":
        return "last query post-softmax attention → text tokens"
    return "last query ↔ text tokens"


def image_text_mass(xb, xl, args):
    """Return raw attention mass for image/text if attention vectors are available."""
    if args.score_type != "attention":
        return {}

    out = {}
    vb = read_attention_vector(xb, args.attn_img_key, "attn_last_to_img_last_layer_heads")
    vl = read_attention_vector(xl, args.attn_img_key, "attn_last_to_img_last_layer_heads")
    tb = read_attention_vector(xb, args.attn_text_key, "attn_last_to_text_last_layer_heads")
    tl = read_attention_vector(xl, args.attn_text_key, "attn_last_to_text_last_layer_heads")

    if vb is not None and vl is not None:
        out["visual_mass_base"] = float(np.sum(np.maximum(vb, 0.0)))
        out["visual_mass_low"] = float(np.sum(np.maximum(vl, 0.0)))
        out["visual_mass_delta"] = out["visual_mass_low"] - out["visual_mass_base"]

    if tb is not None and tl is not None:
        out["text_mass_base"] = float(np.sum(np.maximum(tb, 0.0)))
        out["text_mass_low"] = float(np.sum(np.maximum(tl, 0.0)))
        out["text_mass_delta"] = out["text_mass_low"] - out["text_mass_base"]

    return out


def plot_heatmap_panel(ax, arr, title, args, vmin=None, vmax=None, diff=False):
    if diff:
        vmax_abs = np.max(np.abs(arr)) if vmax is None else vmax
        vmax_abs = max(float(vmax_abs), 1e-12)
        im = ax.imshow(arr, vmin=-vmax_abs, vmax=vmax_abs)
    else:
        if args.score_type == "attention" and not args.visual_normalize:
            im = ax.imshow(arr, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(normalize_view(arr))

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def save_visual_heatmap(rec, xb, xl, out_path, args):
    sim_b, sim_l = get_visual_scores(xb, xl, args)
    diff = sim_l - sim_b

    gb = score_grid(sim_b)
    gl = score_grid(sim_l)
    gd = score_grid(diff)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    mass_extra = ""
    if args.score_type == "attention":
        mass_extra = (
            f", img_mass_base={rec.get('visual_mass_base', float('nan')):.6g}, "
            f"img_mass_low={rec.get('visual_mass_low', float('nan')):.6g}"
        )

    fig.suptitle(
        f"{visual_title_prefix(args)} | sid={rec['sid']} | {rec['group']} | gold={rec['gold']}\n"
        f"JS={rec['visual_js']:.6f}, TV={rec['visual_tv']:.6f}, "
        f"top10_jaccard={rec['visual_top10_jaccard']:.3f}, "
        f"center_shift={rec['visual_center_shift']:.3f}{mass_extra}",
        fontsize=11,
    )

    vmax = max(float(np.max(gb)), float(np.max(gl)), 1e-12)
    diff_vmax = max(float(np.max(np.abs(gd))), 1e-12)

    panels = [
        (gb, "visual base" if args.score_type == "attention" else "base", False),
        (gl, "visual low alpha=0.5" if args.score_type == "attention" else "low alpha=0.5", False),
        (gd, "visual diff low-base" if args.score_type == "attention" else "diff low-base", True),
    ]

    for ax, (arr, title, is_diff) in zip(axes, panels):
        im = plot_heatmap_panel(
            ax,
            arr,
            title,
            args,
            vmin=0.0,
            vmax=vmax,
            diff=is_diff,
        )
        if is_diff:
            im.set_clim(-diff_vmax, diff_vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_text_csv(rec, xb, xl, out_path, args):
    sim_b, sim_l, tokens_b, tokens_l = get_text_scores(xb, xl, args)
    diff = sim_l - sim_b

    order_abs = np.argsort(-np.abs(diff))
    rank_abs = np.empty_like(order_abs)

    for r, idx in enumerate(order_abs):
        rank_abs[idx] = r + 1

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "group",
                "gold",
                "score_type",
                "token_index",
                "token_base",
                "token_low",
                "score_base",
                "score_low",
                "diff_low_minus_base",
                "abs_diff_rank",
            ]
        )

        for i in range(len(sim_b)):
            writer.writerow(
                [
                    rec["sid"],
                    rec["group"],
                    rec["gold"],
                    args.score_type,
                    i,
                    tokens_b[i] if i < len(tokens_b) else f"tok{i}",
                    tokens_l[i] if i < len(tokens_l) else f"tok{i}",
                    float(sim_b[i]),
                    float(sim_l[i]),
                    float(diff[i]),
                    int(rank_abs[i]),
                ]
            )


def save_text_plot(rec, xb, xl, out_path, args, max_tokens=90):
    sim_b, sim_l, tokens_b, _ = get_text_scores(xb, xl, args)
    diff = sim_l - sim_b

    T = len(sim_b)
    idx = np.arange(min(T, max_tokens))
    labels = [tokens_b[i] if i < len(tokens_b) else f"tok{i}" for i in idx]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(12, len(idx) * 0.35), 7),
        sharex=True,
    )

    mass_extra = ""
    if args.score_type == "attention":
        mass_extra = (
            f", text_mass_base={rec.get('text_mass_base', float('nan')):.6g}, "
            f"text_mass_low={rec.get('text_mass_low', float('nan')):.6g}"
        )

    fig.suptitle(
        f"{text_title_prefix(args)} | sid={rec['sid']} | {rec['group']} | gold={rec['gold']}\n"
        f"text_JS={rec['text_js']:.6f}, text_TV={rec['text_tv']:.6f}, "
        f"text_top10_jaccard={rec['text_top10_jaccard']:.3f}{mass_extra}",
        fontsize=11,
    )

    axes[0].plot(idx, sim_b[idx], marker="o", linewidth=1, label="base")
    axes[0].plot(idx, sim_l[idx], marker="o", linewidth=1, label="low alpha=0.5")
    axes[0].set_ylabel(score_label(args))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(idx, diff[idx])
    axes[1].set_ylabel("low - base")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_combined_plot(rec, xb, xl, out_path, args, max_tokens=90):
    visual_b, visual_l = get_visual_scores(xb, xl, args)
    text_b, text_l, tokens_b, _ = get_text_scores(xb, xl, args)

    visual_diff = visual_l - visual_b
    text_diff = text_l - text_b

    gb = score_grid(visual_b)
    gl = score_grid(visual_l)
    gd = score_grid(visual_diff)

    T = len(text_b)
    idx = np.arange(min(T, max_tokens))
    labels = [tokens_b[i] if i < len(tokens_b) else f"tok{i}" for i in idx]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0])

    base_gen = rec.get("base_generation", "")
    low_gen = rec.get("low_generation", "")
    if len(base_gen) > 80:
        base_gen = base_gen[:77] + "..."
    if len(low_gen) > 80:
        low_gen = low_gen[:77] + "..."

    title_score = "post-softmax attention" if args.score_type == "attention" else "hidden cosine similarity"
    fig.suptitle(
        f"sid={rec['sid']} | {rec['group']} | gold={rec['gold']} | {title_score}\n"
        f"base={base_gen} | low={low_gen}",
        fontsize=12,
    )

    vmax = max(float(np.max(gb)), float(np.max(gl)), 1e-12)
    diff_vmax = max(float(np.max(np.abs(gd))), 1e-12)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    im0 = plot_heatmap_panel(ax0, gb, "visual base", args, vmin=0.0, vmax=vmax)
    im1 = plot_heatmap_panel(ax1, gl, "visual low", args, vmin=0.0, vmax=vmax)
    im2 = plot_heatmap_panel(ax2, gd, "visual diff", args, diff=True, vmax=diff_vmax)

    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(idx, text_b[idx], marker="o", linewidth=1, label="text base")
    ax3.plot(idx, text_l[idx], marker="o", linewidth=1, label="text low")
    ax3.set_ylabel(score_label(args))
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, :], sharex=ax3)
    ax4.bar(idx, text_diff[idx])
    ax4.set_ylabel("text low-base")
    ax4.set_xticks(idx)
    ax4.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    visual_dir = os.path.join(args.out_dir, "visual_heatmaps")
    text_csv_dir = os.path.join(args.out_dir, "text_similarity_csv")
    text_plot_dir = os.path.join(args.out_dir, "text_similarity_plots")
    combined_dir = os.path.join(args.out_dir, "combined_plots")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(text_csv_dir, exist_ok=True)
    os.makedirs(text_plot_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    base_json = args.base_json
    mixed_json = args.mixed_json

    if base_json is None:
        base_json = find_one(
            "BASE w1=w2=1",
            [
                f"output/*{args.dataset}*adapt_vis*w1_w11_w21_thr0p4_{args.option}option_{args.test_tag}.json",
                f"output/*{args.dataset}*adapt_vis*w1_w11p0_w21p0_thr0p4_{args.option}option_{args.test_tag}.json",
            ],
        )

    if mixed_json is None:
        mixed_json = find_one(
            "MIXED w1=0.5,w2=1.5",
            [
                f"output/*{args.dataset}*adapt_vis*w1_w10p5_w21p5_thr0p4_{args.option}option_{args.test_tag}.json",
            ],
            exclude=(
                "w1_w10p5_w20p5",
                "w1_w11p5_w21p5",
                "w1_w11_w21",
                "w1_w11p0_w21p0",
            ),
        )

    base = load_json(base_json)
    mixed = load_json(mixed_json)

    if args.generate_features:
        sids_to_generate = selected0p5_sids_for_generation(
            base,
            mixed,
            max_items=args.generate_limit,
        )
        generate_missing_attention_features(args, sids_to_generate)

    groups = OrderedDict(
        [
            ("selected0p5_wrong_to_correct", []),
            ("selected0p5_correct_to_wrong", []),
            ("selected0p5_correct_to_correct", []),
            ("selected0p5_wrong_to_wrong", []),
        ]
    )

    records = []
    missing = []

    for i in range(min(len(base), len(mixed))):
        sid = sid_of(i, base[i])
        sw = selected_weight(mixed[i])

        if sw is None or abs(sw - 0.5) > 1e-6:
            continue

        b_corr = correct(base[i])
        l_corr = correct(mixed[i])
        gold = norm_gold(base[i].get("Golden", base[i].get("gold", "")))

        if (not b_corr) and l_corr:
            group = "selected0p5_wrong_to_correct"
        elif b_corr and (not l_corr):
            group = "selected0p5_correct_to_wrong"
        elif b_corr and l_corr:
            group = "selected0p5_correct_to_correct"
        else:
            group = "selected0p5_wrong_to_wrong"

        bf = feature_file(args.base_feature_dir, sid, "base")
        lf = feature_file(args.mixed_feature_dir, sid, "low")

        if bf is None or lf is None:
            missing.append((sid, bf, lf))
            continue

        xb = np.load(bf, allow_pickle=True)
        xl = np.load(lf, allow_pickle=True)

        sim_visual_b, sim_visual_l = get_visual_scores(xb, xl, args)
        sim_text_b, sim_text_l, tokens_b, _ = get_text_scores(xb, xl, args)

        visual_diff = sim_visual_l - sim_visual_b
        text_diff = sim_text_l - sim_text_b

        rec = {
            "sid": sid,
            "group": group,
            "gold": gold,
            "score_type": args.score_type,
            "base_correct": bool(b_corr),
            "low_correct": bool(l_corr),
            "base_generation": gen_text(base[i]),
            "low_generation": gen_text(mixed[i]),
            "base_feature_file": bf,
            "low_feature_file": lf,

            "last_prompt_cosdist": cosine_dist(
                xb["last_prompt_hidden"],
                xl["last_prompt_hidden"],
            ) if "last_prompt_hidden" in xb.files and "last_prompt_hidden" in xl.files else None,

            "visual_js": js_divergence_values(sim_visual_b, sim_visual_l, args.score_type),
            "visual_tv": tv_distance_values(sim_visual_b, sim_visual_l, args.score_type),
            "visual_entropy_base": entropy_from_values(sim_visual_b, args.score_type),
            "visual_entropy_low": entropy_from_values(sim_visual_l, args.score_type),
            "visual_entropy_delta": entropy_from_values(sim_visual_l, args.score_type)
            - entropy_from_values(sim_visual_b, args.score_type),
            "visual_top5_jaccard": jaccard(
                topk_set(sim_visual_b, 5),
                topk_set(sim_visual_l, 5),
            ),
            "visual_top10_jaccard": jaccard(
                topk_set(sim_visual_b, 10),
                topk_set(sim_visual_l, 10),
            ),
            "visual_top20_jaccard": jaccard(
                topk_set(sim_visual_b, 20),
                topk_set(sim_visual_l, 20),
            ),
            "visual_center_shift": center_shift(sim_visual_b, sim_visual_l, args.score_type),
            "visual_mean_abs_diff": float(np.mean(np.abs(visual_diff))),
            "visual_max_abs_diff": float(np.max(np.abs(visual_diff))),

            "text_js": js_divergence_values(sim_text_b, sim_text_l, args.score_type),
            "text_tv": tv_distance_values(sim_text_b, sim_text_l, args.score_type),
            "text_entropy_base": entropy_from_values(sim_text_b, args.score_type),
            "text_entropy_low": entropy_from_values(sim_text_l, args.score_type),
            "text_entropy_delta": entropy_from_values(sim_text_l, args.score_type)
            - entropy_from_values(sim_text_b, args.score_type),
            "text_top5_jaccard": jaccard(
                topk_set(sim_text_b, 5),
                topk_set(sim_text_l, 5),
            ),
            "text_top10_jaccard": jaccard(
                topk_set(sim_text_b, 10),
                topk_set(sim_text_l, 10),
            ),
            "text_mean_abs_diff": float(np.mean(np.abs(text_diff))),
            "text_max_abs_diff": float(np.max(np.abs(text_diff))),
        }

        rec.update(image_text_mass(xb, xl, args))

        top_text_change = np.argsort(-np.abs(text_diff))[:10]
        rec["top_text_change"] = [
            {
                "token_index": int(j),
                "token": str(tokens_b[j]) if j < len(tokens_b) else f"tok{j}",
                "score_base": float(sim_text_b[j]),
                "score_low": float(sim_text_l[j]),
                "diff_low_minus_base": float(text_diff[j]),
            }
            for j in top_text_change
        ]

        records.append(rec)
        groups[group].append(sid)

    print("\n[LOAD]")
    print("base json:", len(base), base_json)
    print("mixed json:", len(mixed), mixed_json)
    print("base feature dir:", args.base_feature_dir)
    print("mixed feature dir:", args.mixed_feature_dir)
    print("score type:", args.score_type)

    print("\n[GROUP COUNTS | selected weight = 0.5]")
    for k, v in groups.items():
        print(f"{k}: {len(v)}")
        print(v[:120])

    if missing:
        print("\n[MISSING FEATURE FILES]")
        print("missing count:", len(missing))
        for row in missing[:40]:
            print(row)

    metrics = [
        "last_prompt_cosdist",
        "visual_js",
        "visual_tv",
        "visual_entropy_base",
        "visual_entropy_low",
        "visual_entropy_delta",
        "visual_top5_jaccard",
        "visual_top10_jaccard",
        "visual_top20_jaccard",
        "visual_center_shift",
        "visual_mean_abs_diff",
        "visual_max_abs_diff",
        "text_js",
        "text_tv",
        "text_entropy_base",
        "text_entropy_low",
        "text_entropy_delta",
        "text_top5_jaccard",
        "text_top10_jaccard",
        "text_mean_abs_diff",
        "text_max_abs_diff",
    ]

    if args.score_type == "attention":
        metrics += [
            "visual_mass_base",
            "visual_mass_low",
            "visual_mass_delta",
            "text_mass_base",
            "text_mass_low",
            "text_mass_delta",
        ]

    summary = {
        "dataset": args.dataset,
        "option": args.option,
        "score_type": args.score_type,
        "base_json": base_json,
        "mixed_json": mixed_json,
        "base_feature_dir": args.base_feature_dir,
        "mixed_feature_dir": args.mixed_feature_dir,
        "group_counts": {k: len(v) for k, v in groups.items()},
        "metrics": {},
    }

    print("\n[SUMMARY BY GROUP]")
    for g in groups.keys():
        rs = [r for r in records if r["group"] == g]
        print(f"\n===== {g} | n={len(rs)} =====")
        summary["metrics"][g] = {}

        for m in metrics:
            vals = [r.get(m, None) for r in rs]
            mean_v = safe_mean(vals)
            median_v = safe_median(vals)

            print(f"{m:26s} mean={fmt(mean_v)} median={fmt(median_v)}")

            summary["metrics"][g][m] = {
                "mean": mean_v,
                "median": median_v,
            }

    A = [r for r in records if r["group"] == "selected0p5_wrong_to_correct"]
    B = [r for r in records if r["group"] == "selected0p5_correct_to_wrong"]

    print("\n[DIRECT CONTRAST | wrong_to_correct minus correct_to_wrong]")
    print("A wrong_to_correct:", len(A))
    print("B correct_to_wrong:", len(B))

    summary["direct_contrast_AminusB"] = {}

    for m in metrics:
        av = safe_mean([r.get(m, None) for r in A])
        bv = safe_mean([r.get(m, None) for r in B])
        diff = None if av is None or bv is None else av - bv

        print(f"{m:26s} A={fmt(av)} B={fmt(bv)} A-B={fmt(diff)}")

        summary["direct_contrast_AminusB"][m] = {
            "wrong_to_correct_mean": av,
            "correct_to_wrong_mean": bv,
            "A_minus_B": diff,
        }

    records_path = os.path.join(
        args.out_dir,
        "selected0p5_lastquery_attention_records.json" if args.score_type == "attention" else "selected0p5_lastquery_similarity_records.json",
    )
    summary_path = os.path.join(
        args.out_dir,
        "selected0p5_lastquery_attention_summary.json" if args.score_type == "attention" else "selected0p5_lastquery_similarity_summary.json",
    )

    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(records_path)
    print(summary_path)

    rng = random.Random(args.seed)

    for label, rs in [
        ("wrong_to_correct", A),
        ("correct_to_wrong", B),
    ]:
        if not rs:
            continue

        chosen = rs if len(rs) <= args.num_vis else rng.sample(rs, args.num_vis)

        for r in chosen:
            xb = np.load(r["base_feature_file"], allow_pickle=True)
            xl = np.load(r["low_feature_file"], allow_pickle=True)

            suffix = "attention" if args.score_type == "attention" else "lastquery"

            visual_path = os.path.join(
                visual_dir,
                f"{label}_sid{r['sid']:04d}_{suffix}_visual.png",
            )
            save_visual_heatmap(r, xb, xl, visual_path, args)
            print("[VISUAL FIG SAVED]", visual_path)

            text_csv_path = os.path.join(
                text_csv_dir,
                f"{label}_sid{r['sid']:04d}_{suffix}_text.csv",
            )
            save_text_csv(r, xb, xl, text_csv_path, args)
            print("[TEXT CSV SAVED]", text_csv_path)

            text_plot_path = os.path.join(
                text_plot_dir,
                f"{label}_sid{r['sid']:04d}_{suffix}_text.png",
            )
            save_text_plot(r, xb, xl, text_plot_path, args)
            print("[TEXT PLOT SAVED]", text_plot_path)

            if args.save_combined:
                combined_path = os.path.join(
                    combined_dir,
                    f"{label}_sid{r['sid']:04d}_{suffix}_combined.png",
                )
                save_combined_plot(r, xb, xl, combined_path, args)
                print("[COMBINED FIG SAVED]", combined_path)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
