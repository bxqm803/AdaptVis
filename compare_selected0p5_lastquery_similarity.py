import os
import glob
import json
import math
import csv
import random
import argparse
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


def entropy_from_scores(x):
    p = softmax_np(x)
    return float(-np.sum(p * np.log(p + EPS)))


def js_divergence(a, b):
    p = softmax_np(a)
    q = softmax_np(b)
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * (np.log(p + EPS) - np.log(m + EPS)))
        + 0.5 * np.sum(q * (np.log(q + EPS) - np.log(m + EPS)))
    )


def tv_distance(a, b):
    p = softmax_np(a)
    q = softmax_np(b)
    return float(0.5 * np.sum(np.abs(p - q)))


def topk_set(x, k):
    return set(np.argsort(-as_float(x))[:k].tolist())


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def weighted_center(scores):
    p = softmax_np(scores)
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


def center_shift(a, b):
    ax, ay = weighted_center(a)
    bx, by = weighted_center(b)
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


def get_tokens(x):
    if "tokens" in x.files:
        return [clean_token(t) for t in x["tokens"].tolist()]

    T = int(x["text_hidden"].shape[0])
    return [f"tok{i}" for i in range(T)]


def get_text_sims(xb, xl):
    Hb = as_float(xb["text_hidden"])
    Hl = as_float(xl["text_hidden"])

    sim_b = sim_vec_to_tokens(xb["last_prompt_hidden"], Hb)
    sim_l = sim_vec_to_tokens(xl["last_prompt_hidden"], Hl)

    tokens_b = get_tokens(xb)
    tokens_l = get_tokens(xl)

    if len(sim_b) != len(sim_l):
        raise RuntimeError(
            f"text sim length mismatch: base={len(sim_b)} low={len(sim_l)}"
        )

    if len(tokens_b) != len(sim_b):
        tokens_b = [f"tok{i}" for i in range(len(sim_b))]

    if len(tokens_l) != len(sim_l):
        tokens_l = [f"tok{i}" for i in range(len(sim_l))]

    return sim_b, sim_l, tokens_b, tokens_l


def get_visual_sims(xb, xl):
    H_img_b = as_float(xb["image_hidden"])
    H_img_l = as_float(xl["image_hidden"])

    sim_b = sim_vec_to_tokens(xb["last_prompt_hidden"], H_img_b)
    sim_l = sim_vec_to_tokens(xl["last_prompt_hidden"], H_img_l)

    return sim_b, sim_l


def save_visual_heatmap(rec, xb, xl, out_path):
    sim_b, sim_l = get_visual_sims(xb, xl)
    diff = sim_l - sim_b

    gb = score_grid(sim_b)
    gl = score_grid(sim_l)
    gd = score_grid(diff)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    fig.suptitle(
        f"last query ↔ visual tokens | sid={rec['sid']} | {rec['group']} | gold={rec['gold']}\n"
        f"JS={rec['visual_js']:.6f}, TV={rec['visual_tv']:.6f}, "
        f"top10_jaccard={rec['visual_top10_jaccard']:.3f}, "
        f"center_shift={rec['visual_center_shift']:.3f}",
        fontsize=11,
    )

    panels = [
        (gb, "base"),
        (gl, "low alpha=0.5"),
        (gd, "diff low-base"),
    ]

    for ax, (arr, title) in zip(axes, panels):
        if "diff" in title:
            vmax = np.percentile(np.abs(arr), 99)
            vmax = max(float(vmax), 1e-6)
            im = ax.imshow(arr, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(normalize_view(arr))

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_text_csv(rec, xb, xl, out_path):
    sim_b, sim_l, tokens_b, tokens_l = get_text_sims(xb, xl)
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
                "token_index",
                "token_base",
                "token_low",
                "sim_base",
                "sim_low",
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
                    i,
                    tokens_b[i],
                    tokens_l[i],
                    float(sim_b[i]),
                    float(sim_l[i]),
                    float(diff[i]),
                    int(rank_abs[i]),
                ]
            )


def save_text_plot(rec, xb, xl, out_path, max_tokens=90):
    sim_b, sim_l, tokens_b, _ = get_text_sims(xb, xl)
    diff = sim_l - sim_b

    T = len(sim_b)
    idx = np.arange(min(T, max_tokens))
    labels = [tokens_b[i] for i in idx]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(12, len(idx) * 0.35), 7),
        sharex=True,
    )

    fig.suptitle(
        f"last query ↔ text tokens | sid={rec['sid']} | {rec['group']} | gold={rec['gold']}\n"
        f"text_JS={rec['text_js']:.6f}, text_TV={rec['text_tv']:.6f}, "
        f"text_top10_jaccard={rec['text_top10_jaccard']:.3f}",
        fontsize=11,
    )

    axes[0].plot(idx, sim_b[idx], marker="o", linewidth=1, label="base")
    axes[0].plot(idx, sim_l[idx], marker="o", linewidth=1, label="low alpha=0.5")
    axes[0].set_ylabel("cosine sim")
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


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    visual_dir = os.path.join(args.out_dir, "visual_heatmaps")
    text_csv_dir = os.path.join(args.out_dir, "text_similarity_csv")
    text_plot_dir = os.path.join(args.out_dir, "text_similarity_plots")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(text_csv_dir, exist_ok=True)
    os.makedirs(text_plot_dir, exist_ok=True)

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

        sim_visual_b, sim_visual_l = get_visual_sims(xb, xl)
        sim_text_b, sim_text_l, tokens_b, _ = get_text_sims(xb, xl)

        visual_diff = sim_visual_l - sim_visual_b
        text_diff = sim_text_l - sim_text_b

        rec = {
            "sid": sid,
            "group": group,
            "gold": gold,
            "base_correct": bool(b_corr),
            "low_correct": bool(l_corr),
            "base_generation": gen_text(base[i]),
            "low_generation": gen_text(mixed[i]),
            "base_feature_file": bf,
            "low_feature_file": lf,

            "last_prompt_cosdist": cosine_dist(
                xb["last_prompt_hidden"],
                xl["last_prompt_hidden"],
            ),

            "visual_js": js_divergence(sim_visual_b, sim_visual_l),
            "visual_tv": tv_distance(sim_visual_b, sim_visual_l),
            "visual_entropy_base": entropy_from_scores(sim_visual_b),
            "visual_entropy_low": entropy_from_scores(sim_visual_l),
            "visual_entropy_delta": entropy_from_scores(sim_visual_l)
            - entropy_from_scores(sim_visual_b),
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
            "visual_center_shift": center_shift(sim_visual_b, sim_visual_l),
            "visual_mean_abs_diff": float(np.mean(np.abs(visual_diff))),
            "visual_max_abs_diff": float(np.max(np.abs(visual_diff))),

            "text_js": js_divergence(sim_text_b, sim_text_l),
            "text_tv": tv_distance(sim_text_b, sim_text_l),
            "text_entropy_base": entropy_from_scores(sim_text_b),
            "text_entropy_low": entropy_from_scores(sim_text_l),
            "text_entropy_delta": entropy_from_scores(sim_text_l)
            - entropy_from_scores(sim_text_b),
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

        top_text_change = np.argsort(-np.abs(text_diff))[:10]
        rec["top_text_change"] = [
            {
                "token_index": int(j),
                "token": str(tokens_b[j]),
                "sim_base": float(sim_text_b[j]),
                "sim_low": float(sim_text_l[j]),
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

    summary = {
        "dataset": args.dataset,
        "option": args.option,
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
            vals = [r[m] for r in rs]
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
        av = safe_mean([r[m] for r in A])
        bv = safe_mean([r[m] for r in B])
        diff = None if av is None or bv is None else av - bv

        print(f"{m:26s} A={fmt(av)} B={fmt(bv)} A-B={fmt(diff)}")

        summary["direct_contrast_AminusB"][m] = {
            "wrong_to_correct_mean": av,
            "correct_to_wrong_mean": bv,
            "A_minus_B": diff,
        }

    records_path = os.path.join(
        args.out_dir,
        "selected0p5_lastquery_similarity_records.json",
    )
    summary_path = os.path.join(
        args.out_dir,
        "selected0p5_lastquery_similarity_summary.json",
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

            visual_path = os.path.join(
                visual_dir,
                f"{label}_sid{r['sid']:04d}_lastquery_visual.png",
            )
            save_visual_heatmap(r, xb, xl, visual_path)
            print("[VISUAL FIG SAVED]", visual_path)

            text_csv_path = os.path.join(
                text_csv_dir,
                f"{label}_sid{r['sid']:04d}_lastquery_text.csv",
            )
            save_text_csv(r, xb, xl, text_csv_path)
            print("[TEXT CSV SAVED]", text_csv_path)

            text_plot_path = os.path.join(
                text_plot_dir,
                f"{label}_sid{r['sid']:04d}_lastquery_text.png",
            )
            save_text_plot(r, xb, xl, text_plot_path)
            print("[TEXT PLOT SAVED]", text_plot_path)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
