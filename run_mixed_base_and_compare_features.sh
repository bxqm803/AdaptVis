#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Config
# ==============================
GPU="${GPU:-0}"
DATASET="${DATASET:-Controlled_Images_A}"
OPTION="${OPTION:-four}"
MODEL_NAME="${MODEL_NAME:-llava1.5}"
ROOT_DIR="${ROOT_DIR:-data}"
METHOD="${METHOD:-adapt_vis}"
THRESHOLD="${THRESHOLD:-0.4}"

MIXED_FEATURE_DIR="${MIXED_FEATURE_DIR:-output/hidden_features_mixed_w10p5_w21p5_thr0p4}"
BASE_FEATURE_DIR="${BASE_FEATURE_DIR:-output/hidden_features_base_w1}"

ANALYSIS_DIR="${ANALYSIS_DIR:-output/feature_similarity_analysis_selected0p5}"
STATS_SCRIPT="${STATS_SCRIPT:-scripts/compare_selected0p5_feature_change.py}"

mkdir -p output scripts "${ANALYSIS_DIR}"

export TOKENIZERS_PARALLELISM=false
export TEST_MODE=True
export SAVE_HIDDEN_FEATURES=True

echo "[CHECK] compile patched files"
python3 -m py_compile model_zoo/llava/modeling_llava_scal.py
python3 -m py_compile model_zoo/llava15.py

# ==============================
# 1. Run mixed AdaptVis: w1=0.5, w2=1.5
# ==============================
echo
echo "======================================"
echo "[RUN 1] mixed AdaptVis w1=0.5 w2=1.5 threshold=${THRESHOLD}"
echo "Feature dir: ${MIXED_FEATURE_DIR}"
echo "======================================"

export HIDDEN_FEATURE_DIR="${MIXED_FEATURE_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python3 main_aro.py \
  --dataset "${DATASET}" \
  --model-name "${MODEL_NAME}" \
  --download \
  --method "${METHOD}" \
  --weight1 0.5 \
  --weight2 1.5 \
  --threshold "${THRESHOLD}" \
  --option "${OPTION}"

echo "[CHECK] mixed feature count"
find "${MIXED_FEATURE_DIR}" -type f -name "*.npz" | wc -l

# ==============================
# 2. Run base: w1=w2=1.0
# ==============================
echo
echo "======================================"
echo "[RUN 2] base w1=1.0 w2=1.0"
echo "Feature dir: ${BASE_FEATURE_DIR}"
echo "======================================"

export HIDDEN_FEATURE_DIR="${BASE_FEATURE_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python3 main_aro.py \
  --dataset "${DATASET}" \
  --model-name "${MODEL_NAME}" \
  --download \
  --method "${METHOD}" \
  --weight1 1.0 \
  --weight2 1.0 \
  --threshold "${THRESHOLD}" \
  --option "${OPTION}"

echo "[CHECK] base feature count"
find "${BASE_FEATURE_DIR}" -type f -name "*.npz" | wc -l

# ==============================
# 3. Write comparison script
# ==============================
cat > "${STATS_SCRIPT}" <<'PY'
import os
import glob
import json
import math
import random
from collections import OrderedDict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET = os.environ.get("DATASET", "Controlled_Images_A")
OPTION = os.environ.get("OPTION", "four")
TEST_TAG = os.environ.get("TEST_TAG", "True")

MIXED_FEATURE_DIR = os.environ.get("MIXED_FEATURE_DIR", "output/hidden_features_mixed_w10p5_w21p5_thr0p4")
BASE_FEATURE_DIR = os.environ.get("BASE_FEATURE_DIR", "output/hidden_features_base_w1")
ANALYSIS_DIR = os.environ.get("ANALYSIS_DIR", "output/feature_similarity_analysis_selected0p5")

RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "7"))
NUM_VIS = int(os.environ.get("NUM_VIS", "5"))

EPS = 1e-12
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(os.path.join(ANALYSIS_DIR, "figures"), exist_ok=True)


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
        raise FileNotFoundError(f"No file found for {name}:\n" + "\n".join(patterns))
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
    return str(item.get("RawGeneration", item.get("Generation", item.get("generation", ""))))


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


def l2norm(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + EPS)


def cosine_distance_vec(a, b):
    a = as_float(a).reshape(-1)
    b = as_float(b).reshape(-1)
    return float(1.0 - np.dot(a, b) / ((np.linalg.norm(a) + EPS) * (np.linalg.norm(b) + EPS)))


def image_mean_feature(H):
    return np.mean(as_float(H), axis=0)


def mean_patch_cosine_distance(A, B):
    A = l2norm(as_float(A), axis=-1)
    B = l2norm(as_float(B), axis=-1)
    return float(np.mean(1.0 - np.sum(A * B, axis=-1)))


def softmax_np(x):
    x = as_float(x)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + EPS)


def js_divergence_from_scores(a, b):
    p = softmax_np(a)
    q = softmax_np(b)
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * (np.log(p + EPS) - np.log(m + EPS))) + 0.5 * np.sum(q * (np.log(q + EPS) - np.log(m + EPS)))
    return float(js)


def tv_distance_from_scores(a, b):
    p = softmax_np(a)
    q = softmax_np(b)
    return float(0.5 * np.sum(np.abs(p - q)))


def topk_idx(x, k):
    return set(np.argsort(-np.asarray(x))[:k].tolist())


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def weighted_center_from_scores(scores):
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


def center_shift(a_scores, b_scores):
    ax, ay = weighted_center_from_scores(a_scores)
    bx, by = weighted_center_from_scores(b_scores)
    return float(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def pred_relation_from_obj_scores(obj1_scores, obj2_scores):
    x1, y1 = weighted_center_from_scores(obj1_scores)
    x2, y2 = weighted_center_from_scores(obj2_scores)

    dx = x1 - x2
    dy = y1 - y2

    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "under" if dy > 0 else "on"


def relation_correct(pred, gold):
    g = str(gold).strip().lower()
    if g == "above":
        g = "on"
    return str(pred).strip().lower() == g


def image_selfsim(H):
    H = l2norm(as_float(H), axis=-1)
    return H @ H.T


def image_selfsim_delta(H_base, H_low):
    Sb = image_selfsim(H_base)
    Sl = image_selfsim(H_low)

    diff = Sl - Sb
    rel_fro = np.linalg.norm(diff) / (np.linalg.norm(Sb) + EPS)

    sb = Sb.reshape(-1)
    sl = Sl.reshape(-1)
    cos_dist = 1.0 - float(np.dot(sb, sl) / ((np.linalg.norm(sb) + EPS) * (np.linalg.norm(sl) + EPS)))

    return float(rel_fro), float(cos_dist)


def safe_mean(xs):
    xs = [float(x) for x in xs if x is not None and not np.isnan(x)]
    return sum(xs) / len(xs) if xs else None


def safe_median(xs):
    xs = sorted([float(x) for x in xs if x is not None and not np.isnan(x)])
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def fmt(x):
    return "NA" if x is None else f"{x:.6f}"


def score_to_grid(scores):
    scores = np.asarray(scores, dtype=np.float32)
    n = scores.shape[0]
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"Cannot reshape {n} scores to square grid.")
    return scores.reshape(side, side)


def normalize_for_view(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def save_case_figure(rec, xb, xl, out_path):
    sid = rec["sid"]
    group = rec["group"]
    gold = rec["gold"]

    H_img_b = as_float(xb["image_hidden"])
    H_img_l = as_float(xl["image_hidden"])

    sim_obj1_b = as_float(xb["sim_obj1_to_img"])
    sim_obj1_l = as_float(xl["sim_obj1_to_img"])
    sim_obj2_b = as_float(xb["sim_obj2_to_img"])
    sim_obj2_l = as_float(xl["sim_obj2_to_img"])

    obj1 = str(xb["obj1"])
    obj2 = str(xb["obj2"])

    Sb = image_selfsim(H_img_b)
    Sl = image_selfsim(H_img_l)
    Sd = Sl - Sb

    obj1_b = score_to_grid(sim_obj1_b)
    obj1_l = score_to_grid(sim_obj1_l)
    obj1_d = obj1_l - obj1_b

    obj2_b = score_to_grid(sim_obj2_b)
    obj2_l = score_to_grid(sim_obj2_l)
    obj2_d = obj2_l - obj2_b

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))

    fig.suptitle(
        f"sid={sid} | {group} | gold={gold} | "
        f"base_pred={rec['feature_relation_base_pred']} low_pred={rec['feature_relation_low_pred']}\n"
        f"obj1={obj1} | obj2={obj2}",
        fontsize=12,
    )

    panels = [
        (Sb, "patch-patch selfsim base"),
        (Sl, "patch-patch selfsim low α=0.5"),
        (Sd, "patch-patch selfsim diff low-base"),

        (obj1_b, f"obj1 '{obj1}' ↔ patches base"),
        (obj1_l, f"obj1 '{obj1}' ↔ patches low"),
        (obj1_d, f"obj1 diff low-base"),

        (obj2_b, f"obj2 '{obj2}' ↔ patches base"),
        (obj2_l, f"obj2 '{obj2}' ↔ patches low"),
        (obj2_d, f"obj2 diff low-base"),
    ]

    for ax, (arr, title) in zip(axes.flat, panels):
        if "diff" in title:
            vmax = np.percentile(np.abs(arr), 99)
            vmax = max(float(vmax), 1e-6)
            im = ax.imshow(arr, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(normalize_for_view(arr))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    base_json = find_one(
        "BASE fixed w1=w2=1.0",
        [
            f"output/*{DATASET}*adapt_vis*w1_w11_w21_thr0p4_{OPTION}option_{TEST_TAG}.json",
            f"output/*{DATASET}*adapt_vis*w1_w11p0_w21p0_thr0p4_{OPTION}option_{TEST_TAG}.json",
        ],
    )

    mixed_json = find_one(
        "MIXED AdaptVis w1=0.5,w2=1.5",
        [
            f"output/*{DATASET}*adapt_vis*w1_w10p5_w21p5_thr0p4_{OPTION}option_{TEST_TAG}.json",
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
    n = min(len(base), len(mixed))

    print("\n[LOAD]")
    print("base json :", len(base), base_json)
    print("mixed json:", len(mixed), mixed_json)
    print("n         :", n)
    print("base feature dir :", BASE_FEATURE_DIR)
    print("mixed feature dir:", MIXED_FEATURE_DIR)

    groups = OrderedDict([
        ("selected0p5_wrong_to_correct", []),
        ("selected0p5_correct_to_wrong", []),
        ("selected0p5_correct_to_correct", []),
        ("selected0p5_wrong_to_wrong", []),
    ])

    records = []
    missing = []

    for i in range(n):
        sid = sid_of(i, base[i])
        sw = selected_weight(mixed[i])

        if sw is None or abs(sw - 0.5) > 1e-6:
            continue

        b_corr = correct(base[i])
        m_corr = correct(mixed[i])
        gold = norm_gold(base[i].get("Golden", base[i].get("gold", "")))

        if (not b_corr) and m_corr:
            group = "selected0p5_wrong_to_correct"
        elif b_corr and (not m_corr):
            group = "selected0p5_correct_to_wrong"
        elif b_corr and m_corr:
            group = "selected0p5_correct_to_correct"
        else:
            group = "selected0p5_wrong_to_wrong"

        bf = feature_file(BASE_FEATURE_DIR, sid, "base")
        lf = feature_file(MIXED_FEATURE_DIR, sid, "low")

        if bf is None or lf is None:
            missing.append((sid, bf, lf))
            continue

        xb = np.load(bf, allow_pickle=True)
        xl = np.load(lf, allow_pickle=True)

        H_img_b = as_float(xb["image_hidden"])
        H_img_l = as_float(xl["image_hidden"])

        sim_obj1_b = as_float(xb["sim_obj1_to_img"])
        sim_obj1_l = as_float(xl["sim_obj1_to_img"])
        sim_obj2_b = as_float(xb["sim_obj2_to_img"])
        sim_obj2_l = as_float(xl["sim_obj2_to_img"])

        pred_b = pred_relation_from_obj_scores(sim_obj1_b, sim_obj2_b)
        pred_l = pred_relation_from_obj_scores(sim_obj1_l, sim_obj2_l)

        self_rel_fro, self_cos_dist = image_selfsim_delta(H_img_b, H_img_l)

        rec = {
            "sid": sid,
            "group": group,
            "gold": gold,
            "base_correct": b_corr,
            "mixed_correct": m_corr,
            "base_feature_file": bf,
            "low_feature_file": lf,

            "last_prompt_cosdist": cosine_distance_vec(xb["last_prompt_hidden"], xl["last_prompt_hidden"]),
            "obj1_hidden_cosdist": cosine_distance_vec(xb["obj1_hidden"], xl["obj1_hidden"]),
            "obj2_hidden_cosdist": cosine_distance_vec(xb["obj2_hidden"], xl["obj2_hidden"]),
            "image_mean_cosdist": cosine_distance_vec(image_mean_feature(H_img_b), image_mean_feature(H_img_l)),
            "image_patch_mean_cosdist": mean_patch_cosine_distance(H_img_b, H_img_l),

            "obj1_sim_js": js_divergence_from_scores(sim_obj1_b, sim_obj1_l),
            "obj2_sim_js": js_divergence_from_scores(sim_obj2_b, sim_obj2_l),
            "obj1_sim_tv": tv_distance_from_scores(sim_obj1_b, sim_obj1_l),
            "obj2_sim_tv": tv_distance_from_scores(sim_obj2_b, sim_obj2_l),

            "obj1_top5_jaccard": jaccard(topk_idx(sim_obj1_b, 5), topk_idx(sim_obj1_l, 5)),
            "obj2_top5_jaccard": jaccard(topk_idx(sim_obj2_b, 5), topk_idx(sim_obj2_l, 5)),
            "obj1_top10_jaccard": jaccard(topk_idx(sim_obj1_b, 10), topk_idx(sim_obj1_l, 10)),
            "obj2_top10_jaccard": jaccard(topk_idx(sim_obj2_b, 10), topk_idx(sim_obj2_l, 10)),

            "obj1_center_shift": center_shift(sim_obj1_b, sim_obj1_l),
            "obj2_center_shift": center_shift(sim_obj2_b, sim_obj2_l),

            "feature_relation_base_pred": pred_b,
            "feature_relation_low_pred": pred_l,
            "feature_relation_base_correct": relation_correct(pred_b, gold),
            "feature_relation_low_correct": relation_correct(pred_l, gold),

            "image_selfsim_rel_fro_delta": self_rel_fro,
            "image_selfsim_cosdist": self_cos_dist,
        }

        records.append(rec)
        groups[group].append(sid)

    print("\n[SELECTED 0.5 GROUP COUNTS]")
    for k, v in groups.items():
        print(f"{k}: {len(v)}")
        print(v[:120])

    if missing:
        print("\n[MISSING FEATURE FILES]")
        print("missing count:", len(missing))
        for row in missing[:50]:
            print(row)

    metric_names = [
        "last_prompt_cosdist",
        "obj1_hidden_cosdist",
        "obj2_hidden_cosdist",
        "image_mean_cosdist",
        "image_patch_mean_cosdist",

        "obj1_sim_js",
        "obj2_sim_js",
        "obj1_sim_tv",
        "obj2_sim_tv",

        "obj1_top5_jaccard",
        "obj2_top5_jaccard",
        "obj1_top10_jaccard",
        "obj2_top10_jaccard",

        "obj1_center_shift",
        "obj2_center_shift",

        "image_selfsim_rel_fro_delta",
        "image_selfsim_cosdist",
    ]

    summary = {
        "dataset": DATASET,
        "option": OPTION,
        "mixed_json": mixed_json,
        "base_json": base_json,
        "mixed_feature_dir": MIXED_FEATURE_DIR,
        "base_feature_dir": BASE_FEATURE_DIR,
        "group_counts": {k: len(v) for k, v in groups.items()},
        "metrics": {},
    }

    print("\n[SUMMARY BY GROUP]")
    for group_name in groups.keys():
        rs = [r for r in records if r["group"] == group_name]
        print(f"\n===== {group_name} | n={len(rs)} =====")
        summary["metrics"][group_name] = {}

        if not rs:
            continue

        base_rel_acc = sum(r["feature_relation_base_correct"] for r in rs) / len(rs)
        low_rel_acc = sum(r["feature_relation_low_correct"] for r in rs) / len(rs)
        rel_w2c = sum((not r["feature_relation_base_correct"]) and r["feature_relation_low_correct"] for r in rs)
        rel_c2w = sum(r["feature_relation_base_correct"] and (not r["feature_relation_low_correct"]) for r in rs)

        print(f"feature_relation_acc: base={base_rel_acc:.4f} low={low_rel_acc:.4f} delta={low_rel_acc-base_rel_acc:+.4f}")
        print(f"feature_relation wrong->correct: {rel_w2c}")
        print(f"feature_relation correct->wrong: {rel_c2w}")

        summary["metrics"][group_name]["feature_relation_acc_base"] = base_rel_acc
        summary["metrics"][group_name]["feature_relation_acc_low"] = low_rel_acc
        summary["metrics"][group_name]["feature_relation_wrong_to_correct"] = rel_w2c
        summary["metrics"][group_name]["feature_relation_correct_to_wrong"] = rel_c2w

        for m in metric_names:
            vals = [r[m] for r in rs]
            mean_v = safe_mean(vals)
            med_v = safe_median(vals)
            print(f"{m:32s} mean={fmt(mean_v)} median={fmt(med_v)}")
            summary["metrics"][group_name][m] = {
                "mean": mean_v,
                "median": med_v,
            }

    A = [r for r in records if r["group"] == "selected0p5_wrong_to_correct"]
    B = [r for r in records if r["group"] == "selected0p5_correct_to_wrong"]

    print("\n[DIRECT CONTRAST: wrong_to_correct vs correct_to_wrong]")
    print(f"A=wrong_to_correct n={len(A)}")
    print(f"B=correct_to_wrong n={len(B)}")

    summary["direct_contrast_wrong_to_correct_minus_correct_to_wrong"] = {}

    for m in metric_names:
        av = safe_mean([r[m] for r in A])
        bv = safe_mean([r[m] for r in B])
        diff = None if av is None or bv is None else av - bv
        print(f"{m:32s} A_mean={fmt(av)} B_mean={fmt(bv)} A-B={fmt(diff)}")
        summary["direct_contrast_wrong_to_correct_minus_correct_to_wrong"][m] = {
            "wrong_to_correct_mean": av,
            "correct_to_wrong_mean": bv,
            "diff": diff,
        }

    # Save records json.
    records_path = os.path.join(ANALYSIS_DIR, "selected0p5_feature_records.json")
    summary_path = os.path.join(ANALYSIS_DIR, "selected0p5_feature_summary.json")

    serializable_records = []
    for r in records:
        rr = dict(r)
        serializable_records.append(rr)

    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(serializable_records, f, ensure_ascii=False, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SAVED]")
    print(records_path)
    print(summary_path)

    # Random visualizations.
    fig_dir = os.path.join(ANALYSIS_DIR, "figures")
    rng = random.Random(RANDOM_SEED)

    for group_name, rs in [
        ("wrong_to_correct", A),
        ("correct_to_wrong", B),
    ]:
        if not rs:
            continue

        chosen = rs if len(rs) <= NUM_VIS else rng.sample(rs, NUM_VIS)

        for r in chosen:
            xb = np.load(r["base_feature_file"], allow_pickle=True)
            xl = np.load(r["low_feature_file"], allow_pickle=True)

            out_name = f"{group_name}_sid{r['sid']:04d}.png"
            out_path = os.path.join(fig_dir, out_name)
            save_case_figure(r, xb, xl, out_path)
            print("[FIG SAVED]", out_path)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
PY

# ==============================
# 4. Run comparison
# ==============================
echo
echo "======================================"
echo "[RUN 3] Compare selected 0.5 feature changes"
echo "Analysis dir: ${ANALYSIS_DIR}"
echo "======================================"

export DATASET="${DATASET}"
export OPTION="${OPTION}"
export TEST_TAG="True"
export MIXED_FEATURE_DIR="${MIXED_FEATURE_DIR}"
export BASE_FEATURE_DIR="${BASE_FEATURE_DIR}"
export ANALYSIS_DIR="${ANALYSIS_DIR}"

python3 "${STATS_SCRIPT}"

echo
echo "[ALL DONE]"
echo "Summary:"
echo "  ${ANALYSIS_DIR}/selected0p5_feature_summary.json"
echo "Records:"
echo "  ${ANALYSIS_DIR}/selected0p5_feature_records.json"
echo "Figures:"
echo "  ${ANALYSIS_DIR}/figures/"
