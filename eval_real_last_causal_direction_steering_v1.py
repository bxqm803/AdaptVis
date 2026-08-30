#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use the newly learned relation-specific Real-Gray last-token directions to
steer REAL-image generation and test actual accuracy improvement.

Requires the previous script in the repo:
    eval_shared_real_gray_relation_delta_addition_v1.py

Non-oracle experiment:
    1) Use saved middle-layer residual Direction vectors (L14-20 by default)
       to predict a relation on TEST.
    2) Learn new late causal directions s_left/right/above/below from TRAIN
       Real-Gray last-token deltas.
    3) On REAL TEST images, when guide prediction conflicts with baseline
       generation, add the guide-chosen causal direction to last token.
    4) Run actual model.generate() and report W2C / C2W / net accuracy.

Also supports oracle GT selector as a diagnostic upper bound only.
"""

import argparse
import contextlib
import csv
import gc
import importlib
import json
import math
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


RELS = ("left", "right", "above", "below")
RELSET = set(RELS)
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shared-module",
        default="eval_shared_real_gray_relation_delta_addition_v1",
    )
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--direction-key", default="residual")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument(
        "--annotation-json",
        default="data/coco_qa_two_obj.json",
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )
    p.add_argument("--guide-layers", default="14-20")
    p.add_argument("--last-layers", default="25-27")
    p.add_argument(
        "--guide-train-controls",
        default="correct",
        choices=["correct", "all"],
    )
    p.add_argument(
        "--template-train-filter",
        default="real_correct_gray_wrong",
        choices=["real_correct_gray_wrong", "real_correct", "all"],
    )
    p.add_argument(
        "--selectors",
        default="guide,oracle",
        help="guide, oracle, or guide,oracle",
    )
    p.add_argument(
        "--apply-modes",
        default="conflict_only,all",
        help="conflict_only, all, or both",
    )
    p.add_argument(
        "--edit-modes",
        default="add,contrast",
        help="add or contrast. contrast adds s_target - s_baseline_pred.",
    )
    p.add_argument(
        "--windows",
        default="single,multi",
        help="single and/or multi",
    )
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def norm_rel(x):
    s = str(x).strip().lower()
    if re.search(r"\bleft\b", s):
        return "left"
    if re.search(r"\bright\b", s):
        return "right"
    if re.search(r"\babove\b", s) or re.search(r"\bover\b", s):
        return "above"
    if re.search(r"\bbelow\b", s) or re.search(r"\bunder\b", s):
        return "below"
    return s


def parse_layers(spec, n_layers):
    out = []
    for piece in str(spec).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = map(int, piece.split("-", 1))
            out.extend(range(a, b + 1))
        else:
            out.append(int(piece))
    out = sorted(set(out))
    bad = [x for x in out if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(f"bad layers={bad}, valid 0..{n_layers-1}")
    return out


def parse_choices(spec, allowed):
    out = [x.strip() for x in str(spec).split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad:
        raise ValueError(f"bad choices={bad}; allowed={allowed}")
    return out


def safe_mean(xs):
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


# =============================================================================
# OLD middle-layer Direction: only used as non-oracle relation selector
# =============================================================================

def load_direction_bundle(direction_dir, key):
    root = Path(direction_dir)
    with np.load(root / "vectors.npz", allow_pickle=True) as z:
        sids = np.asarray(z["sample_index"], dtype=np.int64)
        labels = np.asarray([norm_rel(x) for x in z["relation"]], dtype=object)
        arr = np.asarray(z[key], dtype=np.float32)

    n = len(sids)
    if arr.shape[0] == n:
        vectors = arr
    elif arr.shape[1] == n:
        vectors = np.transpose(arr, (1, 0, 2))
    else:
        raise RuntimeError(f"cannot align vectors shape={arr.shape} with N={n}")

    split, group = {}, {}
    for r in read_csv(root / "sample_split_and_generation.csv"):
        sid = int(r["sample_index"])
        split[sid] = str(r.get("split", "")).strip().lower()
        group[sid] = str(r.get("generation_group", "")).strip().lower()

    return {
        "sids": sids,
        "labels": labels,
        "vectors": vectors,
        "split": split,
        "group": group,
        "sid_to_idx": {int(s): i for i, s in enumerate(sids.tolist())},
    }


def fit_guide(bundle, layers, controls):
    idx = []
    for i, sid in enumerate(bundle["sids"].tolist()):
        sid = int(sid)
        if bundle["split"].get(sid) != "train":
            continue
        if controls == "correct" and bundle["group"].get(sid) != "correct":
            continue
        idx.append(i)

    codebook = {}
    for l in layers:
        X = bundle["vectors"][idx, l].astype(np.float64)
        Y = bundle["labels"][idx]
        center = X.mean(0)
        protos = {}
        for rel in RELS:
            mu = (X[Y == rel] - center).mean(0)
            mu /= max(np.linalg.norm(mu), EPS)
            protos[rel] = mu.astype(np.float32)
        codebook[l] = {"center": center.astype(np.float32), "protos": protos}
    return codebook


def guide_predict(bundle, codebook, sid, layers):
    i = bundle["sid_to_idx"][sid]
    votes = {r: 0 for r in RELS}
    score_sum = {r: 0.0 for r in RELS}

    for l in layers:
        q = bundle["vectors"][i, l].astype(np.float64)
        q -= codebook[l]["center"].astype(np.float64)
        q /= max(np.linalg.norm(q), EPS)

        scores = {
            rel: float(np.dot(q, codebook[l]["protos"][rel]))
            for rel in RELS
        }
        pred = max(scores, key=scores.get)
        votes[pred] += 1
        for rel in RELS:
            score_sum[rel] += scores[rel]

    max_vote = max(votes.values())
    tied = [r for r in RELS if votes[r] == max_vote]
    pred = max(tied, key=lambda r: score_sum[r])

    return pred, ";".join(f"{r}:{votes[r]}" for r in RELS)


# =============================================================================
# Last-token steering hook
# =============================================================================

class LastSteer:
    def __init__(
        self,
        M,
        decoder_layers,
        templates,
        layers,
        target_rel,
        baseline_rel,
        edit_mode,
        scale,
        last_position,
    ):
        self.M = M
        self.handles = []
        self.done = {}
        self.templates = templates
        self.layers = layers
        self.target_rel = target_rel
        self.baseline_rel = baseline_rel
        self.edit_mode = edit_mode
        self.scale = scale
        self.last_position = last_position

        for l in layers:
            self.done[l] = False
            self.handles.append(
                decoder_layers[l].register_forward_hook(self._hook(l))
            )

    def vec(self, l):
        target = np.asarray(
            self.templates["last"][l]["shared"][self.target_rel],
            dtype=np.float32,
        )

        if self.edit_mode == "add":
            return self.scale * target

        if self.edit_mode == "contrast":
            if self.baseline_rel not in RELSET:
                return self.scale * target
            source = np.asarray(
                self.templates["last"][l]["shared"][self.baseline_rel],
                dtype=np.float32,
            )
            return self.scale * (target - source)

        raise ValueError(self.edit_mode)

    def _hook(self, l):
        def fn(_module, _inputs, output):
            if self.done[l]:
                return output

            h, descriptor = self.M.extract_hidden(output)
            if h.ndim != 3 or self.last_position >= h.shape[1]:
                return output

            v = torch.as_tensor(
                self.vec(l),
                device=h.device,
                dtype=h.dtype,
            )

            y = h.clone()
            y[:, self.last_position, :] += v[None, :]
            self.done[l] = True
            return self.M.replace_hidden(output, descriptor, y)

        return fn

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# =============================================================================
# Evaluation
# =============================================================================

def real_baseline(M, model, processor, records, test_sids, args, device):
    rows = []

    for sid in tqdm(test_sids, desc="REAL baseline"):
        rec = records[sid]
        image = None

        try:
            image = Image.open(rec["image_path"]).convert("RGB")

            batch, ss, rr, last_position = M.prepare_batch_for_image(
                processor,
                image,
                rec,
                args.prompt_template,
                device,
            )

            if ss is None or rr is None:
                continue

            text, pred = M.generate(
                model,
                processor,
                batch,
                args.max_new_tokens,
            )

            rows.append({
                "sid": sid,
                "gt": rec["gt"],
                "pred": pred or "",
                "correct": int(pred == rec["gt"]),
                "last_position": last_position,
                "text": text,
            })

            del batch

        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return rows


def run_condition(
    M,
    model,
    processor,
    decoder_layers,
    templates,
    records,
    baseline,
    guide_preds,
    selected_layers,
    selector,
    apply_mode,
    edit_mode,
    args,
    device,
    name,
):
    rows = []

    for base in tqdm(baseline, desc=name):
        sid = int(base["sid"])
        rec = records[sid]
        base_pred = norm_rel(base["pred"])

        target = (
            guide_preds[sid]
            if selector == "guide"
            else rec["gt"]
        )

        apply_edit = True

        if apply_mode == "conflict_only":
            apply_edit = (
                target in RELSET
                and base_pred in RELSET
                and target != base_pred
            )

        if not apply_edit:
            rows.append({
                "condition": name,
                "sid": sid,
                "gt": rec["gt"],
                "target": target,
                "base_pred": base_pred,
                "edit_pred": base_pred,
                "base_correct": base["correct"],
                "edit_correct": base["correct"],
                "applied": 0,
                "W2C": 0,
                "C2W": 0,
                "changed": 0,
            })
            continue

        image = None

        try:
            image = Image.open(rec["image_path"]).convert("RGB")

            batch, ss, rr, last_position = M.prepare_batch_for_image(
                processor,
                image,
                rec,
                args.prompt_template,
                device,
            )

            with LastSteer(
                M,
                decoder_layers,
                templates,
                selected_layers,
                target,
                base_pred,
                edit_mode,
                args.scale,
                last_position,
            ):
                text, pred = M.generate(
                    model,
                    processor,
                    batch,
                    args.max_new_tokens,
                )

            bc = int(base["correct"])
            ec = int(pred == rec["gt"])

            rows.append({
                "condition": name,
                "sid": sid,
                "gt": rec["gt"],
                "target": target,
                "base_pred": base_pred,
                "edit_pred": pred or "",
                "base_correct": bc,
                "edit_correct": ec,
                "applied": 1,
                "W2C": int(bc == 0 and ec == 1),
                "C2W": int(bc == 1 and ec == 0),
                "changed": int((pred or "") != base_pred),
                "text": text,
            })

            del batch

        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return rows


def summarize(rows, name):
    n = len(rows)
    base_acc = safe_mean(r["base_correct"] for r in rows)
    edit_acc = safe_mean(r["edit_correct"] for r in rows)
    n_wrong = sum(1 - int(r["base_correct"]) for r in rows)
    n_correct = n - n_wrong
    w2c = sum(int(r["W2C"]) for r in rows)
    c2w = sum(int(r["C2W"]) for r in rows)
    applied = sum(int(r["applied"]) for r in rows)

    return {
        "condition": name,
        "N": n,
        "base_acc": base_acc,
        "edit_acc": edit_acc,
        "gain": edit_acc - base_acc,
        "applied": applied,
        "applied_rate": applied / n if n else float("nan"),
        "W2C": w2c,
        "W2C_rate": w2c / n_wrong if n_wrong else float("nan"),
        "C2W": c2w,
        "C2W_rate": c2w / n_correct if n_correct else float("nan"),
        "net": w2c - c2w,
        "changed_rate": safe_mean(r["changed"] for r in rows),
    }


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    selectors = parse_choices(args.selectors, {"guide", "oracle"})
    apply_modes = parse_choices(
        args.apply_modes,
        {"conflict_only", "all"},
    )
    edit_modes = parse_choices(args.edit_modes, {"add", "contrast"})
    windows = parse_choices(args.windows, {"single", "multi"})

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    M = importlib.import_module(args.shared_module)

    direction = load_direction_bundle(
        args.direction_dir,
        args.direction_key,
    )

    split_map = M.load_split(args.direction_dir)

    records = M.load_records(
        args.prompt_jsonl,
        args.annotation_json,
        args.data_root,
        split_map,
    )

    model, processor = M.load_model_and_processor(
        args.model_id,
        M.dtype_from_name(args.dtype),
        args.device,
        args.attn_impl,
    )

    decoder_layers, decoder_path = M.resolve_decoder_layers(model)

    guide_layers = parse_layers(
        args.guide_layers,
        direction["vectors"].shape[1],
    )
    last_layers = parse_layers(
        args.last_layers,
        len(decoder_layers),
    )

    print("[decoder]", decoder_path)
    print("[guide layers]", guide_layers)
    print("[last layers]", last_layers)

    # ------------------------------------------------------------------
    # fit non-oracle relation guide
    # ------------------------------------------------------------------
    guide_codebook = fit_guide(
        direction,
        guide_layers,
        args.guide_train_controls,
    )

    # ------------------------------------------------------------------
    # learn new causal last-token directions from TRAIN Real-Gray delta
    # ------------------------------------------------------------------
    train_sids = sorted(
        sid for sid, rec in records.items()
        if rec["split"] == "train"
    )

    if args.max_train_samples is not None:
        train_sids = train_sids[:args.max_train_samples]

    train_args = argparse.Namespace(
        prompt_template=args.prompt_template,
        gray_value=args.gray_value,
        max_new_tokens=args.max_new_tokens,
        train_filter=args.template_train_filter,
    )

    collected, _ = M.collect_train_deltas(
        model=model,
        processor=processor,
        decoder_layers=decoder_layers,
        records=records,
        train_sids=train_sids,
        pair_layers=[],
        last_layers=last_layers,
        args=train_args,
        device=torch.device(args.device),
        outdir=outdir,
    )

    templates, template_summary = M.fit_shared_templates(collected)

    write_csv(
        outdir / "train_template_summary.csv",
        template_summary,
    )

    # ------------------------------------------------------------------
    # fresh REAL test baseline
    # ------------------------------------------------------------------
    test_sids = sorted(
        sid for sid, rec in records.items()
        if rec["split"] == "test"
    )

    if args.max_test_samples is not None:
        test_sids = test_sids[:args.max_test_samples]

    baseline = real_baseline(
        M,
        model,
        processor,
        records,
        test_sids,
        args,
        torch.device(args.device),
    )

    write_csv(outdir / "baseline.csv", baseline)

    baseline_acc = safe_mean(r["correct"] for r in baseline)

    # ------------------------------------------------------------------
    # guide predictions
    # ------------------------------------------------------------------
    guide_preds = {}
    guide_rows = []

    for r in baseline:
        sid = int(r["sid"])

        pred, votes = guide_predict(
            direction,
            guide_codebook,
            sid,
            guide_layers,
        )

        guide_preds[sid] = pred

        guide_rows.append({
            "sid": sid,
            "gt": records[sid]["gt"],
            "baseline_pred": r["pred"],
            "baseline_correct": r["correct"],
            "guide_pred": pred,
            "guide_correct": int(pred == records[sid]["gt"]),
            "conflict": int(pred != norm_rel(r["pred"])),
            "votes": votes,
        })

    write_csv(outdir / "guide_summary.csv", guide_rows)

    guide_acc = safe_mean(r["guide_correct"] for r in guide_rows)

    conflict = [r for r in guide_rows if int(r["conflict"]) == 1]

    print("\n" + "=" * 120)
    print(
        f"REAL baseline N={len(baseline)} acc={baseline_acc:.4f} | "
        f"guide acc={guide_acc:.4f} | conflicts={len(conflict)}"
    )
    if conflict:
        print(
            "on conflicts | "
            f"baseline correct={safe_mean(r['baseline_correct'] for r in conflict):.4f} | "
            f"guide correct={safe_mean(r['guide_correct'] for r in conflict):.4f}"
        )
    print("=" * 120)

    # ------------------------------------------------------------------
    # steering
    # ------------------------------------------------------------------
    layer_windows = []

    if "single" in windows:
        for l in last_layers:
            layer_windows.append((f"L{l:02d}", [l]))

    if "multi" in windows:
        layer_windows.append(("multi", last_layers))

    details = []
    summaries = []

    for selector in selectors:
        for apply_mode in apply_modes:
            for edit_mode in edit_modes:
                for window_name, selected_layers in layer_windows:

                    name = (
                        f"{selector}_{apply_mode}_{edit_mode}_{window_name}"
                    )

                    rows = run_condition(
                        M,
                        model,
                        processor,
                        decoder_layers,
                        templates,
                        records,
                        baseline,
                        guide_preds,
                        selected_layers,
                        selector,
                        apply_mode,
                        edit_mode,
                        args,
                        torch.device(args.device),
                        name,
                    )

                    details.extend(rows)
                    summaries.append(summarize(rows, name))

                    write_csv(outdir / "steering_details.csv", details)
                    write_csv(outdir / "summary.csv", summaries)

    print("\n" + "=" * 165)
    print(
        "REAL-IMAGE LAST-TOKEN CAUSAL DIRECTION STEERING — ACTUAL model.generate()"
    )
    print("=" * 165)
    print(
        "condition                                      | "
        "acc base->edit gain | applied | W2C/wrong | C2W/correct | net | changed"
    )

    for r in summaries:
        print(
            f"{r['condition']:46s} | "
            f"{r['base_acc']:.4f}->{r['edit_acc']:.4f} "
            f"{r['gain']:+.4f} | "
            f"{r['applied']}/{r['applied_rate']:.3f} | "
            f"{r['W2C']}/{r['W2C_rate']:.3f} | "
            f"{r['C2W']}/{r['C2W_rate']:.3f} | "
            f"{r['net']:+d} | "
            f"{r['changed_rate']:.3f}"
        )

    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "baseline_acc": baseline_acc,
                "guide_acc": guide_acc,
                "guide_layers": guide_layers,
                "last_layers": last_layers,
                "scale": args.scale,
                "note": (
                    "guide selector is the non-oracle experiment; "
                    "oracle selector is only an upper-bound diagnostic."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
