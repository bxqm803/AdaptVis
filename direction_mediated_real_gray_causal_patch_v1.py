#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Direction-Mediated Real-vs-Gray Causal Patch Test

No functions are imported from prior project experiment scripts.

Experiment at selected decoder layers, last prompt token:

    delta_full = h_real - h_gray

Fit a Direction subspace S_l on TRAIN relation vectors:
    default xy:
        x = mu_right - mu_left
        y = mu_above - mu_below

Then decompose the ACTUAL same-sample causal patch:
    delta_dir  = P_S delta_full
    delta_rest = delta_full - delta_dir

Run actual model.generate() under:
    full              : gray + delta_full
    direction_only    : gray + delta_dir
    direction_removed : gray + delta_rest
    random            : gray + matched-norm same-rank random component

Fresh cohort:
    Real correct AND Gray wrong.

Primary paired mediation metrics:
    P(direction_only rescues | full rescues)
    P(direction_removed FAILS | full rescues)

If direction_only ~ random and direction_removed ~ full:
    Direction is likely a readout/tracer rather than the dominant causal mediator.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

RELATIONS = ("left", "right", "above", "below")
REL_SET = set(RELATIONS)
EPS = 1e-10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--direction-key", default="")
    p.add_argument("--list-direction-keys", action="store_true")
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--records-csv", default="")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )
    p.add_argument("--layers", default="25,26,27")
    p.add_argument("--direction-mode", default="xy",
                   choices=["xy", "prototype3"])
    p.add_argument("--train-controls", default="correct",
                   choices=["correct", "all"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--random-seeds", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_float(x):
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def safe_mean(xs):
    vals = []
    for x in xs:
        v = safe_float(x)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(xs):
    vals = []
    for x in xs:
        v = safe_float(x)
        if v is not None:
            vals.append(v)
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0


def norm_rel(x):
    s = str(x).strip().lower()
    m = {
        "left of": "left", "right of": "right",
        "over": "above", "on": "above", "on top of": "above",
        "under": "below", "underneath": "below",
    }
    return m.get(s, s)


def parse_layers(text, n_layers):
    vals = []
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        if "-" in item:
            a, b = map(int, item.split("-", 1))
            step = 1 if b >= a else -1
            vals.extend(range(a, b + step, step))
        else:
            vals.append(int(item))
    vals = sorted(set(vals))
    bad = [x for x in vals if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(f"Invalid layers {bad}; valid 0..{n_layers-1}")
    return vals


def load_direction_metadata(direction_dir: Path):
    vp = direction_dir / "vectors.npz"
    gp = direction_dir / "sample_split_and_generation.csv"
    if not vp.exists():
        raise FileNotFoundError(vp)
    if not gp.exists():
        raise FileNotFoundError(gp)
    with np.load(vp, allow_pickle=True) as z:
        sids = np.asarray(z["sample_index"], dtype=np.int64)
        labels = np.asarray([norm_rel(x) for x in z["relation"]], dtype=object)
    split, group = {}, {}
    for r in read_csv(gp):
        sid = int(r["sample_index"])
        split[sid] = str(r.get("split", "")).strip().lower()
        group[sid] = str(r.get("generation_group", "")).strip().lower()
    return {
        "vectors_path": vp,
        "sids": sids,
        "labels": labels,
        "gt": {int(s): str(labels[i]) for i, s in enumerate(sids.tolist())},
        "split": split,
        "cached_group": group,
    }


def inspect_npz(path):
    with np.load(path, allow_pickle=True) as z:
        for k in z.files:
            a = np.asarray(z[k])
            print(f"{k:40s} shape={str(a.shape):24s} dtype={a.dtype}")


def normalize_direction_array(arr, n_samples, key):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        if arr.shape[0] != n_samples:
            raise ValueError(f"{key}: shape={arr.shape}, N={n_samples}")
        return arr[:, None, :].astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{key}: need [N,L,D] or [L,N,D], got {arr.shape}")
    if arr.shape[0] == n_samples:
        return arr.astype(np.float32)
    if arr.shape[1] == n_samples:
        return np.transpose(arr, (1, 0, 2)).astype(np.float32)
    raise ValueError(f"{key}: cannot align N={n_samples} with {arr.shape}")


def choose_direction_array(path, n_samples, requested_key):
    with np.load(path, allow_pickle=True) as z:
        if requested_key:
            if requested_key not in z.files:
                raise KeyError(f"{requested_key} not found; keys={z.files}")
            return normalize_direction_array(z[requested_key], n_samples, requested_key), requested_key

        cands = []
        for key in z.files:
            arr = np.asarray(z[key])
            if not np.issubdtype(arr.dtype, np.floating):
                continue
            try:
                a = normalize_direction_array(arr, n_samples, key)
            except Exception:
                continue
            name = key.lower()
            score = 0
            for tok, w in [
                ("residual", 12), ("direction", 10), ("r_res", 10),
                ("correct_noimage", 8), ("noimage", 4),
                ("relation", 4), ("vector", 4), ("vec", 2)
            ]:
                if tok in name:
                    score += w
            for tok, w in [
                ("logit", -10), ("prob", -10), ("centroid", -8),
                ("attention", -6)
            ]:
                if tok in name:
                    score += w
            cands.append((score, key, a, arr.shape))
        if not cands:
            raise RuntimeError(
                "No layerwise float Direction array auto-detected. "
                "Use --list-direction-keys then --direction-key."
            )
        cands.sort(key=lambda x: (x[0], x[2].shape[-1]), reverse=True)
        print("\nDirection candidates:")
        for sc, k, a, raw in cands[:10]:
            print(f"  score={sc:3d} {k:35s} raw={raw} -> {a.shape}")
        sc, key, a, raw = cands[0]
        print(f"\n[direction] selected key={key!r}, shape={a.shape}")
        return a, key


def orth_span(cols):
    M = np.stack([np.asarray(x, dtype=np.float64) for x in cols], axis=1)
    Q, R = np.linalg.qr(M, mode="reduced")
    keep = np.abs(np.diag(R)) > 1e-8
    return Q[:, keep].astype(np.float32)


def fit_direction_bases(vectors, meta, layers, controls, mode):
    idxs = []
    for i, sid in enumerate(meta["sids"].tolist()):
        sid = int(sid)
        if meta["split"].get(sid) != "train":
            continue
        if meta["labels"][i] not in REL_SET:
            continue
        if controls == "correct" and meta["cached_group"].get(sid) != "correct":
            continue
        idxs.append(i)
    if not idxs:
        raise RuntimeError("No TRAIN Direction vectors.")

    bases, rows = {}, []
    for li in layers:
        if li >= vectors.shape[1]:
            raise RuntimeError(
                f"Direction array has {vectors.shape[1]} layers; requested L{li}"
            )
        X = vectors[idxs, li].astype(np.float64)
        Y = meta["labels"][idxs]
        means = {}
        for rel in RELATIONS:
            mask = Y == rel
            if not np.any(mask):
                raise RuntimeError(f"No train relation {rel} at L{li}")
            means[rel] = X[mask].mean(0)
        if mode == "xy":
            basis = orth_span([
                means["right"] - means["left"],
                means["above"] - means["below"],
            ])
        else:
            g = np.stack([means[r] for r in RELATIONS]).mean(0)
            M = np.stack([means[r] - g for r in RELATIONS])
            _, S, Vh = np.linalg.svd(M, full_matrices=False)
            rank = min(3, int(np.sum(S > 1e-8 * max(float(S.max()), 1.0))))
            basis = Vh[:rank].T.astype(np.float32)
        if basis.shape[1] == 0:
            raise RuntimeError(f"Degenerate Direction basis at L{li}")
        bases[li] = basis
        rows.append({
            "layer": li,
            "rank": int(basis.shape[1]),
            "n_train": len(idxs),
            "norm_R_minus_L": float(np.linalg.norm(means["right"] - means["left"])),
            "norm_A_minus_B": float(np.linalg.norm(means["above"] - means["below"])),
        })
    return bases, rows


SID_KEYS = ("sample_index", "sid", "id", "index")
IMAGE_KEYS = ("image_path", "image", "filename", "file_name", "img_path")
SUBJ_KEYS = ("subject", "subj", "object1", "obj1", "subject_name")
REF_KEYS = ("reference", "ref", "object2", "obj2", "reference_name")


def first_present(row, keys):
    lower = {str(k).lower(): v for k, v in row.items()}
    for k in keys:
        if k.lower() in lower and str(lower[k.lower()]).strip() != "":
            return lower[k.lower()]
    return None


def parse_record(row, source_dir, data_root):
    sid = first_present(row, SID_KEYS)
    image = first_present(row, IMAGE_KEYS)
    subj = first_present(row, SUBJ_KEYS)
    ref = first_present(row, REF_KEYS)
    if sid is None or image is None or subj is None or ref is None:
        return None
    try:
        sid = int(float(str(sid)))
    except Exception:
        return None
    p = Path(str(image))
    if not p.is_absolute():
        choices = [source_dir / p, data_root / p, p]
        p = next((q for q in choices if q.exists()), source_dir / p)
    return {
        "sid": sid,
        "image_path": str(p),
        "subject": str(subj).strip(),
        "reference": str(ref).strip(),
    }


def load_records_file(path, data_root):
    records = {}
    if path.suffix.lower() == ".csv":
        rows = read_csv(path)
    elif path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    else:
        raise ValueError("records file must be CSV or JSONL")
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        x = parse_record(r, path.parent, data_root)
        if x:
            records[x["sid"]] = x
    return records


def discover_records(explicit, data_root, dataset, target_sids):
    if explicit:
        p = Path(explicit)
        return load_records_file(p, data_root), p
    roots = []
    dp = data_root / dataset
    if dp.exists():
        roots.append(dp)
    roots.append(data_root)
    candidates, seen = [], set()
    for root in roots:
        if not root.exists():
            continue
        paths = []
        if root.is_file():
            paths = [root]
        else:
            paths.extend(root.rglob("*.csv"))
            paths.extend(root.rglob("*.jsonl"))
        for p in paths:
            if p in seen:
                continue
            seen.add(p)
            try:
                recs = load_records_file(p, data_root)
            except Exception:
                continue
            overlap = len(set(recs) & target_sids)
            if overlap:
                candidates.append((overlap, len(recs), p, recs))
    if not candidates:
        raise RuntimeError(
            "Could not auto-find records. Pass --records-csv with fields "
            "sample_index,image_path,subject,reference."
        )
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print("\nRecord candidates:")
    for ov, n, p, _ in candidates[:10]:
        print(f"  overlap={ov:4d} records={n:4d} {p}")
    ov, n, p, recs = candidates[0]
    print(f"\n[records] selected {p}, overlap={ov}")
    return recs, p


def dtype_from_name(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model(model_id, dtype, device, attn_impl):
    names = [
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ]
    cls = next((getattr(transformers, n) for n in names if hasattr(transformers, n)), None)
    if cls is None:
        raise RuntimeError("No supported multimodal generation class in transformers.")
    kw = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {"": device},
    }
    if attn_impl != "none":
        kw["attn_implementation"] = attn_impl
    print(f"[model] class={cls.__name__} id={model_id}")
    try:
        model = cls.from_pretrained(model_id, dtype=dtype, **kw)
    except TypeError:
        model = cls.from_pretrained(model_id, torch_dtype=dtype, **kw)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def get_attr(obj, path):
    cur = obj
    for p in path.split("."):
        cur = getattr(cur, p)
    return cur


def resolve_layers(model):
    for path in [
        "model.language_model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
        "language_model.model.layers",
    ]:
        try:
            x = get_attr(model, path)
            if len(x):
                return x, path
        except Exception:
            pass
    raise RuntimeError("Could not resolve decoder layers.")


def make_gray(image, value):
    v = int(np.clip(value, 0, 255))
    return Image.new("RGB", image.size, color=(v, v, v))


def build_prompt(processor, question, image):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return question


def build_batch(processor, question, image, device):
    prompt = build_prompt(processor, question, image)
    err = None
    for fn in [
        lambda: processor(
            text=[prompt], images=[image], padding=True, return_tensors="pt"
        ),
        lambda: processor(
            text=prompt, images=image, return_tensors="pt"
        ),
    ]:
        try:
            batch = fn()
            break
        except Exception as e:
            err = e
    else:
        raise RuntimeError(f"processor failed: {err}")
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def last_pos(batch):
    if "attention_mask" in batch:
        nz = torch.nonzero(batch["attention_mask"][0], as_tuple=False).flatten()
        if len(nz):
            return int(nz[-1].item())
    return int(batch["input_ids"].shape[1] - 1)


def parse_pred(text):
    s = str(text).strip().lower()
    hits = []
    for rel, pat in [
        ("left", r"\bleft\b"), ("right", r"\bright\b"),
        ("above", r"\babove\b"), ("below", r"\bbelow\b"),
        ("above", r"\bon top of\b"),
        ("below", r"\bunder(?:neath)?\b"),
    ]:
        m = re.search(pat, s)
        if m:
            hits.append((m.start(), rel))
    return sorted(hits)[0][1] if hits else None


def generate(model, processor, batch, max_new_tokens):
    n = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        out = model.generate(
            **batch, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True
        )
    text = processor.tokenizer.decode(out[0, n:], skip_special_tokens=True).strip()
    pred = parse_pred(text)
    del out
    return text, pred


def first_tensor(args):
    for x in args:
        if torch.is_tensor(x):
            return x
    raise RuntimeError("No tensor hook input.")


class Capture:
    def __init__(self, layers, selected, pos):
        self.pos = pos
        self.states = {}
        self.handles = []
        for li in selected:
            self.handles.append(
                layers[li].register_forward_pre_hook(self.make(li))
            )

    def make(self, li):
        def hook(_m, args):
            if li in self.states:
                return None
            x = first_tensor(args)
            if x.ndim == 3 and self.pos < x.shape[1]:
                self.states[li] = (
                    x[0, self.pos].detach().float().cpu().numpy().astype(np.float32)
                )
            return None
        return hook

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def capture_states(model, layers, selected, batch):
    p = last_pos(batch)
    with Capture(layers, selected, p) as cap:
        with torch.inference_mode():
            _ = model(
                **batch,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
    miss = [li for li in selected if li not in cap.states]
    if miss:
        raise RuntimeError(f"Missing captures: {miss}")
    return cap.states, p


class AddDelta:
    def __init__(self, block, pos, delta):
        self.pos = pos
        self.delta = np.asarray(delta, dtype=np.float32)
        self.applied = False
        self.handle = block.register_forward_pre_hook(self.hook)

    def hook(self, _m, args):
        if self.applied:
            return None
        vals = list(args)
        idx, x = next(
            ((i, a) for i, a in enumerate(vals) if torch.is_tensor(a)),
            (None, None),
        )
        if x is None or x.ndim != 3 or self.pos >= x.shape[1]:
            return None
        y = x.clone()
        d = torch.as_tensor(self.delta, device=x.device, dtype=x.dtype)
        y[0, self.pos] = y[0, self.pos] + d
        vals[idx] = y
        self.applied = True
        return tuple(vals)

    def close(self):
        with contextlib.suppress(Exception):
            self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def project(v, B):
    x = np.asarray(v, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    return (B @ (B.T @ x)).astype(np.float32)


def random_orth_basis(dim, rank, D, seed):
    rng = np.random.default_rng(seed)
    D = np.asarray(D, dtype=np.float64)
    cols = []
    for _ in range(rank):
        for _trial in range(200):
            v = rng.standard_normal(dim)
            v = v - D @ (D.T @ v)
            if cols:
                C = np.stack(cols, axis=1)
                v = v - C @ (C.T @ v)
            n = np.linalg.norm(v)
            if n > 1e-8:
                cols.append(v / n)
                break
        else:
            raise RuntimeError("random basis construction failed")
    return np.stack(cols, axis=1).astype(np.float32)


def match_norm(v, target, B, seed):
    x = np.asarray(v, dtype=np.float32)
    if target <= EPS:
        return np.zeros_like(x)
    n = float(np.linalg.norm(x))
    if n > EPS:
        return (x * target / n).astype(np.float32)
    rng = np.random.default_rng(seed)
    y = B @ rng.standard_normal(B.shape[1])
    yn = np.linalg.norm(y)
    return (y / yn * target).astype(np.float32) if yn > EPS else np.zeros_like(x)


def select_test(meta, records, max_samples, seed):
    sids = [
        int(s)
        for s in meta["sids"].tolist()
        if meta["split"].get(int(s)) == "test"
        and int(s) in records
        and meta["gt"].get(int(s)) in REL_SET
    ]
    if max_samples is not None and len(sids) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(sids)
        sids = sids[:max_samples]
    return sorted(sids)


def summarize(baselines, patches, layers):
    rows = []
    for li in layers:
        rr = [r for r in patches if int(r["layer"]) == li]
        maps = {}
        for mode in ["full", "direction_only", "direction_removed"]:
            maps[mode] = {
                int(r["sid"]): int(r["recovered"])
                for r in rr if r["mode"] == mode
            }
        sids = sorted(
            set(maps["full"]) & set(maps["direction_only"]) & set(maps["direction_removed"])
        )
        full = safe_mean(maps["full"][s] for s in sids)
        direct = safe_mean(maps["direction_only"][s] for s in sids)
        removed = safe_mean(maps["direction_removed"][s] for s in sids)

        full_sids = {s for s in sids if maps["full"][s] == 1}
        dir_full = {s for s in full_sids if maps["direction_only"][s] == 1}
        rem_fail = {s for s in full_sids if maps["direction_removed"][s] == 0}

        random_rows = [r for r in rr if r["mode"] == "random"]
        seeds = sorted(set(int(r["random_seed"]) for r in random_rows))
        random_rates = []
        for seed in seeds:
            rseed = [r for r in random_rows if int(r["random_seed"]) == seed]
            random_rates.append(safe_mean(r["recovered"] for r in rseed))

        rows.append({
            "layer": li,
            "n_recoverable": len(sids),
            "full_recovery_rate": full,
            "direction_only_recovery_rate": direct,
            "direction_removed_recovery_rate": removed,
            "random_recovery_mean": safe_mean(random_rates),
            "random_recovery_std": safe_std(random_rates),
            "n_full_rescued": len(full_sids),
            "direction_given_full": (
                len(dir_full) / len(full_sids) if full_sids else float("nan")
            ),
            "n_direction_and_full": len(dir_full),
            "removed_fails_given_full": (
                len(rem_fail) / len(full_sids) if full_sids else float("nan")
            ),
            "n_removed_fails_among_full": len(rem_fail),
            "cmr_rate": (
                (full - removed) / full
                if math.isfinite(full) and full > EPS and math.isfinite(removed)
                else float("nan")
            ),
            "mean_direction_norm_fraction": safe_mean(
                r["direction_norm_fraction"] for r in rr if r["mode"] == "full"
            ),
        })
    return rows


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    meta = load_direction_metadata(Path(args.direction_dir))
    if args.list_direction_keys:
        inspect_npz(meta["vectors_path"])
        return

    direction_vectors, direction_key = choose_direction_array(
        meta["vectors_path"], len(meta["sids"]), args.direction_key
    )

    records, records_source = discover_records(
        args.records_csv,
        Path(args.data_root),
        args.dataset,
        set(int(x) for x in meta["sids"].tolist()),
    )

    model, processor = load_model(
        args.model_id,
        dtype_from_name(args.dtype),
        args.device,
        args.attn_impl,
    )
    decoder_layers, layer_path = resolve_layers(model)
    layers = parse_layers(args.layers, len(decoder_layers))

    bases, basis_rows = fit_direction_bases(
        direction_vectors,
        meta,
        layers,
        args.train_controls,
        args.direction_mode,
    )

    print(f"[decoder] {layer_path}, selected={layers}")
    print("[direction bases]")
    for r in basis_rows:
        print(
            f"  L{r['layer']:02d}: rank={r['rank']}, Ntrain={r['n_train']}, "
            f"||R-L||={r['norm_R_minus_L']:.3f}, ||A-B||={r['norm_A_minus_B']:.3f}"
        )

    out = Path(args.output_dir)
    if args.overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "direction_basis_summary.csv", basis_rows)

    device = torch.device(args.device)
    test_sids = select_test(meta, records, args.max_samples, args.seed)
    baselines, patches, errors = [], [], []
    random_basis_cache = {}

    for sample_i, sid in enumerate(
        tqdm(test_sids, desc="Direction-mediated Real-vs-Gray")
    ):
        rec = records[sid]
        gt = meta["gt"][sid]
        real_img = gray_img = None

        try:
            real_img = Image.open(rec["image_path"]).convert("RGB")
            gray_img = make_gray(real_img, args.gray_value)

            q = args.prompt_template.format(
                subject=rec["subject"],
                reference=rec["reference"],
            )
            real_batch = build_batch(processor, q, real_img, device)
            gray_batch = build_batch(processor, q, gray_img, device)

            rp = last_pos(real_batch)
            gp = last_pos(gray_batch)
            if rp != gp:
                raise RuntimeError(f"Real/Gray last position mismatch {rp} vs {gp}")

            real_text, real_pred = generate(
                model, processor, real_batch, args.max_new_tokens
            )
            gray_text, gray_pred = generate(
                model, processor, gray_batch, args.max_new_tokens
            )

            recoverable = int(real_pred == gt and gray_pred != gt)

            baselines.append({
                "sid": sid,
                "gt": gt,
                "real_pred": real_pred or "",
                "gray_pred": gray_pred or "",
                "real_correct": int(real_pred == gt),
                "gray_correct": int(gray_pred == gt),
                "recoverable": recoverable,
            })

            if not recoverable:
                del real_batch, gray_batch
                continue

            real_states, _ = capture_states(
                model, decoder_layers, layers, real_batch
            )
            gray_states, _ = capture_states(
                model, decoder_layers, layers, gray_batch
            )

            for li in layers:
                full_delta = (real_states[li] - gray_states[li]).astype(np.float32)
                B = bases[li]
                dir_delta = project(full_delta, B)
                rest_delta = (full_delta - dir_delta).astype(np.float32)

                full_norm = float(np.linalg.norm(full_delta))
                dir_norm = float(np.linalg.norm(dir_delta))
                frac = dir_norm / full_norm if full_norm > EPS else 0.0

                for mode, delta in [
                    ("full", full_delta),
                    ("direction_only", dir_delta),
                    ("direction_removed", rest_delta),
                ]:
                    with AddDelta(decoder_layers[li], gp, delta):
                        text, pred = generate(
                            model, processor, gray_batch, args.max_new_tokens
                        )

                    patches.append({
                        "sid": sid,
                        "layer": li,
                        "mode": mode,
                        "random_seed": "",
                        "gt": gt,
                        "real_pred": real_pred or "",
                        "gray_pred": gray_pred or "",
                        "edited_pred": pred or "",
                        "edited_text": text,
                        "recovered": int(pred == gt),
                        "full_norm": full_norm,
                        "direction_norm": dir_norm,
                        "direction_norm_fraction": frac,
                        "edit_norm": float(np.linalg.norm(delta)),
                    })

                for rseed in range(args.random_seeds):
                    key = (li, rseed)
                    if key not in random_basis_cache:
                        random_basis_cache[key] = random_orth_basis(
                            len(full_delta),
                            B.shape[1],
                            B,
                            args.seed + li * 1009 + rseed * 1000003,
                        )

                    RB = random_basis_cache[key]
                    raw = project(full_delta, RB)
                    rnd = match_norm(
                        raw,
                        dir_norm,
                        RB,
                        args.seed + sid * 10000019 + li * 1009 + rseed,
                    )

                    with AddDelta(decoder_layers[li], gp, rnd):
                        text, pred = generate(
                            model, processor, gray_batch, args.max_new_tokens
                        )

                    patches.append({
                        "sid": sid,
                        "layer": li,
                        "mode": "random",
                        "random_seed": rseed,
                        "gt": gt,
                        "real_pred": real_pred or "",
                        "gray_pred": gray_pred or "",
                        "edited_pred": pred or "",
                        "edited_text": text,
                        "recovered": int(pred == gt),
                        "full_norm": full_norm,
                        "direction_norm": dir_norm,
                        "direction_norm_fraction": frac,
                        "edit_norm": float(np.linalg.norm(rnd)),
                    })

            del real_batch, gray_batch, real_states, gray_states

            if args.save_every > 0 and (sample_i + 1) % args.save_every == 0:
                write_csv(out / "baseline_per_sample.csv", baselines)
                write_csv(out / "patch_per_sample.csv", patches)

        except Exception as e:
            errors.append({
                "sid": sid,
                "error_type": type(e).__name__,
                "error": str(e),
            })
            tqdm.write(f"[ERROR sid={sid}] {type(e).__name__}: {e}")

        finally:
            if real_img is not None:
                real_img.close()
            if gray_img is not None:
                gray_img.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(out / "baseline_per_sample.csv", baselines)
    write_csv(out / "patch_per_sample.csv", patches)
    write_csv(out / "errors.csv", errors)

    summary_rows = summarize(baselines, patches, layers)
    write_csv(out / "mediation_summary.csv", summary_rows)

    print("\n" + "=" * 185)
    print("DIRECTION-MEDIATED REAL-vs-GRAY CAUSAL PATCH — ACTUAL model.generate()")
    print("=" * 185)
    print(
        f"TEST N={len(baselines)} | "
        f"fresh Real-correct/Gray-wrong="
        f"{sum(int(r['recoverable']) for r in baselines)}"
    )
    print(
        "\nlayer | fullRec dirOnly removed random(mean±sd) | "
        "P(dir rescues|full) | P(removed FAILS|full) | CMR_rate | ||dir||/||full||"
    )

    for r in summary_rows:
        print(
            f"L{int(r['layer']):02d} | "
            f"{r['full_recovery_rate']:.3f} "
            f"{r['direction_only_recovery_rate']:.3f} "
            f"{r['direction_removed_recovery_rate']:.3f} "
            f"{r['random_recovery_mean']:.3f}±{r['random_recovery_std']:.3f} | "
            f"{r['direction_given_full']:.3f} "
            f"({r['n_direction_and_full']}/{r['n_full_rescued']}) | "
            f"{r['removed_fails_given_full']:.3f} "
            f"({r['n_removed_fails_among_full']}/{r['n_full_rescued']}) | "
            f"{r['cmr_rate']:+.3f} | "
            f"{r['mean_direction_norm_fraction']:.3f}"
        )

    (out / "summary.json").write_text(
        json.dumps({
            "direction_key": direction_key,
            "direction_mode": args.direction_mode,
            "layers": layers,
            "records_source": str(records_source),
            "standalone": True,
            "reused_previous_script_functions": False,
            "primary_metrics": [
                "actual generate recovery",
                "P(direction_only rescues | full rescues)",
                "P(direction_removed fails | full rescues)",
            ],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:")
    for p in [
        out / "direction_basis_summary.csv",
        out / "baseline_per_sample.csv",
        out / "patch_per_sample.csv",
        out / "mediation_summary.csv",
        out / "errors.csv",
        out / "summary.json",
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
