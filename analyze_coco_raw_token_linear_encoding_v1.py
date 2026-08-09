#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear encoding model for RAW A / B / last-token states on COCO-two.

Goal
----
Do NOT alter the prompt and do NOT alter the image.  Keep the original question:

    Where is the {subject} in relation to the {reference}?
    Answer with left, right, above, or below.

At one fixed layer L* (or choose L* once from raw h_A-h_B direction ACC),
ask how much of each raw token state can be linearly explained by ground-truth
image geometry.

Targets (each studied separately)
---------------------------------
    A_raw      = h_A(img)
    B_raw      = h_B(img)
    last_raw   = h_last(img)

and the image-conditioned versions

    A_res      = h_A(img)    - h_A(noimg)
    B_res      = h_B(img)    - h_B(noimg)
    last_res   = h_last(img) - h_last(noimg)

Geometry basis
--------------
Use a non-redundant linear basis:

    pair_center = [(x_A+x_B)/2, (y_A+y_B)/2]
    relative    = [x_A-x_B, y_A-y_B]
    size_A      = [w_A, h_A]
    size_B      = [w_B, h_B]

All bbox quantities are normalized by image width/height.
This avoids the exact collinearity that would occur if we simultaneously used
x_A, x_B and dx=x_A-x_B as separate regressors.

For each split, fit a multi-output ridge encoding model:

    h_i - mean_train(h)  ~=  X_i W

where X contains the standardized geometry variables above.

Main metrics
------------
1) held-out centered cosine(predicted hidden change, true hidden change)
2) held-out R^2 in hidden space
3) held-out relative reconstruction error
4) per-group additive component vectors:
       C_pair_center + C_relative + C_size_A + C_size_B = predicted hidden change
5) drop-one-group delta R^2
6) Shapley allocation of held-out R^2 across the four geometry groups
   (4 groups => only 16 subset fits per split)
7) matched-minus-shuffled cosine for each predicted component

Interpretation
--------------
This is a CORRELATIONAL encoding decomposition on natural COCO images.
It does not establish causal position codes because object identity and geometry
can be correlated in COCO.  Controlled counterfactual image interventions are
still needed for causal claims.

BBox resolution
---------------
The script tries, in order:
  1) bbox-like fields directly present on base.Record
  2) optional --pair-metadata JSON/JSONL/CSV supplied by the user
  3) standard COCO instances annotations under --data-root, resolving only
     unambiguous category-instance pairs (or a unique pair consistent with the
     relation label)

Unresolved samples are skipped and written to bbox_audit.json.

State reuse
-----------
If you already ran analyze_coco_raw_token_prompt_readout_v1.py, reuse its states:

    --state-dir output/coco_raw_token_prompt_readout/qwen-3b/states

Then no model loading / forward pass is needed.

This script imports:
    extract_two_object_relation_states as base
Run from the AdaptVis repo root (or otherwise make that module importable).
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import random
import re
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base

EPS = 1e-12
RELATIONS = ("left", "right", "above", "below")
REL_ALIASES = {
    "left": "left", "right": "right", "above": "above", "below": "below",
    "on": "above", "top": "above", "under": "below", "underneath": "below", "bottom": "below",
}
SLOTS = ("A", "B", "last")
MODES = ("raw", "residual")
GROUPS = ("pair_center", "relative", "size_A", "size_B")
FEATURES = (
    "pair_cx", "pair_cy",
    "dx", "dy",
    "wA", "hA",
    "wB", "hB",
)
GROUP_IDXS = {
    "pair_center": (0, 1),
    "relative": (2, 3),
    "size_A": (4, 5),
    "size_B": (6, 7),
}


def norm_relation(x: Any) -> str:
    k = str(x).strip().lower().replace("-", "_")
    return REL_ALIASES.get(k, k)


def raw_prompt(subject: str, reference: str) -> str:
    return (
        f"Where is the {subject} in relation to the {reference}? "
        "Answer with left, right, above, or below."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="coco_two", choices=["coco_two"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", required=True, choices=sorted(base.SPECS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", choices=["sdpa", "eager", "flash_attention_2", "none"], default="sdpa")
    p.add_argument("--fixed-layer", type=int, default=None)
    p.add_argument("--train-ratio", type=float, default=0.15)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ridge", type=float, default=1e-3,
                   help="Relative ridge strength; effective lambda = ridge * trace(X'X)/p")
    p.add_argument("--shuffle-repeats", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--state-dir", default=None,
                   help="Directory containing raw__correct__all_layers.npz and raw__no_image__all_layers.npz")
    p.add_argument("--pair-metadata", default=None,
                   help="Optional JSON/JSONL/CSV with sample bboxes. sid matching is preferred.")
    p.add_argument("--coco-instances", default=None,
                   help="Optional path to instances_val2017.json. Auto-detected if omitted.")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def atomic_save_npz(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def load_npz(path: Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def build_chat_prompt(processor: Any, text: str, *, with_image: bool) -> str:
    content: List[Dict[str, Any]] = []
    if with_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "apply_chat_template"):
        return processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def process_inputs(processor: Any, rendered: str, image: Optional[Image.Image], device: torch.device) -> Dict[str, Any]:
    if image is None:
        try:
            batch = processor(text=rendered, return_tensors="pt")
        except Exception:
            batch = processor(text=[rendered], return_tensors="pt")
    else:
        try:
            batch = processor(text=rendered, images=image, return_tensors="pt")
        except Exception:
            batch = processor(text=[rendered], images=[image], return_tensors="pt")
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _stack_all_layers(states: Sequence[torch.Tensor], token_index: int, dtype_np: np.dtype) -> np.ndarray:
    return np.stack([
        states[k + 1][0, token_index].detach().float().cpu().numpy()
        for k in range(len(states) - 1)
    ], axis=0).astype(dtype_np)


def extract_raw_all_layers(
    *, args: argparse.Namespace, model: Any, processor: Any, device: torch.device,
    records: Sequence[base.Record], with_image: bool, out_path: Path,
) -> None:
    mode = "correct" if with_image else "no_image"
    if out_path.exists() and not args.overwrite:
        print(f"[reuse] {out_path}")
        return
    if out_path.exists():
        out_path.unlink()

    dtype_np = np.float32 if args.keep_fp32 else np.float16
    sids: List[int] = []
    image_ids: List[str] = []
    subjects: List[str] = []
    references: List[str] = []
    labels: List[str] = []
    apos: List[int] = []
    bpos: List[int] = []
    lastpos: List[int] = []
    Avecs: List[np.ndarray] = []
    Bvecs: List[np.ndarray] = []
    Lvecs: List[np.ndarray] = []
    errors: List[Dict[str, Any]] = []
    blocks_n: Optional[int] = None
    hidden_size: Optional[int] = None

    def save_progress() -> None:
        if not Avecs or blocks_n is None or hidden_size is None:
            return
        atomic_save_npz(out_path, {
            "metadata_json": np.array(json.dumps({
                "model": args.model, "prompt_type": "raw", "vision_mode": mode,
                "prompt_template": raw_prompt("{subject}", "{reference}"),
                "decoder_blocks": blocks_n, "hidden_size": hidden_size,
            }), dtype=object),
            "sample_index": np.asarray(sids, dtype=np.int64),
            "image_id": np.asarray(image_ids, dtype=object),
            "subject": np.asarray(subjects, dtype=object),
            "reference": np.asarray(references, dtype=object),
            "relation": np.asarray(labels, dtype=object),
            "A_position": np.asarray(apos, dtype=np.int64),
            "B_position": np.asarray(bpos, dtype=np.int64),
            "last_position": np.asarray(lastpos, dtype=np.int64),
            "decoder_block_index": np.arange(blocks_n, dtype=np.int32),
            "A_vectors": np.stack(Avecs).astype(dtype_np),
            "B_vectors": np.stack(Bvecs).astype(dtype_np),
            "last_vectors": np.stack(Lvecs).astype(dtype_np),
        })

    desc = f"{args.model}:raw:{mode}:all-layers"
    for rec in tqdm(records, desc=desc, dynamic_ncols=True):
        try:
            image = Image.open(rec.image_path).convert("RGB") if with_image else None
            rendered = build_chat_prompt(processor, raw_prompt(rec.subject, rec.reference), with_image=with_image)
            batch = process_inputs(processor, rendered, image, device)
            input_ids = batch["input_ids"][0].detach().cpu().tolist()
            aidx = base.find_phrase_last_token(processor.tokenizer, input_ids, rec.subject)
            bidx = base.find_phrase_last_token(processor.tokenizer, input_ids, rec.reference)
            if aidx == bidx:
                raise RuntimeError("subject/reference token positions collide")
            lidx = int(batch["attention_mask"][0].sum().item()) - 1 if "attention_mask" in batch else len(input_ids) - 1
            with torch.inference_mode():
                outputs = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
                states = base.hidden_tuple(outputs)
            cur_blocks = len(states) - 1
            if blocks_n is None:
                blocks_n = cur_blocks
                hidden_size = int(states[-1].shape[-1])
                print(f"[{desc}] decoder_blocks={blocks_n}, hidden={hidden_size}")
            Avecs.append(_stack_all_layers(states, int(aidx), dtype_np))
            Bvecs.append(_stack_all_layers(states, int(bidx), dtype_np))
            Lvecs.append(_stack_all_layers(states, int(lidx), dtype_np))
            sids.append(int(rec.sid)); image_ids.append(str(rec.image_id))
            subjects.append(str(rec.subject)); references.append(str(rec.reference))
            labels.append(norm_relation(rec.relation)); apos.append(int(aidx)); bpos.append(int(bidx)); lastpos.append(int(lidx))
            if len(Avecs) % args.save_every == 0:
                save_progress()
            del outputs, states, batch
        except Exception as exc:
            errors.append({
                "sid": int(rec.sid), "image_id": str(rec.image_id), "mode": mode,
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            })
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    save_progress()
    out_path.with_suffix(".errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out_path} | n={len(Avecs)}/{len(records)} | errors={len(errors)}")


def align_raw(correct: Mapping[str, Any], noimg: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[int, int]]:
    cids = np.asarray(correct["sample_index"], dtype=np.int64)
    nids = np.asarray(noimg["sample_index"], dtype=np.int64)
    cm = {int(s): i for i, s in enumerate(cids)}
    nm = {int(s): i for i, s in enumerate(nids)}
    common = np.asarray([s for s in cids.tolist() if int(s) in nm], dtype=np.int64)
    ci = np.asarray([cm[int(s)] for s in common], dtype=np.int64)
    ni = np.asarray([nm[int(s)] for s in common], dtype=np.int64)
    layers = np.asarray(correct["decoder_block_index"], dtype=np.int64)
    layer_to_idx = {int(v): i for i, v in enumerate(layers.tolist())}
    out: Dict[str, np.ndarray] = {}
    for slot, key in (("A", "A_vectors"), ("B", "B_vectors"), ("last", "last_vectors")):
        c = np.asarray(correct[key][ci], dtype=np.float32)
        n = np.asarray(noimg[key][ni], dtype=np.float32)
        out[f"{slot}_correct"] = c
        out[f"{slot}_noimg"] = n
        out[f"{slot}_residual"] = c - n
    y = np.asarray([norm_relation(x) for x in np.asarray(correct["relation"], dtype=object)[ci]], dtype=object)
    return common, out, y, layers, layer_to_idx


# -----------------------------------------------------------------------------
# Baseline layer selection (same centroid direction idea as prior scripts)
# -----------------------------------------------------------------------------


def make_stratified_splits(y: Sequence[str], train_ratio: float, repeats: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, dtype=object)
    rng = np.random.default_rng(seed)
    by = {c: np.where(y == c)[0] for c in RELATIONS}
    splits = []
    for _ in range(repeats):
        tr_parts = []
        for c in RELATIONS:
            idx = by[c].copy(); rng.shuffle(idx)
            ntr = max(1, min(len(idx)-1, int(round(train_ratio * len(idx))))) if len(idx) > 1 else len(idx)
            tr_parts.append(idx[:ntr])
        tr = np.sort(np.concatenate(tr_parts))
        mask = np.ones(len(y), dtype=bool); mask[tr] = False
        te = np.where(mask)[0]
        splits.append((tr, te))
    return splits


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def centroid_direction_acc(X: np.ndarray, y: np.ndarray, splits: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    accs = []
    for tr, te in splits:
        center = X[tr].mean(axis=0, keepdims=True)
        dirs = []
        for c in RELATIONS:
            v = X[tr][y[tr] == c].mean(axis=0) - center[0]
            v = v / max(float(np.linalg.norm(v)), EPS)
            dirs.append(v)
        D = np.stack(dirs, axis=0)
        Xt = _normalize_rows(X[te] - center)
        pred = np.argmax(Xt @ D.T, axis=1)
        gt = np.asarray([RELATIONS.index(str(v)) for v in y[te]], dtype=np.int64)
        accs.append(float(np.mean(pred == gt)))
    return float(np.mean(accs)), float(np.std(accs))


def choose_fixed_layer(raw: Mapping[str, np.ndarray], y: np.ndarray, layers: np.ndarray, args: argparse.Namespace) -> Tuple[int, pd.DataFrame]:
    Xall = raw["A_correct"] - raw["B_correct"]
    splits = make_stratified_splits(y, args.train_ratio, args.repeats, args.seed)
    rows = []
    for j, L in enumerate(layers.tolist()):
        m, s = centroid_direction_acc(Xall[:, j], y, splits)
        rows.append({"layer": int(L), "acc_mean": m, "acc_std": s})
    df = pd.DataFrame(rows)
    best = int(df.loc[df["acc_mean"].idxmax(), "layer"])
    fixed = int(args.fixed_layer) if args.fixed_layer is not None else best
    if fixed not in set(map(int, layers.tolist())):
        raise ValueError(f"fixed layer L{fixed} not present; available={layers.tolist()}")
    return fixed, df


# -----------------------------------------------------------------------------
# BBox resolution
# -----------------------------------------------------------------------------


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping): return obj
    try: return vars(obj)
    except Exception: return {}


def _parse_bbox_value(v: Any) -> Optional[List[float]]:
    if v is None: return None
    if isinstance(v, str):
        try: v = json.loads(v)
        except Exception:
            nums = re.findall(r"[-+]?\d*\.?\d+", v)
            if len(nums) >= 4: v = [float(x) for x in nums[:4]]
    if isinstance(v, Mapping):
        d = v
        if "bbox" in d: return _parse_bbox_value(d["bbox"])
        # x,y,w,h
        keys = {str(k).lower(): k for k in d.keys()}
        if all(k in keys for k in ("x", "y", "w", "h")):
            return [float(d[keys["x"]]), float(d[keys["y"]]), float(d[keys["w"]]), float(d[keys["h"]])]
        if all(k in keys for k in ("x1", "y1", "x2", "y2")):
            x1,y1,x2,y2 = [float(d[keys[k]]) for k in ("x1","y1","x2","y2")]
            return [x1,y1,x2-x1,y2-y1]
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 4:
        vals = [float(x) for x in list(v)[:4]]
        return vals
    return None


def _find_role_bbox(d: Mapping[str, Any], role: str) -> Optional[List[float]]:
    role = role.lower()
    prefixes = {
        "subject": ["subject", "sub", "a", "obj1", "object1", "target"],
        "reference": ["reference", "ref", "b", "obj2", "object2"],
    }[role]
    candidates = []
    for p in prefixes:
        candidates += [f"{p}_bbox", f"{p}_box", f"bbox_{p}", f"box_{p}"]
    low = {str(k).lower(): k for k in d.keys()}
    for c in candidates:
        if c in low:
            b = _parse_bbox_value(d[low[c]])
            if b is not None: return b
    for p in prefixes:
        if p in low and isinstance(d[low[p]], Mapping):
            b = _parse_bbox_value(d[low[p]])
            if b is not None: return b
    # nested common containers
    for container in ("objects", "pair", "entities"):
        if container in low and isinstance(d[low[container]], Mapping):
            nd = d[low[container]]
            ndlow = {str(k).lower(): k for k in nd.keys()}
            for p in prefixes:
                if p in ndlow:
                    b = _parse_bbox_value(nd[ndlow[p]])
                    if b is not None: return b
    return None


def load_pair_metadata(path: Optional[str]) -> Tuple[Dict[int, Mapping[str, Any]], Dict[Tuple[str,str,str,str], Mapping[str, Any]]]:
    by_sid: Dict[int, Mapping[str, Any]] = {}
    by_key: Dict[Tuple[str,str,str,str], Mapping[str, Any]] = {}
    if not path: return by_sid, by_key
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(p)
    rows: List[Mapping[str, Any]] = []
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif p.suffix.lower() in (".jsonl", ".jl"):
        with p.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list): rows = obj
        elif isinstance(obj, Mapping):
            for key in ("records", "data", "samples", "annotations", "items"):
                if isinstance(obj.get(key), list): rows = obj[key]; break
            else: rows = [obj]
    for r in rows:
        if not isinstance(r, Mapping): continue
        sid = r.get("sid", r.get("sample_index", r.get("id")))
        if sid is not None:
            try: by_sid[int(sid)] = r
            except Exception: pass
        image_id = str(r.get("image_id", r.get("image", r.get("coco_image_id", ""))))
        subject = str(r.get("subject", r.get("subject_name", r.get("a", "")))).lower().strip()
        reference = str(r.get("reference", r.get("reference_name", r.get("b", "")))).lower().strip()
        relation = norm_relation(r.get("relation", r.get("label", "")))
        if image_id and subject and reference:
            by_key[(image_id, subject, reference, relation)] = r
    return by_sid, by_key


def auto_find_coco_instances(data_root: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    direct = [
        data_root / "annotations" / "instances_val2017.json",
        data_root / "coco" / "annotations" / "instances_val2017.json",
        data_root / "coco_two" / "annotations" / "instances_val2017.json",
        data_root / "instances_val2017.json",
    ]
    for p in direct:
        if p.exists(): return p
    hits = list(data_root.glob("**/instances_val2017.json"))
    return hits[0] if hits else None


def load_coco_instances(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None: return None
    print(f"[bbox] loading COCO instances: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    cats = {int(c["id"]): str(c["name"]).lower().strip() for c in obj.get("categories", [])}
    cat_to_ids: Dict[str, List[int]] = defaultdict(list)
    for cid, name in cats.items(): cat_to_ids[name].append(cid)
    images = {int(x["id"]): x for x in obj.get("images", [])}
    anns_by_img: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for a in obj.get("annotations", []): anns_by_img[int(a["image_id"])].append(a)
    return {"cats": cats, "cat_to_ids": cat_to_ids, "images": images, "anns_by_img": anns_by_img}


def _name_variants(s: str) -> List[str]:
    s = s.lower().strip()
    vals = [s]
    for art in ("the ", "a ", "an "):
        if s.startswith(art): vals.append(s[len(art):].strip())
    # light singular/plural fallback only
    if s.endswith("s") and len(s) > 3: vals.append(s[:-1])
    else: vals.append(s + "s")
    return list(dict.fromkeys(vals))


def _relation_from_centers(a: Sequence[float], b: Sequence[float], relation: str) -> bool:
    ax, ay = a[0] + a[2]/2.0, a[1] + a[3]/2.0
    bx, by = b[0] + b[2]/2.0, b[1] + b[3]/2.0
    if relation == "left": return ax < bx
    if relation == "right": return ax > bx
    if relation == "above": return ay < by
    if relation == "below": return ay > by
    return False


def resolve_from_coco(rec: Any, coco: Optional[Dict[str, Any]]) -> Optional[Tuple[List[float], List[float], float, float, str]]:
    if coco is None: return None
    try: iid = int(str(rec.image_id).split(".")[0])
    except Exception:
        m = re.search(r"(\d+)", str(rec.image_id))
        if not m: return None
        iid = int(m.group(1))
    if iid not in coco["images"]: return None
    img = coco["images"][iid]; W = float(img["width"]); H = float(img["height"])
    anns = coco["anns_by_img"].get(iid, [])
    def catids_for(name: str) -> set:
        ids = set()
        for v in _name_variants(name): ids.update(coco["cat_to_ids"].get(v, []))
        return ids
    sa = catids_for(str(rec.subject)); rb = catids_for(str(rec.reference))
    if not sa or not rb: return None
    As = [a for a in anns if int(a.get("category_id", -1)) in sa and a.get("bbox")]
    Bs = [b for b in anns if int(b.get("category_id", -1)) in rb and b.get("bbox")]
    if len(As) == 1 and len(Bs) == 1 and int(As[0].get("id", -1)) != int(Bs[0].get("id", -2)):
        return list(map(float, As[0]["bbox"])), list(map(float, Bs[0]["bbox"])), W, H, "coco_unique_category"
    pairs = []
    rel = norm_relation(rec.relation)
    for a in As:
        for b in Bs:
            if int(a.get("id", -1)) == int(b.get("id", -1)): continue
            ba, bb = list(map(float, a["bbox"])), list(map(float, b["bbox"]))
            if _relation_from_centers(ba, bb, rel): pairs.append((ba, bb))
    if len(pairs) == 1:
        return pairs[0][0], pairs[0][1], W, H, "coco_unique_relation_pair"
    return None


def resolve_bbox_record(
    rec: Any, pair_by_sid: Mapping[int, Mapping[str, Any]],
    pair_by_key: Mapping[Tuple[str,str,str,str], Mapping[str, Any]],
    coco: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
    candidates: List[Tuple[Mapping[str, Any], str]] = []
    rd = _as_mapping(rec)
    candidates.append((rd, "record_fields"))
    if int(rec.sid) in pair_by_sid: candidates.append((pair_by_sid[int(rec.sid)], "pair_metadata_sid"))
    key = (str(rec.image_id), str(rec.subject).lower().strip(), str(rec.reference).lower().strip(), norm_relation(rec.relation))
    if key in pair_by_key: candidates.append((pair_by_key[key], "pair_metadata_key"))

    for d, src in candidates:
        ba = _find_role_bbox(d, "subject")
        bb = _find_role_bbox(d, "reference")
        if ba is None or bb is None: continue
        W = d.get("image_width", d.get("width", d.get("W")))
        H = d.get("image_height", d.get("height", d.get("H")))
        if W is None or H is None:
            try:
                with Image.open(rec.image_path) as im: W, H = im.size
            except Exception: continue
        W=float(W); H=float(H)
        return bbox_to_features(ba, bb, W, H), {"source": src}

    got = resolve_from_coco(rec, coco)
    if got is not None:
        ba, bb, W, H, src = got
        return bbox_to_features(ba, bb, W, H), {"source": src}
    return None, {"source": "unresolved"}


def bbox_to_features(ba: Sequence[float], bb: Sequence[float], W: float, H: float) -> Dict[str, float]:
    if W <= 0 or H <= 0: raise ValueError("invalid image size")
    xa = (float(ba[0]) + float(ba[2])/2.0) / W
    ya = (float(ba[1]) + float(ba[3])/2.0) / H
    xb = (float(bb[0]) + float(bb[2])/2.0) / W
    yb = (float(bb[1]) + float(bb[3])/2.0) / H
    wa, ha = float(ba[2])/W, float(ba[3])/H
    wb, hb = float(bb[2])/W, float(bb[3])/H
    return {
        "xA": xa, "yA": ya, "xB": xb, "yB": yb,
        "pair_cx": 0.5*(xa+xb), "pair_cy": 0.5*(ya+yb),
        "dx": xa-xb, "dy": ya-yb,
        "wA": wa, "hA": ha, "wB": wb, "hB": hb,
    }


# -----------------------------------------------------------------------------
# Linear encoding model
# -----------------------------------------------------------------------------


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a*b, axis=1)
    den = np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), EPS)
    return num/den


def global_r2(pred: np.ndarray, target: np.ndarray, center: np.ndarray) -> float:
    num = float(np.sum((target-pred)**2))
    den = float(np.sum((target-center)**2))
    return 1.0 - num/max(den, EPS)


def relerr(pred_centered: np.ndarray, target_centered: np.ndarray) -> float:
    return float(np.linalg.norm(pred_centered-target_centered) / max(float(np.linalg.norm(target_centered)), EPS))


def ridge_fit_predict(
    X: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray,
    ridge: float, feature_indices: Sequence[int],
) -> Dict[str, Any]:
    idx = np.asarray(feature_indices, dtype=np.int64)
    ymu = Y[tr].mean(axis=0, keepdims=True)
    Ytr = Y[tr] - ymu
    Yte = Y[te]
    Yte_c = Yte - ymu
    if len(idx) == 0:
        pred_c = np.zeros_like(Yte_c)
        return {
            "pred": np.repeat(ymu, len(te), axis=0), "pred_c": pred_c,
            "target": Yte, "target_c": Yte_c, "ymu": ymu,
            "W": np.zeros((0, Y.shape[1]), dtype=np.float64),
            "xmu": np.zeros((0,), dtype=np.float64), "xsd": np.ones((0,), dtype=np.float64),
            "idx": idx,
        }
    xmu = X[tr][:, idx].mean(axis=0)
    xsd = X[tr][:, idx].std(axis=0)
    xsd = np.where(xsd < 1e-8, 1.0, xsd)
    Xtr = (X[tr][:, idx] - xmu)/xsd
    Xte = (X[te][:, idx] - xmu)/xsd
    gram = Xtr.T @ Xtr
    lam = float(ridge) * float(np.trace(gram)/max(len(idx),1))
    W = np.linalg.solve(gram + lam*np.eye(len(idx)), Xtr.T @ Ytr)
    pred_c = Xte @ W
    pred = ymu + pred_c
    return {"pred": pred, "pred_c": pred_c, "target": Yte, "target_c": Yte_c,
            "ymu": ymu, "W": W, "xmu": xmu, "xsd": xsd, "idx": idx}


def metrics_from_fit(f: Mapping[str, Any]) -> Dict[str, float]:
    pred=np.asarray(f["pred"]); pc=np.asarray(f["pred_c"]); target=np.asarray(f["target"]); tc=np.asarray(f["target_c"]); ymu=np.asarray(f["ymu"])
    return {
        "full_cos": float(np.mean(cosine_rows(pred, target))),
        "centered_cos": float(np.mean(cosine_rows(pc, tc))),
        "r2": global_r2(pred, target, ymu),
        "relative_error": relerr(pc, tc),
    }


def group_component_from_full_fit(f: Mapping[str, Any], X: np.ndarray, te: np.ndarray, group: str) -> np.ndarray:
    # f is full 8-feature fit, so coefficient rows correspond to FEATURES order.
    xmu=np.asarray(f["xmu"]); xsd=np.asarray(f["xsd"]); W=np.asarray(f["W"])
    Xte=(X[te]-xmu)/xsd
    idx=np.asarray(GROUP_IDXS[group], dtype=np.int64)
    return Xte[:, idx] @ W[idx]


def shuffled_delta(comp: np.ndarray, target_c: np.ndarray, repeats: int, rng: np.random.Generator) -> float:
    matched=float(np.mean(cosine_rows(comp, target_c)))
    vals=[]
    n=len(comp)
    for _ in range(repeats):
        perm=rng.permutation(n)
        vals.append(float(np.mean(cosine_rows(comp[perm], target_c))))
    return matched-float(np.mean(vals))


def all_subset_indices(groups_subset: Iterable[str]) -> List[int]:
    out=[]
    for g in groups_subset: out.extend(GROUP_IDXS[g])
    return sorted(out)


def shapley_r2_for_split(X: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray, ridge: float) -> Tuple[Dict[str,float], Dict[frozenset,float]]:
    subset_r2: Dict[frozenset,float] = {}
    for r in range(len(GROUPS)+1):
        for comb in itertools.combinations(GROUPS, r):
            S=frozenset(comb)
            f=ridge_fit_predict(X,Y,tr,te,ridge,all_subset_indices(S))
            subset_r2[S]=metrics_from_fit(f)["r2"]
    phi={g:0.0 for g in GROUPS}
    m=len(GROUPS)
    fact=math.factorial
    for g in GROUPS:
        others=[x for x in GROUPS if x!=g]
        for r in range(len(others)+1):
            for comb in itertools.combinations(others,r):
                S=frozenset(comb)
                weight=fact(len(S))*fact(m-len(S)-1)/fact(m)
                phi[g]+=weight*(subset_r2[S|{g}]-subset_r2[S])
    return phi, subset_r2


def analyze_target(
    X: np.ndarray, Y: np.ndarray, splits: Sequence[Tuple[np.ndarray,np.ndarray]],
    args: argparse.Namespace, target_name: str,
) -> Tuple[Dict[str,Any], List[Dict[str,Any]], List[Dict[str,Any]]]:
    rng=np.random.default_rng(args.seed+12345)
    full_metrics=[]
    comp_acc=defaultdict(list)
    comp_norm=defaultdict(list)
    comp_match=defaultdict(list)
    drop_delta=defaultdict(list)
    shapley=defaultdict(list)
    split_rows=[]

    all_idx=list(range(len(FEATURES)))
    for si,(tr,te) in enumerate(splits):
        full=ridge_fit_predict(X,Y,tr,te,args.ridge,all_idx)
        fm=metrics_from_fit(full); full_metrics.append(fm)
        # additive components of the FULL fitted model
        comps={g:group_component_from_full_fit(full,X,te,g) for g in GROUPS}
        recon=sum(comps.values())
        algebra=float(np.max(np.abs(recon-full["pred_c"])))
        for g,c in comps.items():
            comp_acc[g].append(float(np.mean(cosine_rows(c,full["target_c"]))))
            comp_norm[g].append(float(np.mean(np.linalg.norm(c,axis=1))))
            comp_match[g].append(shuffled_delta(c,full["target_c"],args.shuffle_repeats,rng))
            keep=[i for i in all_idx if i not in GROUP_IDXS[g]]
            red=ridge_fit_predict(X,Y,tr,te,args.ridge,keep)
            drop_delta[g].append(fm["r2"]-metrics_from_fit(red)["r2"])
        phi,_=shapley_r2_for_split(X,Y,tr,te,args.ridge)
        for g,v in phi.items(): shapley[g].append(float(v))
        split_rows.append({"target":target_name,"split":si,**fm,"algebra_max_abs":algebra})

    def ms(vals: Sequence[float]) -> Tuple[float,float]: return float(np.mean(vals)),float(np.std(vals))
    result={"target":target_name}
    for k in ("full_cos","centered_cos","r2","relative_error"):
        m,s=ms([x[k] for x in full_metrics]); result[k+"_mean"]=m; result[k+"_std"]=s
    component_rows=[]
    for g in GROUPS:
        cm,cs=ms(comp_acc[g]); nm,ns=ms(comp_norm[g]); mm,msd=ms(comp_match[g]); dm,ds=ms(drop_delta[g]); sm,ss=ms(shapley[g])
        component_rows.append({
            "target":target_name,"group":g,
            "component_cos_mean":cm,"component_cos_std":cs,
            "component_norm_mean":nm,"component_norm_std":ns,
            "matched_minus_shuffled_mean":mm,"matched_minus_shuffled_std":msd,
            "drop_one_delta_r2_mean":dm,"drop_one_delta_r2_std":ds,
            "shapley_r2_mean":sm,"shapley_r2_std":ss,
        })
    return result, component_rows, split_rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args=parse_args()
    if not (0 < args.train_ratio < 1): raise ValueError("--train-ratio must be in (0,1)")
    if args.repeats < 1 or args.shuffle_repeats < 1: raise ValueError("repeats must be >=1")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    state_dir=Path(args.state_dir) if args.state_dir else (out/"states")
    state_dir.mkdir(parents=True,exist_ok=True)
    correct_path=state_dir/"raw__correct__all_layers.npz"
    noimg_path=state_dir/"raw__no_image__all_layers.npz"

    records,audit=base.load_records(args.dataset,Path(args.data_root),args.max_samples)
    records=[r for r in records if norm_relation(r.relation) in RELATIONS]
    print(f"[{args.dataset}] n={len(records)} counts={dict(Counter(norm_relation(r.relation) for r in records))}")
    (out/"dataset.audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")

    # Extract only if reused states are absent.
    if not (correct_path.exists() and noimg_path.exists()):
        if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        spec=base.SPECS[args.model]
        model_cls=getattr(transformers,spec.model_class,None)
        if model_cls is None: raise RuntimeError(f"transformers=={transformers.__version__} has no {spec.model_class}")
        load_kwargs={"torch_dtype":base.resolve_dtype(spec.dtype_name),"low_cpu_mem_usage":True,"trust_remote_code":spec.trust_remote_code,"device_map":{"":args.device}}
        if args.attn_impl!="none": load_kwargs["attn_implementation"]=args.attn_impl
        model=model_cls.from_pretrained(spec.repo_id,**load_kwargs); model.eval()
        processor=AutoProcessor.from_pretrained(spec.repo_id,trust_remote_code=spec.trust_remote_code)
        base.configure_processor(model,processor); device=torch.device(args.device)
        extract_raw_all_layers(args=args,model=model,processor=processor,device=device,records=records,with_image=True,out_path=correct_path)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        extract_raw_all_layers(args=args,model=model,processor=processor,device=device,records=records,with_image=False,out_path=noimg_path)
        del model,processor; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print(f"[reuse states] {correct_path}")
        print(f"[reuse states] {noimg_path}")

    correct=load_npz(correct_path); noimg=load_npz(noimg_path)
    common,raw,y,layers,layer_to_idx=align_raw(correct,noimg)
    fixed,scan=choose_fixed_layer(raw,y,layers,args); scan.to_csv(out/"baseline_layer_scan.csv",index=False)
    br=scan.loc[scan["acc_mean"].idxmax()]; fr=scan.loc[scan["layer"]==fixed].iloc[0]
    print("\n"+"="*112); print("FIXED LAYER"); print("="*112)
    print(f"raw h_A-h_B best : L{int(br.layer)} acc={100*float(br.acc_mean):.2f}%±{100*float(br.acc_std):.2f}%")
    print(f"analysis layer    : L{fixed} acc={100*float(fr.acc_mean):.2f}%±{100*float(fr.acc_std):.2f}%")

    # Resolve geometry for raw common sids.
    rec_by_sid={int(r.sid):r for r in records}
    pair_sid,pair_key=load_pair_metadata(args.pair_metadata)
    coco_path=auto_find_coco_instances(Path(args.data_root),args.coco_instances)
    coco=load_coco_instances(coco_path)
    bbox_rows=[]; unresolved=[]; source_counts=Counter()
    for sid in common.tolist():
        rec=rec_by_sid.get(int(sid))
        if rec is None:
            unresolved.append({"sid":int(sid),"reason":"record_missing"}); continue
        feat,meta=resolve_bbox_record(rec,pair_sid,pair_key,coco)
        if feat is None:
            unresolved.append({"sid":int(sid),"image_id":str(rec.image_id),"subject":str(rec.subject),"reference":str(rec.reference),"relation":norm_relation(rec.relation),"reason":"bbox_unresolved"})
            continue
        source_counts[meta["source"]]+=1
        bbox_rows.append({"sid":int(sid),"image_id":str(rec.image_id),"subject":str(rec.subject),"reference":str(rec.reference),"relation":norm_relation(rec.relation),"bbox_source":meta["source"],**feat})
    bbox_df=pd.DataFrame(bbox_rows)
    bbox_df.to_csv(out/"resolved_geometry.csv",index=False)
    (out/"bbox_audit.json").write_text(json.dumps({"resolved":len(bbox_rows),"unresolved":len(unresolved),"source_counts":dict(source_counts),"unresolved_examples":unresolved[:100],"coco_instances":str(coco_path) if coco_path else None},indent=2,ensure_ascii=False),encoding="utf-8")
    print("\n"+"="*112); print("BBOX RESOLUTION"); print("="*112)
    print(f"resolved={len(bbox_rows)}/{len(common)} | sources={dict(source_counts)}")
    if unresolved: print(f"unresolved={len(unresolved)} (see {out/'bbox_audit.json'})")
    if len(bbox_rows) < 40:
        raise RuntimeError("Too few samples with resolved A/B bboxes. Supply --pair-metadata with exact subject/reference bboxes.")

    # Align states to geometry rows.
    sid_to_raw={int(s):i for i,s in enumerate(common.tolist())}
    keep_sids=bbox_df["sid"].to_numpy(dtype=np.int64)
    ridx=np.asarray([sid_to_raw[int(s)] for s in keep_sids],dtype=np.int64)
    X=bbox_df[list(FEATURES)].to_numpy(dtype=np.float64)
    y_keep=y[ridx]
    li=layer_to_idx[fixed]
    targets={}
    for slot in SLOTS:
        targets[f"{slot}_raw"]=raw[f"{slot}_correct"][ridx,li].astype(np.float64)
        targets[f"{slot}_residual"]=raw[f"{slot}_residual"][ridx,li].astype(np.float64)

    splits=make_stratified_splits(y_keep,args.train_ratio,args.repeats,args.seed)
    summary_rows=[]; component_rows=[]; split_rows=[]
    for name,Y in targets.items():
        res,comps,srows=analyze_target(X,Y,splits,args,name)
        summary_rows.append(res); component_rows.extend(comps); split_rows.extend(srows)

    summary_df=pd.DataFrame(summary_rows); comps_df=pd.DataFrame(component_rows); split_df=pd.DataFrame(split_rows)
    summary_df.to_csv(out/"encoding_summary.csv",index=False)
    comps_df.to_csv(out/"encoding_components.csv",index=False)
    split_df.to_csv(out/"encoding_splits.csv",index=False)

    print("\n"+"="*112); print(f"LINEAR ENCODING MODEL @ L{fixed}"); print("geometry basis = pair_center + relative(dx,dy) + size_A + size_B"); print("metrics are HELD-OUT; hidden targets are centered by TRAIN mean"); print("="*112)
    for row in summary_rows:
        print(f"{row['target']:14s} | centered cos={row['centered_cos_mean']:+.4f}±{row['centered_cos_std']:.4f} | R2={row['r2_mean']:+.4f}±{row['r2_std']:.4f} | relerr={row['relative_error_mean']:.4f} | full-cos={row['full_cos_mean']:+.4f}")

    print("\n"+"="*112); print("ADDITIVE GEOMETRY COMPONENTS"); print("component_cos = cosine(component vector, true centered hidden state)"); print("drop-one ΔR2 = full R2 - R2(without this group)"); print("Shapley R2 = fair allocation of held-out explained variance across correlated groups"); print("="*112)
    for target in targets:
        print(f"\n{target}")
        d=comps_df[comps_df.target==target]
        for _,r in d.iterrows():
            print(f"  {r['group']:12s} | cos={r['component_cos_mean']:+.4f}±{r['component_cos_std']:.4f} | match-shuf={r['matched_minus_shuffled_mean']:+.4f} | ΔR2={r['drop_one_delta_r2_mean']:+.4f} | ShapleyR2={r['shapley_r2_mean']:+.4f} | norm={r['component_norm_mean']:.3f}")

    # Compact scientific interpretation table: focus on residual targets.
    print("\n"+"="*112); print("RESIDUAL TARGET QUICK VIEW (Image - NoImage)"); print("="*112)
    for slot in SLOTS:
        target=f"{slot}_residual"
        s=summary_df[summary_df.target==target].iloc[0]
        d=comps_df[comps_df.target==target].set_index("group")
        print(f"{target:14s} | full R2={float(s.r2_mean):+.4f} | cos={float(s.centered_cos_mean):+.4f} | Shapley: center={float(d.loc['pair_center','shapley_r2_mean']):+.4f}, relative={float(d.loc['relative','shapley_r2_mean']):+.4f}, sizeA={float(d.loc['size_A','shapley_r2_mean']):+.4f}, sizeB={float(d.loc['size_B','shapley_r2_mean']):+.4f}")

    summary={
        "model":args.model,"dataset":args.dataset,"fixed_layer":fixed,
        "n_raw_common":int(len(common)),"n_bbox_resolved":int(len(bbox_df)),
        "bbox_source_counts":dict(source_counts),"features":list(FEATURES),"groups":{k:list(v) for k,v in GROUP_IDXS.items()},
        "train_ratio":args.train_ratio,"repeats":args.repeats,"ridge":args.ridge,
        "results":summary_df.to_dict(orient="records"),"components":comps_df.to_dict(orient="records"),
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print("\nSaved:")
    for p in ("baseline_layer_scan.csv","resolved_geometry.csv","bbox_audit.json","encoding_summary.csv","encoding_components.csv","encoding_splits.csv","summary.json"):
        print(f"  {out/p}")


if __name__ == "__main__":
    main()
