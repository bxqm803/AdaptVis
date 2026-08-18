#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen3b_head_passport_L18_L28_v1.py

Pilot: build a READ-WRITE-EFFECT-CONTEXT "passport" for every attention head
in Qwen2.5-VL-3B, restricted to decoder layers L18..L28 (inclusive), on
COCO_two.

The script is designed to run from the AdaptVis llava16 repo root and reuses
helpers already used by multimodel_spatial_grounding_ablation_v1.py.

Primary features
----------------
READ
  visual_mass
  subject_mass                # bbox enrichment; NaN if bbox unavailable
  reference_mass              # bbox enrichment; NaN if bbox unavailable
  cross_object_attention      # bbox enrichment; NaN if bbox unavailable
  spatial_entropy
  hflip_equivariance

WRITE (pre-W_O head representation)
  direction_acc
  direction_margin
  gray_drop                   # direction_acc_real - direction_acc_gray
  gray_margin_drop

EFFECT (all w.r.t. one differentiable final relation margin)
  dla                          # direct logit attribution at prompt-last position
  grad_activation_signed
  grad_activation_abs
  grad_attention_*            # edge-group Grad x Attention

CONTEXT
  mlp_grad_activation_signed  # same-layer MLP output Grad x Activation
  mlp_grad_activation_abs

Unified differentiable target
-----------------------------
For the four relation scores s_r at the prompt-last position:

    M = s_GT - logsumexp({s_r : r != GT})

Larger M means the final state more strongly supports the ground-truth relation.
All gradient EFFECT quantities use this exact M.

Important semantics
-------------------
* WRITE uses PRE-W_O per-head output z_h.
* EFFECT uses the actual head contribution to the residual stream.  For
  Grad x Activation, computing x_h * dM/dx_h at the input to W_O is exactly the
  first-order contribution of that head through the linear W_O map (ignoring
  the shared o_proj bias).
* DLA is a direct readout diagnostic, not total causal effect.
* Grad x Activation / Grad x Attention are first-order local attributions, not
  substitutes for later activation-patching validation.
* subject/reference/cross-object mass require object bounding boxes.  The
  loader tries to discover them from the COCO_two record / prompt metadata.
  If unavailable, those fields are written as NaN rather than silently changing
  their definition.

Outputs
-------
  output/head_passport_qwen3b_L18_L28_v1/
    config.json
    split.csv
    bbox_audit.json
    direction_codebook.npz
    head_passport_sample.csv.gz
    head_passport_summary.csv
    layer_context_summary.csv
    errors.jsonl
    DONE

Suggested pilot
---------------
CUDA_VISIBLE_DEVICES=0 python -u qwen3b_head_passport_L18_L28_v1.py \
  --data-root data \
  --max-samples 200 \
  --output-root output/head_passport_qwen3b_L18_L28_v1

Full COCO_two
-------------
CUDA_VISIBLE_DEVICES=0 python -u qwen3b_head_passport_L18_L28_v1.py \
  --data-root data \
  --max-samples 0 \
  --output-root output/head_passport_qwen3b_L18_L28_v1_full
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import random
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

# Existing AdaptVis helpers.
import extract_two_object_relation_states as base
import analyze_coco_centroid_generation_step1_v4 as centroid_base
import analyze_coco_head_object_residual_direction_probe_v1 as direction_base
import scan_multimodel_spatial_logitlens_matrix_v1 as lens_base


SCRIPT_VERSION = "qwen3b-head-passport-L18-L28-v1"
MODEL_ALIAS = "qwen-3b"
DATASET = "coco_two"
RELATIONS = ("left", "right", "above", "below")
RID = {r: i for i, r in enumerate(RELATIONS)}
PROMPT_FILE = Path("prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

@dataclass
class Example:
    sid: int
    relation: str
    subject: str
    reference: str
    question: str
    image_path: Path
    subject_bbox_raw: Any = None
    reference_bbox_raw: Any = None
    subject_bbox_key: str = ""
    reference_bbox_key: str = ""

    def image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")


CANON_REL = {
    "left": "left",
    "right": "right",
    "above": "above",
    "on": "above",
    "over": "above",
    "top": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "beneath": "below",
    "bottom": "below",
}


def canon_relation(x: Any) -> Optional[str]:
    if isinstance(x, (list, tuple)):
        x = x[0] if x else ""
    key = str(x).strip().lower().replace("_", " ").replace("-", " ")
    key = " ".join(key.split())
    return CANON_REL.get(key)


def _obj_to_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    try:
        return vars(x)
    except Exception:
        return {}


def _find_key_recursive(obj: Any, keys: Sequence[str], prefix: str = "") -> Tuple[Any, str]:
    """Conservative recursive metadata lookup, depth <= 3."""
    wanted = {k.lower() for k in keys}

    def rec(x: Any, p: str, depth: int) -> Tuple[Any, str]:
        if depth > 3:
            return None, ""
        if isinstance(x, dict):
            for k, v in x.items():
                if str(k).lower() in wanted:
                    return v, f"{p}.{k}" if p else str(k)
            for k, v in x.items():
                if isinstance(v, (dict, list, tuple)):
                    out, kp = rec(v, f"{p}.{k}" if p else str(k), depth + 1)
                    if out is not None:
                        return out, kp
        return None, ""

    return rec(obj, prefix, 0)


def discover_bbox_pair(record: Any, prompt: Mapping[str, Any], subject: str, reference: str):
    """
    Try common bbox schemas without inventing geometry.
    Returned boxes remain raw; normalization happens after image size is known.
    """
    sources = [("prompt", prompt), ("record", _obj_to_dict(record))]
    sub_keys = (
        "subject_bbox", "subject_box", "sub_bbox", "sub_box", "bbox_subject",
        "box_subject", "subject_bounding_box", "bbox1", "box1", "bbox_a", "box_a",
    )
    ref_keys = (
        "reference_bbox", "reference_box", "ref_bbox", "ref_box", "bbox_reference",
        "box_reference", "reference_bounding_box", "bbox2", "box2", "bbox_b", "box_b",
    )
    for name, src in sources:
        sb, sk = _find_key_recursive(src, sub_keys, name)
        rb, rk = _find_key_recursive(src, ref_keys, name)
        if sb is not None and rb is not None:
            return sb, rb, sk, rk

        # Dict keyed by object name.
        boxes, bk = _find_key_recursive(src, ("bboxes", "boxes", "object_bboxes", "object_boxes"), name)
        if isinstance(boxes, dict):
            low = {str(k).strip().lower(): (k, v) for k, v in boxes.items()}
            s = low.get(subject.strip().lower())
            r = low.get(reference.strip().lower())
            if s is not None and r is not None:
                return s[1], r[1], f"{bk}.{s[0]}", f"{bk}.{r[0]}"

    return None, None, "", ""


def load_examples(data_root: Path, max_samples: int, seed: int) -> Tuple[List[Example], List[Any]]:
    records, audit = base.load_records("coco_two", data_root, None)
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Missing {PROMPT_FILE}; run from AdaptVis repo root")
    prompts = centroid_base.load_standard_prompts(PROMPT_FILE)

    out: List[Example] = []
    for r in records:
        rel = canon_relation(getattr(r, "relation", None))
        if rel not in RELATIONS:
            continue
        sid = int(r.sid)
        p = prompts.get(sid)
        if p is None:
            continue
        subject = str(p["subject"])
        reference = str(p["reference"])
        sb, rb, sk, rk = discover_bbox_pair(r, p, subject, reference)
        out.append(
            Example(
                sid=sid,
                relation=rel,
                subject=subject,
                reference=reference,
                question=str(p["question_text"]),
                image_path=Path(r.image_path),
                subject_bbox_raw=sb,
                reference_bbox_raw=rb,
                subject_bbox_key=sk,
                reference_bbox_key=rk,
            )
        )

    if max_samples and max_samples > 0 and max_samples < len(out):
        out = stratified_cap(out, max_samples, seed)
    return sorted(out, key=lambda x: x.sid), list(audit)


def stratified_cap(examples: Sequence[Example], n: int, seed: int) -> List[Example]:
    rng = random.Random(seed)
    by = defaultdict(list)
    for x in examples:
        by[x.relation].append(x)
    for g in by.values():
        rng.shuffle(g)
    ptr = {r: 0 for r in RELATIONS}
    out: List[Example] = []
    while len(out) < n:
        moved = False
        for rel in RELATIONS:
            i = ptr[rel]
            if i < len(by.get(rel, [])) and len(out) < n:
                out.append(by[rel][i])
                ptr[rel] += 1
                moved = True
        if not moved:
            break
    return out


def stratified_split(examples: Sequence[Example], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    by = defaultdict(list)
    for x in examples:
        by[x.relation].append(x)
    train, val, test = [], [], []
    for rel in RELATIONS:
        g = list(by[rel])
        rng.shuffle(g)
        n = len(g)
        nt = max(1, int(round(n * train_ratio)))
        nv = max(1, int(round(n * val_ratio)))
        if nt + nv >= n:
            nv = max(1, n - nt - 1)
        train += g[:nt]
        val += g[nt:nt + nv]
        test += g[nt + nv:]
    return (
        sorted(train, key=lambda x: x.sid),
        sorted(val, key=lambda x: x.sid),
        sorted(test, key=lambda x: x.sid),
    )


# -----------------------------------------------------------------------------
# BBox parsing / visual geometry
# -----------------------------------------------------------------------------

def _flatten_box(raw: Any) -> Optional[Tuple[float, float, float, float, str]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        d = {str(k).lower(): v for k, v in raw.items()}
        # xyxy schemas
        xyxy_sets = [
            ("xmin", "ymin", "xmax", "ymax"),
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
        ]
        for ks in xyxy_sets:
            if all(k in d for k in ks):
                return tuple(float(d[k]) for k in ks) + ("xyxy",)
        # xywh schemas
        xywh_sets = [
            ("x", "y", "w", "h"),
            ("x", "y", "width", "height"),
            ("left", "top", "width", "height"),
        ]
        for ks in xywh_sets:
            if all(k in d for k in ks):
                return tuple(float(d[k]) for k in ks) + ("xywh",)
        for k in ("bbox", "box", "coordinates", "coords"):
            if k in d:
                return _flatten_box(d[k])
    if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) >= 4:
        vals = [float(raw[i]) for i in range(4)]
        return vals[0], vals[1], vals[2], vals[3], "unknown"
    return None


def normalize_bbox(raw: Any, key_hint: str, image_size: Tuple[int, int], bbox_format: str) -> Optional[Tuple[float,float,float,float]]:
    """Return normalized xyxy in [0,1]."""
    p = _flatten_box(raw)
    if p is None:
        return None
    a, b, c, d, inferred = p
    W, H = image_size
    fmt = bbox_format
    if fmt == "auto":
        if inferred in ("xyxy", "xywh"):
            fmt = inferred
        else:
            kh = key_hint.lower()
            if "xyxy" in kh:
                fmt = "xyxy"
            elif "xywh" in kh:
                fmt = "xywh"
            else:
                # COCO generic bbox convention is xywh.  This is used only for
                # metadata whose format is otherwise unspecified; override with
                # --bbox-format xyxy if your local prompt stores xyxy arrays.
                fmt = "xywh"

    vals = np.asarray([a,b,c,d], dtype=np.float64)
    normalized_input = float(np.nanmax(np.abs(vals))) <= 1.5
    if fmt == "xywh":
        x1, y1, w, h = vals
        x2, y2 = x1 + w, y1 + h
    else:
        x1, y1, x2, y2 = vals

    if not normalized_input:
        x1, x2 = x1 / max(W, 1), x2 / max(W, 1)
        y1, y2 = y1 / max(H, 1), y2 / max(H, 1)

    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    c = np.asarray(coords, dtype=np.float64).copy()
    for j in range(2):
        lo, hi = float(np.min(c[:, j])), float(np.max(c[:, j]))
        if hi > lo:
            c[:, j] = (c[:, j] - lo) / (hi - lo)
        else:
            c[:, j] = 0.5
    return c


def bbox_token_mask(coords01: np.ndarray, bbox: Optional[Tuple[float,float,float,float]]) -> Optional[np.ndarray]:
    if bbox is None:
        return None
    x1,y1,x2,y2 = bbox
    x, y = coords01[:,0], coords01[:,1]
    m = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    if not np.any(m):
        # nearest token to bbox center, so tiny boxes do not become undefined
        cx, cy = (x1+x2)/2, (y1+y2)/2
        j = int(np.argmin((x-cx)**2 + (y-cy)**2))
        m[j] = True
    return m


def mirror_permutation(coords01: np.ndarray) -> np.ndarray:
    """Map each original visual token index j to its horizontal-mirror index k."""
    c = np.asarray(coords01, dtype=np.float64)
    target = np.stack([1.0 - c[:,0], c[:,1]], axis=1)
    # Nvis is modest; nearest-neighbour mapping is cached per visual grid by caller.
    d2 = ((target[:,None,:] - c[None,:,:])**2).sum(axis=-1)
    return np.argmin(d2, axis=1).astype(np.int64)


def cosine_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    num = np.sum(a*b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, eps)


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------

def resolve_dtype(name: str) -> torch.dtype:
    return {"float16":torch.float16, "bfloat16":torch.bfloat16, "float32":torch.float32}[name]


def load_model(device: str):
    actual = lens_base.resolve_model_alias(MODEL_ALIAS, base.SPECS)
    spec = base.SPECS[actual]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers=={transformers.__version__} has no {spec.model_class}")
    model = model_cls.from_pretrained(
        spec.repo_id,
        torch_dtype=resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": device},
        attn_implementation="eager",
    )
    model.eval()
    # No parameter gradients: we cut the graph at L18 input and require grad there.
    for p in model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    centroid_base.configure_processor(model, processor)
    return actual, spec, model, processor


def build_batch(processor, question: str, image: Image.Image, device: torch.device):
    rendered = direction_base.build_chat_prompt(processor, question, True)
    return direction_base.process_inputs(processor, rendered, image, device)




def hidden_states_from_outputs(outputs: Any) -> Tuple[torch.Tensor, ...]:
    candidates = [
        getattr(outputs, "hidden_states", None),
        getattr(getattr(outputs, "language_model_outputs", None), "hidden_states", None),
        getattr(getattr(outputs, "text_model_output", None), "hidden_states", None),
    ]
    for states in candidates:
        if isinstance(states, (tuple, list)) and states and torch.is_tensor(states[-1]):
            return tuple(states)
    raise RuntimeError("No language decoder hidden_states returned")


def attentions_from_outputs(outputs: Any) -> Tuple[Any, ...]:
    candidates = [
        getattr(outputs, "attentions", None),
        getattr(getattr(outputs, "language_model_outputs", None), "attentions", None),
        getattr(getattr(outputs, "text_model_output", None), "attentions", None),
    ]
    for attn in candidates:
        if isinstance(attn, (tuple, list)) and len(attn) > 0:
            return tuple(attn)
    raise RuntimeError("No language attentions returned; eager attention is required")


def relation_token_map(tokenizer) -> Dict[str, List[int]]:
    out = {}
    for rel in RELATIONS:
        ids = set()
        for text in (rel, " "+rel, rel.capitalize(), " "+rel.capitalize()):
            tok = tokenizer.encode(text, add_special_tokens=False)
            if len(tok) == 1:
                ids.add(int(tok[0]))
        if not ids:
            tok = tokenizer.encode(" "+rel, add_special_tokens=False)
            if not tok:
                raise RuntimeError(f"Cannot tokenize relation {rel}")
            ids.add(int(tok[-1]))
        out[rel] = sorted(ids)
    return out


def relation_scores_torch(logits_last: torch.Tensor, token_map: Mapping[str, Sequence[int]]) -> torch.Tensor:
    vals = []
    for rel in RELATIONS:
        ids = torch.as_tensor(list(token_map[rel]), device=logits_last.device, dtype=torch.long)
        vals.append(torch.logsumexp(logits_last.float().index_select(0, ids), dim=0))
    return torch.stack(vals, dim=0)


def relation_stats_np(scores: np.ndarray, gt: str) -> Dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64).reshape(4)
    g = RID[gt]
    pred = int(np.argmax(s))
    wrong = max((i for i in range(4) if i != g), key=lambda i: float(s[i]))
    return {
        "pred": RELATIONS[pred],
        "correct": int(pred == g),
        "margin": float(s[g] - s[wrong]),
    }


def unembedding_relation_vectors(lm_head, token_map: Mapping[str, Sequence[int]]) -> torch.Tensor:
    W = lm_head.weight.detach().float()
    vecs = []
    for rel in RELATIONS:
        ids = torch.as_tensor(list(token_map[rel]), device=W.device, dtype=torch.long)
        vecs.append(W.index_select(0, ids).mean(dim=0))
    return torch.stack(vecs, dim=0)  # [4,D]


# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------

class SelectedCapture:
    def __init__(self, decoder_layers: Sequence[Any], selected_layers: Sequence[int], cut_layer: Optional[int], need_grad: bool):
        self.decoder_layers = decoder_layers
        self.layers = list(selected_layers)
        self.cut_layer = cut_layer
        self.need_grad = need_grad
        self.prewo: Dict[int, torch.Tensor] = {}
        self.mlp: Dict[int, torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        if self.need_grad and self.cut_layer is not None:
            layer = self.decoder_layers[self.cut_layer]
            def cut_hook(module, args, kwargs):
                if args:
                    h = args[0].detach().requires_grad_(True)
                    return (h,) + tuple(args[1:]), kwargs
                if "hidden_states" in kwargs:
                    kw = dict(kwargs)
                    kw["hidden_states"] = kw["hidden_states"].detach().requires_grad_(True)
                    return args, kw
                raise RuntimeError("Could not find hidden_states at cut layer")
            self.handles.append(layer.register_forward_pre_hook(cut_hook, with_kwargs=True))

        for l in self.layers:
            layer = self.decoder_layers[l]
            if not hasattr(layer, "self_attn") or not hasattr(layer.self_attn, "o_proj"):
                raise RuntimeError(f"Layer {l} does not expose self_attn.o_proj")
            def make_o_hook(li):
                def hook(module, args):
                    x = args[0]
                    if self.need_grad and x.requires_grad:
                        x.retain_grad()
                    self.prewo[li] = x
                return hook
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_o_hook(l)))

            if not hasattr(layer, "mlp"):
                raise RuntimeError(f"Layer {l} has no .mlp")
            def make_mlp_hook(li):
                def hook(module, args, output):
                    y = output[0] if isinstance(output, (tuple,list)) else output
                    if self.need_grad and torch.is_tensor(y) and y.requires_grad:
                        y.retain_grad()
                    self.mlp[li] = y
                return hook
            self.handles.append(layer.mlp.register_forward_hook(make_mlp_hook(l)))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        for h in self.handles:
            try: h.remove()
            except Exception: pass
        self.handles.clear()


# -----------------------------------------------------------------------------
# Forward passes
# -----------------------------------------------------------------------------

def locate_positions(processor, batch, subject: str, reference: str):
    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    subject_span, reference_span = centroid_base.locate_object_spans(processor.tokenizer, ids, subject, reference)
    subject_index = int(subject_span[1])
    reference_index = int(reference_span[1])
    subject_positions = direction_base.locate_phrase_positions(processor.tokenizer, ids, subject)
    reference_positions = direction_base.locate_phrase_positions(processor.tokenizer, ids, reference)
    return ids, subject_index, reference_index, subject_positions, reference_positions


def prewo_to_heads(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    # Qwen o_proj input normally [B,T,H*Dh].
    if x.ndim != 3:
        raise RuntimeError(f"Expected o_proj input [B,T,D], got {tuple(x.shape)}")
    if x.shape[-1] != n_heads * head_dim:
        raise RuntimeError(f"o_proj input dim {x.shape[-1]} != {n_heads}*{head_dim}")
    return x.view(x.shape[0], x.shape[1], n_heads, head_dim)


def phrase_mean(heads: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    idx = torch.as_tensor(list(positions), device=heads.device, dtype=torch.long)
    return heads[0].index_select(0, idx).mean(dim=0)  # [H,Dh]


def forward_direction_only(model, processor, device, decoder_layers, layers, n_heads, head_dim, ex: Example, image: Image.Image):
    batch = build_batch(processor, ex.question, image, device)
    _, _, _, sp, rp = locate_positions(processor, batch, ex.subject, ex.reference)
    with SelectedCapture(decoder_layers, layers, cut_layer=None, need_grad=False) as cap:
        with torch.inference_mode():
            _ = model(**batch, output_attentions=False, output_hidden_states=False, use_cache=False, return_dict=True)
    arr = []
    for l in layers:
        h = prewo_to_heads(cap.prewo[l], n_heads, head_dim)
        d = phrase_mean(h, sp) - phrase_mean(h, rp)
        arr.append(d.detach().float().cpu().numpy())
    del batch
    return np.stack(arr, axis=0).astype(np.float32)  # [Ls,H,Dh]


def normalize_attn_tensor(a: torch.Tensor, input_len: int) -> torch.Tensor:
    # Reuse repo normalization; expected [H,Q,K] after helper.
    return centroid_base.normalize_attention_tensor(a, expected_query_length=input_len)


def visual_maps_from_attn(model, processor, batch, attn_selected: Mapping[int, torch.Tensor], layers: Sequence[int], subject_index: int, reference_index: int):
    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
    vis = centroid_base.resolve_visual_indices(model, processor, batch, ids)
    coords = centroid_base.visual_coordinates(model, batch, len(vis), batch["input_ids"].device)
    if coords is None:
        raise RuntimeError(f"Could not construct visual coordinates for {len(vis)} visual tokens")
    vis_idx = torch.as_tensor(vis, device=batch["input_ids"].device, dtype=torch.long)
    maps = []
    masses = []
    ent = []
    for l in layers:
        A = normalize_attn_tensor(attn_selected[l], len(ids))  # [H,Q,K]
        rows = A[:, [subject_index, reference_index], :]       # [H,2,K]
        v = rows.index_select(-1, vis_idx)                     # [H,2,Nv]
        mass = v.sum(dim=-1)
        p = v / v.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        Hn = -(p.clamp_min(1e-12).log() * p).sum(dim=-1) / max(math.log(max(len(vis),2)), 1e-12)
        maps.append(p.detach().float().cpu().numpy())
        masses.append(mass.detach().float().cpu().numpy())
        ent.append(Hn.detach().float().cpu().numpy())
    return (
        np.stack(maps, axis=0),       # [Ls,H,2,Nv]
        np.stack(masses, axis=0),     # [Ls,H,2]
        np.stack(ent, axis=0),        # [Ls,H,2]
        np.asarray(vis, dtype=np.int64),
        coords.detach().float().cpu().numpy(),
    )


def forward_real_with_effects(
    *, model, processor, device, decoder_layers, layers, n_heads, head_dim,
    token_map, relation_u, ex: Example, image: Image.Image, bbox_format: str,
):
    batch = build_batch(processor, ex.question, image, device)
    ids, subject_index, reference_index, sp, rp = locate_positions(processor, batch, ex.subject, ex.reference)

    # Parameters are frozen.  Cut the graph exactly at L18 input: forward values
    # are unchanged, but backward stores only L18+ decoder graph.
    with torch.enable_grad():
        with SelectedCapture(decoder_layers, layers, cut_layer=min(layers), need_grad=True) as cap:
            outputs = model(
                **batch,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            all_attn = attentions_from_outputs(outputs)
            attn_raw = {l: all_attn[l] for l in layers}
            for a in attn_raw.values():
                if a is not None and a.requires_grad:
                    a.retain_grad()

            rel_scores = relation_scores_torch(outputs.logits[0,-1], token_map)
            g = RID[ex.relation]
            wrong = torch.stack([rel_scores[i] for i in range(4) if i != g])
            margin = rel_scores[g] - torch.logsumexp(wrong, dim=0)

            # READ values before backward.
            real_maps, visual_mass_pair, entropy_pair, vis_indices, coords = visual_maps_from_attn(
                model, processor, batch, attn_raw, layers, subject_index, reference_index
            )

            # Direction, DLA values.
            direction = []
            dla = []
            hidden = hidden_states_from_outputs(outputs)
            u_gt = relation_u[g]
            u_wrong = torch.stack([relation_u[i] for i in range(4) if i != g], dim=0).mean(dim=0)
            u_dir = (u_gt - u_wrong).to(device)

            for l in layers:
                heads = prewo_to_heads(cap.prewo[l], n_heads, head_dim)
                d = phrase_mean(heads, sp) - phrase_mean(heads, rp)
                direction.append(d.detach().float().cpu().numpy())

                ans = heads[0, -1].float()  # [H,Dh]
                W = decoder_layers[l].self_attn.o_proj.weight.detach().float()  # [Dmodel,H*Dh]
                W_heads = W.view(W.shape[0], n_heads, head_dim).permute(1,2,0).contiguous()  # [H,Dh,Dmodel]
                o_heads = torch.einsum("hd,hdm->hm", ans, W_heads)  # [H,Dmodel]
                # hidden_states[l] is block input; scaling is only to remove layer scale.
                resid = hidden[l][0,-1].detach().float()
                rms = torch.sqrt(torch.mean(resid*resid)).clamp_min(1e-6)
                dla_l = (o_heads @ u_dir.float()) / rms
                dla.append(dla_l.detach().cpu().numpy())

            # Backward once for ALL selected heads/layers.
            margin.backward()

            grad_act = []
            mlp_ga = []
            grad_attn_profiles = []
            vis_idx_t = torch.as_tensor(vis_indices, device=device, dtype=torch.long)
            object_text_idx = torch.as_tensor([subject_index, reference_index], device=device, dtype=torch.long)

            # bbox masks used both for READ and Grad x Attention edge grouping.
            coords01 = normalize_coords(coords)
            sb = normalize_bbox(ex.subject_bbox_raw, ex.subject_bbox_key, image.size, bbox_format)
            rb = normalize_bbox(ex.reference_bbox_raw, ex.reference_bbox_key, image.size, bbox_format)
            smask_np = bbox_token_mask(coords01, sb)
            rmask_np = bbox_token_mask(coords01, rb)
            smask = torch.as_tensor(np.where(smask_np)[0], device=device, dtype=torch.long) if smask_np is not None else None
            rmask = torch.as_tensor(np.where(rmask_np)[0], device=device, dtype=torch.long) if rmask_np is not None else None

            read_bbox = []
            for li, l in enumerate(layers):
                pre = cap.prewo[l]
                if pre.grad is None:
                    raise RuntimeError(f"No gradient retained for pre-W_O L{l}")
                h = prewo_to_heads(pre, n_heads, head_dim)
                hg = prewo_to_heads(pre.grad, n_heads, head_dim)
                ga = (h.float() * hg.float()).sum(dim=(0,1,3))  # [H]
                grad_act.append(ga.detach().cpu().numpy())

                mo = cap.mlp[l]
                if mo.grad is None:
                    raise RuntimeError(f"No gradient retained for MLP L{l}")
                mga = (mo.float() * mo.grad.float()).sum()
                mlp_ga.append(float(mga.detach().cpu()))

                # READ bbox enrichment from visual-only normalized maps.
                p_np = real_maps[li]  # [H,2,Nv]
                if smask_np is None or rmask_np is None:
                    own_s = np.full(n_heads, np.nan, dtype=np.float32)
                    own_r = np.full(n_heads, np.nan, dtype=np.float32)
                    cross = np.full(n_heads, np.nan, dtype=np.float32)
                else:
                    sbase = max(float(np.mean(smask_np)), 1e-12)
                    rbase = max(float(np.mean(rmask_np)), 1e-12)
                    own_s = p_np[:,0,smask_np].sum(axis=-1) / sbase
                    own_r = p_np[:,1,rmask_np].sum(axis=-1) / rbase
                    cross_sr = p_np[:,0,rmask_np].sum(axis=-1) / rbase
                    cross_rs = p_np[:,1,smask_np].sum(axis=-1) / sbase
                    cross = 0.5*(cross_sr + cross_rs)
                read_bbox.append(np.stack([own_s, own_r, cross], axis=-1))  # [H,3]

                # Grad x Attention. Retained grad is on original returned attention.
                Araw = attn_raw[l]
                if Araw.grad is None:
                    raise RuntimeError(f"No attention gradient at L{l}; returned attention may be detached in this transformers build")
                A = normalize_attn_tensor(Araw, len(ids))
                G = normalize_attn_tensor(Araw.grad, len(ids))
                GE = A.float() * G.float()  # [H,Q,K]

                # answer -> all visual
                ans_vis = GE[:, -1, :].index_select(-1, vis_idx_t).sum(dim=-1)
                # answer -> subject/reference text tokens
                ans_objtxt = GE[:, -1, :].index_select(-1, object_text_idx).sum(dim=-1)

                if smask is not None and rmask is not None:
                    # map local visual mask indices -> decoder key indices
                    sub_keys = vis_idx_t.index_select(0, smask)
                    ref_keys = vis_idx_t.index_select(0, rmask)
                    own = 0.5*(
                        GE[:, subject_index, :].index_select(-1, sub_keys).sum(dim=-1) +
                        GE[:, reference_index, :].index_select(-1, ref_keys).sum(dim=-1)
                    )
                    cross_e = 0.5*(
                        GE[:, subject_index, :].index_select(-1, ref_keys).sum(dim=-1) +
                        GE[:, reference_index, :].index_select(-1, sub_keys).sum(dim=-1)
                    )
                else:
                    own = torch.full((n_heads,), float("nan"), device=device)
                    cross_e = torch.full((n_heads,), float("nan"), device=device)
                prof = torch.stack([own, cross_e, ans_vis, ans_objtxt], dim=-1)
                grad_attn_profiles.append(prof.detach().cpu().numpy())

            rel_scores_np = rel_scores.detach().float().cpu().numpy()
            out = {
                "direction": np.stack(direction, axis=0).astype(np.float32),
                "real_maps": real_maps.astype(np.float32),
                "visual_mass_pair": visual_mass_pair.astype(np.float32),
                "entropy_pair": entropy_pair.astype(np.float32),
                "read_bbox": np.stack(read_bbox, axis=0).astype(np.float32),
                "coords": coords.astype(np.float32),
                "dla": np.stack(dla, axis=0).astype(np.float32),
                "grad_act": np.stack(grad_act, axis=0).astype(np.float32),
                "grad_attn": np.stack(grad_attn_profiles, axis=0).astype(np.float32),
                "mlp_ga": np.asarray(mlp_ga, dtype=np.float32),
                "final_relation_scores": rel_scores_np,
                "final_margin_smooth": float(margin.detach().cpu()),
                "bbox_subject_norm": sb,
                "bbox_reference_norm": rb,
            }

    del outputs, batch
    return out


def forward_hflip_maps(model, processor, device, decoder_layers, layers, n_heads, head_dim, ex: Example, image: Image.Image):
    batch = build_batch(processor, ex.question, image, device)
    ids, si, ri, sp, rp = locate_positions(processor, batch, ex.subject, ex.reference)
    with SelectedCapture(decoder_layers, layers, cut_layer=None, need_grad=False) as cap:
        with torch.inference_mode():
            outputs = model(**batch, output_attentions=True, output_hidden_states=False, use_cache=False, return_dict=True)
    all_attn = attentions_from_outputs(outputs)
    attn = {l: all_attn[l] for l in layers}
    maps, masses, entropy, vis, coords = visual_maps_from_attn(model, processor, batch, attn, layers, si, ri)
    direction = []
    for l in layers:
        heads = prewo_to_heads(cap.prewo[l], n_heads, head_dim)
        d = phrase_mean(heads, sp) - phrase_mean(heads, rp)
        direction.append(d.detach().float().cpu().numpy())
    del outputs, batch
    return np.stack(direction, axis=0).astype(np.float32), maps.astype(np.float32), coords.astype(np.float32)


# -----------------------------------------------------------------------------
# Direction codebook
# -----------------------------------------------------------------------------

def fit_direction_codebooks(train_dirs: np.ndarray, labels: Sequence[str]):
    # [N,Ls,H,D]
    N,L,H,D = train_dirs.shape
    centers = np.zeros((L,H,D), dtype=np.float32)
    dirs = np.zeros((L,H,4,D), dtype=np.float32)
    y = np.asarray(labels)
    for l in range(L):
        for h in range(H):
            c,d = direction_base.fit_codebook(train_dirs[:,l,h,:].astype(np.float32), y)
            centers[l,h] = c
            dirs[l,h] = d
    return centers, dirs


def direction_scores(X: np.ndarray, centers: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    # X [Ls,H,D] -> [Ls,H,4]
    Z = X.astype(np.float32) - centers
    Z = Z / np.maximum(np.linalg.norm(Z, axis=-1, keepdims=True), 1e-12)
    return np.einsum("lhd,lhkd->lhk", Z, dirs)


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def mean_or_nan(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    if a.size == 0 or np.all(np.isnan(a)):
        return float("nan")
    return float(np.nanmean(a))


def summarize(sample_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (layer, head), g in sample_df.groupby(["layer","head"], sort=True):
        row = {
            "layer": int(layer),
            "head": int(head),
            "head_name": f"L{int(layer)}H{int(head):02d}",
            "n": int(len(g)),
            "visual_mass": mean_or_nan(g.visual_mass),
            "subject_mass": mean_or_nan(g.subject_mass),
            "reference_mass": mean_or_nan(g.reference_mass),
            "cross_object_attention": mean_or_nan(g.cross_object_attention),
            "spatial_entropy": mean_or_nan(g.spatial_entropy),
            "hflip_equivariance": mean_or_nan(g.hflip_equivariance),
            "direction_acc": float(np.mean(g.direction_correct)),
            "direction_margin": mean_or_nan(g.direction_margin),
            "gray_direction_acc": float(np.mean(g.gray_direction_correct)),
            "gray_direction_margin": mean_or_nan(g.gray_direction_margin),
            "gray_drop": float(np.mean(g.direction_correct) - np.mean(g.gray_direction_correct)),
            "gray_margin_drop": mean_or_nan(g.direction_margin - g.gray_direction_margin),
            "dla": mean_or_nan(g.dla),
            "dla_abs": mean_or_nan(np.abs(g.dla)),
            "grad_activation_signed": mean_or_nan(g.grad_activation),
            "grad_activation_abs": mean_or_nan(np.abs(g.grad_activation)),
            "grad_attention_own_vis": mean_or_nan(g.grad_attention_own_vis),
            "grad_attention_cross_vis": mean_or_nan(g.grad_attention_cross_vis),
            "grad_attention_answer_visual": mean_or_nan(g.grad_attention_answer_visual),
            "grad_attention_answer_object_text": mean_or_nan(g.grad_attention_answer_object_text),
            "mlp_grad_activation_signed": mean_or_nan(g.mlp_grad_activation),
            "mlp_grad_activation_abs": mean_or_nan(np.abs(g.mlp_grad_activation)),
            "final_correct_rate": float(np.mean(g.final_correct)),
        }
        # Effect sign can be different on correct/wrong samples; store both.
        for tag, mask in (("correct", g.final_correct == 1), ("wrong", g.final_correct == 0)):
            gg = g[mask]
            row[f"grad_activation_{tag}_signed"] = mean_or_nan(gg.grad_activation)
            row[f"dla_{tag}_signed"] = mean_or_nan(gg.dla)
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--output-root", type=Path, default=Path("output/head_passport_qwen3b_L18_L28_v1"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer-start", type=int, default=18)
    ap.add_argument("--layer-end", type=int, default=28)
    ap.add_argument("--max-samples", type=int, default=200, help="0 = all COCO_two")
    ap.add_argument("--train-ratio", type=float, default=0.15)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--bbox-format", choices=("auto","xyxy","xywh"), default="auto")
    ap.add_argument("--skip-hflip", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--empty-cache-every", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out = args.output_root
    if out.exists() and args.force:
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    examples, audit = load_examples(args.data_root, args.max_samples, args.seed)
    if len(examples) < 20:
        raise RuntimeError(f"Only {len(examples)} usable examples")
    train, val, test = stratified_split(examples, args.train_ratio, args.val_ratio, args.seed)

    split_rows = []
    for name, xs in (("train",train),("val",val),("test",test)):
        for x in xs:
            split_rows.append({"sid":x.sid,"split":name,"relation":x.relation,"subject":x.subject,"reference":x.reference})
    pd.DataFrame(split_rows).to_csv(out/"split.csv", index=False)

    bbox_audit = {
        "examples": len(examples),
        "with_bbox_pair_raw": int(sum(x.subject_bbox_raw is not None and x.reference_bbox_raw is not None for x in examples)),
        "subject_bbox_keys": dict(Counter(x.subject_bbox_key for x in examples if x.subject_bbox_key)),
        "reference_bbox_keys": dict(Counter(x.reference_bbox_key for x in examples if x.reference_bbox_key)),
        "bbox_format_requested": args.bbox_format,
        "note": "If coverage is zero, bbox-dependent passport fields are NaN; the script does not silently replace their definition.",
    }
    (out/"bbox_audit.json").write_text(json.dumps(bbox_audit, indent=2, ensure_ascii=False), encoding="utf-8")

    actual, spec, model, processor = load_model(args.device)
    device = torch.device(args.device)
    decoder_layers, decoder_path = lens_base.resolve_decoder_layers(model)
    n_total_layers = len(decoder_layers)
    n_heads, head_dim = direction_base.scan_shape(model, decoder_layers)
    if args.layer_start < 0 or args.layer_end >= n_total_layers or args.layer_start > args.layer_end:
        raise ValueError(f"Invalid layer range {args.layer_start}..{args.layer_end}; model has {n_total_layers} layers")
    layers = list(range(args.layer_start, args.layer_end+1))

    lm_head, lm_head_path = lens_base.resolve_output_embeddings(model)
    token_map = relation_token_map(processor.tokenizer)
    relation_u = unembedding_relation_vectors(lm_head, token_map)

    config = {
        "script_version": SCRIPT_VERSION,
        "model_alias": MODEL_ALIAS,
        "model_actual": actual,
        "repo_id": spec.repo_id,
        "dataset": DATASET,
        "N": len(examples),
        "train_N": len(train),
        "val_N": len(val),
        "test_N": len(test),
        "layers": layers,
        "n_total_layers": n_total_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "decoder_path": decoder_path,
        "lm_head_path": lm_head_path,
        "token_map": token_map,
        "target": "M = s_GT - logsumexp(s_wrong)",
        "write_space": "pre-W_O",
        "effect_space": "actual W_O-coupled first-order effect",
        "bbox_audit": bbox_audit,
    }
    (out/"config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out/"dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"[model] {actual} repo={spec.repo_id}")
    print(f"[shape] layers={n_total_layers}, heads={n_heads}, head_dim={head_dim}; scanning {layers[0]}..{layers[-1]}")
    print(f"[data] N={len(examples)} train={len(train)} val={len(val)} test={len(test)}")
    print(f"[bbox] raw pair coverage={bbox_audit['with_bbox_pair_raw']}/{len(examples)}")

    # ------------------------------------------------------------------
    # Stage 1: fit WRITE direction codebook on Real TRAIN only.
    # ------------------------------------------------------------------
    train_dirs = []
    train_labels = []
    errors = []
    for ex in tqdm(train, desc="train-direction-real"):
        try:
            d = forward_direction_only(model, processor, device, decoder_layers, layers, n_heads, head_dim, ex, ex.image())
            train_dirs.append(d)
            train_labels.append(ex.relation)
        except Exception as e:
            errors.append({"stage":"train_direction","sid":ex.sid,"error":repr(e),"traceback":traceback.format_exc()})
    if len(train_dirs) < 8:
        raise RuntimeError(f"Only {len(train_dirs)} successful train directions")
    Xtr = np.stack(train_dirs, axis=0)
    centers, dirs = fit_direction_codebooks(Xtr, train_labels)
    np.savez_compressed(out/"direction_codebook.npz", centers=centers, directions=dirs, layers=np.asarray(layers), labels=np.asarray(RELATIONS))
    del Xtr, train_dirs
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage 2: TEST passport. Each sample: Real backward + Gray + HFlip.
    # ------------------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    layer_context_rows: List[Dict[str, Any]] = []

    for ii, ex in enumerate(tqdm(test, desc="test-passport")):
        try:
            real_img = ex.image()
            real = forward_real_with_effects(
                model=model, processor=processor, device=device,
                decoder_layers=decoder_layers, layers=layers,
                n_heads=n_heads, head_dim=head_dim,
                token_map=token_map, relation_u=relation_u,
                ex=ex, image=real_img, bbox_format=args.bbox_format,
            )
            real_dscore = direction_scores(real["direction"], centers, dirs)  # [Ls,H,4]

            gray_img = Image.new("RGB", real_img.size, color=(128,128,128))
            gray_dir = forward_direction_only(model, processor, device, decoder_layers, layers, n_heads, head_dim, ex, gray_img)
            gray_dscore = direction_scores(gray_dir, centers, dirs)

            if not args.skip_hflip:
                flip_dir, flip_maps, flip_coords = forward_hflip_maps(
                    model, processor, device, decoder_layers, layers, n_heads, head_dim,
                    ex, ImageOps.mirror(real_img)
                )
                c01 = normalize_coords(real["coords"])
                fc01 = normalize_coords(flip_coords)
                # Geometry should have the same token grid; use real-grid mirror map
                # and compare to the returned flip-grid ordering.
                perm = mirror_permutation(c01)
                expected = np.empty_like(real["real_maps"])
                expected[..., perm] = real["real_maps"]
                # if flip grid ordering differs slightly, this still works for Qwen's
                # same-size HFlip; warn via value degradation rather than remapping by GT.
                hflip_eq = cosine_rows(expected.reshape(len(layers),n_heads,2,-1), flip_maps.reshape(len(layers),n_heads,2,-1)).mean(axis=-1)
            else:
                hflip_eq = np.full((len(layers), n_heads), np.nan, dtype=np.float32)

            final_stats = relation_stats_np(real["final_relation_scores"], ex.relation)

            for li,l in enumerate(layers):
                layer_context_rows.append({
                    "sid":ex.sid,
                    "layer":l,
                    "relation":ex.relation,
                    "final_correct":final_stats["correct"],
                    "final_margin_smooth":real["final_margin_smooth"],
                    "mlp_grad_activation":float(real["mlp_ga"][li]),
                })
                for h in range(n_heads):
                    ds = real_dscore[li,h]
                    gs = gray_dscore[li,h]
                    dst = relation_stats_np(ds, ex.relation)
                    gst = relation_stats_np(gs, ex.relation)
                    visual_mass = float(np.mean(real["visual_mass_pair"][li,h]))
                    entropy = float(np.mean(real["entropy_pair"][li,h]))
                    bb = real["read_bbox"][li,h]
                    ge = real["grad_attn"][li,h]
                    rows.append({
                        "sid": ex.sid,
                        "relation": ex.relation,
                        "subject": ex.subject,
                        "reference": ex.reference,
                        "layer": l,
                        "head": h,
                        "head_name": f"L{l}H{h:02d}",
                        # READ
                        "visual_mass": visual_mass,
                        "subject_mass": float(bb[0]),
                        "reference_mass": float(bb[1]),
                        "cross_object_attention": float(bb[2]),
                        "spatial_entropy": entropy,
                        "hflip_equivariance": float(hflip_eq[li,h]),
                        # WRITE
                        "direction_pred": dst["pred"],
                        "direction_correct": dst["correct"],
                        "direction_margin": dst["margin"],
                        "gray_direction_pred": gst["pred"],
                        "gray_direction_correct": gst["correct"],
                        "gray_direction_margin": gst["margin"],
                        "gray_margin_drop": float(dst["margin"] - gst["margin"]),
                        # EFFECT
                        "dla": float(real["dla"][li,h]),
                        "grad_activation": float(real["grad_act"][li,h]),
                        "grad_attention_own_vis": float(ge[0]),
                        "grad_attention_cross_vis": float(ge[1]),
                        "grad_attention_answer_visual": float(ge[2]),
                        "grad_attention_answer_object_text": float(ge[3]),
                        # CONTEXT / final
                        "mlp_grad_activation": float(real["mlp_ga"][li]),
                        "final_pred": final_stats["pred"],
                        "final_correct": final_stats["correct"],
                        "final_margin_hard": final_stats["margin"],
                        "final_margin_smooth": real["final_margin_smooth"],
                        "bbox_available": int(real["bbox_subject_norm"] is not None and real["bbox_reference_norm"] is not None),
                    })

            del real, real_dscore, gray_dir, gray_dscore
            if not args.skip_hflip:
                del flip_dir, flip_maps

        except Exception as e:
            errors.append({"stage":"test_passport","sid":ex.sid,"error":repr(e),"traceback":traceback.format_exc()})

        if args.empty_cache_every > 0 and (ii+1) % args.empty_cache_every == 0:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No successful TEST passport rows")

    sample_df = pd.DataFrame(rows)
    sample_df.to_csv(out/"head_passport_sample.csv.gz", index=False, compression="gzip")
    summary_df = summarize(sample_df)
    summary_df.to_csv(out/"head_passport_summary.csv", index=False)

    lc = pd.DataFrame(layer_context_rows)
    lc_sum = lc.groupby("layer", as_index=False).agg(
        n=("sid","count"),
        mlp_grad_activation_signed=("mlp_grad_activation","mean"),
        mlp_grad_activation_abs=("mlp_grad_activation", lambda x: float(np.mean(np.abs(x)))),
        final_correct_rate=("final_correct","mean"),
    )
    lc.to_csv(out/"layer_context_sample.csv.gz", index=False, compression="gzip")
    lc_sum.to_csv(out/"layer_context_summary.csv", index=False)

    with (out/"errors.jsonl").open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False, default=str)+"\n")

    done = {
        "script_version": SCRIPT_VERSION,
        "sample_rows": int(len(sample_df)),
        "head_summary_rows": int(len(summary_df)),
        "successful_test_sids": int(sample_df.sid.nunique()),
        "requested_test_sids": len(test),
        "errors": len(errors),
        "bbox_available_test_fraction": float(sample_df.groupby("sid").bbox_available.first().mean()),
    }
    (out/"DONE").write_text(json.dumps(done, indent=2), encoding="utf-8")

    print("\n[DONE]")
    print(json.dumps(done, indent=2))
    print(f"sample : {out/'head_passport_sample.csv.gz'}")
    print(f"summary: {out/'head_passport_summary.csv'}")


if __name__ == "__main__":
    main()
