#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_causal_top7_reader_transporter_heads_standalone_v1.py

Standalone Qwen2.5-VL causal-state -> head analysis.

NO AdaptVis Python imports.

Goal
====
Start from EACH SAMPLE'S OWN strongest causal states instead of asking spatial
heads to discover them.

Default causal set:
    core50_candidates_all440.csv
    -> within each sample sort by mediation strength
    -> keep strongest Top7 TEXT causal states.

For every causal state:
    (source layer L, token position p)

ask two separate questions.

====================================================================
A) DIRECT READER HEADS
====================================================================
The exact causal state h_L[p] is the INPUT to attention layer H=L+1.
Therefore DIRECT readers are ONLY heads at H=L+1.

For head h and future text query q:

    c_RG(h,q<-p)
      = A_real[h,q,p] V_real[h,p]
        -
        A_gray[h,q,p] V_gray[h,p]

Reader strength for causal state p:

    R_h(L,p)
      = max_{q in future text queries} || c_RG(h,q<-p) ||_2

We also calculate:
    - max real attention A_real[h,q,p]
    - max |A_real-A_gray|
    - source percentile:
        rank causal p against ALL eligible text source positions for that head,
        using the same max-over-future-query Real-Gray A*V norm.
    - head rank:
        rank all query heads at H=L+1 by R_h(L,p).

Default STRONG READER:
    source_percentile >= 0.90
    AND
    head_rank_within_aligned_layer <= 3

This is strict same-layer:
    causal L -> reader H=L+1.

====================================================================
B) TRANSPORTER HEADS
====================================================================
Now ask which downstream heads carry the SAME TOKEN POSITION toward prompt-last.

For each downstream attention layer H >= L+1 and query head h:

    m_RG(H,h,p->last)
      = W_O^{H,h} [
            A_real[h,last,p] V_real[h,p]
            -
            A_gray[h,last,p] V_gray[h,p]
        ]

Transport strength:

    T_{H,h}(L,p) = || m_RG(H,h,p->last) ||_2

For each transporter we calculate:
    - raw post-W_O Real-Gray message norm
    - target causal-token source percentile among all text source positions
      for that same head's prompt-last message
    - rank within that transport layer
    - GLOBAL rank among all scanned downstream heads H>=L+1
    - hop offset H-(L+1)

Default STRONG TRANSPORTER:
    source_percentile >= 0.90
    AND
    global_downstream_head_rank <= 3

Important interpretation:
- H=L+1 is direct transport of the exact causal state.
- H>L+1 is downstream transport of the SAME TOKEN POSITION after intervening
  transformer blocks have updated it.  It is a token-lineage transport proxy,
  not proof that the exact original residual vector survived unchanged.

====================================================================
Why source percentile AND head rank?
====================================================================
Raw message norm alone can prefer heads with globally large output scales.
We require BOTH:

1) causal token is unusually strong among source tokens for that head;
2) head is unusually strong for this causal state.

This is meant to surface genuinely selective readers / transporters.

====================================================================
Outputs
====================================================================
causal_top7_states.csv
reader_all_heads.csv
strong_readers.csv
reader_head_summary.csv
reader_by_causal_layer.csv
reader_by_token_class.csv

transporter_all_heads.csv
strong_transporters.csv
transporter_head_summary.csv
transporter_by_causal_layer.csv
transporter_by_token_class.csv

causal_state_best_heads.csv
analysis_summary.txt
metadata.json
errors.jsonl

====================================================================
Recommended
====================================================================
CUDA_VISIBLE_DEVICES=0 python -u \
  scan_causal_top7_reader_transporter_heads_standalone_v1.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --data-root data \
  --prompt-jsonl prompts/COCO_QA_two_obj_with_answer_four_options.jsonl \
  --oracle-bank-dir output/qwen3b_oracle_core50_top10_bank_all440 \
  --causal-bank core50 \
  --causal-top-k 7 \
  --causal-domain text \
  --reader-source-percentile 0.90 \
  --reader-top-heads 3 \
  --transport-layers 21-31 \
  --transport-source-percentile 0.90 \
  --transport-top-heads 3 \
  --require-n 440 \
  --output-dir output/qwen3b_causal_top7_reader_transporter_v1 \
  --overwrite

For the old GLOBAL causal Top7 instead of Core50-first Top7:
    --causal-bank top10 --causal-top-k 7
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import re
import shutil
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")


REL = ("left", "right", "above", "below")
EPS = 1e-12

STANDARD_OBJECT_RE = re.compile(
    r"Where\s+(?:is|are)\s+the\s+(.+?)\s+in\s+relation\s+to\s+the\s+(.+?)\?\s*Answer\s+with",
    flags=re.IGNORECASE | re.DOTALL,
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument(
        "--model-class",
        default="Qwen2_5_VLForConditionalGeneration",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )

    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--oracle-bank-dir", required=True)
    p.add_argument(
        "--causal-bank",
        choices=["core50", "top10"],
        default="core50",
    )
    p.add_argument("--causal-top-k", type=int, default=7)
    p.add_argument(
        "--causal-domain",
        choices=["text", "all"],
        default="text",
    )
    p.add_argument(
        "--causal-order",
        choices=["mediation", "rank"],
        default="mediation",
        help="Within causal bank, choose strongest states by mediation or original rank.",
    )

    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument(
        "--reader-source-percentile",
        type=float,
        default=0.90,
    )
    p.add_argument("--reader-top-heads", type=int, default=3)

    p.add_argument(
        "--transport-layers",
        default="21-31",
        help="Downstream attention layers to scan, e.g. 21-31.",
    )
    p.add_argument(
        "--transport-source-percentile",
        type=float,
        default=0.90,
    )
    p.add_argument("--transport-top-heads", type=int, default=3)

    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--require-n", type=int, default=0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--save-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save all per-state/per-head rows, not just strong rows.",
    )
    return p.parse_args()


# =============================================================================
# Generic helpers
# =============================================================================

def resolve_dtype(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def parse_layer_spec(text: str) -> List[int]:
    out = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def safe_mean(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def safe_median(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def hname(L, h):
    return f"L{int(L)}H{int(h):02d}"


def boolify(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"true", "1", "yes", "t", "y"}


def canon_rel(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    aliases = {
        "left": "left",
        "right": "right",
        "above": "above",
        "on": "above",
        "over": "above",
        "below": "below",
        "under": "below",
        "underneath": "below",
        "beneath": "below",
    }
    if s in aliases:
        return aliases[s]
    for token, rel in (
        ("left", "left"),
        ("right", "right"),
        ("below", "below"),
        ("under", "below"),
        ("above", "above"),
        ("over", "above"),
    ):
        if re.search(rf"\b{token}\b", s):
            return rel
    return None


def append_jsonl(path: Path, row: Mapping[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def percentile_desc(values: np.ndarray, target_index: int) -> float:
    """
    1.0 = strongest, 0.0 = weakest.
    Average percentile for ties.
    """
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0 or not (0 <= target_index < len(x)):
        return float("nan")
    v = x[target_index]
    if not np.isfinite(v):
        return float("nan")
    finite = x[np.isfinite(x)]
    if len(finite) <= 1:
        return 1.0
    greater = float(np.sum(finite > v))
    equal = float(np.sum(finite == v))
    avg_rank0 = greater + 0.5 * max(0.0, equal - 1.0)
    return float(1.0 - avg_rank0 / (len(finite) - 1.0))


def ranks_desc(values: Sequence[float]) -> List[int]:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(-x, kind="mergesort")
    ranks = np.empty(len(x), dtype=int)
    for i, idx in enumerate(order, 1):
        ranks[idx] = i
    return ranks.tolist()


# =============================================================================
# Dataset
# =============================================================================

@dataclass(frozen=True)
class CocoRecord:
    sid: int
    image_path: Path


def load_coco_records(data_root: Path) -> Dict[int, CocoRecord]:
    ann = data_root / "coco_qa_two_obj.json"
    img_dir = data_root / "val2017"
    if not ann.exists():
        raise FileNotFoundError(ann)
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)

    raw = json.loads(ann.read_text(encoding="utf-8"))
    out = {}
    for sid, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or not row:
            continue
        image_id = int(row[0])
        path = img_dir / f"{image_id:012d}.jpg"
        if path.exists():
            out[sid] = CocoRecord(sid=sid, image_path=path)
    return out


def extract_standard_user_text(text):
    text = str(text).strip()
    text = re.sub(r"^\s*<image>\s*", "", text, flags=re.IGNORECASE)
    m = re.search(
        r"\bUSER\s*:\s*(.*?)(?:\s*\bASSISTANT\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        text = m.group(1)
    return text.strip()


def parse_standard_objects(question_text):
    compact = re.sub(r"\s+", " ", str(question_text)).strip()
    m = STANDARD_OBJECT_RE.search(compact)
    if not m:
        raise ValueError(f"Cannot parse objects from: {compact!r}")
    return m.group(1).strip(), m.group(2).strip()


def load_prompts(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = int(row["id"])
            q = extract_standard_user_text(row["question"])
            subject, reference = parse_standard_objects(q)
            answer = row["answer"]
            if isinstance(answer, (list, tuple)):
                answer = answer[0] if answer else None
            gt = canon_rel(answer)
            out[sid] = {
                "sid": sid,
                "question_text": q,
                "subject": subject,
                "reference": reference,
                "gt": gt,
            }
    return out


# =============================================================================
# Causal bank
# =============================================================================

def resolve_rank_col(df):
    for c in (
        "oracle_rank_k36",
        "causal_rank",
        "rank",
        "rank_k36",
        "rank_in_sample",
    ):
        if c in df.columns:
            return c
    return None


def resolve_mediation_col(df):
    for c in (
        "mediation",
        "causal_M",
        "M",
        "mediation_score",
        "score",
    ):
        if c in df.columns:
            return c
    return None


def load_causal_topk(
    root: Path,
    bank_name: str,
    top_k: int,
    domain: str,
    order_mode: str,
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], str, Optional[str]]:
    filename = (
        "core50_candidates_all440.csv"
        if bank_name == "core50"
        else "top10_candidates_all440.csv"
    )
    path = root / filename
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {"sid", "source_layer", "position"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path} missing {sorted(missing)}")

    for c in ("sid", "source_layer", "position"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)

    if "is_text" in df.columns:
        df["is_text"] = df["is_text"].map(boolify)
    else:
        broad = df.get(
            "broad_category",
            pd.Series([""] * len(df)),
        ).astype(str)
        cat = df.get(
            "category",
            pd.Series([""] * len(df)),
        ).astype(str)
        df["is_text"] = (~broad.eq("visual")) & (~cat.eq("visual"))

    if domain == "text":
        df = df[df["is_text"]].copy()

    rank_col = resolve_rank_col(df)
    med_col = resolve_mediation_col(df)

    if order_mode == "mediation" and med_col is None and rank_col is None:
        raise RuntimeError("No mediation or rank column available.")
    if order_mode == "rank" and rank_col is None and med_col is None:
        raise RuntimeError("No rank or mediation column available.")

    selected_groups = {}
    rows = []

    for sid, g in df.groupby("sid"):
        q = g.copy()

        if order_mode == "mediation" and med_col is not None:
            q[med_col] = pd.to_numeric(q[med_col], errors="coerce")
            q = q.sort_values(
                [med_col] + ([rank_col] if rank_col else []),
                ascending=[False] + ([True] if rank_col else []),
            )
        elif rank_col is not None:
            q[rank_col] = pd.to_numeric(q[rank_col], errors="coerce")
            q = q.sort_values(rank_col, ascending=True)
        elif med_col is not None:
            q[med_col] = pd.to_numeric(q[med_col], errors="coerce")
            q = q.sort_values(med_col, ascending=False)

        q = q.head(int(top_k)).copy()
        q["causal_topk_rank"] = np.arange(1, len(q) + 1)
        selected_groups[int(sid)] = q
        rows.append(q)

    selected = pd.concat(rows, ignore_index=True) if rows else df.iloc[:0].copy()
    return selected, selected_groups, rank_col, med_col


def token_class_from_row(row) -> Tuple[str, str]:
    for c in ("canonical_token_class",):
        if hasattr(row, c):
            v = getattr(row, c)
            if pd.notna(v) and str(v):
                key = getattr(row, "canonical_token_key", "")
                return str(v), str(key)

    category = str(getattr(row, "category", ""))
    broad = str(getattr(row, "broad_category", ""))
    token = str(getattr(row, "token", ""))

    low = category.lower()
    if "subject" in low:
        return "subject", "SUBJECT"
    if "reference" in low:
        return "reference", "REFERENCE"
    if "relation" in low:
        return "relation_word", token
    return broad or "other_text", token


# =============================================================================
# Processor / positions
# =============================================================================

def configure_processor(model, processor):
    cfg = getattr(model, "config", None)
    vc = getattr(cfg, "vision_config", None)
    if vc is not None and hasattr(processor, "patch_size"):
        ps = getattr(vc, "patch_size", None)
        if ps is not None:
            processor.patch_size = int(ps)


def build_prompt(processor, question_text):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question_text},
        ],
    }]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def make_batch(processor, image, question_text, device):
    rendered = build_prompt(processor, question_text)
    batch = processor(
        text=[rendered],
        images=[image],
        return_tensors="pt",
    )
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def candidate_token_id(tokenizer, token):
    try:
        i = int(tokenizer.convert_tokens_to_ids(token))
    except Exception:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and i == int(unk):
        return None
    return i


def resolve_visual_indices(model, processor, batch, input_ids):
    mm = batch.get("mm_token_type_ids")
    if torch.is_tensor(mm) and mm.ndim == 2:
        idx = torch.nonzero(mm[0] == 1, as_tuple=False).flatten().tolist()
        if idx:
            return [int(x) for x in idx]

    tt = batch.get("token_type_ids")
    if torch.is_tensor(tt) and tt.ndim == 2:
        vals = set(int(x) for x in tt[0].detach().cpu().tolist())
        if 1 in vals:
            idx = torch.nonzero(tt[0] == 1, as_tuple=False).flatten().tolist()
            if idx:
                return [int(x) for x in idx]

    token_ids = set()
    for obj in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        processor,
        getattr(processor, "tokenizer", None),
    ):
        if obj is None:
            continue
        for name in ("image_token_id", "image_token_index"):
            v = getattr(obj, name, None)
            if isinstance(v, (int, np.integer)) and int(v) >= 0:
                token_ids.add(int(v))

    for token in ("<|image_pad|>", "<image>", "<IMG_CONTEXT>"):
        i = candidate_token_id(processor.tokenizer, token)
        if i is not None:
            token_ids.add(i)

    idx = [i for i, x in enumerate(input_ids) if int(x) in token_ids]
    if idx:
        return idx
    raise RuntimeError("Could not locate visual-token positions.")


def token_strings(processor, ids):
    out = []
    tok = processor.tokenizer
    for x in ids:
        try:
            s = tok.convert_ids_to_tokens(int(x))
        except Exception:
            s = str(x)
        out.append(str(s).replace("\n", "\\n"))
    return out


# =============================================================================
# Model internals / capture
# =============================================================================

def get_attr_path(root, path):
    x = root
    for part in path.split("."):
        if not hasattr(x, part):
            return None
        x = getattr(x, part)
    return x


def resolve_decoder_layers(model):
    preferred = [
        "model.language_model.layers",
        "model.model.language_model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "model.language_model.model.layers",
        "model.model.layers",
        "model.layers",
    ]
    for path in preferred:
        x = get_attr_path(model, path)
        if isinstance(x, (torch.nn.ModuleList, list, tuple)) and len(x) >= 4:
            return x, path

    found = []
    for name, module in model.named_modules():
        x = getattr(module, "layers", None)
        if isinstance(x, torch.nn.ModuleList) and len(x) >= 4:
            found.append((name, x))
    if not found:
        raise RuntimeError("Cannot resolve decoder layers.")
    found.sort(key=lambda z: -len(z[1]))
    return found[0][1], found[0][0]


def get_text_config(model):
    cfg = getattr(model, "config", None)
    for x in (
        getattr(cfg, "text_config", None),
        getattr(cfg, "language_config", None),
        cfg,
    ):
        if x is not None and getattr(x, "num_attention_heads", None) is not None:
            return x
    raise RuntimeError("Cannot resolve text config.")


def resolve_attn(layer):
    for name in ("self_attn", "attention", "attn"):
        x = getattr(layer, name, None)
        if x is not None:
            return x
    raise RuntimeError("Cannot resolve attention module.")


def resolve_o_proj(attn):
    for name in ("o_proj", "out_proj", "proj"):
        x = getattr(attn, name, None)
        if isinstance(x, torch.nn.Module):
            return x
    raise RuntimeError("Cannot resolve o_proj.")


def resolve_v_proj(attn):
    for name in ("v_proj", "value", "value_proj"):
        x = getattr(attn, name, None)
        if isinstance(x, torch.nn.Module):
            return x
    raise RuntimeError("Cannot resolve v_proj.")


def extract_attentions(outputs):
    candidates = [
        getattr(outputs, "attentions", None),
        getattr(
            getattr(outputs, "language_model_outputs", None),
            "attentions",
            None,
        ),
        getattr(
            getattr(outputs, "text_model_output", None),
            "attentions",
            None,
        ),
    ]
    for x in candidates:
        if isinstance(x, (list, tuple)) and len(x):
            return tuple(x)
    raise RuntimeError("No attentions returned; use eager attention.")


def norm_attn(x):
    if x.ndim == 4:
        x = x[0]
    return x.detach().float().cpu().numpy().astype(np.float32)


def all_head_v(vout, n_heads, head_dim):
    x = vout[0]
    nkv = x.shape[-1] // head_dim
    x = x.reshape(x.shape[0], nkv, head_dim)
    if nkv == n_heads:
        return x
    if n_heads % nkv:
        raise RuntimeError(
            f"query heads={n_heads} not divisible by kv heads={nkv}"
        )
    return np.repeat(x, n_heads // nkv, axis=1)


class VCapture:
    def __init__(self, decoder_layers, scan_layers):
        self.v = {}
        self.handles = []
        for L in scan_layers:
            vp = resolve_v_proj(resolve_attn(decoder_layers[L]))

            def make_hook(layer_idx):
                def hook(_m, _inp, output):
                    self.v[layer_idx] = (
                        output.detach().float().cpu().numpy().astype(np.float32)
                    )
                return hook

            self.handles.append(
                vp.register_forward_hook(make_hook(L))
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


@torch.inference_mode()
def run_trace(model, decoder_layers, batch, scan_layers):
    cap = VCapture(decoder_layers, scan_layers)
    try:
        kw = dict(batch)
        kw["use_cache"] = False
        kw["return_dict"] = True
        kw["output_attentions"] = True
        out = model(**kw)
        aa = extract_attentions(out)
        return {
            "attn": {L: norm_attn(aa[L]) for L in scan_layers},
            "v": cap.v,
        }
    finally:
        cap.close()


def o_proj_head_matrix(decoder_layers, L, h, head_dim):
    op = resolve_o_proj(resolve_attn(decoder_layers[L]))
    W = op.weight.detach().float().cpu().numpy()
    return W[:, h * head_dim:(h + 1) * head_dim].astype(np.float32)


# =============================================================================
# Reader / transporter scores
# =============================================================================

def eligible_text_positions(input_ids, visual_indices):
    visual = set(map(int, visual_indices))
    prompt_last = len(input_ids) - 1
    return [
        p for p in range(len(input_ids))
        if p not in visual and p != prompt_last
    ]


def future_text_queries(text_positions, p, prompt_last):
    qs = [q for q in text_positions if q > int(p)]
    if prompt_last > int(p):
        qs.append(prompt_last)
    return sorted(set(qs))


def reader_scores_for_state(
    real_trace,
    gray_trace,
    H,
    p,
    text_positions,
    prompt_last,
    n_heads,
    head_dim,
):
    """
    Direct aligned reader at H=L+1.

    For each source s:
      source_score_h[s] = max future text q ||A_R V_R - A_G V_G||.

    Return one row per head for target p.
    """
    Ar = real_trace["attn"][H]
    Ag = gray_trace["attn"][H]
    Vr = all_head_v(real_trace["v"][H], n_heads, head_dim)
    Vg = all_head_v(gray_trace["v"][H], n_heads, head_dim)

    n = min(Ar.shape[-1], Ag.shape[-1], Vr.shape[0], Vg.shape[0])
    sources = [s for s in text_positions if 0 <= s < n]
    if int(p) not in sources:
        return []

    qs = [
        q for q in future_text_queries(text_positions, p, prompt_last)
        if 0 <= q < Ar.shape[1]
    ]
    if not qs:
        return []

    p_idx = sources.index(int(p))
    rows = []

    for h in range(n_heads):
        # [Q,S,D]
        rr = (
            Ar[h, qs][:, sources, None]
            * Vr[np.asarray(sources), h][None, :, :]
        )
        gg = (
            Ag[h, qs][:, sources, None]
            * Vg[np.asarray(sources), h][None, :, :]
        )
        diff = rr - gg
        norms = np.linalg.norm(diff, axis=-1)  # [Q,S]
        source_score = norms.max(axis=0)

        target_score = float(source_score[p_idx])
        source_pct = percentile_desc(source_score, p_idx)

        real_attn_target = float(
            np.max(Ar[h, qs, int(p)])
        )
        delta_attn_target = float(
            np.max(np.abs(
                Ar[h, qs, int(p)] - Ag[h, qs, int(p)]
            ))
        )

        rows.append({
            "head_layer": int(H),
            "head": int(h),
            "head_name": hname(H, h),
            "reader_av_rg_norm": target_score,
            "reader_source_percentile": source_pct,
            "reader_real_attn_max": real_attn_target,
            "reader_abs_delta_attn_max": delta_attn_target,
            "reader_query_N": len(qs),
            "reader_source_N": len(sources),
        })

    ranks = ranks_desc([r["reader_av_rg_norm"] for r in rows])
    for r, rank in zip(rows, ranks):
        r["reader_head_rank"] = int(rank)
        r["reader_head_percentile"] = (
            1.0
            if len(rows) <= 1
            else 1.0 - (rank - 1) / (len(rows) - 1)
        )
    return rows


def transporter_scores_for_state(
    decoder_layers,
    real_trace,
    gray_trace,
    scan_layers,
    aligned_H,
    p,
    text_positions,
    prompt_last,
    n_heads,
    head_dim,
):
    """
    Prompt-last transport scan.

    For each downstream H>=aligned_H and head h:
       post-WO norm of Real-Gray source p -> prompt-last message.

    Also source percentile among all text source positions for same head/layer.
    """
    rows = []

    for H in scan_layers:
        if H < aligned_H:
            continue

        Ar = real_trace["attn"][H]
        Ag = gray_trace["attn"][H]
        Vr = all_head_v(real_trace["v"][H], n_heads, head_dim)
        Vg = all_head_v(gray_trace["v"][H], n_heads, head_dim)

        n = min(Ar.shape[-1], Ag.shape[-1], Vr.shape[0], Vg.shape[0])
        if not (0 <= int(p) < n):
            continue
        if not (0 <= prompt_last < Ar.shape[1]):
            continue

        sources = [s for s in text_positions if 0 <= s < n]
        if int(p) not in sources:
            continue
        p_idx = sources.index(int(p))

        layer_rows = []
        src_idx = np.asarray(sources, dtype=int)

        for h in range(n_heads):
            # [S,D]
            diff = (
                Ar[h, prompt_last, src_idx, None]
                * Vr[src_idx, h, :]
                -
                Ag[h, prompt_last, src_idx, None]
                * Vg[src_idx, h, :]
            )

            W_h = o_proj_head_matrix(
                decoder_layers,
                H,
                h,
                head_dim,
            )  # [hidden,D]

            # [S,hidden]
            post = diff @ W_h.T
            norms = np.linalg.norm(post, axis=-1)

            score = float(norms[p_idx])
            src_pct = percentile_desc(norms, p_idx)

            layer_rows.append({
                "transport_layer": int(H),
                "head": int(h),
                "head_name": hname(H, h),
                "hop_offset": int(H - aligned_H),
                "transport_postwo_rg_norm": score,
                "transport_source_percentile": src_pct,
                "transport_real_attention": float(
                    Ar[h, prompt_last, int(p)]
                ),
                "transport_abs_delta_attention": float(
                    abs(
                        Ar[h, prompt_last, int(p)]
                        - Ag[h, prompt_last, int(p)]
                    )
                ),
                "transport_source_N": len(sources),
            })

        layer_ranks = ranks_desc([
            r["transport_postwo_rg_norm"]
            for r in layer_rows
        ])
        for r, rank in zip(layer_rows, layer_ranks):
            r["transport_layer_head_rank"] = int(rank)

        rows.extend(layer_rows)

    global_ranks = ranks_desc([
        r["transport_postwo_rg_norm"]
        for r in rows
    ])
    for r, rank in zip(rows, global_ranks):
        r["transport_global_head_rank"] = int(rank)
        r["transport_global_head_percentile"] = (
            1.0
            if len(rows) <= 1
            else 1.0 - (rank - 1) / (len(rows) - 1)
        )

    return rows


# =============================================================================
# Summaries
# =============================================================================

def summarize_head_rows(
    df,
    head_col,
    strong_col,
    source_pct_col,
    rank_col,
    sample_col="sid",
):
    rows = []
    if len(df) == 0:
        return pd.DataFrame()

    for head, g in df.groupby(head_col):
        strong = g[strong_col].map(boolify)
        rows.append({
            "head_name": head,
            "state_N": len(g),
            "sample_N": int(g[sample_col].nunique()),
            "strong_N": int(strong.sum()),
            "strong_sample_N": int(
                g.loc[strong, sample_col].nunique()
            ),
            "strong_fraction": float(strong.mean()),
            "mean_source_percentile": safe_mean(
                g[source_pct_col]
            ),
            "median_source_percentile": safe_median(
                g[source_pct_col]
            ),
            "mean_rank": safe_mean(g[rank_col]),
            "median_rank": safe_median(g[rank_col]),
        })
    return pd.DataFrame(rows).sort_values(
        ["strong_N", "strong_fraction", "mean_source_percentile"],
        ascending=[False, False, False],
    )


def grouped_strong_summary(df, group_cols, strong_col):
    rows = []
    if len(df) == 0:
        return pd.DataFrame()
    for key, g in df.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        strong = g[strong_col].map(boolify)
        row.update({
            "state_head_rows": len(g),
            "unique_states": int(
                g[["sid", "causal_topk_rank"]]
                .drop_duplicates()
                .shape[0]
            ),
            "strong_rows": int(strong.sum()),
            "strong_fraction": float(strong.mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    a = parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    transport_layers = parse_layer_spec(a.transport_layers)
    if not transport_layers:
        raise ValueError("No transport layers.")

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    errors_path = outdir / "errors.jsonl"

    prompts = load_prompts(Path(a.prompt_jsonl))
    records = load_coco_records(Path(a.data_root))

    (
        causal_selected,
        causal_by_sid,
        rank_col,
        mediation_col,
    ) = load_causal_topk(
        Path(a.oracle_bank_dir),
        a.causal_bank,
        a.causal_top_k,
        a.causal_domain,
        a.causal_order,
    )

    valid_sids = sorted(
        set(causal_by_sid)
        & set(prompts)
        & set(records)
    )

    if a.max_samples > 0:
        valid_sids = valid_sids[: int(a.max_samples)]

    if a.require_n and len(valid_sids) != int(a.require_n):
        raise RuntimeError(
            f"Expected N={a.require_n}, got {len(valid_sids)}"
        )

    # Save exact causal states used.
    causal_out_rows = []
    for sid in valid_sids:
        for row in causal_by_sid[sid].itertuples():
            cls, key = token_class_from_row(row)
            causal_out_rows.append({
                "sid": sid,
                "gt": prompts[sid]["gt"],
                "causal_topk_rank": int(row.causal_topk_rank),
                "source_layer": int(row.source_layer),
                "position": int(row.position),
                "token": str(getattr(row, "token", "")),
                "category": str(getattr(row, "category", "")),
                "broad_category": str(
                    getattr(row, "broad_category", "")
                ),
                "canonical_token_class": cls,
                "canonical_token_key": key,
                "original_rank": (
                    getattr(row, rank_col)
                    if rank_col else np.nan
                ),
                "mediation": (
                    getattr(row, mediation_col)
                    if mediation_col else np.nan
                ),
            })
    causal_top_df = pd.DataFrame(causal_out_rows)
    causal_top_df.to_csv(
        outdir / "causal_top7_states.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Model.
    # -------------------------------------------------------------------------
    model_cls = getattr(transformers, a.model_class)
    kwargs = dict(
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        device_map={"": a.device},
        dtype=resolve_dtype(a.dtype),
    )
    if a.attn_impl != "none":
        kwargs["attn_implementation"] = a.attn_impl

    print(f"Loading {a.model_id}", flush=True)
    try:
        model = model_cls.from_pretrained(
            a.model_id,
            **kwargs,
        )
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = model_cls.from_pretrained(
            a.model_id,
            **kwargs,
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(
        a.model_id,
        trust_remote_code=False,
    )
    configure_processor(model, processor)

    decoder_layers, decoder_path = resolve_decoder_layers(model)
    cfg = get_text_config(model)
    n_heads = int(cfg.num_attention_heads)
    hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        hidden_size = int(
            resolve_o_proj(resolve_attn(decoder_layers[0])).in_features
        )
    head_dim = hidden_size // n_heads

    # Need every aligned reader H=L+1 plus requested transport layers.
    causal_layers = sorted(
        set(
            int(x)
            for sid in valid_sids
            for x in causal_by_sid[sid]["source_layer"].tolist()
        )
    )
    reader_layers = sorted({L + 1 for L in causal_layers})
    scan_layers = sorted(
        set(reader_layers)
        | set(transport_layers)
    )

    for L in scan_layers:
        if not (0 <= L < len(decoder_layers)):
            raise ValueError(f"Bad scan layer={L}")

    print("=" * 180)
    print("CAUSAL TOP7 -> STRONG READER / TRANSPORTER HEADS")
    print("=" * 180)
    print(
        f"N={len(valid_sids)} | bank={a.causal_bank} | "
        f"topK={a.causal_top_k} | domain={a.causal_domain} | "
        f"order={a.causal_order}"
    )
    print(
        f"causal layers={causal_layers} | reader layers={reader_layers} | "
        f"transport scan={transport_layers}"
    )
    print(
        f"strong reader: source_pct>={a.reader_source_percentile:.2f} "
        f"AND aligned-head-rank<={a.reader_top_heads}"
    )
    print(
        f"strong transport: source_pct>={a.transport_source_percentile:.2f} "
        f"AND global-downstream-head-rank<={a.transport_top_heads}"
    )
    print()

    reader_rows = []
    transport_rows = []
    best_rows = []

    try:
        for sid in tqdm(valid_sids, desc="causal Top7 head scan"):
            real = gray = None
            try:
                meta = prompts[sid]
                image_path = records[sid].image_path

                real = Image.open(image_path).convert("RGB")
                gray = Image.new(
                    "RGB",
                    real.size,
                    (a.gray_value, a.gray_value, a.gray_value),
                )

                rb = make_batch(
                    processor,
                    real,
                    meta["question_text"],
                    torch.device(a.device),
                )
                gb = make_batch(
                    processor,
                    gray,
                    meta["question_text"],
                    torch.device(a.device),
                )

                ids = rb["input_ids"][0].detach().cpu().tolist()
                toks = token_strings(processor, ids)
                visual = resolve_visual_indices(
                    model,
                    processor,
                    rb,
                    ids,
                )
                text_pos = eligible_text_positions(ids, visual)
                prompt_last = len(ids) - 1

                rt = run_trace(
                    model,
                    decoder_layers,
                    rb,
                    scan_layers,
                )
                gt = run_trace(
                    model,
                    decoder_layers,
                    gb,
                    scan_layers,
                )

                sample_causal = causal_by_sid[sid]

                for crow in sample_causal.itertuples():
                    causal_rank = int(crow.causal_topk_rank)
                    source_layer = int(crow.source_layer)
                    p = int(crow.position)
                    aligned_H = source_layer + 1

                    cls, key = token_class_from_row(crow)
                    token = (
                        toks[p]
                        if 0 <= p < len(toks)
                        else str(getattr(crow, "token", ""))
                    )

                    base = {
                        "sid": sid,
                        "gt": meta["gt"],
                        "causal_topk_rank": causal_rank,
                        "causal_source_layer": source_layer,
                        "causal_position": p,
                        "causal_token": token,
                        "canonical_token_class": cls,
                        "canonical_token_key": key,
                        "causal_mediation": (
                            getattr(crow, mediation_col)
                            if mediation_col else np.nan
                        ),
                        "causal_original_rank": (
                            getattr(crow, rank_col)
                            if rank_col else np.nan
                        ),
                    }

                    # ---------------------------------------------------------
                    # Direct readers.
                    # ---------------------------------------------------------
                    rr = reader_scores_for_state(
                        rt,
                        gt,
                        aligned_H,
                        p,
                        text_pos,
                        prompt_last,
                        n_heads,
                        head_dim,
                    )

                    for r in rr:
                        strong = (
                            float(r["reader_source_percentile"])
                            >= a.reader_source_percentile
                            and int(r["reader_head_rank"])
                            <= a.reader_top_heads
                        )
                        r.update(base)
                        r["strong_reader"] = strong
                        reader_rows.append(r)

                    strong_rr = [r for r in rr if (
                        float(r["reader_source_percentile"])
                        >= a.reader_source_percentile
                        and int(r["reader_head_rank"])
                        <= a.reader_top_heads
                    )]
                    strong_rr.sort(
                        key=lambda x: (
                            x["reader_head_rank"],
                            -x["reader_source_percentile"],
                        )
                    )

                    # ---------------------------------------------------------
                    # Transporters.
                    # ---------------------------------------------------------
                    tr = transporter_scores_for_state(
                        decoder_layers,
                        rt,
                        gt,
                        transport_layers,
                        aligned_H,
                        p,
                        text_pos,
                        prompt_last,
                        n_heads,
                        head_dim,
                    )

                    for r in tr:
                        strong = (
                            float(r["transport_source_percentile"])
                            >= a.transport_source_percentile
                            and int(r["transport_global_head_rank"])
                            <= a.transport_top_heads
                        )
                        r.update(base)
                        r["aligned_reader_layer"] = aligned_H
                        r["strong_transporter"] = strong
                        r["direct_immediate_transport"] = (
                            int(r["transport_layer"]) == aligned_H
                        )
                        transport_rows.append(r)

                    strong_tr = [r for r in tr if (
                        float(r["transport_source_percentile"])
                        >= a.transport_source_percentile
                        and int(r["transport_global_head_rank"])
                        <= a.transport_top_heads
                    )]
                    strong_tr.sort(
                        key=lambda x: (
                            x["transport_global_head_rank"],
                            -x["transport_source_percentile"],
                        )
                    )

                    best_reader = (
                        min(
                            rr,
                            key=lambda x: x["reader_head_rank"],
                        )
                        if rr else None
                    )
                    best_transport = (
                        min(
                            tr,
                            key=lambda x: x["transport_global_head_rank"],
                        )
                        if tr else None
                    )

                    best_rows.append({
                        **base,
                        "aligned_reader_layer": aligned_H,
                        "best_reader_head": (
                            best_reader["head_name"]
                            if best_reader else ""
                        ),
                        "best_reader_rank": (
                            best_reader["reader_head_rank"]
                            if best_reader else np.nan
                        ),
                        "best_reader_source_percentile": (
                            best_reader["reader_source_percentile"]
                            if best_reader else np.nan
                        ),
                        "strong_reader_heads": ",".join(
                            x["head_name"] for x in strong_rr
                        ),
                        "strong_reader_N": len(strong_rr),
                        "best_transporter_head": (
                            best_transport["head_name"]
                            if best_transport else ""
                        ),
                        "best_transport_layer": (
                            best_transport["transport_layer"]
                            if best_transport else np.nan
                        ),
                        "best_transport_global_rank": (
                            best_transport["transport_global_head_rank"]
                            if best_transport else np.nan
                        ),
                        "best_transport_source_percentile": (
                            best_transport["transport_source_percentile"]
                            if best_transport else np.nan
                        ),
                        "strong_transporter_heads": ",".join(
                            x["head_name"] for x in strong_tr
                        ),
                        "strong_transporter_N": len(strong_tr),
                        "has_strong_reader": bool(strong_rr),
                        "has_strong_transporter": bool(strong_tr),
                    })

                del rt, gt, rb, gb

            except Exception as exc:
                append_jsonl(errors_path, {
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().splitlines()[-16:],
                })
                tqdm.write(
                    f"[ERROR] sid={sid} {type(exc).__name__}: {exc}"
                )
            finally:
                if real is not None:
                    with contextlib.suppress(Exception):
                        real.close()
                if gray is not None:
                    with contextlib.suppress(Exception):
                        gray.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        reader_df = pd.DataFrame(reader_rows)
        transport_df = pd.DataFrame(transport_rows)
        best_df = pd.DataFrame(best_rows)

        if a.save_all:
            reader_df.to_csv(
                outdir / "reader_all_heads.csv",
                index=False,
            )
            transport_df.to_csv(
                outdir / "transporter_all_heads.csv",
                index=False,
            )

        strong_reader_df = (
            reader_df[reader_df["strong_reader"]]
            .copy()
            if len(reader_df)
            else pd.DataFrame()
        )
        strong_transport_df = (
            transport_df[transport_df["strong_transporter"]]
            .copy()
            if len(transport_df)
            else pd.DataFrame()
        )

        strong_reader_df.to_csv(
            outdir / "strong_readers.csv",
            index=False,
        )
        strong_transport_df.to_csv(
            outdir / "strong_transporters.csv",
            index=False,
        )
        best_df.to_csv(
            outdir / "causal_state_best_heads.csv",
            index=False,
        )

        reader_summary = summarize_head_rows(
            reader_df,
            "head_name",
            "strong_reader",
            "reader_source_percentile",
            "reader_head_rank",
        )
        reader_summary.to_csv(
            outdir / "reader_head_summary.csv",
            index=False,
        )

        transport_summary = summarize_head_rows(
            transport_df,
            "head_name",
            "strong_transporter",
            "transport_source_percentile",
            "transport_global_head_rank",
        )
        if len(transport_summary):
            # Add layer/head columns for easier sorting.
            parts = transport_summary["head_name"].str.extract(
                r"L(\d+)H(\d+)"
            )
            transport_summary["layer"] = pd.to_numeric(parts[0])
            transport_summary["head"] = pd.to_numeric(parts[1])
        transport_summary.to_csv(
            outdir / "transporter_head_summary.csv",
            index=False,
        )

        if len(reader_df):
            grouped_strong_summary(
                reader_df,
                ["causal_source_layer", "head_name"],
                "strong_reader",
            ).to_csv(
                outdir / "reader_by_causal_layer.csv",
                index=False,
            )
            grouped_strong_summary(
                reader_df,
                ["canonical_token_class", "head_name"],
                "strong_reader",
            ).to_csv(
                outdir / "reader_by_token_class.csv",
                index=False,
            )

        if len(transport_df):
            grouped_strong_summary(
                transport_df,
                ["causal_source_layer", "head_name"],
                "strong_transporter",
            ).to_csv(
                outdir / "transporter_by_causal_layer.csv",
                index=False,
            )
            grouped_strong_summary(
                transport_df,
                ["canonical_token_class", "head_name"],
                "strong_transporter",
            ).to_csv(
                outdir / "transporter_by_token_class.csv",
                index=False,
            )

        # -------------------------------------------------------------
        # Compact report.
        # -------------------------------------------------------------
        lines = []
        lines.append("=" * 180)
        lines.append("CAUSAL TOP7 -> STRONG READER / TRANSPORTER HEADS")
        lines.append("=" * 180)
        lines.append(
            f"N samples={len(valid_sids)} | causal states analyzed={len(best_df)} | "
            f"bank={a.causal_bank} | topK={a.causal_top_k} | domain={a.causal_domain}"
        )
        lines.append(
            f"Reader strong: sourcePct>={a.reader_source_percentile:.2f} "
            f"and aligned head rank<={a.reader_top_heads}"
        )
        lines.append(
            f"Transport strong: sourcePct>={a.transport_source_percentile:.2f} "
            f"and global downstream rank<={a.transport_top_heads}"
        )
        lines.append("")

        if len(best_df):
            lines.append("STATE COVERAGE")
            lines.append("-" * 180)
            lines.append(
                f"causal states with >=1 strong reader      : "
                f"{best_df['has_strong_reader'].mean():.4f}"
            )
            lines.append(
                f"causal states with >=1 strong transporter : "
                f"{best_df['has_strong_transporter'].mean():.4f}"
            )
            lines.append(
                f"mean strong readers/state                 : "
                f"{best_df['strong_reader_N'].mean():.3f}"
            )
            lines.append(
                f"mean strong transporters/state            : "
                f"{best_df['strong_transporter_N'].mean():.3f}"
            )
            lines.append("")

        lines.append("TOP STRONG DIRECT READERS")
        lines.append("-" * 180)
        if len(reader_summary):
            show = [
                "head_name",
                "state_N",
                "sample_N",
                "strong_N",
                "strong_sample_N",
                "strong_fraction",
                "mean_source_percentile",
                "mean_rank",
            ]
            lines.append(
                reader_summary.head(30)[show].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )
        lines.append("")

        lines.append("TOP STRONG TRANSPORTERS TO PROMPT-LAST")
        lines.append("-" * 180)
        if len(transport_summary):
            show = [
                "head_name",
                "state_N",
                "sample_N",
                "strong_N",
                "strong_sample_N",
                "strong_fraction",
                "mean_source_percentile",
                "mean_rank",
            ]
            lines.append(
                transport_summary.head(40)[show].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )
        lines.append("")

        if len(best_df):
            lines.append("BY CAUSAL RANK")
            lines.append("-" * 180)
            for rank, g in best_df.groupby("causal_topk_rank"):
                lines.append(
                    f"Top{int(rank)} state: N={len(g)} | "
                    f"strong-reader={g['has_strong_reader'].mean():.4f} | "
                    f"strong-transfer={g['has_strong_transporter'].mean():.4f}"
                )

        report = "\n".join(lines) + "\n"
        (outdir / "analysis_summary.txt").write_text(
            report,
            encoding="utf-8",
        )
        print(report)

        metadata = {
            "model_id": a.model_id,
            "decoder_path": decoder_path,
            "causal_bank": a.causal_bank,
            "causal_top_k": a.causal_top_k,
            "causal_domain": a.causal_domain,
            "causal_order": a.causal_order,
            "rank_column": rank_col,
            "mediation_column": mediation_col,
            "causal_layers": causal_layers,
            "reader_layers": reader_layers,
            "transport_layers": transport_layers,
            "reader_definition": (
                "Strict direct reader at H=L+1. For each head, max over future "
                "text queries of ||A_real[q,p]V_real[p] - A_gray[q,p]V_gray[p]||."
            ),
            "strong_reader_definition": (
                f"reader_source_percentile >= {a.reader_source_percentile} "
                f"AND reader_head_rank <= {a.reader_top_heads}"
            ),
            "transport_definition": (
                "For each H>=L+1, prompt-last post-W_O Real-Gray message norm "
                "||W_O^h(A_real[last,p]V_real[p]-A_gray[last,p]V_gray[p])||."
            ),
            "strong_transporter_definition": (
                f"transport_source_percentile >= {a.transport_source_percentile} "
                f"AND global downstream head rank <= {a.transport_top_heads}"
            ),
            "interpretation_warning": (
                "Transport at H=L+1 directly consumes the exact causal state. "
                "H>L+1 follows the same token position after intermediate updates "
                "and is therefore token-lineage transport, not proof that the exact "
                "original residual vector survived unchanged."
            ),
        }
        (outdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print("Saved:", outdir)

    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
