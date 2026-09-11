#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_signed_layerwise_rn_update_gating_v1.py

Question
========
For a selected behaviorally causal text state c=(C,p), define the aligned
Real-NoImage state at every layer:

    d_l(p) = h_real[l,p] - h_noimage[l,q]

where q is the exact-token LCS-aligned NoImage position.

The FINAL causal-token RN target is:

    t_c = d_C(p)

The net RN update contributed between consecutive decoder block outputs is:

    u_l(p) = d_l(p) - d_{l-1}(p)

Because transformer blocks are residual, u_l is the block-level change in the
Real-NoImage displacement at this token.

We score whether that update helps build the FINAL causal RN state:

    s_{l,c}
      = <u_l(p), t_c> / ||t_c||^2

Interpretation:
    s > 0 : layer update builds the final causal RN direction
    s < 0 : layer update erodes / opposes the final causal RN direction
    s ~ 0 : little target-axis effect

This script first diagnoses correct vs wrong samples, then performs actual
greedy generation under cached, clean-trajectory interventions.

Interventions
=============
For a unique (layer l, token p), scores from all selected causal targets that
use the same token are aggregated (default: mean). Let sbar be the aggregate.

positive_only:
    if sbar > tau:
        h[l,p] <- h[l,p] + alpha * u_l

negative_cancel:
    if sbar < -tau:
        h[l,p] <- h[l,p] - alpha * u_l

signed_gated:
    positive -> +alpha*u_l
    negative -> -alpha*u_l

all_amplify control:
    h[l,p] <- h[l,p] + alpha*u_l
    regardless of sign

direct_rn reference:
    at each selected causal state c=(C,p):
        h[C,p] <- h[C,p] + beta*t_c

The direct_rn condition is the known oracle causal-token reference. The signed
conditions intervene BEFORE / ALONG the formation trajectory using the layer's
own RN update.

IMPORTANT
=========
This is an ORACLE mechanism experiment:
  1) selected causal states come from the existing oracle causal ranking;
  2) sign is judged using the final causal target t_c.

The point is to test whether selective amplification of productive RN updates
and cancellation of destructive RN updates can improve generation. If it works,
the next problem is to replace the oracle sign detector with a non-oracle one.

Cached-update caveat
====================
u_l is measured on the CLEAN Real/NoImage trajectories. During a multi-layer
intervention, downstream states have already changed, but later cached u_l
vectors are still added/subtracted. This is intentional: it tests whether the
clean trajectory contains reusable productive/destructive RN update directions.
It is not an exact replay of a counterfactual forward pass.

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 python -u eval_signed_layerwise_rn_update_gating_v1.py \
  --model qwen-3b \
  --ranked-causal \
    output/qwen3b_oracle_causal_scan_L1_L26_all440/ranked_k36_tokens.csv \
  --causal-layers 20-26 \
  --causal-top-k 7 \
  --update-layers 8-26 \
  --scales 0.5,1.0 \
  --score-threshold 0 \
  --score-aggregate mean \
  --eval-max-samples 40 \
  --output-dir output/qwen3b_signed_rn_update_gating_n40_v1 \
  --overwrite

Useful conservative follow-up:
    --score-threshold 0.01

Outputs
=======
selected_causal_states.csv
alignment_summary.csv
per_target_layer_updates.csv
gated_update_map.csv
sample_accumulation_summary.csv
accumulation_by_baseline_correctness.csv
layer_update_summary.csv
layer_update_by_baseline_correctness.csv
generation_per_sample.csv
generation_summary.csv
generation_by_relation.csv
analysis_summary.txt
metadata.json
errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import shutil
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoProcessor
from tqdm import tqdm

import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL = ("left", "right", "above", "below")
EPS = 1e-12


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b","qwen-7b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--ranked-causal", required=True)

    p.add_argument("--causal-layers", default="20-26")
    p.add_argument("--causal-top-k", type=int, default=7)
    p.add_argument(
        "--causal-categories",
        default="",
        help="Optional broad_category filter; empty keeps all nonvisual/nonlast selected states.",
    )

    p.add_argument(
        "--update-layers",
        default="8-26",
        help=(
            "Layers l whose RN update u_l=d_l-d_{l-1} is scored/intervened on. "
            "All requested layers must be >=1."
        ),
    )
    p.add_argument(
        "--scales",
        default="0.5,1.0",
        help="alpha values for positive/negative/signed/all-amplify conditions.",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Gate only if |aggregated score| exceeds this threshold.",
    )
    p.add_argument(
        "--score-aggregate",
        default="mean",
        choices=["mean","maxabs","first"],
        help=(
            "How to combine signs when the same token position appears in multiple "
            "selected causal target states."
        ),
    )
    p.add_argument(
        "--conditions",
        default="positive_only,negative_cancel,signed_gated,all_amplify,direct_rn",
        help=(
            "Comma-separated subset of: "
            "positive_only,negative_cancel,signed_gated,all_amplify,direct_rn"
        ),
    )
    p.add_argument(
        "--direct-beta",
        type=float,
        default=1.0,
        help="Scale for direct causal-token RN reference.",
    )

    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-max-samples", type=int, default=40)

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager","sdpa","flash_attention_2","none"],
    )

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def parse_layers(text: str) -> List[int]:
    out=set()

    for part in str(text).split(","):
        part=part.strip().upper().replace("L","")
        if not part:
            continue

        if "-" in part:
            a,b=part.split("-",1)
            a,b=int(a),int(b)
            out.update(range(min(a,b),max(a,b)+1))
        else:
            out.add(int(part))

    return sorted(out)


def parse_floats(text: str) -> List[float]:
    return [
        float(x.strip())
        for x in str(text).split(",")
        if x.strip()
    ]


def parse_set(text: str) -> List[str]:
    return [
        x.strip()
        for x in str(text).split(",")
        if x.strip()
    ]


def safe_mean(xs):
    vals=[]
    for x in xs:
        try:
            v=float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)

    return float(np.mean(vals)) if vals else float("nan")


def safe_median(xs):
    vals=[]
    for x in xs:
        try:
            v=float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)

    return float(np.median(vals)) if vals else float("nan")


def cosine_np(a,b):
    a=np.asarray(a,np.float32)
    b=np.asarray(b,np.float32)

    na=float(np.linalg.norm(a))
    nb=float(np.linalg.norm(b))

    if na<=EPS or nb<=EPS:
        return float("nan")

    return float(np.dot(a,b)/(na*nb))


def append_jsonl(path,row):
    with open(path,"a",encoding="utf-8") as f:
        f.write(
            json.dumps(dict(row),ensure_ascii=False)
            + "\n"
        )


def write_json(path,obj):
    Path(path).write_text(
        json.dumps(obj,ensure_ascii=False,indent=2),
        encoding="utf-8",
    )


# =============================================================================
# Data / model
# =============================================================================

def load_data(a):
    two=base.import_two_object_module()
    prompts=base.load_standard_prompts(
        Path(a.prompt_jsonl)
    )
    records,_audit=two.load_records(
        "coco_two",
        Path(a.data_root),
        None,
    )
    rec_by_sid={
        int(r.sid):r
        for r in records
    }

    meta=[]

    for rec in records:
        sid=int(rec.sid)

        if sid not in prompts:
            continue

        p=prompts[sid]
        gt=traj.normalize_relation(
            base,
            p["answer_raw"],
        )

        if gt not in REL:
            continue

        meta.append({
            "sid":sid,
            "gt":gt,
            "subject":str(p["subject"]),
            "reference":str(p["reference"]),
            "question_text":str(p["question_text"]),
        })

    meta=traj.stratified_cap(
        meta,
        a.max_samples,
        a.seed,
    )

    if a.eval_max_samples>0:
        meta=traj.stratified_cap(
            meta,
            a.eval_max_samples,
            a.seed+1,
        )

    return two,meta,rec_by_sid


def load_model(a,two):
    spec=base.merged_model_specs(two)[a.model]
    cls=getattr(
        transformers,
        spec.model_class,
    )

    kw=dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"":a.device},
    )

    if a.attn_impl!="none":
        kw["attn_implementation"]=a.attn_impl

    print(
        f"[model] loading {spec.repo_id}",
        flush=True,
    )

    try:
        model=cls.from_pretrained(
            spec.repo_id,
            **kw,
        )
    except TypeError:
        kw["torch_dtype"]=kw.pop("dtype")
        model=cls.from_pretrained(
            spec.repo_id,
            **kw,
        )

    model.eval()

    processor=AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )

    base.configure_processor(
        model,
        processor,
    )

    for p in model.parameters():
        p.requires_grad_(False)

    decoder_layers,decoder_path=base.resolve_decoder_layers(
        model
    )

    return (
        model,
        processor,
        decoder_layers,
        decoder_path,
        spec,
    )


def load_causal_selection(
    path: Path,
    allowed_sids: set,
    causal_layers: Sequence[int],
    top_k: int,
    categories: Sequence[str],
):
    d=pd.read_csv(path)

    required={
        "sid",
        "rank",
        "source_layer",
        "position",
        "token",
        "category",
        "broad_category",
    }

    missing=required-set(d.columns)

    if missing:
        raise RuntimeError(
            f"{path} missing columns {sorted(missing)}"
        )

    for c in (
        "sid",
        "rank",
        "source_layer",
        "position",
    ):
        d[c]=pd.to_numeric(
            d[c],
            errors="raise",
        ).astype(int)

    d=d[
        d["sid"].isin(allowed_sids)
    ].copy()

    d=d[
        d["source_layer"].isin(
            set(map(int,causal_layers))
        )
    ].copy()

    d=d[
        d["broad_category"].astype(str)!="visual"
    ].copy()

    d=d[
        d["broad_category"].astype(str)!="last"
    ].copy()

    if categories:
        wanted=set(map(str,categories))
        d=d[
            d["broad_category"].astype(str).isin(wanted)
        ].copy()

    rows=[]

    for sid,g in d.groupby("sid"):
        z=(
            g.sort_values("rank")
            .head(int(top_k))
            .copy()
        )
        z["causal_text_rank"]=np.arange(
            1,
            len(z)+1,
        )
        rows.append(z)

    if rows:
        return pd.concat(
            rows,
            ignore_index=True,
        )

    return d.iloc[:0].copy()


# =============================================================================
# NoImage / LCS alignment
# =============================================================================

def move_batch(batch,device):
    return {
        k:(
            v.to(device)
            if torch.is_tensor(v)
            else v
        )
        for k,v in batch.items()
    }


def build_noimage_batch(
    processor,
    question_text,
    device,
):
    messages=[
        {
            "role":"user",
            "content":[
                {
                    "type":"text",
                    "text":str(question_text),
                }
            ],
        }
    ]

    try:
        prompt=processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt=str(question_text)

    last_error=None

    for fn in [
        lambda: processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ),
        lambda: processor(
            text=prompt,
            return_tensors="pt",
        ),
    ]:
        try:
            return move_batch(
                fn(),
                device,
            )
        except Exception as exc:
            last_error=exc

    raise RuntimeError(
        f"NoImage processor failed: {last_error}"
    )


def lcs_token_map(
    real_ids: List[int],
    no_ids: List[int],
) -> Dict[int,int]:
    a=list(map(int,real_ids))
    b=list(map(int,no_ids))

    n,m=len(a),len(b)

    dp=np.zeros(
        (n+1,m+1),
        dtype=np.uint16,
    )

    for i in range(n-1,-1,-1):
        ai=a[i]
        row=dp[i]
        below=dp[i+1]

        for j in range(m-1,-1,-1):
            if ai==b[j]:
                row[j]=1+below[j+1]
            else:
                x=below[j]
                y=row[j+1]
                row[j]=x if x>=y else y

    out={}
    i=j=0

    while i<n and j<m:
        if (
            a[i]==b[j]
            and
            dp[i,j]==1+dp[i+1,j+1]
        ):
            out[i]=j
            i+=1
            j+=1

        elif dp[i+1,j]>=dp[i,j+1]:
            i+=1

        else:
            j+=1

    return out


# =============================================================================
# Block capture / patch
# =============================================================================

def first_tensor(output):
    if torch.is_tensor(output):
        return output

    if (
        isinstance(output,(tuple,list))
        and output
        and torch.is_tensor(output[0])
    ):
        return output[0]

    raise RuntimeError(
        f"Unsupported layer output type: "
        f"{type(output).__name__}"
    )


def replace_first_tensor(
    output,
    tensor,
):
    if torch.is_tensor(output):
        return tensor

    if isinstance(output,tuple):
        return (
            tensor,
            *output[1:],
        )

    if isinstance(output,list):
        return [
            tensor,
            *output[1:],
        ]

    raise RuntimeError(
        f"Unsupported layer output type: "
        f"{type(output).__name__}"
    )


class BlockCapture:
    def __init__(
        self,
        decoder_layers,
        layers,
    ):
        self.states={}
        self.handles=[]

        for L in sorted(
            set(map(int,layers))
        ):
            def make_hook(layer):
                def hook(
                    _m,
                    _inp,
                    out,
                ):
                    x=first_tensor(out)

                    self.states[layer]=(
                        x.detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    return None

                return hook

            self.handles.append(
                decoder_layers[L]
                .register_forward_hook(
                    make_hook(L)
                )
            )

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()

        self.handles=[]


@torch.inference_mode()
def capture_blocks(
    model,
    decoder_layers,
    batch,
    layers,
):
    cap=BlockCapture(
        decoder_layers,
        layers,
    )

    try:
        kw=dict(batch)
        kw["use_cache"]=False
        kw["return_dict"]=True

        _=model(**kw)

        missing=[
            L
            for L in layers
            if L not in cap.states
        ]

        if missing:
            raise RuntimeError(
                f"Missing block captures: {missing}"
            )

        return dict(cap.states)

    finally:
        cap.close()


class MultiLayerResidualAdd:
    """
    Add cached vectors to decoder block OUTPUTS during prompt prefill only.

    patch_map:
        layer -> {
            real_position -> np.ndarray[hidden]
        }
    """
    def __init__(
        self,
        *,
        decoder_layers,
        patch_map,
        prompt_len,
    ):
        self.handles=[]
        self.prompt_len=int(prompt_len)

        for L,pos_map in sorted(
            patch_map.items()
        ):
            if not pos_map:
                continue

            def make_hook(local_pos_map):
                def hook(
                    _m,
                    _inp,
                    out,
                ):
                    x=first_tensor(out)

                    # Do not patch decode-token steps.
                    if int(x.shape[1])!=self.prompt_len:
                        return None

                    y=x.clone()

                    for p,vec in local_pos_map.items():
                        p=int(p)

                        if 0<=p<int(y.shape[1]):
                            y[0,p]+=torch.as_tensor(
                                vec,
                                device=y.device,
                                dtype=y.dtype,
                            )

                    return replace_first_tensor(
                        out,
                        y,
                    )

                return hook

            self.handles.append(
                decoder_layers[int(L)]
                .register_forward_hook(
                    make_hook(
                        dict(pos_map)
                    )
                )
            )

    def __enter__(self):
        return self

    def __exit__(
        self,
        *args,
    ):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()

        self.handles=[]


@torch.inference_mode()
def generate_with_patch(
    *,
    model,
    processor,
    decoder_layers,
    batch,
    patch_map,
    max_new_tokens,
):
    prompt_len=int(
        batch["input_ids"].shape[1]
    )

    ctx=(
        MultiLayerResidualAdd(
            decoder_layers=decoder_layers,
            patch_map=patch_map,
            prompt_len=prompt_len,
        )
        if patch_map
        else contextlib.nullcontext()
    )

    with ctx:
        text=base.generate_text(
            model,
            processor,
            batch,
            max_new_tokens=max_new_tokens,
        )

    pred=traj.normalize_relation(
        base,
        text,
    )

    return pred,text


# =============================================================================
# Causal targets and RN layer updates
# =============================================================================

def prepare_targets(
    *,
    causal_rows,
    r2n,
    real_states,
    no_states,
    update_layers,
):
    """
    Each target has a final RN direction t_c = d_C(p).

    We keep only update layers L <= C, and require both L and L-1 captures.
    """
    targets=[]

    for r in causal_rows.itertuples():
        C=int(r.source_layer)
        p=int(r.position)

        q=r2n.get(p,None)

        if q is None:
            continue

        if (
            C not in real_states
            or C not in no_states
        ):
            continue

        R=real_states[C]
        N=no_states[C]

        if not (
            0<=p<R.shape[1]
            and
            0<=q<N.shape[1]
        ):
            continue

        hR=R[0,p].astype(np.float32)
        hN=N[0,q].astype(np.float32)
        target=(hR-hN).astype(np.float32)

        tn=float(
            np.linalg.norm(target)
        )

        if tn<=EPS:
            continue

        valid_update_layers=[]

        for L in update_layers:
            L=int(L)

            if L>C:
                continue

            if L<1:
                continue

            if (
                L not in real_states
                or
                L-1 not in real_states
                or
                L not in no_states
                or
                L-1 not in no_states
            ):
                continue

            if not (
                0<=p<real_states[L].shape[1]
                and
                0<=p<real_states[L-1].shape[1]
                and
                0<=q<no_states[L].shape[1]
                and
                0<=q<no_states[L-1].shape[1]
            ):
                continue

            valid_update_layers.append(L)

        if not valid_update_layers:
            continue

        targets.append({
            "sid":int(r.sid),
            "rank":int(r.rank),
            "causal_text_rank":int(
                r.causal_text_rank
            ),
            "target_layer":C,
            "real_position":p,
            "noimage_position":int(q),
            "token":str(r.token),
            "category":str(r.category),
            "broad_category":str(
                r.broad_category
            ),
            "target":target,
            "target_norm":tn,
            "valid_update_layers":valid_update_layers,
        })

    return targets


def rn_state(
    *,
    target,
    layer,
    real_states,
    no_states,
):
    L=int(layer)
    p=int(target["real_position"])
    q=int(target["noimage_position"])

    return (
        real_states[L][0,p].astype(np.float32)
        -
        no_states[L][0,q].astype(np.float32)
    )


def score_target_updates(
    *,
    sid,
    gt,
    baseline_correct,
    target,
    real_states,
    no_states,
):
    """
    For each L:
      d_prev = RN state after block L-1
      d_cur  = RN state after block L
      u_L    = d_cur - d_prev
      score  = <u_L,t_C>/||t_C||^2
    """
    rows=[]
    t=target["target"]
    tn=float(target["target_norm"])

    for L in target["valid_update_layers"]:
        d0=rn_state(
            target=target,
            layer=L-1,
            real_states=real_states,
            no_states=no_states,
        )
        d1=rn_state(
            target=target,
            layer=L,
            real_states=real_states,
            no_states=no_states,
        )

        u=(d1-d0).astype(np.float32)

        u_norm=float(
            np.linalg.norm(u)
        )
        d0_proj=float(
            np.dot(d0,t)/(tn*tn)
        )
        d1_proj=float(
            np.dot(d1,t)/(tn*tn)
        )
        score=float(
            np.dot(u,t)/(tn*tn)
        )

        # Numerical identity check:
        # d1_proj - d0_proj == score.
        identity_error=float(
            abs(
                (d1_proj-d0_proj)
                -
                score
            )
        )

        rows.append({
            "sid":int(sid),
            "gt":str(gt),
            "baseline_correct":bool(
                baseline_correct
            ),
            "target_rank":int(
                target["rank"]
            ),
            "causal_text_rank":int(
                target["causal_text_rank"]
            ),
            "target_layer":int(
                target["target_layer"]
            ),
            "update_layer":int(L),
            "distance_to_target":int(
                target["target_layer"]-L
            ),
            "real_position":int(
                target["real_position"]
            ),
            "noimage_position":int(
                target["noimage_position"]
            ),
            "token":str(target["token"]),
            "category":str(
                target["category"]
            ),
            "broad_category":str(
                target["broad_category"]
            ),
            "target_rn_norm":tn,
            "rn_prev_norm":float(
                np.linalg.norm(d0)
            ),
            "rn_cur_norm":float(
                np.linalg.norm(d1)
            ),
            "update_norm":u_norm,
            "rn_prev_projection":d0_proj,
            "rn_cur_projection":d1_proj,
            "update_target_score":score,
            "update_target_cosine":cosine_np(
                u,
                t,
            ),
            "identity_error":identity_error,
            "_update_vector":u,
        })

    return rows


# =============================================================================
# Gate aggregation and patch-map construction
# =============================================================================

def aggregate_scores(
    rows,
    mode,
):
    """
    rows share the same (update_layer, real_position).

    Their update vectors are the same clean u_l(p); only target directions differ.
    """
    if not rows:
        raise ValueError("aggregate_scores got no rows")

    if mode=="mean":
        return float(
            np.mean([
                float(r["update_target_score"])
                for r in rows
            ])
        )

    if mode=="maxabs":
        best=max(
            rows,
            key=lambda r:abs(
                float(
                    r["update_target_score"]
                )
            ),
        )
        return float(
            best["update_target_score"]
        )

    if mode=="first":
        best=min(
            rows,
            key=lambda r:int(
                r["target_rank"]
            ),
        )
        return float(
            best["update_target_score"]
        )

    raise ValueError(mode)


def build_gated_entries(
    target_update_rows,
    aggregate_mode,
):
    """
    Dedupe by unique (layer, position), to avoid double-adding the same clean u_l
    merely because that token appears as more than one selected causal state.
    """
    grouped={}

    for r in target_update_rows:
        key=(
            int(r["update_layer"]),
            int(r["real_position"]),
        )
        grouped.setdefault(
            key,
            [],
        ).append(r)

    entries=[]

    for (
        L,
        p,
    ),rows in sorted(grouped.items()):
        score=aggregate_scores(
            rows,
            aggregate_mode,
        )

        # Same (L,p) => same u vector. Take first.
        u=np.asarray(
            rows[0]["_update_vector"],
            np.float32,
        )

        entries.append({
            "update_layer":L,
            "real_position":p,
            "token":str(rows[0]["token"]),
            "aggregated_score":float(score),
            "n_target_votes":len(rows),
            "positive_vote_fraction":float(
                np.mean([
                    float(
                        r["update_target_score"]
                    )>0
                    for r in rows
                ])
            ),
            "mean_target_score":float(
                np.mean([
                    float(
                        r["update_target_score"]
                    )
                    for r in rows
                ])
            ),
            "maxabs_target_score":float(
                max(
                    (
                        float(
                            r["update_target_score"]
                        )
                        for r in rows
                    ),
                    key=abs,
                )
            ),
            "update_norm":float(
                np.linalg.norm(u)
            ),
            "_update_vector":u,
        })

    return entries


def add_vec_to_patch_map(
    patch_map,
    layer,
    position,
    vec,
):
    patch_map.setdefault(
        int(layer),
        {},
    )

    if int(position) in patch_map[int(layer)]:
        patch_map[int(layer)][int(position)]=(
            patch_map[int(layer)][int(position)]
            +
            np.asarray(vec,np.float32)
        ).astype(np.float32)

    else:
        patch_map[int(layer)][int(position)]=(
            np.asarray(
                vec,
                np.float32,
            ).copy()
        )


def build_condition_patch_map(
    *,
    entries,
    condition,
    scale,
    threshold,
):
    patch_map={}
    n_pos=n_neg=n_zero=0

    for e in entries:
        s=float(
            e["aggregated_score"]
        )
        u=np.asarray(
            e["_update_vector"],
            np.float32,
        )

        vec=None

        if s>threshold:
            n_pos+=1
        elif s<(-threshold):
            n_neg+=1
        else:
            n_zero+=1

        if condition=="positive_only":
            if s>threshold:
                vec=float(scale)*u

        elif condition=="negative_cancel":
            if s<(-threshold):
                vec=-float(scale)*u

        elif condition=="signed_gated":
            if s>threshold:
                vec=float(scale)*u
            elif s<(-threshold):
                vec=-float(scale)*u

        elif condition=="all_amplify":
            vec=float(scale)*u

        else:
            raise ValueError(condition)

        if vec is not None:
            add_vec_to_patch_map(
                patch_map,
                e["update_layer"],
                e["real_position"],
                vec,
            )

    return (
        patch_map,
        {
            "n_positive_entries":n_pos,
            "n_negative_entries":n_neg,
            "n_neutral_entries":n_zero,
            "n_patched_entries":sum(
                len(v)
                for v in patch_map.values()
            ),
        },
    )


def build_direct_rn_patch_map(
    targets,
    beta,
):
    patch_map={}

    for t in targets:
        add_vec_to_patch_map(
            patch_map,
            t["target_layer"],
            t["real_position"],
            float(beta)*np.asarray(
                t["target"],
                np.float32,
            ),
        )

    return patch_map


# =============================================================================
# Summaries
# =============================================================================

def accumulation_sample_row(
    *,
    sid,
    gt,
    baseline_correct,
    target_update_rows,
    gated_entries,
):
    scores=np.asarray(
        [
            float(
                r["update_target_score"]
            )
            for r in target_update_rows
        ],
        dtype=np.float64,
    )

    agg_scores=np.asarray(
        [
            float(
                e["aggregated_score"]
            )
            for e in gated_entries
        ],
        dtype=np.float64,
    )

    positive_mass=float(
        np.maximum(
            scores,
            0,
        ).sum()
    )
    negative_mass=float(
        np.maximum(
            -scores,
            0,
        ).sum()
    )
    total_abs=positive_mass+negative_mass
    net=float(
        scores.sum()
    )

    agg_pos=float(
        np.maximum(
            agg_scores,
            0,
        ).sum()
    )
    agg_neg=float(
        np.maximum(
            -agg_scores,
            0,
        ).sum()
    )
    agg_abs=agg_pos+agg_neg
    agg_net=float(
        agg_scores.sum()
    )

    return {
        "sid":int(sid),
        "gt":str(gt),
        "baseline_correct":bool(
            baseline_correct
        ),
        "N_target_update_edges":len(
            target_update_rows
        ),
        "N_unique_layer_token_updates":len(
            gated_entries
        ),
        "positive_update_fraction":float(
            np.mean(scores>0)
        ) if len(scores) else float("nan"),
        "negative_update_fraction":float(
            np.mean(scores<0)
        ) if len(scores) else float("nan"),
        "positive_mass":positive_mass,
        "negative_mass":negative_mass,
        "net_mass":net,
        "accumulation_efficiency":(
            net/total_abs
            if total_abs>EPS
            else float("nan")
        ),
        "aggregated_positive_fraction":float(
            np.mean(agg_scores>0)
        ) if len(agg_scores) else float("nan"),
        "aggregated_negative_fraction":float(
            np.mean(agg_scores<0)
        ) if len(agg_scores) else float("nan"),
        "aggregated_positive_mass":agg_pos,
        "aggregated_negative_mass":agg_neg,
        "aggregated_net_mass":agg_net,
        "aggregated_accumulation_efficiency":(
            agg_net/agg_abs
            if agg_abs>EPS
            else float("nan")
        ),
    }


def summarize_accumulation_by_correctness(
    df,
):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]

    for correct,g in df.groupby(
        "baseline_correct",
        dropna=False,
    ):
        rows.append({
            "baseline_correct":bool(correct),
            "N":len(g),
            "mean_positive_update_fraction":safe_mean(
                g["positive_update_fraction"]
            ),
            "mean_negative_update_fraction":safe_mean(
                g["negative_update_fraction"]
            ),
            "mean_positive_mass":safe_mean(
                g["positive_mass"]
            ),
            "mean_negative_mass":safe_mean(
                g["negative_mass"]
            ),
            "mean_net_mass":safe_mean(
                g["net_mass"]
            ),
            "mean_accumulation_efficiency":safe_mean(
                g["accumulation_efficiency"]
            ),
            "median_accumulation_efficiency":safe_median(
                g["accumulation_efficiency"]
            ),
            "mean_aggregated_positive_fraction":safe_mean(
                g["aggregated_positive_fraction"]
            ),
            "mean_aggregated_negative_fraction":safe_mean(
                g["aggregated_negative_fraction"]
            ),
            "mean_aggregated_accumulation_efficiency":safe_mean(
                g["aggregated_accumulation_efficiency"]
            ),
        })

    return pd.DataFrame(rows)


def summarize_layer_updates(
    df,
    extra_keys=None,
):
    if len(df)==0:
        return pd.DataFrame()

    extra_keys=list(
        extra_keys or []
    )

    keys=(
        extra_keys
        +
        ["update_layer"]
    )

    rows=[]

    for key,g in df.groupby(
        keys,
        dropna=False,
    ):
        if not isinstance(key,tuple):
            key=(key,)

        row={
            k:v
            for k,v in zip(
                keys,
                key,
            )
        }

        score=g[
            "update_target_score"
        ].astype(float).to_numpy()

        row.update({
            "N_samples":g["sid"].nunique(),
            "N_target_updates":len(g),
            "mean_update_target_score":float(
                np.mean(score)
            ),
            "median_update_target_score":float(
                np.median(score)
            ),
            "positive_fraction":float(
                np.mean(score>0)
            ),
            "negative_fraction":float(
                np.mean(score<0)
            ),
            "mean_positive_score":safe_mean(
                x
                for x in score
                if x>0
            ),
            "mean_negative_score":safe_mean(
                x
                for x in score
                if x<0
            ),
            "mean_update_target_cosine":safe_mean(
                g["update_target_cosine"]
            ),
            "mean_update_norm":safe_mean(
                g["update_norm"]
            ),
            "mean_rn_prev_projection":safe_mean(
                g["rn_prev_projection"]
            ),
            "mean_rn_cur_projection":safe_mean(
                g["rn_cur_projection"]
            ),
            "max_identity_error":float(
                np.max(
                    g["identity_error"]
                    .astype(float)
                    .to_numpy()
                )
            ),
        })

        rows.append(row)

    out=pd.DataFrame(rows)

    return out.sort_values(
        keys
    ).reset_index(drop=True)


def summarize_generation(
    df,
):
    if len(df)==0:
        return pd.DataFrame()

    b=(
        df[
            df["condition"]=="baseline"
        ]
        .drop_duplicates("sid")
        .set_index("sid")
    )

    rows=[]

    for (
        cond,
        scale,
    ),g in (
        df[
            df["condition"]!="baseline"
        ]
        .groupby(
            ["condition","scale"],
            dropna=False,
        )
    ):
        x=g.set_index("sid")

        common=sorted(
            set(b.index)
            &
            set(x.index)
        )

        if not common:
            continue

        bc=(
            b.loc[
                common,
                "correct",
            ]
            .astype(bool)
            .to_numpy()
        )

        pc=(
            x.loc[
                common,
                "correct",
            ]
            .astype(bool)
            .to_numpy()
        )

        bp=(
            b.loc[
                common,
                "prediction",
            ]
            .astype(str)
            .to_numpy()
        )

        pp=(
            x.loc[
                common,
                "prediction",
            ]
            .astype(str)
            .to_numpy()
        )

        w2c=int(
            np.sum(
                (~bc)&pc
            )
        )

        c2w=int(
            np.sum(
                bc&(~pc)
            )
        )

        rows.append({
            "condition":str(cond),
            "scale":float(scale),
            "N":len(common),
            "baseline_accuracy":float(
                np.mean(bc)
            ),
            "patched_accuracy":float(
                np.mean(pc)
            ),
            "gain":float(
                np.mean(pc)
                -
                np.mean(bc)
            ),
            "wrong_to_correct":w2c,
            "correct_to_wrong":c2w,
            "net":w2c-c2w,
            "changed":int(
                np.sum(bp!=pp)
            ),
            "repair_rate_on_wrong":(
                w2c
                /
                max(
                    int((~bc).sum()),
                    1,
                )
            ),
            "preserve_rate_on_correct":(
                1
                -
                c2w
                /
                max(
                    int(bc.sum()),
                    1,
                )
            ),
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "patched_accuracy",
                "condition",
                "scale",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def summarize_generation_by_relation(
    df,
):
    if len(df)==0:
        return pd.DataFrame()

    rows=[]

    for (
        cond,
        scale,
        gt,
    ),g in df.groupby(
        [
            "condition",
            "scale",
            "gt",
        ],
        dropna=False,
    ):
        rows.append({
            "condition":str(cond),
            "scale":float(scale),
            "relation":str(gt),
            "N":len(g),
            "accuracy":float(
                g["correct"]
                .astype(bool)
                .mean()
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "condition",
                "scale",
                "relation",
            ]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# Main
# =============================================================================

def main():
    a=parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    causal_layers=parse_layers(
        a.causal_layers
    )
    update_layers=parse_layers(
        a.update_layers
    )
    scales=parse_floats(
        a.scales
    )
    categories=parse_set(
        a.causal_categories
    )
    conditions=parse_set(
        a.conditions
    )

    valid_conditions={
        "positive_only",
        "negative_cancel",
        "signed_gated",
        "all_amplify",
        "direct_rn",
    }

    unknown=set(
        conditions
    )-valid_conditions

    if unknown:
        raise ValueError(
            f"Unknown conditions: {sorted(unknown)}"
        )

    if any(
        int(L)<1
        for L in update_layers
    ):
        raise ValueError(
            "--update-layers must be >=1 "
            "because u_L needs block L-1."
        )

    outdir=Path(
        a.output_dir
    )

    if (
        a.overwrite
        and outdir.exists()
    ):
        shutil.rmtree(outdir)

    if (
        outdir.exists()
        and any(outdir.iterdir())
    ):
        raise RuntimeError(
            f"Non-empty output dir: {outdir}"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    error_path=(
        outdir/"errors.jsonl"
    )

    two,meta,rec_by_sid=load_data(a)
    eval_sids={
        int(m["sid"])
        for m in meta
    }

    causal_sel=load_causal_selection(
        Path(a.ranked_causal),
        eval_sids,
        causal_layers,
        a.causal_top_k,
        categories,
    )

    causal_sel.to_csv(
        outdir/"selected_causal_states.csv",
        index=False,
    )

    causal_by_sid={
        int(sid):(
            g.sort_values("rank")
            .copy()
        )
        for sid,g in causal_sel.groupby(
            "sid"
        )
    }

    model=processor=None

    alignment_rows=[]
    target_update_rows_all=[]
    gated_rows_all=[]
    sample_acc_rows=[]
    generation_rows=[]

    try:
        (
            model,
            processor,
            decoder_layers,
            decoder_path,
            spec,
        )=load_model(
            a,
            two,
        )

        device=torch.device(
            a.device
        )
        n_layers=len(
            decoder_layers
        )

        capture_layers=sorted(
            set(
                causal_layers
                +
                update_layers
                +
                [
                    int(L)-1
                    for L in update_layers
                ]
            )
        )

        bad=[
            L
            for L in capture_layers
            if not 0<=L<n_layers
        ]

        if bad:
            raise ValueError(
                f"Requested/capture layers outside "
                f"0..{n_layers-1}: {bad}"
            )

        print("="*196)
        print(
            "SIGNED LAYERWISE RN UPDATE GATING"
        )
        print("="*196)
        print(
            f"model={a.model} repo={spec.repo_id}"
        )
        print(
            f"N={len(meta)}"
        )
        print(
            f"causal_layers={causal_layers} "
            f"topK={a.causal_top_k}"
        )
        print(
            f"update_layers={update_layers}"
        )
        print(
            f"scales={scales}"
        )
        print(
            f"threshold={a.score_threshold}"
        )
        print(
            f"score_aggregate={a.score_aggregate}"
        )
        print(
            f"conditions={conditions}"
        )
        print()

        for m in tqdm(
            meta,
            desc="SIGNED RN UPDATE GATING",
        ):
            sid=int(
                m["sid"]
            )

            if sid not in causal_by_sid:
                continue

            image=None

            try:
                image=base.record_image(
                    rec_by_sid[sid]
                )

                if hasattr(
                    image,
                    "convert",
                ):
                    image=image.convert(
                        "RGB"
                    )

                rb=base.make_question_batch(
                    processor=processor,
                    image=image,
                    question_text=m[
                        "question_text"
                    ],
                    device=device,
                )

                nb=build_noimage_batch(
                    processor,
                    m["question_text"],
                    device,
                )

                rid=(
                    rb["input_ids"][0]
                    .detach()
                    .cpu()
                    .tolist()
                )
                nid=(
                    nb["input_ids"][0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                r2n=lcs_token_map(
                    rid,
                    nid,
                )

                real_states=capture_blocks(
                    model,
                    decoder_layers,
                    rb,
                    capture_layers,
                )

                no_states=capture_blocks(
                    model,
                    decoder_layers,
                    nb,
                    capture_layers,
                )

                targets=prepare_targets(
                    causal_rows=causal_by_sid[sid],
                    r2n=r2n,
                    real_states=real_states,
                    no_states=no_states,
                    update_layers=update_layers,
                )

                if not targets:
                    raise RuntimeError(
                        "No selected causal target survived "
                        "REAL-NoImage alignment/update-layer filtering"
                    )

                # ---------------------------------------------------------
                # Baseline REAL generation
                # ---------------------------------------------------------
                base_text=base.generate_text(
                    model,
                    processor,
                    rb,
                    max_new_tokens=a.max_new_tokens,
                )

                base_pred=traj.normalize_relation(
                    base,
                    base_text,
                )

                base_correct=(
                    base_pred==m["gt"]
                )

                generation_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "condition":"baseline",
                    "scale":0.0,
                    "prediction":base_pred,
                    "correct":base_correct,
                    "n_patched_entries":0,
                    "n_positive_entries":0,
                    "n_negative_entries":0,
                    "text":base_text,
                })

                alignment_rows.append({
                    "sid":sid,
                    "gt":m["gt"],
                    "real_seq_len":len(rid),
                    "noimage_seq_len":len(nid),
                    "lcs_matches":len(r2n),
                    "selected_targets_requested":len(
                        causal_by_sid[sid]
                    ),
                    "selected_targets_aligned":len(
                        targets
                    ),
                })

                # ---------------------------------------------------------
                # Score clean RN update trajectory for each target
                # ---------------------------------------------------------
                sample_target_update_rows=[]

                for t in targets:
                    rows=score_target_updates(
                        sid=sid,
                        gt=m["gt"],
                        baseline_correct=base_correct,
                        target=t,
                        real_states=real_states,
                        no_states=no_states,
                    )

                    sample_target_update_rows.extend(
                        rows
                    )

                if not sample_target_update_rows:
                    raise RuntimeError(
                        "No layerwise RN updates were scored"
                    )

                gated_entries=build_gated_entries(
                    sample_target_update_rows,
                    a.score_aggregate,
                )

                if not gated_entries:
                    raise RuntimeError(
                        "No unique gated layer-token entries"
                    )

                # Raw CSV rows cannot keep ndarray helper columns.
                for r in sample_target_update_rows:
                    out_row={
                        k:v
                        for k,v in r.items()
                        if k!="_update_vector"
                    }
                    target_update_rows_all.append(
                        out_row
                    )

                for e in gated_entries:
                    gated_rows_all.append({
                        "sid":sid,
                        "gt":m["gt"],
                        "baseline_correct":base_correct,
                        "update_layer":int(
                            e["update_layer"]
                        ),
                        "real_position":int(
                            e["real_position"]
                        ),
                        "token":str(
                            e["token"]
                        ),
                        "aggregated_score":float(
                            e["aggregated_score"]
                        ),
                        "n_target_votes":int(
                            e["n_target_votes"]
                        ),
                        "positive_vote_fraction":float(
                            e["positive_vote_fraction"]
                        ),
                        "mean_target_score":float(
                            e["mean_target_score"]
                        ),
                        "maxabs_target_score":float(
                            e["maxabs_target_score"]
                        ),
                        "update_norm":float(
                            e["update_norm"]
                        ),
                        "gate_sign":(
                            "positive"
                            if float(
                                e["aggregated_score"]
                            )>a.score_threshold
                            else (
                                "negative"
                                if float(
                                    e["aggregated_score"]
                                )<(-a.score_threshold)
                                else "neutral"
                            )
                        ),
                    })

                sample_acc_rows.append(
                    accumulation_sample_row(
                        sid=sid,
                        gt=m["gt"],
                        baseline_correct=base_correct,
                        target_update_rows=sample_target_update_rows,
                        gated_entries=gated_entries,
                    )
                )

                # ---------------------------------------------------------
                # Generation conditions
                # ---------------------------------------------------------
                for cond in conditions:
                    if cond=="direct_rn":
                        patch_map=build_direct_rn_patch_map(
                            targets,
                            a.direct_beta,
                        )

                        pred,text=generate_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            patch_map=patch_map,
                            max_new_tokens=a.max_new_tokens,
                        )

                        generation_rows.append({
                            "sid":sid,
                            "gt":m["gt"],
                            "condition":"direct_rn",
                            "scale":float(
                                a.direct_beta
                            ),
                            "prediction":pred,
                            "correct":pred==m["gt"],
                            "n_patched_entries":sum(
                                len(v)
                                for v in patch_map.values()
                            ),
                            "n_positive_entries":np.nan,
                            "n_negative_entries":np.nan,
                            "text":text,
                        })

                        continue

                    for scale in scales:
                        (
                            patch_map,
                            counts,
                        )=build_condition_patch_map(
                            entries=gated_entries,
                            condition=cond,
                            scale=scale,
                            threshold=a.score_threshold,
                        )

                        pred,text=generate_with_patch(
                            model=model,
                            processor=processor,
                            decoder_layers=decoder_layers,
                            batch=rb,
                            patch_map=patch_map,
                            max_new_tokens=a.max_new_tokens,
                        )

                        generation_rows.append({
                            "sid":sid,
                            "gt":m["gt"],
                            "condition":cond,
                            "scale":float(scale),
                            "prediction":pred,
                            "correct":pred==m["gt"],
                            "n_patched_entries":counts[
                                "n_patched_entries"
                            ],
                            "n_positive_entries":counts[
                                "n_positive_entries"
                            ],
                            "n_negative_entries":counts[
                                "n_negative_entries"
                            ],
                            "text":text,
                        })

            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "phase":"sample",
                        "sid":sid,
                        "error":(
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                        "traceback_tail":(
                            traceback
                            .format_exc()
                            .splitlines()[-60:]
                        ),
                    },
                )

                tqdm.write(
                    f"[ERROR] sid={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # =================================================================
        # Save raw tables
        # =================================================================
        align_df=pd.DataFrame(
            alignment_rows
        )
        upd_df=pd.DataFrame(
            target_update_rows_all
        )
        gate_df=pd.DataFrame(
            gated_rows_all
        )
        sample_acc_df=pd.DataFrame(
            sample_acc_rows
        )
        gen_df=pd.DataFrame(
            generation_rows
        )

        align_df.to_csv(
            outdir/"alignment_summary.csv",
            index=False,
        )

        upd_df.to_csv(
            outdir/"per_target_layer_updates.csv",
            index=False,
        )

        gate_df.to_csv(
            outdir/"gated_update_map.csv",
            index=False,
        )

        sample_acc_df.to_csv(
            outdir/"sample_accumulation_summary.csv",
            index=False,
        )

        gen_df.to_csv(
            outdir/"generation_per_sample.csv",
            index=False,
        )

        # =================================================================
        # Summaries
        # =================================================================
        accum_by_correct=summarize_accumulation_by_correctness(
            sample_acc_df
        )

        accum_by_correct.to_csv(
            outdir/"accumulation_by_baseline_correctness.csv",
            index=False,
        )

        layer_summary=summarize_layer_updates(
            upd_df
        )

        layer_summary.to_csv(
            outdir/"layer_update_summary.csv",
            index=False,
        )

        layer_by_correct=summarize_layer_updates(
            upd_df,
            extra_keys=[
                "baseline_correct"
            ],
        )

        layer_by_correct.to_csv(
            outdir/"layer_update_by_baseline_correctness.csv",
            index=False,
        )

        gen_summary=summarize_generation(
            gen_df
        )

        gen_summary.to_csv(
            outdir/"generation_summary.csv",
            index=False,
        )

        gen_rel=summarize_generation_by_relation(
            gen_df
        )

        gen_rel.to_csv(
            outdir/"generation_by_relation.csv",
            index=False,
        )

        # =================================================================
        # Text report
        # =================================================================
        report=[
            "="*200,
            "SIGNED LAYERWISE RN UPDATE GATING",
            "="*200,
            f"model={a.model} repo={spec.repo_id}",
            f"N requested={len(meta)}",
            f"causal layers={causal_layers} topK={a.causal_top_k}",
            f"update layers={update_layers}",
            f"scales={scales}",
            f"threshold={a.score_threshold}",
            f"score aggregate={a.score_aggregate}",
            "",
            "ACCUMULATION BY BASELINE GENERATION CORRECTNESS",
            "-"*200,
            accum_by_correct.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(accum_by_correct) else "EMPTY",
            "",
            "LAYERWISE RN UPDATE SCORE",
            "-"*200,
            layer_summary.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(layer_summary) else "EMPTY",
            "",
            "GENERATION",
            "-"*200,
            gen_summary.to_string(
                index=False,
                float_format=lambda x:f"{x:.4f}",
            ) if len(gen_summary) else "EMPTY",
            "",
            "Definitions:",
            "  d_L = h_real[L,p] - h_noimage[L,q]",
            "  u_L = d_L - d_{L-1}",
            "  score = <u_L, t_C>/||t_C||^2, where t_C=d_C.",
            "",
            "accumulation_efficiency = sum(score) / sum(abs(score))",
            "  close to +1 : mostly constructive RN accumulation",
            "  close to  0 : strong positive/negative cancellation",
            "  below 0     : destructive updates dominate on the chosen target axis",
            "",
            "Generation conditions:",
            "  positive_only : amplify only constructive clean RN updates",
            "  negative_cancel: subtract only destructive clean RN updates",
            "  signed_gated  : do both",
            "  all_amplify   : amplify every RN update regardless of sign (control)",
            "  direct_rn     : known oracle final causal-state RN reference",
            "",
            "Most informative outcomes:",
            "  1) wrong samples have more negative mass / lower accumulation efficiency;",
            "  2) negative_cancel or signed_gated improves W2C with limited C2W;",
            "  3) signed_gated > all_amplify, showing that update sign matters rather than",
            "     merely increasing overall RN magnitude.",
            "",
            "Caution:",
            "  This is oracle because the final causal target t_C determines update sign.",
            "  If signed gating works, the next task is a non-oracle sign/quality estimator.",
        ]

        report_text="\n".join(
            report
        )+"\n"

        print(
            report_text
        )

        (
            outdir/"analysis_summary.txt"
        ).write_text(
            report_text,
            encoding="utf-8",
        )

        write_json(
            outdir/"metadata.json",
            {
                "script":"eval_signed_layerwise_rn_update_gating_v1.py",
                "model":a.model,
                "repo_id":spec.repo_id,
                "decoder_path":decoder_path,
                "eval_requested_N":len(meta),
                "causal_layers":causal_layers,
                "causal_top_k":a.causal_top_k,
                "update_layers":update_layers,
                "scales":scales,
                "score_threshold":a.score_threshold,
                "score_aggregate":a.score_aggregate,
                "conditions":conditions,
                "direct_beta":a.direct_beta,
                "rn_state":"d_L = h_real[L,p]-h_noimage[L,q]",
                "rn_update":"u_L = d_L-d_{L-1}",
                "oracle_score":"dot(u_L,t_C)/||t_C||^2 with t_C=d_C",
                "positive_intervention":"h[L,p] += alpha*u_L",
                "negative_intervention":"h[L,p] -= alpha*u_L",
                "alignment":"exact token-ID LCS",
                "oracle_note":(
                    "Causal target states come from prior oracle ranking and final "
                    "target t_C is used to label each clean RN update productive or destructive."
                ),
            },
        )

    finally:
        if model is not None:
            del model

        if processor is not None:
            del processor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__=="__main__":
    main()
