#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Attention vs MLP vs residual-carry -> DAS-U causal comparison.

At each Stage-I block L, capture natural clean/source components:

    x = block input residual
    a = attention sublayer output
    m = MLP sublayer output
    y = block output

For Qwen-style residual blocks: y = x + a + m.

All interventions are made at the SAME causal cut: block output.

    residual_carry: y' = y + (x_source - x_target)
    attention_out : y' = y + (a_source - a_target)
    mlp_out       : y' = y + (m_source - m_target)
    block_output  : y' = y_source

ROLE alignment:
    swapped/source B -> target A
    swapped/source A -> target B

IDENTITY control:
    source A -> target A
    source B -> target B

Then let all later layers run live and measure:
    * DAS-U source_progress at L22/L23/L24/L25
    * final next-token source/opposite follow
    * source-minus-target margin

Interpretation:
    attention_out strongest -> attention locally writes relation-U
    mlp_out strongest       -> MLP locally writes/refines relation-U
    residual_carry strongest-> relation mostly formed earlier and carried
    block_output strong but pieces modest -> distributed/additive block effect

Recommended:
CUDA_VISIBLE_DEVICES=0 python -u validate_attention_mlp_residual_to_das_u_v1.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --das-dir output/qwen3b_relation_das_full \
  --intervention-layers 18,19,20,21,22,23 \
  --u-layers 22,23,24,25 \
  --u-dim 16 \
  --components residual_carry,attention_out,mlp_out,block_output \
  --sample-scope das_eval \
  --pair-status both_correct \
  --max-samples 0 \
  --replacement-mode tokenwise_resample \
  --device cuda:0 \
  --output-dir output/qwen3b_attention_mlp_residual_to_das_u \
  --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import random
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

VERSION = "attention-mlp-residual-to-das-u-v1"
RELATIONS = ("left", "right", "above", "below")
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
OPPOSITE = {"left":"right","right":"left","above":"below","below":"above"}
VALID_COMPONENTS = ("residual_carry","attention_out","mlp_out","block_output")


def import_file(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def first_3d(output):
    if torch.is_tensor(output) and output.ndim == 3:
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x) and x.ndim == 3:
                return x
    raise RuntimeError("No 3D hidden tensor in module output")


def replace_first_3d(output, hidden):
    if torch.is_tensor(output) and output.ndim == 3:
        return hidden
    if isinstance(output, tuple):
        z = list(output)
        for i, x in enumerate(z):
            if torch.is_tensor(x) and x.ndim == 3:
                z[i] = hidden
                return tuple(z)
    if isinstance(output, list):
        z = list(output)
        for i, x in enumerate(z):
            if torch.is_tensor(x) and x.ndim == 3:
                z[i] = hidden
                return z
    raise RuntimeError("Could not replace hidden tensor")


def parse_ints(text: str):
    out = []
    for s in str(text).split(","):
        s = s.strip()
        if s:
            v = int(s)
            if v not in out:
                out.append(v)
    if not out:
        raise ValueError("empty integer list")
    return sorted(out)


def parse_components(text: str):
    out = []
    for s in str(text).split(","):
        s = s.strip()
        if not s:
            continue
        if s not in VALID_COMPONENTS:
            raise ValueError(f"unknown component={s}; valid={VALID_COMPONENTS}")
        if s not in out:
            out.append(s)
    return out


def safe_mean(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def safe_std(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.std()) if a.size else float("nan")


def write_csv(path: Path, rows):
    import csv
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


def append_jsonl(path: Path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def token_rows(hidden, positions, dtype):
    idx = torch.as_tensor(
        sorted(set(map(int, positions))),
        device=hidden.device,
        dtype=torch.long,
    )
    return (
        hidden[0].index_select(0, idx)
        .detach().float().cpu().numpy()
        .astype(dtype, copy=False)
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--object-state", default="mean")
    p.add_argument(
        "--source-output-dir",
        default="output/spatial_storage_transport_utilization/coco/qwen-3b",
    )
    p.add_argument("--das-dir", default="output/qwen3b_relation_das_full")
    p.add_argument("--writer-helper", default="validate_writer_to_das_u_v1.py")
    p.add_argument("--intervention-layers", default="18,19,20,21,22,23")
    p.add_argument("--u-layers", default="22,23,24,25")
    p.add_argument("--u-dim", type=int, default=16)
    p.add_argument(
        "--components",
        default="residual_carry,attention_out,mlp_out,block_output",
    )
    p.add_argument(
        "--sample-scope",
        default="das_eval",
        choices=("das_eval","all"),
    )
    p.add_argument(
        "--pair-status",
        default="both_correct",
        choices=("all","both_correct","original_only","swapped_only","both_wrong"),
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=17)
    p.add_argument(
        "--replacement-mode",
        default="tokenwise_resample",
        choices=("tokenwise_resample","pooled_broadcast","mean_shift"),
    )
    p.add_argument(
        "--run-identity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--cache-dtype", choices=("float16","float32"), default="float16")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--empty-cache-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


class BranchCapture:
    """Capture x,a,m,y at intervention layers and pooled y at DAS-U layers."""

    def __init__(
        self, decoder_layers, attention_helper,
        intervention_layers, u_layers,
        a_positions, b_positions, storage_dtype,
    ):
        self.decoder_layers = decoder_layers
        self.attention_helper = attention_helper
        self.layers = list(intervention_layers)
        self.u_layers = list(u_layers)
        self.ap = list(a_positions)
        self.bp = list(b_positions)
        self.dtype = storage_dtype
        self.handles = []
        self.components = defaultdict(dict)
        self.u_states = {}

    def store(self, layer, name, hidden):
        self.components[layer][name] = {
            "A_tokens": token_rows(hidden, self.ap, self.dtype),
            "B_tokens": token_rows(hidden, self.bp, self.dtype),
        }

    def __enter__(self):
        for layer in self.layers:
            block = self.decoder_layers[layer]
            attn = self.attention_helper.resolve_self_attention(block)
            mlp = getattr(block, "mlp", None)
            if mlp is None:
                raise RuntimeError(f"L{layer}: block has no .mlp")

            def mk_pre(L):
                def hook(_m, inputs):
                    if inputs and torch.is_tensor(inputs[0]) and inputs[0].ndim == 3:
                        self.store(L, "residual_carry", inputs[0])
                return hook

            def mk_attn(L):
                def hook(_m, _i, out):
                    self.store(L, "attention_out", first_3d(out))
                return hook

            def mk_mlp(L):
                def hook(_m, _i, out):
                    self.store(L, "mlp_out", first_3d(out))
                return hook

            def mk_block(L):
                def hook(_m, _i, out):
                    self.store(L, "block_output", first_3d(out))
                return hook

            self.handles += [
                block.register_forward_pre_hook(mk_pre(layer)),
                attn.register_forward_hook(mk_attn(layer)),
                mlp.register_forward_hook(mk_mlp(layer)),
                block.register_forward_hook(mk_block(layer)),
            ]

        for layer in self.u_layers:
            block = self.decoder_layers[layer]
            def mk_u(L):
                def hook(_m, _i, out):
                    h = first_3d(out)
                    ai = torch.as_tensor(
                        sorted(set(map(int, self.ap))),
                        device=h.device, dtype=torch.long,
                    )
                    bi = torch.as_tensor(
                        sorted(set(map(int, self.bp))),
                        device=h.device, dtype=torch.long,
                    )
                    self.u_states[L] = {
                        "A": h[0].index_select(0, ai).mean(0).detach()
                              .float().cpu().numpy().astype(self.dtype, copy=False),
                        "B": h[0].index_select(0, bi).mean(0).detach()
                              .float().cpu().numpy().astype(self.dtype, copy=False),
                    }
                return hook
            self.handles.append(block.register_forward_hook(mk_u(layer)))
        return self

    def validate(self):
        miss = []
        for L in self.layers:
            for c in VALID_COMPONENTS:
                if c not in self.components.get(L, {}):
                    miss.append(f"L{L}:{c}")
        for L in self.u_layers:
            if L not in self.u_states:
                miss.append(f"U@L{L}")
        if miss:
            raise RuntimeError(f"missing captures: {miss}")

    def __exit__(self, *args):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


def aligned_source_rows(helper, src_state, tgt_state, alignment, mode):
    if alignment == "role":
        sa_np = np.asarray(src_state["B_tokens"], np.float32)
        sb_np = np.asarray(src_state["A_tokens"], np.float32)
    elif alignment == "identity":
        sa_np = np.asarray(src_state["A_tokens"], np.float32)
        sb_np = np.asarray(src_state["B_tokens"], np.float32)
    else:
        raise ValueError(alignment)

    ta = torch.as_tensor(np.asarray(tgt_state["A_tokens"], np.float32))
    tb = torch.as_tensor(np.asarray(tgt_state["B_tokens"], np.float32))
    sa = torch.as_tensor(sa_np)
    sb = torch.as_tensor(sb_np)

    ma, _ = helper.map_source_rows_to_target(
        source_rows=sa, target_rows=ta, mode=mode
    )
    mb, _ = helper.map_source_rows_to_target(
        source_rows=sb, target_rows=tb, mode=mode
    )
    return ma.numpy().astype(np.float32), mb.numpy().astype(np.float32)


class ComponentPatch:
    """Post-block causal cut; patch one additive component, capture downstream U."""

    def __init__(
        self, helper, decoder_layers, layer, component, alignment,
        src_components, tgt_components,
        a_positions, b_positions, u_layers, replacement_mode,
    ):
        self.helper = helper
        self.decoder_layers = decoder_layers
        self.layer = int(layer)
        self.component = component
        self.alignment = alignment
        self.src = src_components
        self.tgt = tgt_components
        self.ap = sorted(set(map(int, a_positions)))
        self.bp = sorted(set(map(int, b_positions)))
        self.u_layers = [u for u in u_layers if u >= self.layer]
        self.mode = replacement_mode
        self.handles = []
        self.u_states = {}
        self.fired = 0

    def __enter__(self):
        block = self.decoder_layers[self.layer]

        def patch_hook(_m, _i, out):
            h = first_3d(out)
            y = h.float().clone()
            ai = torch.as_tensor(self.ap, device=y.device, dtype=torch.long)
            bi = torch.as_tensor(self.bp, device=y.device, dtype=torch.long)

            ya = y[0].index_select(0, ai)
            yb = y[0].index_select(0, bi)

            src_state = self.src[self.component]
            tgt_state = self.tgt[self.component]
            sa_np, sb_np = aligned_source_rows(
                self.helper, src_state, tgt_state,
                self.alignment, self.mode,
            )
            sa = torch.as_tensor(sa_np, device=y.device, dtype=torch.float32)
            sb = torch.as_tensor(sb_np, device=y.device, dtype=torch.float32)

            if self.component == "block_output":
                na, nb = sa, sb
            else:
                ta = torch.as_tensor(
                    np.asarray(tgt_state["A_tokens"], np.float32),
                    device=y.device, dtype=torch.float32,
                )
                tb = torch.as_tensor(
                    np.asarray(tgt_state["B_tokens"], np.float32),
                    device=y.device, dtype=torch.float32,
                )
                na = ya + (sa - ta)
                nb = yb + (sb - tb)

            y[0, ai, :] = na
            y[0, bi, :] = nb
            self.fired += 1

            if self.layer in self.u_layers:
                self.u_states[self.layer] = {
                    "A": na.mean(0).detach().cpu().numpy().astype(np.float32),
                    "B": nb.mean(0).detach().cpu().numpy().astype(np.float32),
                }
            return replace_first_3d(out, y.to(h.dtype))

        self.handles.append(block.register_forward_hook(patch_hook))

        for U in self.u_layers:
            if U == self.layer:
                continue
            ublock = self.decoder_layers[U]
            def mk_u(L):
                def hook(_m, _i, out):
                    h = first_3d(out)
                    ai = torch.as_tensor(self.ap, device=h.device, dtype=torch.long)
                    bi = torch.as_tensor(self.bp, device=h.device, dtype=torch.long)
                    self.u_states[L] = {
                        "A": h[0].index_select(0, ai).mean(0).detach()
                              .float().cpu().numpy().astype(np.float32),
                        "B": h[0].index_select(0, bi).mean(0).detach()
                              .float().cpu().numpy().astype(np.float32),
                    }
                return hook
            self.handles.append(ublock.register_forward_hook(mk_u(U)))
        return self

    def validate(self):
        if self.fired < 1:
            raise RuntimeError("component patch did not fire")
        miss = [U for U in self.u_layers if U not in self.u_states]
        if miss:
            raise RuntimeError(f"missing U captures {miss}")

    def __exit__(self, *args):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()


def additivity_audit(cache_components, layers):
    rows = []
    for L in layers:
        c = cache_components[L]
        for obj in ("A_tokens","B_tokens"):
            x = np.asarray(c["residual_carry"][obj], np.float32)
            a = np.asarray(c["attention_out"][obj], np.float32)
            m = np.asarray(c["mlp_out"][obj], np.float32)
            y = np.asarray(c["block_output"][obj], np.float32)
            diff = (x + a + m) - y
            rows.append({
                "layer": L,
                "object": obj[0],
                "relative_l2_error": float(
                    np.linalg.norm(diff) / (np.linalg.norm(y) + 1e-12)
                ),
                "max_abs_error": float(np.abs(diff).max()),
            })
    return rows


def build_cache(
    args, rows, layers, u_layers,
    model, processor, decoder_layers, relation_token_map,
    records_by_sid, prompt_rows, base, v3, receiver,
    attention_helper, helper, error_path,
):
    dtype = np.float16 if args.cache_dtype == "float16" else np.float32
    cache, ok_rows, audit = {}, [], []

    print("\nCaching clean/source component states...", flush=True)
    for i, row in enumerate(tqdm(rows, desc="component-cache"), 1):
        pair = None
        try:
            pair = receiver.prepare_pair(
                args=args, row=row,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
                base=base, v3=v3, processor=processor,
                device=torch.device(args.device),
            )

            cc = BranchCapture(
                decoder_layers, attention_helper, layers, u_layers,
                pair.original_a_positions, pair.original_b_positions, dtype,
            )
            with cc:
                oc = model(
                    **pair.original_batch, use_cache=False,
                    output_attentions=False, output_hidden_states=False,
                    return_dict=True,
                )
            cc.validate()
            cpred, cscores = helper.relation_scores(
                oc.logits[0, -1], relation_token_map
            )
            del oc

            sc = BranchCapture(
                decoder_layers, attention_helper, layers, u_layers,
                pair.swapped_a_positions, pair.swapped_b_positions, dtype,
            )
            with sc:
                os = model(
                    **pair.swapped_batch, use_cache=False,
                    output_attentions=False, output_hidden_states=False,
                    return_dict=True,
                )
            sc.validate()
            spred, sscores = helper.relation_scores(
                os.logits[0, -1], relation_token_map
            )
            del os

            for branch, comp in (("clean",cc.components),("source",sc.components)):
                for r in additivity_audit(comp, layers):
                    r.update({"sid": int(pair.sid), "branch": branch})
                    audit.append(r)

            cache[int(pair.sid)] = {
                "clean_components": cc.components,
                "source_components": sc.components,
                "clean_u_states": cc.u_states,
                "source_u_states": sc.u_states,
                "clean_prediction": cpred,
                "source_prediction": spred,
                "clean_scores": cscores,
                "source_scores": sscores,
            }
            ok_rows.append(dict(row))

        except Exception as e:
            append_jsonl(error_path, {
                "phase":"cache","sid":int(row["sid"]),
                "error_type":type(e).__name__,"error":str(e),
                "traceback":traceback.format_exc(),
            })
            if args.fail_fast:
                raise
        finally:
            if pair is not None:
                receiver.release_pair(pair)
            if torch.cuda.is_available() and args.empty_cache_every > 0 and i % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

    return cache, ok_rows, audit


def run_condition(
    args, layer, component, alignment, rows, cache, u_layers, bases,
    model, processor, decoder_layers, relation_token_map,
    records_by_sid, prompt_rows, base, v3, receiver, helper,
    error_path, sample_path,
):
    downstream = [u for u in u_layers if u >= layer]
    per_u = defaultdict(list)
    n = src_hits = tgt_hits = changes = clean_n = clean_to_src = 0
    margins = []

    tag = f"L{layer}_{component}_{alignment}"

    for i, row in enumerate(tqdm(rows, desc=tag, leave=False), 1):
        sid = int(row["sid"])
        pair = None
        try:
            pair = receiver.prepare_pair(
                args=args, row=row,
                records_by_sid=records_by_sid,
                prompt_rows=prompt_rows,
                base=base, v3=v3, processor=processor,
                device=torch.device(args.device),
            )

            cap = ComponentPatch(
                helper, decoder_layers, layer, component, alignment,
                cache[sid]["source_components"][layer],
                cache[sid]["clean_components"][layer],
                pair.original_a_positions, pair.original_b_positions,
                u_layers, args.replacement_mode,
            )
            with cap:
                out = model(
                    **pair.original_batch, use_cache=False,
                    output_attentions=False, output_hidden_states=False,
                    return_dict=True,
                )
            cap.validate()
            pred, scores = helper.relation_scores(
                out.logits[0, -1], relation_token_map
            )
            del out

            gt = str(pair.gt)
            src_gt = OPPOSITE[gt]
            clean_pred = str(cache[sid]["clean_prediction"])

            n += 1
            src_hits += int(pred == src_gt)
            tgt_hits += int(pred == gt)
            changes += int(pred != clean_pred)
            if clean_pred == gt:
                clean_n += 1
                clean_to_src += int(pred == src_gt)

            margins.append(
                float(scores[REL_TO_ID[src_gt]] - scores[REL_TO_ID[gt]])
            )

            sample = {
                "sid":sid,"tag":tag,"gt_target":gt,"gt_source":src_gt,
                "clean_prediction":clean_pred,"patched_prediction":pred,
            }

            for U in downstream:
                q = bases[U]
                t = helper.role_pair_coords(
                    cache[sid]["clean_u_states"][U], q, source_branch=False
                )
                s = helper.role_pair_coords(
                    cache[sid]["source_u_states"][U], q, source_branch=True
                )
                p = helper.role_pair_coords(
                    cap.u_states[U], q, source_branch=False
                )
                g = helper.u_geometry(t, s, p)
                per_u[U].append(g)
                sample[f"U{U}_progress"] = g["source_progress"]
                sample[f"U{U}_closed"] = g["fraction_distance_closed"]
                sample[f"U{U}_source_side"] = g["source_side"]

            append_jsonl(sample_path, sample)

        except Exception as e:
            append_jsonl(error_path, {
                "phase":"eval","tag":tag,"sid":sid,
                "error_type":type(e).__name__,"error":str(e),
                "traceback":traceback.format_exc(),
            })
            if args.fail_fast:
                raise
        finally:
            if pair is not None:
                receiver.release_pair(pair)
            if torch.cuda.is_available() and args.empty_cache_every > 0 and i % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

    urows = []
    for U in downstream:
        vals = per_u[U]
        urows.append({
            "intervention_layer":layer,"component":component,
            "alignment":alignment,"u_layer":U,"N":len(vals),
            "source_progress_mean":safe_mean(v["source_progress"] for v in vals),
            "source_progress_std":safe_std(v["source_progress"] for v in vals),
            "fraction_distance_closed_mean":safe_mean(
                v["fraction_distance_closed"] for v in vals
            ),
            "source_side_rate":safe_mean(v["source_side"] for v in vals),
            "off_axis_ratio_mean":safe_mean(v["off_axis_ratio"] for v in vals),
        })

    final = {
        "intervention_layer":layer,"component":component,
        "alignment":alignment,"N":n,
        "next_token_source_follow":src_hits/n if n else float("nan"),
        "next_token_target_accuracy":tgt_hits/n if n else float("nan"),
        "next_token_change_vs_clean":changes/n if n else float("nan"),
        "clean_correct_to_source_rate":clean_to_src/clean_n if clean_n else float("nan"),
        "source_minus_target_margin_mean":safe_mean(margins),
    }
    return urows, final


def comparisons(urows, finals):
    ul = {
        (int(r["intervention_layer"]),str(r["component"]),int(r["u_layer"]),str(r["alignment"])):r
        for r in urows
    }
    keys = sorted({
        (int(r["intervention_layer"]),str(r["component"]),int(r["u_layer"]))
        for r in urows if str(r["alignment"])=="role"
    })
    uc = []
    for L,c,U in keys:
        rr = ul[(L,c,U,"role")]
        ii = ul.get((L,c,U,"identity"))
        ip = float(ii["source_progress_mean"]) if ii else float("nan")
        uc.append({
            "intervention_layer":L,"component":c,"u_layer":U,"N":int(rr["N"]),
            "role_progress":float(rr["source_progress_mean"]),
            "identity_progress":ip,
            "role_minus_identity":float(rr["source_progress_mean"])-ip,
            "role_fraction_closed":float(rr["fraction_distance_closed_mean"]),
            "role_source_side":float(rr["source_side_rate"]),
            "role_off_axis":float(rr["off_axis_ratio_mean"]),
        })

    fl = {
        (int(r["intervention_layer"]),str(r["component"]),str(r["alignment"])):r
        for r in finals
    }
    fkeys = sorted({
        (int(r["intervention_layer"]),str(r["component"]))
        for r in finals if str(r["alignment"])=="role"
    })
    fc = []
    for L,c in fkeys:
        rr = fl[(L,c,"role")]
        ii = fl.get((L,c,"identity"))
        fc.append({
            "intervention_layer":L,"component":c,"N":int(rr["N"]),
            "role_source_follow":float(rr["next_token_source_follow"]),
            "identity_source_follow":float(ii["next_token_source_follow"]) if ii else float("nan"),
            "role_target_accuracy":float(rr["next_token_target_accuracy"]),
            "role_change":float(rr["next_token_change_vs_clean"]),
            "role_cleanGT_to_source":float(rr["clean_correct_to_source_rate"]),
            "role_margin":float(rr["source_minus_target_margin_mean"]),
        })
    return uc, fc


def print_tables(uc, fc):
    print("\n" + "="*122)
    print("ATTENTION vs MLP vs RESIDUAL CARRY -> DAS-U")
    print("="*122)
    print(f"{'L':>4} {'component':<16} {'U':>5} {'ROLE':>9} {'identity':>10} {'role-id':>10} {'closed':>9} {'srcSide':>9}")
    print("-"*122)
    for r in uc:
        print(
            f"L{r['intervention_layer']:<3d} {r['component']:<16} "
            f"L{r['u_layer']:<4d} {r['role_progress']:>+8.3f} "
            f"{r['identity_progress']:>+9.3f} {r['role_minus_identity']:>+9.3f} "
            f"{100*r['role_fraction_closed']:>8.2f}% {100*r['role_source_side']:>8.2f}%"
        )

    print("\n" + "="*116)
    print("FINAL NEXT-TOKEN")
    print("="*116)
    print(f"{'L':>4} {'component':<16} {'ROLE src':>10} {'ID src':>9} {'ROLE tgt':>10} {'change':>9} {'cleanGT->src':>13} {'margin':>10}")
    print("-"*116)
    for r in fc:
        print(
            f"L{r['intervention_layer']:<3d} {r['component']:<16} "
            f"{100*r['role_source_follow']:>9.2f}% {100*r['identity_source_follow']:>8.2f}% "
            f"{100*r['role_target_accuracy']:>9.2f}% {100*r['role_change']:>8.2f}% "
            f"{100*r['role_cleanGT_to_source']:>12.2f}% {r['role_margin']:>+9.4f}"
        )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    layers = parse_ints(args.intervention_layers)
    u_layers = parse_ints(args.u_layers)
    components = parse_components(args.components)

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"output dir not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / "errors.jsonl"

    helper = import_file(Path(args.writer_helper), "_amr_helper")
    ioi = import_file(Path("analyze_coco_ioi_backward_circuit_v1.py"), "_amr_ioi")
    producer = import_file(Path("analyze_coco_producer_qk_ov_v1.py"), "_amr_prod")
    receiver = import_file(Path("analyze_coco_receiver_qkv_v1.py"), "_amr_recv")
    v3 = import_file(Path("analyze_spatial_storage_transport_utilization_v3.py"), "_amr_v3")
    base = import_file(Path("analyze_coco_centroid_generation_step1_v4.py"), "_amr_base")
    attention_helper = import_file(Path("analyze_coco_flip_attention_spatial_vectors_v1.py"), "_amr_attn")

    rows = [
        r for r in helper.read_jsonl(Path(args.source_output_dir)/"extraction.jsonl")
        if str(r.get("gt")) in RELATIONS
    ]

    dasdir = Path(args.das_dir)
    if args.sample_scope == "das_eval":
        cfg = json.loads((dasdir/"config.json").read_text(encoding="utf-8"))
        eval_sids = set(map(int, cfg.get("eval_sids", [])))
        rows = [r for r in rows if int(r["sid"]) in eval_sids]

    if args.pair_status != "all":
        rows = [r for r in rows if str(r.get("generation_pair_status","")) == args.pair_status]

    rows = helper.stratified_subset(
        rows=rows, limit=args.max_samples, seed=args.sample_seed
    )
    if not rows:
        raise RuntimeError("no samples after filtering")

    bases = helper.load_das_bases(
        das_dir=dasdir, u_layers=u_layers, u_dim=args.u_dim
    )

    model = processor = None
    try:
        model, processor, spec, decoder_layers, decoder_path, relation_token_map = (
            producer.load_model_bundle(args=args, base=base)
        )
        model.eval()

        saved = args.max_samples
        args.max_samples = None
        try:
            records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)
        finally:
            args.max_samples = saved

        print("\n"+"="*120)
        print("ATTENTION vs MLP vs RESIDUAL-CARRY")
        print("="*120)
        print("N                    :", len(rows))
        print("intervention layers  :", layers)
        print("components           :", components)
        print("U layers / dim       :", u_layers, "/", args.u_dim)
        print("pair status          :", args.pair_status)
        print("same causal cut      : post-block residual output")
        print("="*120, flush=True)

        cfg = {
            "version":VERSION,"model":args.model,"N_requested":len(rows),
            "intervention_layers":layers,"components":components,
            "u_layers":u_layers,"u_dim":args.u_dim,
            "pair_status":args.pair_status,"sample_scope":args.sample_scope,
            "definitions":{
                "residual_carry":"y'=y+(x_source-x_target)",
                "attention_out":"y'=y+(a_source-a_target)",
                "mlp_out":"y'=y+(m_source-m_target)",
                "block_output":"y'=y_source",
            },
            "audit":audit,
        }
        (outdir/"config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")

        cache, rows, add = build_cache(
            args, rows, layers, u_layers, model, processor, decoder_layers,
            relation_token_map, records_by_sid, prompt_rows, base, v3,
            receiver, attention_helper, helper, error_path,
        )
        write_csv(outdir/"component_additivity_audit.csv", add)

        print("\nAdditivity audit (mean relative L2):")
        for L in layers:
            vals = [r["relative_l2_error"] for r in add if r["layer"] == L]
            print(f"  L{L}: {safe_mean(vals):.6e}")

        all_u, all_f = [], []
        alignments = ["role","identity"] if args.run_identity else ["role"]

        for L in layers:
            if not any(U >= L for U in u_layers):
                continue
            for c in components:
                for a in alignments:
                    print(f"\n>>> L{L} {c} {a.upper()}", flush=True)
                    ur, fr = run_condition(
                        args,L,c,a,rows,cache,u_layers,bases,
                        model,processor,decoder_layers,relation_token_map,
                        records_by_sid,prompt_rows,base,v3,receiver,helper,
                        error_path,outdir/f"samples_L{L}_{c}_{a}.jsonl",
                    )
                    all_u += ur
                    all_f.append(fr)
                    write_csv(outdir/"component_u_effects_all.csv", all_u)
                    write_csv(outdir/"component_final_effects_all.csv", all_f)

            uc, fc = comparisons(
                [r for r in all_u if r["intervention_layer"] == L],
                [r for r in all_f if r["intervention_layer"] == L],
            )
            print_tables(uc, fc)

        uc, fc = comparisons(all_u, all_f)
        write_csv(outdir/"attention_mlp_residual_u_comparison.csv", uc)
        write_csv(outdir/"attention_mlp_residual_final_comparison.csv", fc)

        # Per-layer/component best downstream U.
        best = []
        for L in layers:
            for c in components:
                rr = [r for r in uc if r["intervention_layer"] == L and r["component"] == c]
                if not rr:
                    continue
                b = max(rr, key=lambda x: x["role_progress"])
                best.append({
                    "intervention_layer":L,"component":c,
                    "best_u_layer":b["u_layer"],
                    "best_role_progress":b["role_progress"],
                    "best_identity_progress":b["identity_progress"],
                    "best_role_minus_identity":b["role_minus_identity"],
                    "mean_role_progress":safe_mean(x["role_progress"] for x in rr),
                })
        write_csv(outdir/"attention_mlp_residual_best_by_layer.csv", best)

        print_tables(uc, fc)

        print("\n"+"="*110)
        print("BEST LOCAL COMPONENT BY LAYER (excluding full block reference)")
        print("="*110)
        for L in layers:
            rr = [r for r in best if r["intervention_layer"] == L and r["component"] != "block_output"]
            if not rr:
                continue
            b = max(rr, key=lambda x:x["best_role_progress"])
            br = next((r for r in best if r["intervention_layer"] == L and r["component"]=="block_output"), None)
            print(
                f"L{L}: {b['component']} "
                f"bestProg={b['best_role_progress']:+.3f} @U=L{b['best_u_layer']} "
                f"| full-block={br['best_role_progress']:+.3f}" if br else
                f"L{L}: {b['component']} bestProg={b['best_role_progress']:+.3f}"
            )

        report = [
            f"version: {VERSION}",
            f"N: {len(rows)}",
            f"layers: {layers}",
            f"U: {u_layers}, dim={args.u_dim}",
            "",
            "All component interventions use the same post-block causal cut.",
            "",
            "Interpretation:",
            "attention_out dominant -> attention locally writes relation-U",
            "mlp_out dominant -> MLP locally writes/refines relation-U",
            "residual_carry dominant -> relation mostly formed earlier and carried",
            "block_output >> every component -> distributed/additive block effect",
            "",
            "Use ROLE >> IDENTITY to establish relation-specificity.",
        ]
        (outdir/"report.txt").write_text("\n".join(report)+"\n",encoding="utf-8")

        print("\nSaved:")
        for name in (
            "attention_mlp_residual_u_comparison.csv",
            "attention_mlp_residual_final_comparison.csv",
            "attention_mlp_residual_best_by_layer.csv",
            "component_additivity_audit.csv",
            "report.txt",
        ):
            print(" ", outdir/name)

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
