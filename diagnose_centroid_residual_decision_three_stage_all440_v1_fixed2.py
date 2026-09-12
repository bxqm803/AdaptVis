#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_centroid_residual_decision_three_stage_all440_v1.py

All-440 three-stage diagnostic for Qwen2.5-VL-3B on COCO_two:

    C = attention-centroid spatial geometry
    R = object-pair residual spatial geometry
    D = baseline generated decision

Goal
====
Test whether attention-centroid geometry and the Synthetic-defined residual H/V
spatial subspace carry the same sample-specific spatial signal, rather than
merely both correlating with the GT relation.

The script performs THREE levels of analysis:

1) RAW CONTINUOUS ALIGNMENT
   For the same sample:

       centroid c = (c_H, c_V) = (dx, -dy)
       residual z = mean_{L in residual layers} (z_H,L, z_V,L)

   where +H=right and +V=above for BOTH coordinate systems.

   Report:
       corr(c_H, z_H), corr(c_V, z_V)
       cross corr(c_H, z_V), corr(c_V, z_H)
       sign agreement and 2D cosine.

2) WITHIN-RELATION ALIGNMENT (critical control)
   Remove each GT relation's mean separately:

       c~_i = c_i - mean(c | GT_i)
       z~_i = z_i - mean(z | GT_i)

   Then recompute the correlations.  If same-axis correlations survive this
   centering, centroid explains sample-specific residual geometry beyond the
   trivial fact that both readouts encode left/right/above/below.

   A repeated stratified 30/70 held-out linear transport test is also included:

       relation-only:       z_hat = mean_train(z | GT)
       + centroid residual: z_hat = mean_train(z | GT)
                              + A [c - mean_train(c | GT)]

   The improvement in held-out MSE tests whether centroid contains information
   about residual state beyond GT relation identity.

3) DISCRETE THREE-STAGE C -> R -> D TABLE
   For each sample:
       C_correct = centroid top1 == GT
       R_correct = residual top1 == GT
       D_correct = baseline generation == GT

   Report all 8 cells and, importantly:
       C=GT, R=GT, D!=GT
   which is the cleanest downstream decision-mismatch cohort.

Attention centroid variants
===========================
A SINGLE model forward with output_attentions=True is reused to compute several
centroid readouts.  Defaults for Qwen-3B:

    24:mean   = L24 head-mean centroid (the repo's centroid-steering reader)
    24:5      = L24H05, an established strong centroid head
    27:10     = L27H10, an established strong centroid head

The primary C stage defaults to L24H05, but all specified variants receive
continuous-correlation and three-stage summaries.

Residual R
==========
The residual coordinate system exactly matches the previous continuous spatial
geometry diagnostic:

    q_L = [(h_sub-h_ref)_REAL] - [(h_sub-h_ref)_NOIMAGE]

Synthetic-400 defines H/V axes at every layer.  Target COCO q_L is projected
into these axes and normalized by half the Synthetic left-right / above-below
class gaps.  L20-L26 are averaged by default.

No intervention is performed.  Target GT is used only for diagnosis/evaluation
(and for the explicitly diagnostic within-relation centering / held-out control).

Recommended all-440 run
=======================
CUDA_VISIBLE_DEVICES=0 python -u \
  diagnose_centroid_residual_decision_three_stage_all440_v1.py \
  --model qwen-3b \
  --source-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_synthetic_hsub_href_all.npz \
  --target-spatial-npz \
    output/qwen3b_hsub_href_spatial_cache/qwen-3b_coco_two_hsub_href_all_originalprompt_mean.npz \
  --baseline-csv \
    output/qwen3b_synthetic400_to_coco440_originalprompt_mean_v2/per_sample_candidate_repair.csv \
  --residual-layers 20-26 \
  --centroid-specs 24:mean,24:5,27:10 \
  --primary-centroid 24:5 \
  --expected-n 440 \
  --device cuda:0 \
  --output-dir output/qwen3b_centroid_residual_decision_all440_v1 \
  --overwrite

If centroid_all_samples.csv already exists from a previous run, skip model
loading by adding:

  --centroid-csv path/to/centroid_all_samples.csv

Main outputs
============
centroid_all_samples.csv
per_sample_three_stage.csv
continuous_relationship_by_centroid.csv
continuous_relationship_by_residual_layer.csv
within_relation_relationship_by_gt.csv
three_stage_8cells.csv
cr_conditioned_decision.csv
three_stage_by_gt.csv
linear_transport_cv.csv
linear_transport_summary.csv
analysis_summary.txt
metadata.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REL = ("left", "right", "above", "below")
EPS = 1e-10


def norm_rel(x: Any) -> str:
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")
    s = str(x).strip().lower().replace("-", "_")
    table = {
        "left": "left", "left of": "left", "l": "left",
        "right": "right", "right of": "right", "r": "right",
        "above": "above", "on": "above", "over": "above", "top": "above",
        "below": "below", "under": "below", "beneath": "below", "bottom": "below",
    }
    return table.get(s, s)


def parse_layers(text: str) -> List[int]:
    out = set()
    for part in str(text).split(","):
        part = part.strip().upper().replace("L", "")
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("No layers parsed")
    return sorted(out)


def parse_centroid_spec_one(text: str) -> Tuple[int, Optional[int], str]:
    raw = str(text).strip().lower().replace("l", "")
    if ":" not in raw:
        raise ValueError(f"Bad centroid spec {text!r}; use e.g. 24:mean or 24:5")
    ltxt, htxt = raw.split(":", 1)
    layer = int(ltxt)
    htxt = htxt.strip().replace("h", "")
    if htxt == "mean":
        head = None
        name = f"L{layer}_mean"
    else:
        head = int(htxt)
        name = f"L{layer}H{head:02d}"
    return layer, head, name


def parse_centroid_specs(text: str) -> List[Tuple[int, Optional[int], str]]:
    specs = []
    seen = set()
    for part in str(text).split(","):
        if not part.strip():
            continue
        spec = parse_centroid_spec_one(part)
        if spec[2] not in seen:
            specs.append(spec)
            seen.add(spec[2])
    if not specs:
        raise ValueError("No centroid specs parsed")
    return specs


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise RuntimeError("Cannot normalize near-zero vector")
    return v / n


def axis_evidence(z: Sequence[float]) -> Dict[str, float]:
    h, v = float(z[0]), float(z[1])
    return {"left": -h, "right": h, "above": v, "below": -v}


def relation_from_hv(h: float, v: float) -> str:
    # +H=right, +V=above
    if abs(h) >= abs(v):
        return "right" if h >= 0 else "left"
    return "above" if v >= 0 else "below"


def safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x, y) -> float:
    x = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").to_numpy()
    y = pd.Series(np.asarray(y, dtype=np.float64)).rank(method="average").to_numpy()
    return safe_corr(x, y)


def mean_cosine_2d(A: np.ndarray, B: np.ndarray) -> float:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    good = den > EPS
    if not np.any(good):
        return float("nan")
    return float(np.mean(num[good] / den[good]))


def load_state_npz(path: Path, require_labels: bool):
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        if "relation_vectors" in keys:
            X = np.asarray(z["relation_vectors"], dtype=np.float32)
            definition = str(z["vector_definition"].item()) if "vector_definition" in keys else "relation_vectors"
        elif {"img", "no_image"}.issubset(keys):
            X = np.asarray(z["img"], dtype=np.float32) - np.asarray(z["no_image"], dtype=np.float32)
            definition = "img_minus_no_image"
        else:
            raise RuntimeError(f"Bad NPZ keys in {path}: {sorted(keys)}")

        if "decoder_block_index" not in keys:
            raise RuntimeError(f"{path} missing decoder_block_index")
        layers = [int(v) for v in np.asarray(z["decoder_block_index"]).tolist()]
        sids = (
            np.asarray(z["sample_index"], dtype=np.int64)
            if "sample_index" in keys else np.arange(len(X), dtype=np.int64)
        )
        labels = None
        if "relation" in keys:
            labels = np.asarray([norm_rel(v) for v in z["relation"].tolist()], dtype=object)
        elif require_labels:
            raise RuntimeError(f"{path} requires relation labels")

    if X.ndim != 3:
        raise RuntimeError(f"Expected [N,L,D], got {X.shape}")
    return X, labels, layers, sids, definition


def _first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def load_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    cols = set(df.columns)
    sid_col = _first_existing(cols, ["sid", "sample_index", "sample_id", "index"])
    gt_col = _first_existing(cols, ["gt", "relation", "ground_truth", "answer"])
    pred_col = _first_existing(cols, [
        "baseline_prediction", "baseline_pred", "baseline_relation",
        "base_prediction", "base_pred", "prediction",
    ])
    correct_col = _first_existing(cols, ["baseline_correct", "base_correct", "correct"])
    if sid_col is None or gt_col is None or pred_col is None:
        raise RuntimeError(
            f"baseline CSV needs sid/GT/baseline-pred columns; got {sorted(cols)}"
        )
    out = pd.DataFrame({
        "sid": pd.to_numeric(df[sid_col], errors="raise").astype(int),
        "gt": [norm_rel(x) for x in df[gt_col]],
        "decision_pred": [norm_rel(x) for x in df[pred_col]],
    })
    if correct_col is not None:
        vals = df[correct_col]
        if vals.dtype == bool:
            out["decision_correct"] = vals.astype(bool).to_numpy()
        else:
            s = vals.astype(str).str.strip().str.lower()
            out["decision_correct"] = s.isin(["1", "true", "t", "yes", "y"]).to_numpy()
    else:
        out["decision_correct"] = out["decision_pred"] == out["gt"]

    if out["sid"].duplicated().any():
        raise RuntimeError("baseline CSV has duplicate sid")
    bad_gt = sorted(set(out["gt"]) - set(REL))
    bad_pred = sorted(set(out["decision_pred"]) - set(REL))
    if bad_gt or bad_pred:
        raise RuntimeError(f"Bad labels: gt={bad_gt}, pred={bad_pred}")
    return out.sort_values("sid").reset_index(drop=True)


def fit_hv_geometry(X, y, source_layers, wanted_layers):
    lmap = {L: i for i, L in enumerate(source_layers)}
    geom = {}
    rows = []
    for L in wanted_layers:
        if L not in lmap:
            raise RuntimeError(f"Source missing L{L}")
        Xf = X[:, lmap[L]].astype(np.float64)
        center = Xf.mean(axis=0)
        mus = {r: Xf[y == r].mean(axis=0) for r in REL}
        dirs = {r: unit(mus[r] - center) for r in REL}
        dH = unit(dirs["right"] - dirs["left"])
        dV = unit(dirs["above"] - dirs["below"])
        gapH = float(np.dot(mus["right"] - mus["left"], dH))
        gapV = float(np.dot(mus["above"] - mus["below"], dV))
        if gapH < 0:
            dH, gapH = -dH, -gapH
        if gapV < 0:
            dV, gapV = -dV, -gapV
        B = np.stack([dH, dV], axis=1)
        dual = B @ np.linalg.inv(B.T @ B)
        geom[L] = {
            "center": center,
            "dual": dual,
            "halfH": max(gapH / 2.0, EPS),
            "halfV": max(gapV / 2.0, EPS),
            "layer_index": lmap[L],
        }
        rows.append({
            "layer": L,
            "axis_H_dot_axis_V": float(np.dot(dH, dV)),
            "full_gap_H": gapH,
            "full_gap_V": gapV,
        })
    return geom, pd.DataFrame(rows)


def read_coord(x, g):
    res = np.asarray(x, dtype=np.float64) - g["center"]
    c = g["dual"].T @ res
    return np.asarray([c[0] / g["halfH"], c[1] / g["halfV"]], dtype=np.float64)


def build_residual_table(
    X_target: np.ndarray,
    target_layers: Sequence[int],
    target_sids: np.ndarray,
    geom: Mapping[int, Mapping[str, Any]],
    residual_layers: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lmap = {L: i for i, L in enumerate(target_layers)}
    layer_rows = []
    sample_rows = []
    for i, sid in enumerate(target_sids):
        zs = []
        for L in residual_layers:
            if L not in lmap:
                raise RuntimeError(f"Target missing L{L}")
            z = read_coord(X_target[i, lmap[L]], geom[L])
            zs.append(z)
            ev = axis_evidence(z)
            layer_rows.append({
                "sid": int(sid), "layer": int(L),
                "z_H": float(z[0]), "z_V": float(z[1]),
                **{f"e_{r}": float(ev[r]) for r in REL},
                "residual_pred_layer": relation_from_hv(float(z[0]), float(z[1])),
            })
        Z = np.stack(zs, axis=0)
        zmean = Z.mean(axis=0)
        evmean = axis_evidence(zmean)
        rpred = relation_from_hv(float(zmean[0]), float(zmean[1]))
        sample_rows.append({
            "sid": int(sid),
            "z_H": float(zmean[0]), "z_V": float(zmean[1]),
            "residual_norm": float(np.linalg.norm(zmean)),
            **{f"residual_e_{r}": float(evmean[r]) for r in REL},
            "residual_pred": rpred,
        })
    return pd.DataFrame(sample_rows), pd.DataFrame(layer_rows)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b", "qwen-7b"])
    p.add_argument("--source-spatial-npz", required=True)
    p.add_argument("--target-spatial-npz", required=True)
    p.add_argument("--baseline-csv", required=True)
    p.add_argument("--residual-layers", default="20-26")
    p.add_argument(
        "--centroid-specs", default="24:mean,24:5,27:10",
        help="Comma list: layer:mean or layer:head, e.g. 24:mean,24:5,27:10",
    )
    p.add_argument(
        "--primary-centroid", default="24:5",
        help="One centroid spec used as the primary C stage in per-sample/8-cell tables.",
    )
    p.add_argument(
        "--centroid-csv", default=None,
        help="Optional previously generated centroid_all_samples.csv; skips model loading.",
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--expected-n", type=int, default=440)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all; for smoke test only")
    p.add_argument("--transport-fit-frac", type=float, default=0.30)
    p.add_argument("--transport-repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    a.residual_layers_parsed = parse_layers(a.residual_layers)
    a.centroid_specs_parsed = parse_centroid_specs(a.centroid_specs)
    a.primary_centroid_parsed = parse_centroid_spec_one(a.primary_centroid)
    names = [x[2] for x in a.centroid_specs_parsed]
    if a.primary_centroid_parsed[2] not in names:
        p.error("--primary-centroid must also appear in --centroid-specs")
    if not (0 < a.transport_fit_frac < 1):
        p.error("--transport-fit-frac must be in (0,1)")
    if a.transport_repeats < 1:
        p.error("--transport-repeats must be >=1")
    return a


def import_centroid_repo_module():
    try:
        import eval_coco_centroid_select_late_direction_qwen25_v1 as csel
    except Exception as exc:
        raise SystemExit(
            "Could not import eval_coco_centroid_select_late_direction_qwen25_v1.py. "
            "Run from the AdaptVis llava16 repository root.\n"
            f"{type(exc).__name__}: {exc}"
        )
    return csel


def extract_centroid_specs_one_sample(
    csel,
    model,
    processor,
    image,
    record,
    specs: Sequence[Tuple[int, Optional[int], str]],
    device: str,
) -> Dict[str, Dict[str, Any]]:
    import torch

    dev = torch.device(device)
    batch = csel.make_batch(processor, image, record, dev)
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    input_length = len(input_ids)
    subject_span, reference_span = csel.cent.locate_object_spans(
        processor.tokenizer,
        input_ids,
        record["subject"],
        record["reference"],
    )
    subject_index = int(subject_span[1])
    reference_index = int(reference_span[1])
    visual_indices = csel.cent.resolve_visual_indices(
        model, processor, batch, input_ids
    )
    coords = csel.cent.visual_coordinates(
        model, batch, len(visual_indices), batch["input_ids"].device
    )
    if coords is None:
        raise RuntimeError(f"Could not construct visual coordinates, n={len(visual_indices)}")

    with torch.inference_mode():
        outputs = model(
            **batch,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )
    attentions = csel.resolve_attention_tuple(outputs)
    coords_np = coords.detach().float().cpu().numpy().astype(np.float32)

    by_layer: Dict[int, Dict[str, Any]] = {}
    for layer, _, _ in specs:
        if layer in by_layer:
            continue
        if not (0 <= int(layer) < len(attentions)):
            raise RuntimeError(f"Centroid L{layer} unavailable; attentions={len(attentions)}")
        prompt_tensor = csel.cent.normalize_attention_tensor(
            attentions[int(layer)], expected_query_length=input_length
        )
        rows = prompt_tensor[:, [subject_index, reference_index], :]
        metrics = csel.cent.query_attention_metrics(
            rows, visual_indices, coords, subject_index, reference_index
        )
        maps = metrics["visual_maps"].detach().float().cpu().numpy().astype(np.float32)
        visual_mass = metrics["visual_mass"].detach().float().mean(dim=0).cpu().numpy()
        by_layer[int(layer)] = {
            "maps": maps,
            "visual_mass": visual_mass,
        }

    result = {}
    for layer, head, name in specs:
        maps = by_layer[layer]["maps"]
        if head is None:
            m = maps.mean(axis=0)
        else:
            if not (0 <= head < maps.shape[0]):
                raise RuntimeError(f"{name}: head outside 0..{maps.shape[0]-1}")
            m = maps[head]
        centroids = np.einsum("ov,vd->od", m, coords_np).astype(np.float32)
        dx = float(centroids[0, 0] - centroids[1, 0])
        dy = float(centroids[0, 1] - centroids[1, 1])
        pred, axis_conf = csel.cent.relation_from_centroids(dx, dy)
        # Align coordinate convention with residual geometry: +V = above.
        cH = dx
        cV = -dy
        result[name] = {
            "centroid_pred": norm_rel(pred),
            "dx": dx, "dy": dy,
            "c_H": cH, "c_V": cV,
            "centroid_norm": float(math.hypot(cH, cV)),
            "axis_confidence": float(axis_conf),
            "subject_x": float(centroids[0, 0]),
            "subject_y": float(centroids[0, 1]),
            "reference_x": float(centroids[1, 0]),
            "reference_y": float(centroids[1, 1]),
            "subject_visual_mass": float(by_layer[layer]["visual_mass"][0]),
            "reference_visual_mass": float(by_layer[layer]["visual_mass"][1]),
            "n_visual_tokens": int(len(visual_indices)),
            "n_heads": int(maps.shape[0]),
        }

    del outputs, attentions, batch
    return result


def compute_centroid_all(args, outdir: Path) -> pd.DataFrame:
    csel = import_centroid_repo_module()
    # csel.load_coco_records expects these args.
    load_args = argparse.Namespace(
        data_root=args.data_root,
        prompt_jsonl=args.prompt_jsonl,
        max_samples=(args.max_samples if args.max_samples > 0 else None),
        model=args.model,
        dtype=args.dtype,
        device=args.device,
    )
    records, _audit = csel.load_coco_records(load_args)
    model, processor, _layers, decoder_path, spec = csel.load_model_and_processor(load_args)
    print(f"[centroid] model={spec.repo_id} decoder={decoder_path} N={len(records)}")
    print(f"[centroid] specs={[x[2] for x in args.centroid_specs_parsed]}")

    rows = []
    for j, record in enumerate(records, 1):
        image = None
        try:
            image = csel.open_record_image(record)
            vals = extract_centroid_specs_one_sample(
                csel, model, processor, image, record,
                args.centroid_specs_parsed, args.device,
            )
            row = {
                "sid": int(record["sid"]),
                "gt": norm_rel(record["relation"]),
            }
            for name, d in vals.items():
                for k, v in d.items():
                    row[f"{name}__{k}"] = v
            rows.append(row)
        except Exception as exc:
            print(f"[centroid ERROR] sid={record['sid']} {type(exc).__name__}: {exc}")
            raise
        finally:
            if image is not None:
                image.close()
            csel.cleanup()
        if j % 20 == 0 or j == len(records):
            print(f"[centroid] {j}/{len(records)}")

    df = pd.DataFrame(rows).sort_values("sid").reset_index(drop=True)
    df.to_csv(outdir / "centroid_all_samples.csv", index=False)
    del model, processor
    gc.collect()
    return df


def load_centroid_csv(path: Path, specs) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sid" not in df.columns:
        raise RuntimeError("centroid csv missing sid")
    if "gt" not in df.columns and "relation" in df.columns:
        df["gt"] = [norm_rel(x) for x in df["relation"]]
    if "gt" in df.columns:
        df["gt"] = [norm_rel(x) for x in df["gt"]]
    for _, _, name in specs:
        for suffix in ["centroid_pred", "c_H", "c_V"]:
            col = f"{name}__{suffix}"
            if col not in df.columns:
                raise RuntimeError(f"centroid csv missing {col}")
        df[f"{name}__centroid_pred"] = [norm_rel(x) for x in df[f"{name}__centroid_pred"]]
    return df.sort_values("sid").reset_index(drop=True)


def center_within_relation(df: pd.DataFrame, cols: Sequence[str], gt_col="gt") -> np.ndarray:
    X = df[list(cols)].to_numpy(dtype=np.float64)
    out = np.empty_like(X)
    gts = df[gt_col].to_numpy(dtype=object)
    for r in REL:
        idx = np.where(gts == r)[0]
        if len(idx) == 0:
            continue
        out[idx] = X[idx] - X[idx].mean(axis=0, keepdims=True)
    return out


def relationship_row(name: str, C: np.ndarray, Z: np.ndarray, prefix: str) -> Dict[str, Any]:
    hcorr = safe_corr(C[:, 0], Z[:, 0])
    vcorr = safe_corr(C[:, 1], Z[:, 1])
    hv = safe_corr(C[:, 0], Z[:, 1])
    vh = safe_corr(C[:, 1], Z[:, 0])
    return {
        "centroid": name,
        "mode": prefix,
        "N": len(C),
        "pearson_cH_zH": hcorr,
        "pearson_cV_zV": vcorr,
        "pearson_cH_zV_cross": hv,
        "pearson_cV_zH_cross": vh,
        "same_axis_mean_abs_corr": float(np.nanmean(np.abs([hcorr, vcorr]))),
        "cross_axis_mean_abs_corr": float(np.nanmean(np.abs([hv, vh]))),
        "same_minus_cross_abs_corr": float(
            np.nanmean(np.abs([hcorr, vcorr])) - np.nanmean(np.abs([hv, vh]))
        ),
        "spearman_cH_zH": spearman_corr(C[:, 0], Z[:, 0]),
        "spearman_cV_zV": spearman_corr(C[:, 1], Z[:, 1]),
        "sign_agree_H": float(np.mean(np.sign(C[:, 0]) == np.sign(Z[:, 0]))),
        "sign_agree_V": float(np.mean(np.sign(C[:, 1]) == np.sign(Z[:, 1]))),
        "mean_cosine_2d": mean_cosine_2d(C, Z),
    }


def stratified_train_indices(gt: np.ndarray, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    tr, te = [], []
    for r in REL:
        idx = np.where(gt == r)[0].tolist()
        rng.shuffle(idx)
        n = int(round(len(idx) * frac))
        n = max(1, min(n, len(idx) - 1))
        tr.extend(idx[:n])
        te.extend(idx[n:])
    return np.asarray(sorted(tr), dtype=int), np.asarray(sorted(te), dtype=int)


def linear_transport_cv(df: pd.DataFrame, centroid_name: str, repeats: int, fit_frac: float, seed: int):
    C = df[[f"{centroid_name}__c_H", f"{centroid_name}__c_V"]].to_numpy(dtype=np.float64)
    Z = df[["z_H", "z_V"]].to_numpy(dtype=np.float64)
    gt = df["gt"].to_numpy(dtype=object)
    rows = []
    for rep in range(repeats):
        tr, te = stratified_train_indices(gt, fit_frac, seed + rep * 9973)
        cmean, zmean = {}, {}
        for r in REL:
            ridx = tr[gt[tr] == r]
            cmean[r] = C[ridx].mean(axis=0)
            zmean[r] = Z[ridx].mean(axis=0)
        Ctr = np.stack([C[i] - cmean[gt[i]] for i in tr])
        Ztr = np.stack([Z[i] - zmean[gt[i]] for i in tr])
        # 2D -> 2D centered linear map; no intercept after relation centering.
        A, *_ = np.linalg.lstsq(Ctr, Ztr, rcond=None)
        Z0 = np.stack([zmean[gt[i]] for i in te])
        Cte_centered = np.stack([C[i] - cmean[gt[i]] for i in te])
        Z1 = Z0 + Cte_centered @ A
        Y = Z[te]
        mse0 = float(np.mean((Y - Z0) ** 2))
        mse1 = float(np.mean((Y - Z1) ** 2))
        # Correlation of relation-centered predicted deviations with actual deviations.
        actual_dev = Y - Z0
        pred_dev = Z1 - Z0
        rows.append({
            "centroid": centroid_name,
            "repeat": rep,
            "N_train": len(tr), "N_test": len(te),
            "relation_only_mse": mse0,
            "centroid_transport_mse": mse1,
            "relative_mse_improvement": (mse0 - mse1) / max(mse0, EPS),
            "dev_corr_H": safe_corr(actual_dev[:, 0], pred_dev[:, 0]),
            "dev_corr_V": safe_corr(actual_dev[:, 1], pred_dev[:, 1]),
            "A_H_to_H": float(A[0, 0]),
            "A_H_to_V": float(A[0, 1]),
            "A_V_to_H": float(A[1, 0]),
            "A_V_to_V": float(A[1, 1]),
        })
    return pd.DataFrame(rows)


def summarize_three_stage(df: pd.DataFrame, centroid_name: str):
    ccol = f"{centroid_name}__centroid_correct"
    work = df.copy()
    work["C"] = work[ccol].astype(bool)
    work["R"] = work["residual_correct"].astype(bool)
    work["D"] = work["decision_correct"].astype(bool)

    rows8 = []
    for C in [True, False]:
        for R in [True, False]:
            for D in [True, False]:
                s = work[(work.C == C) & (work.R == R) & (work.D == D)]
                rows8.append({
                    "centroid": centroid_name,
                    "C_centroid_correct": C,
                    "R_residual_correct": R,
                    "D_decision_correct": D,
                    "N": len(s),
                    "fraction_all": len(s) / max(len(work), 1),
                })

    cr_rows = []
    for C in [True, False]:
        for R in [True, False]:
            s = work[(work.C == C) & (work.R == R)]
            cr_rows.append({
                "centroid": centroid_name,
                "C_centroid_correct": C,
                "R_residual_correct": R,
                "N": len(s),
                "fraction_all": len(s) / max(len(work), 1),
                "decision_accuracy": float(s.D.mean()) if len(s) else float("nan"),
                "decision_wrong_N": int((~s.D).sum()) if len(s) else 0,
            })

    bygt = []
    for r in REL:
        g = work[work["gt"] == r]
        for C in [True, False]:
            for R in [True, False]:
                s = g[(g.C == C) & (g.R == R)]
                bygt.append({
                    "centroid": centroid_name,
                    "gt": r,
                    "C_centroid_correct": C,
                    "R_residual_correct": R,
                    "N": len(s),
                    "gt_N": len(g),
                    "fraction_within_gt": len(s) / max(len(g), 1),
                    "decision_accuracy": float(s.D.mean()) if len(s) else float("nan"),
                    "decision_wrong_N": int((~s.D).sum()) if len(s) else 0,
                })

    summary = {
        "centroid": centroid_name,
        "N": len(work),
        "centroid_accuracy": float(work.C.mean()),
        "residual_accuracy": float(work.R.mean()),
        "decision_accuracy": float(work.D.mean()),
        "C_equals_R_rate": float((work[f"{centroid_name}__centroid_pred"] == work["residual_pred"]).mean()),
        "C_equals_D_rate": float((work[f"{centroid_name}__centroid_pred"] == work["decision_pred"]).mean()),
        "R_equals_D_rate": float((work["residual_pred"] == work["decision_pred"]).mean()),
        "C_R_both_GT_N": int((work.C & work.R).sum()),
        "C_R_both_GT_decision_accuracy": float(work.loc[work.C & work.R, "D"].mean()) if (work.C & work.R).any() else float("nan"),
        "C_GT_R_GT_D_wrong_N": int((work.C & work.R & (~work.D)).sum()),
        "C_GT_R_wrong_D_wrong_N": int((work.C & (~work.R) & (~work.D)).sum()),
        "C_wrong_R_GT_D_wrong_N": int(((~work.C) & work.R & (~work.D)).sum()),
        "C_wrong_R_wrong_D_wrong_N": int(((~work.C) & (~work.R) & (~work.D)).sum()),
    }
    return summary, pd.DataFrame(rows8), pd.DataFrame(cr_rows), pd.DataFrame(bygt)


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline(Path(args.baseline_csv))
    if args.max_samples > 0:
        keep = set(baseline.sid.sort_values().head(args.max_samples).tolist())
        baseline = baseline[baseline.sid.isin(keep)].copy()
    if args.expected_n > 0 and args.max_samples == 0 and baseline.sid.nunique() != args.expected_n:
        raise RuntimeError(
            f"Expected {args.expected_n} baseline sids, got {baseline.sid.nunique()}. "
            "Do not accidentally use an N80 baseline file."
        )

    Xs, ys, Ls, sids_s, source_def = load_state_npz(Path(args.source_spatial_npz), require_labels=True)
    Xt, yt, Lt, sids_t, target_def = load_state_npz(Path(args.target_spatial_npz), require_labels=False)
    geom, geom_df = fit_hv_geometry(Xs, ys, Ls, args.residual_layers_parsed)
    geom_df.to_csv(outdir / "residual_hv_geometry.csv", index=False)
    residual, residual_layer = build_residual_table(
        Xt, Lt, sids_t, geom, args.residual_layers_parsed
    )

    if args.centroid_csv:
        centroid = load_centroid_csv(Path(args.centroid_csv), args.centroid_specs_parsed)
        centroid.to_csv(outdir / "centroid_all_samples.csv", index=False)
    else:
        centroid = compute_centroid_all(args, outdir)

    # Inner join should be exact for all-440.
    df = baseline.merge(residual, on="sid", how="inner").merge(centroid, on="sid", how="inner", suffixes=("", "_centroid"))
    if "gt_centroid" in df.columns:
        mismatch = df[df["gt"] != df["gt_centroid"]]
        if len(mismatch):
            raise RuntimeError(f"GT mismatch baseline vs centroid for {len(mismatch)} samples")
        df = df.drop(columns=["gt_centroid"])
    if args.max_samples == 0 and args.expected_n > 0 and len(df) != args.expected_n:
        raise RuntimeError(f"Merged N={len(df)} != expected {args.expected_n}")

    # Attach layer table only to retained sids and GT.
    residual_layer = residual_layer[residual_layer.sid.isin(set(df.sid))].merge(df[["sid", "gt"]], on="sid", how="left")

    df["residual_correct"] = df["residual_pred"] == df["gt"]
    for _, _, name in args.centroid_specs_parsed:
        df[f"{name}__centroid_correct"] = df[f"{name}__centroid_pred"] == df["gt"]
        df[f"{name}__equals_residual"] = df[f"{name}__centroid_pred"] == df["residual_pred"]
        df[f"{name}__equals_decision"] = df[f"{name}__centroid_pred"] == df["decision_pred"]

    primary = args.primary_centroid_parsed[2]
    df["primary_centroid"] = primary
    df["C_pred"] = df[f"{primary}__centroid_pred"]
    df["C_correct"] = df[f"{primary}__centroid_correct"]
    df["R_pred"] = df["residual_pred"]
    df["R_correct"] = df["residual_correct"]
    df["D_pred"] = df["decision_pred"]
    df["D_correct"] = df["decision_correct"]
    # Build the categorical C/R/D pattern in pure Python.
    # Do NOT concatenate NumPy unicode arrays with `+`: NumPy 2.x raises
    # UFuncNoLoopError for that operation.
    df["three_stage_pattern"] = [
        f"{'C+' if bool(c) else 'C-'}/{'R+' if bool(r) else 'R-'}/{'D+' if bool(d) else 'D-'}"
        for c, r, d in zip(
            df["C_correct"].tolist(),
            df["R_correct"].tolist(),
            df["D_correct"].tolist(),
        )
    ]
    df.to_csv(outdir / "per_sample_three_stage.csv", index=False)
    residual_layer.to_csv(outdir / "per_sample_residual_layer.csv", index=False)

    # ------------------------------------------------------------------
    # Continuous C <-> R relationship for every centroid variant.
    # ------------------------------------------------------------------
    rel_rows = []
    within_gt_rows = []
    layer_rel_rows = []
    Z = df[["z_H", "z_V"]].to_numpy(dtype=np.float64)
    Zc = center_within_relation(df, ["z_H", "z_V"])

    layer_pivot_h = residual_layer.pivot(index="sid", columns="layer", values="z_H")
    layer_pivot_v = residual_layer.pivot(index="sid", columns="layer", values="z_V")

    for _, _, name in args.centroid_specs_parsed:
        C = df[[f"{name}__c_H", f"{name}__c_V"]].to_numpy(dtype=np.float64)
        Cc = center_within_relation(df, [f"{name}__c_H", f"{name}__c_V"])
        rel_rows.append(relationship_row(name, C, Z, "raw"))
        rel_rows.append(relationship_row(name, Cc, Zc, "within_relation_centered"))

        for r in REL:
            idx = np.where(df["gt"].to_numpy(dtype=object) == r)[0]
            if len(idx) >= 3:
                rr = relationship_row(name, C[idx] - C[idx].mean(0), Z[idx] - Z[idx].mean(0), f"within_{r}")
                rr["gt"] = r
                within_gt_rows.append(rr)

        # Layerwise residual relation: same C against z_L.
        ordered_sid = df.sid.to_numpy(dtype=int)
        for L in args.residual_layers_parsed:
            ZL = np.stack([
                layer_pivot_h.loc[int(sid), L] for sid in ordered_sid
            ]), np.stack([
                layer_pivot_v.loc[int(sid), L] for sid in ordered_sid
            ])
            ZL = np.stack(ZL, axis=1).astype(np.float64)
            temp = df[["sid", "gt"]].copy()
            temp["zH"] = ZL[:, 0]
            temp["zV"] = ZL[:, 1]
            ZLc = center_within_relation(temp, ["zH", "zV"])
            rawr = relationship_row(name, C, ZL, "raw")
            rawr["residual_layer"] = L
            cenr = relationship_row(name, Cc, ZLc, "within_relation_centered")
            cenr["residual_layer"] = L
            layer_rel_rows.extend([rawr, cenr])

    relationship_df = pd.DataFrame(rel_rows)
    relationship_df.to_csv(outdir / "continuous_relationship_by_centroid.csv", index=False)
    pd.DataFrame(layer_rel_rows).to_csv(outdir / "continuous_relationship_by_residual_layer.csv", index=False)
    pd.DataFrame(within_gt_rows).to_csv(outdir / "within_relation_relationship_by_gt.csv", index=False)

    # ------------------------------------------------------------------
    # Three-stage tables for all centroid variants.
    # ------------------------------------------------------------------
    stage_summary_rows = []
    cells8_all, cr_all, bygt_all = [], [], []
    for _, _, name in args.centroid_specs_parsed:
        summ, cells8, cr, bygt = summarize_three_stage(df, name)
        stage_summary_rows.append(summ)
        cells8_all.append(cells8)
        cr_all.append(cr)
        bygt_all.append(bygt)
    stage_summary = pd.DataFrame(stage_summary_rows)
    cells8 = pd.concat(cells8_all, ignore_index=True)
    cr_table = pd.concat(cr_all, ignore_index=True)
    bygt_table = pd.concat(bygt_all, ignore_index=True)
    stage_summary.to_csv(outdir / "three_stage_summary.csv", index=False)
    cells8.to_csv(outdir / "three_stage_8cells.csv", index=False)
    cr_table.to_csv(outdir / "cr_conditioned_decision.csv", index=False)
    bygt_table.to_csv(outdir / "three_stage_by_gt.csv", index=False)

    # ------------------------------------------------------------------
    # Held-out relation-controlled linear transport C -> R.
    # ------------------------------------------------------------------
    cv_frames = []
    for _, _, name in args.centroid_specs_parsed:
        cv_frames.append(linear_transport_cv(
            df, name, args.transport_repeats, args.transport_fit_frac, args.seed
        ))
    cv = pd.concat(cv_frames, ignore_index=True)
    cv.to_csv(outdir / "linear_transport_cv.csv", index=False)
    cv_summary = (
        cv.groupby("centroid", as_index=False)
        .agg(
            repeats=("repeat", "count"),
            mean_relation_only_mse=("relation_only_mse", "mean"),
            mean_centroid_transport_mse=("centroid_transport_mse", "mean"),
            mean_relative_mse_improvement=("relative_mse_improvement", "mean"),
            std_relative_mse_improvement=("relative_mse_improvement", "std"),
            mean_dev_corr_H=("dev_corr_H", "mean"),
            mean_dev_corr_V=("dev_corr_V", "mean"),
            mean_A_H_to_H=("A_H_to_H", "mean"),
            mean_A_H_to_V=("A_H_to_V", "mean"),
            mean_A_V_to_H=("A_V_to_H", "mean"),
            mean_A_V_to_V=("A_V_to_V", "mean"),
        )
    )
    cv_summary.to_csv(outdir / "linear_transport_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Terminal summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 180)
    print("THREE-STAGE C(ATTENTION CENTROID) -> R(RESIDUAL SPATIAL) -> D(DECISION)")
    print("=" * 180)
    print(f"N={len(df)} | decision_acc={df.decision_correct.mean():.4f} | residual_top1_acc={df.residual_correct.mean():.4f}")
    print(f"residual_layers={args.residual_layers_parsed} | primary_centroid={primary}")

    print("\nCENTROID VARIANTS / STAGE AGREEMENT")
    print("-" * 180)
    show_cols = [
        "centroid", "centroid_accuracy", "residual_accuracy", "decision_accuracy",
        "C_equals_R_rate", "C_equals_D_rate", "R_equals_D_rate",
        "C_R_both_GT_N", "C_R_both_GT_decision_accuracy", "C_GT_R_GT_D_wrong_N",
    ]
    print(stage_summary[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCONTINUOUS CENTROID <-> RESIDUAL RELATION")
    print("-" * 180)
    show_rel = [
        "centroid", "mode", "pearson_cH_zH", "pearson_cV_zV",
        "pearson_cH_zV_cross", "pearson_cV_zH_cross",
        "same_minus_cross_abs_corr", "mean_cosine_2d",
    ]
    print(relationship_df[show_rel].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nPRIMARY C/R CONDITION -> DECISION ACCURACY")
    print("-" * 180)
    print(cr_table[cr_table.centroid == primary].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nPRIMARY THREE-STAGE 8 CELLS")
    print("-" * 180)
    print(cells8[cells8.centroid == primary].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nPRIMARY BY GT: C/R CONDITION -> DECISION")
    print("-" * 180)
    print(bygt_table[bygt_table.centroid == primary].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nRELATION-CONTROLLED HELD-OUT LINEAR TRANSPORT C -> R")
    print("-" * 180)
    print(cv_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Compact text summary.
    ps = stage_summary[stage_summary.centroid == primary].iloc[0]
    pr_raw = relationship_df[(relationship_df.centroid == primary) & (relationship_df["mode"] == "raw")].iloc[0]
    pr_ctr = relationship_df[(relationship_df.centroid == primary) & (relationship_df["mode"] == "within_relation_centered")].iloc[0]
    pcv = cv_summary[cv_summary.centroid == primary].iloc[0]
    text = []
    text.append(f"N={len(df)}")
    text.append(f"primary_centroid={primary}")
    text.append(f"centroid_accuracy={ps['centroid_accuracy']:.6f}")
    text.append(f"residual_accuracy={ps['residual_accuracy']:.6f}")
    text.append(f"decision_accuracy={ps['decision_accuracy']:.6f}")
    text.append(f"C_equals_R_rate={ps['C_equals_R_rate']:.6f}")
    text.append(f"C_R_both_GT_N={int(ps['C_R_both_GT_N'])}")
    text.append(f"C_R_both_GT_decision_accuracy={ps['C_R_both_GT_decision_accuracy']:.6f}")
    text.append(f"C_GT_R_GT_D_wrong_N={int(ps['C_GT_R_GT_D_wrong_N'])}")
    text.append(f"raw_corr_H={pr_raw['pearson_cH_zH']:.6f}")
    text.append(f"raw_corr_V={pr_raw['pearson_cV_zV']:.6f}")
    text.append(f"centered_corr_H={pr_ctr['pearson_cH_zH']:.6f}")
    text.append(f"centered_corr_V={pr_ctr['pearson_cV_zV']:.6f}")
    text.append(f"transport_relative_mse_improvement={pcv['mean_relative_mse_improvement']:.6f}")
    (outdir / "analysis_summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")

    metadata = {
        "script": Path(__file__).name,
        "model": args.model,
        "source_spatial_npz": args.source_spatial_npz,
        "target_spatial_npz": args.target_spatial_npz,
        "source_definition": source_def,
        "target_definition": target_def,
        "baseline_csv": args.baseline_csv,
        "residual_layers": args.residual_layers_parsed,
        "centroid_specs": [x[2] for x in args.centroid_specs_parsed],
        "primary_centroid": primary,
        "centroid_csv_input": args.centroid_csv,
        "N": len(df),
        "transport_fit_frac": args.transport_fit_frac,
        "transport_repeats": args.transport_repeats,
        "seed": args.seed,
        "note": "GT is used only for post-hoc diagnosis, within-relation centering, and diagnostic transport CV.",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\n[saved] {outdir}")


if __name__ == "__main__":
    main()
