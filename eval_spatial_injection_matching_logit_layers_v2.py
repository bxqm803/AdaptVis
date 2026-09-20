#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_spatial_injection_matching_logit_layers_v2.py

Standalone-on-repo diagnostic that reuses the EXACT spatial geometry and patch
operator from:

    eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.py

No Figure-B helper scripts are required.

Goal
====
For each source layer L, causally move the object-pair spatial representation
toward / away from each semantic relation and read ONLY that relation's own
final decision logit.

Desired rule, independently for each relation r:

    toward r  -> logit(r) increases
    away r    -> logit(r) decreases

No opposite-relation logit contrast is used as the primary endpoint.

Geometry / intervention
=======================
This imports v6 and uses its exact Synthetic-400 Real-Gray geometry:

    q_L = (h_sub_real - h_ref_real) - (h_sub_gray - h_ref_gray)

The v6 geometry fits H/V directions and an orthonormal basis Q spanning that
same 2-D spatial subspace.

We reconstruct the actual semantic axes from v6's saved B=[d_H,d_V], express
them in Q-coordinates, and call v6.SpatialPatch.  Therefore the hidden-state
edit is exactly:

    h_sub <- h_sub + delta_z/2
    h_ref <- h_ref - delta_z/2

with delta_z constrained to the SAME spatial subspace and applied by the SAME
patch operator used by the recovery method.

For each layer:

    toward left  = -strength * d_H
    toward right = +strength * d_H
    toward under = -strength * d_V
    toward on    = +strength * d_V

and "away" is the sign reversal.

Decision logit
==============
For COCO, this uses the SAME fast relation-token scoring convention as v6:
for each relation, take the maximum final-prompt logit across its valid
single-token surface variants.

This is deliberately a fixed-strength mechanism probe, not the full recovery
optimizer.  Recovery uses the same spatial edit operator/subspace but chooses
the layerwise coordinates adaptively with its minimum-norm optimization.

Recommended first run
=====================
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u \
  eval_spatial_injection_matching_logit_layers_v2.py \
  --layers 20-26 \
  --strengths 1.0 \
  --coco-max-samples 80 \
  --output-dir output/qwen3b_spatial_matching_logit_L20_26_n80_v2 \
  --overwrite

If strength=1 is too small, rerun a dose sweep, e.g.:

  --strengths 1,5,10

Outputs
=======
per_sample_effects.csv
layer_relation_summary.csv
layer_principle_summary.csv
synthetic400_HV_geometry.csv
analysis_summary.txt
metadata.json
errors.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

try:
    import eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6 as ctrl
except Exception as exc:
    raise SystemExit(
        "Could not import eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.py.\n"
        "Run this script from the AdaptVis llava16 repo root.\n"
        f"{type(exc).__name__}: {exc}"
    )


REL = ("left", "right", "above", "below")
DISPLAY = {
    "left": "left",
    "right": "right",
    "above": "on",
    "below": "under",
}
SCRIPT_VERSION = "spatial-injection-matching-logit-layers-v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # v6 / head-scan compatible fields.
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--coco-prompt-jsonl",
        "--prompt-jsonl",
        dest="coco_prompt_jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--controlled-json", default="data/controlled_images_dataset.json")
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default=None)
    p.add_argument("--source-max-samples", type=int, default=None)
    p.add_argument("--target-max-samples", type=int, default=None)

    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument("--revision", default="main")
    p.add_argument("--control", default="gray", choices=["gray"])
    p.add_argument("--geometry-control", default="gray", choices=["gray"])
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--pool", default="mean", choices=["mean", "last"])
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--seed", type=int, default=17)

    # Fields expected by common loader even though Qwen does not use them.
    p.add_argument("--internvl-input-size", type=int, default=448)
    p.add_argument("--internvl-max-num-tiles", type=int, default=12)
    p.add_argument(
        "--internvl-use-thumbnail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--layers", default="20-26")
    p.add_argument(
        "--strengths",
        default="1.0",
        help=(
            "Comma-separated residual-space L2 magnitudes along the semantic "
            "H/V axes. v6 itself does not use fixed natural-gap scaling."
        ),
    )
    p.add_argument("--coco-max-samples", type=int, default=80)
    p.add_argument("--eval-seed", type=int, default=17)

    p.add_argument(
        "--geometry-cache",
        default="",
        help=(
            "Optional v6-compatible Synthetic geometry NPZ. If omitted, "
            "<output-dir>/geometry_cache/synthetic400_residual_real_minus_gray.npz "
            "is built/reused."
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args()


def parse_layers(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    if not out:
        raise ValueError("No --layers supplied")
    return out


def parse_strengths(text: str) -> List[float]:
    vals = sorted(set(float(x) for x in str(text).replace(" ", "").split(",") if x))
    if not vals or any(x <= 0 for x in vals):
        raise ValueError("--strengths must be positive")
    return vals


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(str(k))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sem(values: Sequence[float]) -> float:
    a = np.asarray(values, dtype=np.float64)
    if len(a) <= 1:
        return 0.0
    return float(a.std(ddof=1) / math.sqrt(len(a)))


def stratified_cap(
    rows: Sequence[Mapping[str, Any]],
    n: int,
    seed: int,
) -> List[dict]:
    rows = [dict(x) for x in rows]
    if n <= 0 or n >= len(rows):
        return rows

    rng = random.Random(seed)
    buckets: Dict[str, List[dict]] = {r: [] for r in REL}
    other: List[dict] = []
    for row in rows:
        r = ctrl.norm_rel(row.get("relation"))
        if r in buckets:
            buckets[r].append(row)
        else:
            other.append(row)

    for v in buckets.values():
        rng.shuffle(v)

    chosen: List[dict] = []
    base_n = n // 4
    rem = n % 4
    for j, r in enumerate(REL):
        take = base_n + (1 if j < rem else 0)
        chosen.extend(buckets[r][:take])

    if len(chosen) < n:
        used = {int(x["sid"]) for x in chosen}
        leftovers = [x for x in rows if int(x["sid"]) not in used]
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: n - len(chosen)])

    return sorted(chosen[:n], key=lambda x: int(x["sid"]))


def relation_axis_coords(geom_L: Mapping[str, Any], relation: str) -> np.ndarray:
    """
    Express v6's semantic axis d_H or d_V in v6's orthonormal Q coordinates.

    Since Q spans exactly span(d_H,d_V), Q @ (Q^T d_axis) = d_axis.
    """
    Q = np.asarray(geom_L["control_basis"], dtype=np.float64)
    B = np.asarray(geom_L["B"], dtype=np.float64)
    dH = B[:, 0]
    dV = B[:, 1]

    if relation == "left":
        dz = -dH
    elif relation == "right":
        dz = +dH
    elif relation == "below":
        dz = -dV
    elif relation == "above":
        dz = +dV
    else:
        raise ValueError(relation)

    c = Q.T @ dz
    recon = Q @ c
    err = float(np.linalg.norm(recon - dz))
    if err > 1e-5:
        raise RuntimeError(
            f"Axis reconstruction error for {relation}: {err:.3e}"
        )
    return c.astype(np.float64)


def fast_scores(
    *,
    model: Any,
    decoder_layers: Sequence[Any],
    prepared: Any,
    token_ids: Mapping[str, Sequence[int]],
    layer: Optional[int] = None,
    geom: Optional[Mapping[int, Any]] = None,
    coords: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if layer is None:
        with torch.inference_mode():
            out = model(**prepared.batch, use_cache=False, return_dict=True)
    else:
        c = torch.as_tensor(
            np.asarray(coords, dtype=np.float32).reshape(1, 2),
            device=prepared.batch["input_ids"].device,
            dtype=torch.float32,
        )
        with ctrl.SpatialPatch(
            decoder_layers=decoder_layers,
            layers=[int(layer)],
            geom=geom,
            sub_pos=prepared.sub_pos,
            ref_pos=prepared.ref_pos,
            coords=c,
        ), torch.inference_mode():
            out = model(**prepared.batch, use_cache=False, return_dict=True)

    last = out.logits[0, -1].float()
    scores = {
        r: float(
            torch.stack([last[int(tok)] for tok in token_ids[r]])
            .max()
            .detach()
            .item()
        )
        for r in REL
    }
    del out, last
    return scores


def summarize_relation(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out: List[dict] = []

    for (L, strength, rel), g in df.groupby(
        ["layer", "strength", "relation"], sort=True
    ):
        toward = g["delta_logit_toward"].to_numpy(dtype=float)
        away = g["delta_logit_away"].to_numpy(dtype=float)
        out.append({
            "layer": int(L),
            "strength": float(strength),
            "relation": str(rel),
            "relation_display": DISPLAY[str(rel)],
            "N": int(len(g)),
            "toward_mean": float(toward.mean()),
            "toward_sem": sem(toward),
            "away_mean": float(away.mean()),
            "away_sem": sem(away),
            "paired_toward_minus_away_mean": float((toward - away).mean()),
            "toward_positive_rate": float(np.mean(toward > 0)),
            "away_negative_rate": float(np.mean(away < 0)),
            "both_sign_rate": float(np.mean((toward > 0) & (away < 0))),
            "mean_sign_rule_pass": int(toward.mean() > 0 and away.mean() < 0),
        })
    return pd.DataFrame(out)


def summarize_layers(rel_df: pd.DataFrame) -> pd.DataFrame:
    out: List[dict] = []
    for (L, strength), g in rel_df.groupby(
        ["layer", "strength"], sort=True
    ):
        row: Dict[str, Any] = {
            "layer": int(L),
            "strength": float(strength),
            "relations_passing_mean_sign_rule": int(
                g["mean_sign_rule_pass"].sum()
            ),
            "all_four_pass": int(
                len(g) == 4 and int(g["mean_sign_rule_pass"].sum()) == 4
            ),
            "mean_toward_logit": float(g["toward_mean"].mean()),
            "mean_away_logit": float(g["away_mean"].mean()),
            "mean_paired_toward_minus_away": float(
                g["paired_toward_minus_away_mean"].mean()
            ),
            "mean_both_sign_rate": float(g["both_sign_rate"].mean()),
        }
        for rel in REL:
            q = g[g["relation"] == rel]
            if len(q):
                rr = q.iloc[0]
                name = DISPLAY[rel]
                row[f"{name}_toward"] = float(rr["toward_mean"])
                row[f"{name}_away"] = float(rr["away_mean"])
                row[f"{name}_both_sign_rate"] = float(rr["both_sign_rate"])
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    a = parse_args()
    layers = parse_layers(a.layers)
    strengths = parse_strengths(a.strengths)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    outdir = Path(a.output_dir)
    if a.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / "errors.jsonl"

    model = processor = None

    try:
        # --------------------------------------------------------------
        # Exact v6 model loader.
        # --------------------------------------------------------------
        model, processor, decoder_layers, _nh, _hd, spec = (
            ctrl.hs.load_model_bundle(a.model, a)
        )
        for p in model.parameters():
            p.requires_grad_(False)

        n_layers = len(decoder_layers)
        if min(layers) < 0 or max(layers) >= n_layers:
            raise RuntimeError(
                f"Requested layers={layers}, but model has {n_layers} layers"
            )

        # --------------------------------------------------------------
        # Exact v6 Synthetic-400 geometry.
        # --------------------------------------------------------------
        source_rows = ctrl.hs.load_synthetic(a)

        if a.geometry_cache:
            geom_cache = Path(a.geometry_cache)
        else:
            geom_cache = (
                outdir
                / "geometry_cache"
                / "synthetic400_residual_real_minus_gray.npz"
            )

        Xg, yg = ctrl.build_synthetic_geometry_cache(
            model_alias=a.model,
            model=model,
            processor_or_backend=processor,
            decoder_layers=decoder_layers,
            layers=layers,
            source_rows=source_rows,
            cache_path=geom_cache,
            args=a,
        )
        geom, axis_df = ctrl.fit_geometry(Xg, yg, layers)
        axis_df.to_csv(
            outdir / "synthetic400_HV_geometry.csv",
            index=False,
        )

        # --------------------------------------------------------------
        # Exact v6 relation-logit surface convention.
        # --------------------------------------------------------------
        fast_token_ids, details = ctrl.encode_fast_relation_tokens(
            processor.tokenizer, "coco"
        )
        if fast_token_ids is None:
            raise RuntimeError(
                "COCO relation surfaces were not all single-token under the "
                "v6 scoring convention. This diagnostic currently requires "
                "the same v6 fast-logit path.\n"
                + json.dumps(details, ensure_ascii=False, indent=2)
            )

        # --------------------------------------------------------------
        # COCO target rows.
        # --------------------------------------------------------------
        target_rows = ctrl.hs.load_target("coco", a)
        target_rows = stratified_cap(
            target_rows,
            int(a.coco_max_samples),
            int(a.eval_seed),
        )
        if not target_rows:
            raise RuntimeError("No COCO target rows")

        print("\n" + "=" * 132)
        print("SPATIAL INJECTION -> OWN FINAL RELATION LOGIT (v6 geometry/operator)")
        print("=" * 132)
        print(f"model={a.model} repo={getattr(spec, 'repo_id', spec)}")
        print(f"layers={layers}")
        print(f"strengths={strengths}")
        print(f"Synthetic geometry N={len(source_rows)}")
        print(f"COCO eval N={len(target_rows)}")
        print(f"relation token variants={fast_token_ids}")
        print(
            "Rule: toward relation -> its OWN logit rises; "
            "away from relation -> its OWN logit falls."
        )
        print(
            "Spatial subspace + SpatialPatch are imported directly from recovery v6."
        )
        print("=" * 132 + "\n")

        rows: List[dict] = []

        for row in tqdm(target_rows, desc="COCO signed spatial-logit test"):
            sid = int(row["sid"])
            image = prepared = None
            try:
                image = Image.open(str(row["image_path"])).convert("RGB")
                prepared = ctrl.prepare_standard(
                    model=model,
                    processor=processor,
                    decoder_layers=decoder_layers,
                    row=row,
                    image=image,
                    args=a,
                )

                base_scores = fast_scores(
                    model=model,
                    decoder_layers=decoder_layers,
                    prepared=prepared,
                    token_ids=fast_token_ids,
                )

                for L in layers:
                    # Four semantic directions represented in the EXACT v6 Q basis.
                    rel_coord = {
                        r: relation_axis_coords(geom[L], r)
                        for r in REL
                    }

                    # Only four unique signed axis forwards are required:
                    # +H, -H, +V, -V. Cache by coordinate tuple.
                    for strength in strengths:
                        score_cache: Dict[Tuple[float, float], Dict[str, float]] = {}

                        def get_scores(c: np.ndarray) -> Dict[str, float]:
                            c = np.asarray(c, dtype=np.float64).reshape(2)
                            key = tuple(np.round(c, 12).tolist())
                            if key not in score_cache:
                                score_cache[key] = fast_scores(
                                    model=model,
                                    decoder_layers=decoder_layers,
                                    prepared=prepared,
                                    token_ids=fast_token_ids,
                                    layer=L,
                                    geom=geom,
                                    coords=c,
                                )
                            return score_cache[key]

                        for rel in REL:
                            c_t = float(strength) * rel_coord[rel]
                            c_a = -c_t

                            toward_scores = get_scores(c_t)
                            away_scores = get_scores(c_a)

                            d_t = float(toward_scores[rel] - base_scores[rel])
                            d_a = float(away_scores[rel] - base_scores[rel])

                            rows.append({
                                "sid": sid,
                                "gt": ctrl.norm_rel(row.get("relation")),
                                "layer": int(L),
                                "strength": float(strength),
                                "relation": rel,
                                "relation_display": DISPLAY[rel],
                                "base_logit": float(base_scores[rel]),
                                "toward_logit": float(toward_scores[rel]),
                                "away_logit": float(away_scores[rel]),
                                "delta_logit_toward": d_t,
                                "delta_logit_away": d_a,
                                "paired_toward_minus_away": d_t - d_a,
                                "toward_positive": int(d_t > 0),
                                "away_negative": int(d_a < 0),
                                "both_sign_rule": int(d_t > 0 and d_a < 0),
                            })

            except Exception as exc:
                ctrl.append_jsonl(
                    error_path,
                    {
                        "sid": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc().splitlines()[-60:],
                    },
                )
                tqdm.write(
                    f"[ERROR sid={sid}] {type(exc).__name__}: {exc}"
                )
                if a.fail_fast:
                    raise
            finally:
                if prepared is not None:
                    prepared.close()
                if image is not None:
                    image.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not rows:
            raise RuntimeError("No successful evaluation rows")

        write_csv(outdir / "per_sample_effects.csv", rows)

        rel_df = summarize_relation(rows)
        layer_df = summarize_layers(rel_df)

        rel_df.to_csv(
            outdir / "layer_relation_summary.csv",
            index=False,
        )
        layer_df.to_csv(
            outdir / "layer_principle_summary.csv",
            index=False,
        )

        # --------------------------------------------------------------
        # Console report.
        # --------------------------------------------------------------
        lines: List[str] = []
        lines.append("=" * 132)
        lines.append(
            "SIGNED RULE: TOWARD r -> own logit UP; AWAY r -> own logit DOWN"
        )
        lines.append("=" * 132)

        for strength in strengths:
            lines.append(f"[strength={strength:g}]")
            g = layer_df[layer_df["strength"] == float(strength)]
            for _, rr in g.iterrows():
                chunks = []
                for name in ("left", "right", "on", "under"):
                    chunks.append(
                        f"{name}({rr.get(name + '_toward', float('nan')):+.4f}/"
                        f"{rr.get(name + '_away', float('nan')):+.4f})"
                    )
                lines.append(
                    f"L{int(rr.layer):02d} | "
                    f"pass={int(rr.relations_passing_mean_sign_rule)}/4 | "
                    f"mean toward={rr.mean_toward_logit:+.5f} | "
                    f"mean away={rr.mean_away_logit:+.5f} | "
                    f"both-sign={rr.mean_both_sign_rate:.3f}"
                )
                lines.append("      " + " ".join(chunks))
            lines.append("")

        lines.append(
            "Each relation tuple is (toward delta-own-logit / away delta-own-logit)."
        )
        lines.append("Desired pattern for every tuple: (+ / -).")

        report = "\n".join(lines) + "\n"
        print("\n" + report)
        (outdir / "analysis_summary.txt").write_text(
            report, encoding="utf-8"
        )

        meta = {
            "script_version": SCRIPT_VERSION,
            "model": a.model,
            "layers": layers,
            "strengths": strengths,
            "synthetic_geometry_source": "Synthetic-400 Real-Gray",
            "geometry_function": (
                "eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.fit_geometry"
            ),
            "patch_operator": (
                "eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.SpatialPatch"
            ),
            "spatial_basis": (
                "exact v6 orthonormal control basis spanning semantic H/V axes"
            ),
            "semantic_edit": (
                "v6 B=[d_H,d_V] axes reconstructed exactly in Q coordinates"
            ),
            "decision_readout": (
                "same v6 fast single-token relation-logit convention"
            ),
            "uses_opposite_relation_logit": False,
            "full_recovery_optimizer_used": False,
            "interpretation": (
                "fixed-strength mechanism probe using the same spatial edit "
                "subspace/operator as recovery; recovery adaptively optimizes "
                "multi-layer coordinates"
            ),
            "coco_eval_N": len(target_rows),
            "relation_token_ids": fast_token_ids,
        }
        (outdir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"[SAVED] {outdir / 'per_sample_effects.csv'}")
        print(f"[SAVED] {outdir / 'layer_relation_summary.csv'}")
        print(f"[SAVED] {outdir / 'layer_principle_summary.csv'}")

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
