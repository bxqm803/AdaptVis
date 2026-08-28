#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Attention-vs-MLP Direction contribution scan for Qwen-7B.

This script uses the existing Direction representation only as a diagnostic
coordinate system. It does NOT repair the model.

Correct / wrong grouping is based on the cached ACTUAL model.generate() result
from:
    <direction-dir>/sample_split_and_generation.csv

For decoder block l:

    r_l = (h_sub^img - h_ref^img)
          - (h_sub^noimg - h_ref^noimg)

and the train-set Direction codebook gives:
    mu_left, mu_right, mu_above, mu_below.

Inside block l we capture the actual residual updates produced by Attention
and MLP:

    a_l = [(Attn_img_sub - Attn_img_ref)
           - (Attn_noimg_sub - Attn_noimg_ref)]

    m_l = [(MLP_img_sub - MLP_img_ref)
           - (MLP_noimg_sub - MLP_noimg_ref)]

For a competitor c:

    C_attn = a_l dot (mu_GT - mu_c)
    C_mlp  = m_l dot (mu_GT - mu_c)

C > 0: pushes toward GT relative to competitor.
C < 0: pushes toward competitor relative to GT.

Two competitors are saved:
1) strongest non-GT Direction competitor at this layer (all samples);
2) actual final generated wrong relation (generation-wrong samples only).

For l > 0 we also sanity-check:
    observed block residual update ~= attention update + MLP update.

Recommended run:

CUDA_VISIBLE_DEVICES=0 python analyze_attn_mlp_direction_failure_scan_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers all \
  --split test \
  --output-dir output/qwen7b_attn_mlp_direction_failure_v1 \
  --overwrite
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import analyze_layerwise_direction_failure_scan_v1 as direction


RELATIONS = ("left", "right", "above", "below")
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
EPS = 1e-10


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument("--model", default="qwen-7b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )
    p.add_argument("--layers", default="all")
    p.add_argument("--split", default="test", choices=["train", "test", "all"])
    p.add_argument(
        "--generation-groups",
        default="correct,wrong",
        help="Use cached actual model.generate() groups.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=25)
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


def mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def frac(xs: Iterable[bool]) -> float:
    vals = [bool(x) for x in xs]
    return float(np.mean(vals)) if vals else float("nan")


def norm_relation(x: Any) -> str:
    return direction.norm_relation(x)


def parse_layers(text: str, n_layers: int) -> List[int]:
    if str(text).strip().lower() == "all":
        return list(range(n_layers))
    out = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        li = int(x)
        if not 0 <= li < n_layers:
            raise ValueError(f"Layer {li} outside [0,{n_layers-1}]")
        out.append(li)
    out = list(dict.fromkeys(out))
    if not out:
        raise ValueError("No layers selected.")
    return out


def get_attr_path(obj: Any, path: str):
    cur = obj
    for piece in path.split("."):
        cur = getattr(cur, piece)
    return cur


def resolve_decoder_layers(model):
    candidates = [
        "model.language_model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
        "language_model.model.layers",
    ]
    errors = []
    for path in candidates:
        try:
            layers = get_attr_path(model, path)
            if hasattr(layers, "__len__") and len(layers) > 0:
                block = layers[0]
                if hasattr(block, "self_attn") and hasattr(block, "mlp"):
                    return layers, path
        except Exception as e:
            errors.append(f"{path}: {type(e).__name__}")
    raise RuntimeError(
        "Could not resolve decoder layers. Tried: " + "; ".join(errors)
    )


def first_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for item in x:
            if torch.is_tensor(item):
                return item
    raise RuntimeError(f"No tensor found in module output type={type(x)}")


def pool_positions(tensor: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    valid = [int(p) for p in positions if 0 <= int(p) < int(tensor.shape[1])]
    if not valid:
        raise RuntimeError("No valid object-token positions.")
    idx = torch.as_tensor(valid, device=tensor.device, dtype=torch.long)
    return tensor[0].index_select(0, idx).mean(dim=0)


class ModuleDiffCollector:
    def __init__(
        self,
        decoder_layers,
        selected_layers: Sequence[int],
        subject_positions: Sequence[int],
        reference_positions: Sequence[int],
    ):
        self.layers = decoder_layers
        self.selected = list(selected_layers)
        self.subj = list(map(int, subject_positions))
        self.ref = list(map(int, reference_positions))
        self.attn: Dict[int, torch.Tensor] = {}
        self.mlp: Dict[int, torch.Tensor] = {}
        self.handles = []

    def _make_hook(self, li: int, kind: str):
        def hook(_module, _args, output):
            x = first_tensor(output)
            hs = pool_positions(x, self.subj)
            hr = pool_positions(x, self.ref)
            d = (hs - hr).detach().float().cpu()
            if kind == "attn":
                self.attn[li] = d
            else:
                self.mlp[li] = d
            return output
        return hook

    def __enter__(self):
        for li in self.selected:
            block = self.layers[li]
            self.handles.append(
                block.self_attn.register_forward_hook(
                    self._make_hook(li, "attn")
                )
            )
            self.handles.append(
                block.mlp.register_forward_hook(
                    self._make_hook(li, "mlp")
                )
            )
        return self

    def validate(self):
        for li in self.selected:
            if li not in self.attn:
                raise RuntimeError(f"Missing Attention output at L{li}")
            if li not in self.mlp:
                raise RuntimeError(f"Missing MLP output at L{li}")

    def close(self):
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []

    def __exit__(self, *_args):
        self.close()


def fit_codebook(X_train: np.ndarray, y_train: np.ndarray):
    center = X_train.mean(axis=0)
    Xc = X_train - center
    protos = []
    for rel in RELATIONS:
        mask = y_train == rel
        if not np.any(mask):
            raise RuntimeError(f"No training examples for {rel}")
        p = Xc[mask].mean(axis=0)
        p = p / max(float(np.linalg.norm(p)), EPS)
        protos.append(p)
    return center.astype(np.float32), np.stack(protos).astype(np.float32)


def load_direction_assets(direction_dir: Path):
    vec_path = direction_dir / "vectors.npz"
    split_path = direction_dir / "sample_split_and_generation.csv"
    if not vec_path.exists():
        raise FileNotFoundError(vec_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    with np.load(vec_path, allow_pickle=True) as z:
        arrays = {k: z[k] for k in z.files}

    split_rows = read_csv(split_path)
    sids = arrays["sample_index"].astype(np.int64)
    labels = np.asarray([norm_relation(x) for x in arrays["relation"]])
    residual = np.asarray(arrays["residual"], dtype=np.float32)

    idx_by_sid = {int(s): i for i, s in enumerate(sids.tolist())}
    split_by_sid = {
        int(r["sample_index"]): str(r["split"]).strip()
        for r in split_rows
    }
    generation_by_sid = {
        int(r["sample_index"]): {
            "generation_group": str(r.get("generation_group", "")).strip(),
            "generation_pred": norm_relation(r.get("generation_pred", "")),
            "generation_text": str(r.get("generation_text", "")),
        }
        for r in split_rows
    }

    train_idx = np.asarray(
        [
            idx_by_sid[int(r["sample_index"])]
            for r in split_rows
            if str(r["split"]).strip() == "train"
            and int(r["sample_index"]) in idx_by_sid
        ],
        dtype=np.int64,
    )
    if len(train_idx) == 0:
        raise RuntimeError("No training samples in cached split.")

    n_layers = int(residual.shape[1])
    codebooks = {}
    for li in range(n_layers):
        center, protos = fit_codebook(
            residual[train_idx, li, :],
            labels[train_idx],
        )
        codebooks[li] = {"center": center, "protos": protos}

    return {
        "labels": labels,
        "residual": residual,
        "idx_by_sid": idx_by_sid,
        "split_by_sid": split_by_sid,
        "generation_by_sid": generation_by_sid,
        "codebooks": codebooks,
        "n_layers": n_layers,
    }


def capture_module_diffs(
    *,
    model,
    processor,
    device,
    decoder_layers,
    selected_layers,
    question: str,
    subject: str,
    reference: str,
    image: Optional[Image.Image],
):
    rendered = direction.build_chat_prompt(
        processor, question, image is not None
    )
    batch = direction.process_inputs(
        processor, rendered, image, device
    )
    ids = [
        int(x)
        for x in batch["input_ids"][0].detach().cpu().tolist()
    ]
    subj_pos = direction.locate_phrase_positions(
        processor.tokenizer, ids, subject
    )
    ref_pos = direction.locate_phrase_positions(
        processor.tokenizer, ids, reference
    )

    with ModuleDiffCollector(
        decoder_layers,
        selected_layers,
        subj_pos,
        ref_pos,
    ) as col:
        with torch.inference_mode():
            _ = model(
                **batch,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
        col.validate()

    attn = {
        li: col.attn[li].numpy().astype(np.float32)
        for li in selected_layers
    }
    mlp = {
        li: col.mlp[li].numpy().astype(np.float32)
        for li in selected_layers
    }
    del batch
    return attn, mlp


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def module_metrics(
    vec: np.ndarray,
    *,
    protos: np.ndarray,
    gt: str,
    competitor: str,
):
    g = protos[REL2ID[gt]]
    c = protos[REL2ID[competitor]]
    gt_support = float(vec @ g)
    comp_support = float(vec @ c)
    return {
        "gt_support": gt_support,
        "competitor_support": comp_support,
        "margin_contribution": gt_support - comp_support,
        "wrong_drive": comp_support - gt_support,
    }


def bootstrap_gap(
    correct_vals,
    wrong_vals,
    n_boot: int,
    rng: np.random.Generator,
):
    a = np.asarray(correct_vals, dtype=np.float64)
    b = np.asarray(wrong_vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")

    obs = float(b.mean() - a.mean())
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        boots[i] = bb.mean() - aa.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def summarize_direction(per_direction_rows, out_dir):
    buckets = defaultdict(list)
    for r in per_direction_rows:
        buckets[(int(r["layer"]), str(r["generation_group"]))].append(r)

    rows = []
    for (li, gg), rs in sorted(buckets.items()):
        rows.append({
            "layer": li,
            "generation_group": gg,
            "n": len(rs),
            "mean_gt_signal": mean(r["direction_gt_signal"] for r in rs),
            "mean_best_wrong_signal": mean(
                r["direction_best_wrong_signal"] for r in rs
            ),
            "mean_gt_vs_best_margin": mean(
                r["direction_gt_vs_best_margin"] for r in rs
            ),
            "mean_gt_vs_generation_pred_margin": mean(
                r["direction_gt_vs_generation_pred_margin"] for r in rs
            ),
        })

    write_csv(
        out_dir / "direction_trajectory_by_generation_group.csv",
        rows,
    )

    print("\n" + "=" * 120)
    print("DIRECTION TRAJECTORY BY ACTUAL GENERATION GROUP")
    print("=" * 120)
    print(
        "layer group    N    GTsignal  bestWrong  GT-bestMargin  "
        "GT-finalWrongMargin"
    )
    for r in rows:
        print(
            f"L{int(r['layer']):02d}  "
            f"{str(r['generation_group']):<7s} "
            f"{int(r['n']):3d}  "
            f"{float(r['mean_gt_signal']):+.3f}   "
            f"{float(r['mean_best_wrong_signal']):+.3f}    "
            f"{float(r['mean_gt_vs_best_margin']):+.3f}         "
            f"{float(r['mean_gt_vs_generation_pred_margin']):+.3f}"
        )


def summarize_modules(per_module_rows, out_dir, bootstrap, seed):
    buckets = defaultdict(list)
    for r in per_module_rows:
        buckets[(int(r["layer"]), str(r["module"]))].append(r)

    rng = np.random.default_rng(seed)
    rows = []

    for (li, module), rs in sorted(buckets.items()):
        correct = [r for r in rs if r["generation_group"] == "correct"]
        wrong = [r for r in rs if r["generation_group"] == "wrong"]

        cvals = [float(r["wrong_drive_best"]) for r in correct]
        wvals = [float(r["wrong_drive_best"]) for r in wrong]
        gap, lo, hi = bootstrap_gap(cvals, wvals, bootstrap, rng)

        wrong_genpred = [
            r for r in wrong
            if math.isfinite(float(r["wrong_drive_generation_pred"]))
        ]

        rows.append({
            "layer": li,
            "module": module,
            "n_correct": len(correct),
            "n_wrong": len(wrong),

            "mean_wrong_drive_best_correct": mean(cvals),
            "mean_wrong_drive_best_wrong": mean(wvals),
            "wrong_minus_correct_wrong_drive_gap": gap,
            "bootstrap95_lo": lo,
            "bootstrap95_hi": hi,

            "fraction_wrong_driving_best_correct": frac(
                float(r["wrong_drive_best"]) > 0 for r in correct
            ),
            "fraction_wrong_driving_best_wrong": frac(
                float(r["wrong_drive_best"]) > 0 for r in wrong
            ),

            "mean_wrong_drive_generation_pred_on_wrong": mean(
                r["wrong_drive_generation_pred"] for r in wrong_genpred
            ),
            "fraction_drives_generation_pred_on_wrong": frac(
                float(r["wrong_drive_generation_pred"]) > 0
                for r in wrong_genpred
            ),

            "mean_margin_contrib_generation_pred_on_wrong": mean(
                r["margin_contribution_generation_pred"]
                for r in wrong_genpred
            ),

            "mean_reconstruction_cosine": mean(
                r["reconstruction_cosine"] for r in rs
            ),
            "mean_reconstruction_relative_error": mean(
                r["reconstruction_relative_error"] for r in rs
            ),
        })

    write_csv(out_dir / "module_layer_summary.csv", rows)

    ranked = sorted(
        rows,
        key=lambda r: (
            float(r["wrong_minus_correct_wrong_drive_gap"])
            if math.isfinite(float(r["wrong_minus_correct_wrong_drive_gap"]))
            else -1e9
        ),
        reverse=True,
    )
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
        r["ci_excludes_zero_positive"] = int(
            math.isfinite(float(r["bootstrap95_lo"]))
            and float(r["bootstrap95_lo"]) > 0
        )
    write_csv(out_dir / "ranked_candidate_modules.csv", ranked)

    print("\n" + "=" * 145)
    print("TOP ATTENTION / MLP FAILURE CANDIDATES")
    print("=" * 145)
    print(
        "rank layer module  wrong-correct gap      95%CI             "
        "drive(finalGeneratedWrong)  drivesWrong%  reconCos"
    )
    for r in ranked[:20]:
        print(
            f"{int(r['rank']):>3d}  "
            f"L{int(r['layer']):02d}  "
            f"{str(r['module']):<5s}   "
            f"{float(r['wrong_minus_correct_wrong_drive_gap']):+.4f}   "
            f"[{float(r['bootstrap95_lo']):+.4f},"
            f"{float(r['bootstrap95_hi']):+.4f}]      "
            f"{float(r['mean_wrong_drive_generation_pred_on_wrong']):+.4f}                 "
            f"{float(r['fraction_drives_generation_pred_on_wrong']):.3f}      "
            f"{float(r['mean_reconstruction_cosine']):.4f}"
        )

    return ranked


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    direction_dir = Path(args.direction_dir)
    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = load_direction_assets(direction_dir)
    wanted_groups = {
        x.strip()
        for x in args.generation_groups.split(",")
        if x.strip()
    }

    eligible_sids = []
    for sid, split in assets["split_by_sid"].items():
        if args.split != "all" and split != args.split:
            continue
        gg = assets["generation_by_sid"].get(sid, {}).get(
            "generation_group", ""
        )
        if gg in wanted_groups:
            eligible_sids.append(sid)

    if args.max_samples is not None and len(eligible_sids) > args.max_samples:
        rng = random.Random(args.seed)
        rng.shuffle(eligible_sids)
        eligible_sids = eligible_sids[: int(args.max_samples)]

    keep_sids = set(eligible_sids)
    order = {sid: i for i, sid in enumerate(eligible_sids)}

    records, _audit = base.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    records = [
        r for r in records
        if int(r.sid) in keep_sids
        and norm_relation(r.relation) in REL2ID
    ]
    records.sort(key=lambda r: order[int(r.sid)])

    if not records:
        raise RuntimeError("No records selected.")

    print(
        f"[data] split={args.split}, groups={sorted(wanted_groups)}, "
        f"N={len(records)}"
    )
    print(
        "[cached actual generation groups]",
        dict(
            Counter(
                assets["generation_by_sid"][int(r.sid)]["generation_group"]
                for r in records
            )
        ),
    )

    spec = base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)
    kw: Dict[str, Any] = {
        "torch_dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    print(f"[model] loading {spec.repo_id} on {args.device}")
    model = cls.from_pretrained(spec.repo_id, **kw)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, layer_path = resolve_decoder_layers(model)
    if len(decoder_layers) != assets["n_layers"]:
        raise RuntimeError(
            f"Layer mismatch model={len(decoder_layers)} "
            f"cache={assets['n_layers']}"
        )
    selected_layers = parse_layers(args.layers, len(decoder_layers))
    print(
        f"[decoder] {layer_path}; n_layers={len(decoder_layers)}; "
        f"selected={selected_layers}"
    )

    per_module_rows = []
    per_direction_rows = []
    errors = []

    module_path = out_dir / "per_sample_module_contributions.csv"
    direction_path = out_dir / "per_sample_direction_trajectory.csv"

    done_samples = 0

    for rec in tqdm(records, desc="attn/mlp direction scan"):
        image = None
        try:
            sid = int(rec.sid)
            idx = assets["idx_by_sid"][sid]
            gt = norm_relation(rec.relation)
            gen = assets["generation_by_sid"][sid]
            gg = gen["generation_group"]
            gen_pred = gen["generation_pred"]

            question = args.prompt_template.format(
                subject=rec.subject,
                reference=rec.reference,
            )
            image = Image.open(rec.image_path).convert("RGB")

            real_attn, real_mlp = capture_module_diffs(
                model=model,
                processor=processor,
                device=device,
                decoder_layers=decoder_layers,
                selected_layers=selected_layers,
                question=question,
                subject=str(rec.subject),
                reference=str(rec.reference),
                image=image,
            )

            noimg_attn, noimg_mlp = capture_module_diffs(
                model=model,
                processor=processor,
                device=device,
                decoder_layers=decoder_layers,
                selected_layers=selected_layers,
                question=question,
                subject=str(rec.subject),
                reference=str(rec.reference),
                image=None,
            )

            residual_sample = assets["residual"][idx]

            for li in selected_layers:
                cb = assets["codebooks"][li]
                center = cb["center"]
                protos = cb["protos"]

                q = residual_sample[li] - center
                scores = q @ protos.T
                gt_i = REL2ID[gt]

                tmp = scores.copy()
                tmp[gt_i] = -np.inf
                comp_i = int(np.argmax(tmp))
                best_comp = RELATIONS[comp_i]

                gt_signal = float(scores[gt_i])
                best_wrong_signal = float(scores[comp_i])

                if gg == "wrong" and gen_pred in REL2ID and gen_pred != gt:
                    gen_wrong_signal = float(scores[REL2ID[gen_pred]])
                    gt_vs_gen = gt_signal - gen_wrong_signal
                else:
                    gen_wrong_signal = float("nan")
                    gt_vs_gen = float("nan")

                per_direction_rows.append({
                    "sid": sid,
                    "layer": li,
                    "gt": gt,
                    "generation_group": gg,
                    "generation_pred": gen_pred,
                    "best_direction_competitor": best_comp,
                    "direction_gt_signal": gt_signal,
                    "direction_best_wrong_signal": best_wrong_signal,
                    "direction_gt_vs_best_margin":
                        gt_signal - best_wrong_signal,
                    "direction_generation_pred_signal":
                        gen_wrong_signal,
                    "direction_gt_vs_generation_pred_margin":
                        gt_vs_gen,
                })

                attn_res = real_attn[li] - noimg_attn[li]
                mlp_res = real_mlp[li] - noimg_mlp[li]

                if li > 0:
                    observed = (
                        residual_sample[li] - residual_sample[li - 1]
                    )
                    recon = attn_res + mlp_res
                    recon_cos = cosine(observed, recon)
                    recon_relerr = float(
                        np.linalg.norm(observed - recon)
                        / max(float(np.linalg.norm(observed)), EPS)
                    )
                else:
                    recon_cos = float("nan")
                    recon_relerr = float("nan")

                for module, vec in (("attn", attn_res), ("mlp", mlp_res)):
                    best = module_metrics(
                        vec,
                        protos=protos,
                        gt=gt,
                        competitor=best_comp,
                    )

                    if gg == "wrong" and gen_pred in REL2ID and gen_pred != gt:
                        actual_wrong = module_metrics(
                            vec,
                            protos=protos,
                            gt=gt,
                            competitor=gen_pred,
                        )
                    else:
                        actual_wrong = {
                            "gt_support": float("nan"),
                            "competitor_support": float("nan"),
                            "margin_contribution": float("nan"),
                            "wrong_drive": float("nan"),
                        }

                    per_module_rows.append({
                        "sid": sid,
                        "layer": li,
                        "module": module,
                        "gt": gt,
                        "generation_group": gg,
                        "generation_pred": gen_pred,
                        "best_direction_competitor": best_comp,

                        "gt_support_best": best["gt_support"],
                        "competitor_support_best":
                            best["competitor_support"],
                        "margin_contribution_best":
                            best["margin_contribution"],
                        "wrong_drive_best": best["wrong_drive"],

                        "gt_support_generation_pred":
                            actual_wrong["gt_support"],
                        "generation_pred_support":
                            actual_wrong["competitor_support"],
                        "margin_contribution_generation_pred":
                            actual_wrong["margin_contribution"],
                        "wrong_drive_generation_pred":
                            actual_wrong["wrong_drive"],

                        "module_residual_norm":
                            float(np.linalg.norm(vec)),
                        "reconstruction_cosine": recon_cos,
                        "reconstruction_relative_error": recon_relerr,
                    })

            done_samples += 1
            if done_samples % args.save_every == 0:
                write_csv(module_path, per_module_rows)
                write_csv(direction_path, per_direction_rows)

        except Exception as e:
            errors.append({
                "sid": int(getattr(rec, "sid", -1)),
                "error_type": type(e).__name__,
                "error": str(e),
            })
            tqdm.write(
                f"[ERROR sid={getattr(rec,'sid','?')}] "
                f"{type(e).__name__}: {e}"
            )
        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(module_path, per_module_rows)
    write_csv(direction_path, per_direction_rows)
    (out_dir / "errors.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summarize_direction(per_direction_rows, out_dir)
    summarize_modules(
        per_module_rows,
        out_dir,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    relation_buckets = defaultdict(list)
    for r in per_module_rows:
        relation_buckets[
            (
                int(r["layer"]),
                str(r["module"]),
                str(r["generation_group"]),
                str(r["gt"]),
            )
        ].append(r)

    relation_rows = []
    for (li, module, gg, gt), rs in sorted(relation_buckets.items()):
        relation_rows.append({
            "layer": li,
            "module": module,
            "generation_group": gg,
            "gt": gt,
            "n": len(rs),
            "mean_wrong_drive_best": mean(
                r["wrong_drive_best"] for r in rs
            ),
            "fraction_wrong_driving_best": frac(
                float(r["wrong_drive_best"]) > 0 for r in rs
            ),
            "mean_wrong_drive_generation_pred": mean(
                r["wrong_drive_generation_pred"] for r in rs
            ),
        })
    write_csv(
        out_dir / "module_summary_by_relation.csv",
        relation_rows,
    )

    meta = {
        "experiment":
            "generation-conditioned Attention-vs-MLP Direction contribution scan",
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "n_records": len(records),
        "n_success": done_samples,
        "n_errors": len(errors),
        "selected_layers": selected_layers,
        "correctness_definition":
            "cached actual model.generate() result from "
            "sample_split_and_generation.csv",
        "wrong_drive_definition":
            "module residual update dot (mu_competitor - mu_GT)",
        "note":
            "positive wrong_drive is diagnostic evidence that the module update "
            "pushes toward competitor relative to GT; causal proof requires "
            "module-level intervention.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved to:", out_dir)
    print("  per_sample_module_contributions.csv")
    print("  per_sample_direction_trajectory.csv")
    print("  direction_trajectory_by_generation_group.csv")
    print("  module_layer_summary.csv")
    print("  ranked_candidate_modules.csv")
    print("  module_summary_by_relation.csv")
    print("  errors.json")
    print(
        "\nNext step: causally patch/remove the top candidate Attention/MLP "
        "modules. Any claimed improvement should then be evaluated with "
        "actual model.generate(), not only restricted first-step logits."
    )

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
