#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intro diagnostic using ONLY layer hidden state -> final norm -> LM head.

For every COCO example and decoder layer, read the REAL prompt-last hidden state h_l,
apply the model's own final norm and LM head, and score A/B/C/D.

For a target relation t (default left) and opposite o (default right), with the
sample-specific random mapping pi_i:

  probability-gap view:
      P_i,l = softmax_ABCD(z_i,l)[pi_i(t)] - softmax_ABCD(z_i,l)[pi_i(o)]

  target-logit view:
      Z_i,l = z_i,l[pi_i(t)]

The script searches for ONE final-correct target sample and ONE final-wrong->opposite
sample whose MID-layer trajectories are similar, but whose LATE decision trajectories
diverge.  It outputs two figures for the SAME pair:

  intro_pair_prob_gap.png      : mapped-target vs mapped-opposite probability gap
  intro_pair_target_logit.png  : raw mapped-target logit

No learned probe / centroid reader is used in either figure.

Run from the AdaptVis repo root.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import random
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# Existing repo helpers (already used by the current AdaptVis scripts).
import analyze_coco_centroid_generation_step1_v4 as base
import eval_coco_multilayer_relation_trajectory_repair_v1 as traj


REL = ("left", "right", "above", "below")
LETTERS = ("A", "B", "C", "D")
DISPLAY_REL = {"left": "left", "right": "right", "above": "on", "below": "under"}
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}
SCRIPT_VERSION = "intro-hiddenstate-decision-pair-v2"


def canonical_relation(x: Any) -> str:
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    table = {
        "left": "left", "left_of": "left",
        "right": "right", "right_of": "right",
        "above": "above", "on": "above", "over": "above", "top": "above",
        "below": "below", "under": "below", "beneath": "below", "bottom": "below",
    }
    if s not in table:
        raise ValueError(f"Unknown relation: {x!r}")
    return table[s]


def disp_rel(r: str) -> str:
    return DISPLAY_REL.get(str(r), str(r))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--model", default="qwen-3b", choices=["qwen-3b"])
    p.add_argument("--layers", nargs="+", type=int, default=list(range(18, 33)))
    p.add_argument("--target-relation", default="left")
    p.add_argument("--opposite-relation", default="right")
    p.add_argument("--mid-layers", nargs="+", type=int, default=[18,19,20,21,22,23,24,25,26])
    p.add_argument("--late-layers", nargs="+", type=int, default=[27,28,29,30,31,32])
    p.add_argument("--min-mid-target-prob-frac", type=float, default=0.60,
                   help="Fraction of mid layers where mapped target prob must exceed mapped opposite prob for BOTH samples.")
    p.add_argument("--max-mid-probgap-rmse", type=float, default=-1.0,
                   help="Hard cap; <0 disables. Try 0.08 or 0.05 for stricter matching.")
    p.add_argument("--max-mid-targetlogit-rmse", type=float, default=-1.0,
                   help="Hard cap; <0 disables. Keeps raw target-logit trajectories close in the middle.")
    p.add_argument("--require-late-sign-split", action="store_true", default=True,
                   help="Require late mean prob-gap >0 for correct and <0 for wrong->opposite.")
    p.add_argument("--no-require-late-sign-split", dest="require_late_sign_split", action="store_false")
    p.add_argument("--min-late-probgap-separation", type=float, default=0.40)
    p.add_argument("--same-option-pair", action="store_true", default=True,
                   help="Require target/opposite to map to the same two letters in both samples.")
    p.add_argument("--allow-different-option-pair", dest="same_option_pair", action="store_false")
    p.add_argument("--weight-mid-probgap", type=float, default=3.0)
    p.add_argument("--weight-mid-targetlogit", type=float, default=0.15)
    p.add_argument("--weight-late-separation", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--coco-max-samples", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def stratified_cap(items: Sequence[Mapping[str, Any]], max_samples: int, seed: int) -> List[Dict[str, Any]]:
    xs = [dict(x) for x in items]
    if max_samples <= 0 or len(xs) <= max_samples:
        return xs
    rng = random.Random(seed)
    by: Dict[str, List[Dict[str, Any]]] = {}
    for x in xs:
        by.setdefault(str(x["gt"]), []).append(x)
    for v in by.values():
        rng.shuffle(v)
    out: List[Dict[str, Any]] = []
    keys = sorted(by.keys())
    while len(out) < max_samples:
        moved = False
        for k in keys:
            if by[k] and len(out) < max_samples:
                out.append(by[k].pop())
                moved = True
        if not moved:
            break
    return out


def assign_relation_balanced_mappings(items: Sequence[Mapping[str, Any]], seed: int) -> Dict[int, Dict[str, str]]:
    """Deterministic per-sample random relation->A/B/C/D mapping, balanced by GT relation."""
    rng = random.Random(seed)
    perms = list(__import__("itertools").permutations(LETTERS))
    out: Dict[int, Dict[str, str]] = {}
    by_rel: Dict[str, List[Mapping[str, Any]]] = {r: [] for r in REL}
    for m in items:
        by_rel[str(m["gt"])].append(m)
    for rel, group in by_rel.items():
        rng.shuffle(group)
        pp = perms.copy()
        rng.shuffle(pp)
        for j, m in enumerate(group):
            perm = pp[j % len(pp)]
            out[int(m["sid"])] = {r: perm[k] for k, r in enumerate(REL)}
    return out


def build_randmap_prompt(subject: str, reference: str, rel_to_letter: Mapping[str, str]) -> str:
    letter_to_rel = {letter: rel for rel, letter in rel_to_letter.items()}
    option_text = ", ".join(f"{L}: {disp_rel(letter_to_rel[L])}" for L in LETTERS)
    return (
        f"Where is the {subject} relative to the {reference}?\n"
        f"Options: {option_text}.\n"
        f"Answer with only one letter: A, B, C, or D."
    )


def mapping_string(mp: Mapping[str, str]) -> str:
    return ";".join(f"{r}:{mp[r]}" for r in REL)


def build_option_token_map(tokenizer: Any) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for a in LETTERS:
        ids: List[int] = []
        for s in (a, " " + a):
            tok = tokenizer.encode(s, add_special_tokens=False)
            if len(tok) == 1:
                ids.append(int(tok[0]))
        ids = sorted(set(ids))
        if not ids:
            raise RuntimeError(f"Could not find single-token spelling for option {a}")
        out[a] = ids
    return out


def make_batch(processor: Any, device: torch.device, image: Image.Image, prompt: str) -> Dict[str, Any]:
    return base.make_question_batch(processor=processor, image=image, question_text=prompt, device=device)


class LastTokenCapture:
    def __init__(self, decoder_layers: Sequence[Any], layers: Sequence[int]):
        self.states: Dict[int, np.ndarray] = {}
        self.handles = [decoder_layers[int(L)].register_forward_hook(self._hook(int(L))) for L in layers]

    def _hook(self, L: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            h = traj.first_tensor(output)
            if h.ndim != 3:
                raise RuntimeError(f"Expected [B,T,D] at L{L}, got {tuple(h.shape)}")
            self.states[L] = h[0, -1, :].detach().float().cpu().numpy().astype(np.float32)
            return output
        return hook

    def close(self) -> None:
        for h in self.handles:
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


def capture_real_last_states(
    *, model: Any, decoder_layers: Sequence[Any], batch: Mapping[str, Any], layers: Sequence[int]
) -> Tuple[Dict[int, np.ndarray], torch.Tensor]:
    cap = LastTokenCapture(decoder_layers, layers)
    try:
        with torch.no_grad():
            outputs = model(**batch, use_cache=False, return_dict=True)
        missing = [L for L in layers if L not in cap.states]
        if missing:
            raise RuntimeError(f"Last-token capture missed layers: {missing}")
        final_logits = outputs.logits[0, -1].detach().float().cpu()
        return dict(cap.states), final_logits
    finally:
        cap.close()


def _get_nested(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def resolve_norm_and_head(model: Any) -> Tuple[Any, Any, str, str]:
    norm_paths = [
        "model.language_model.norm", "model.norm", "language_model.model.norm", "language_model.norm"
    ]
    head_paths = ["lm_head", "language_model.lm_head", "model.lm_head"]
    norm = head = None
    norm_path = head_path = ""
    for p in norm_paths:
        try:
            norm = _get_nested(model, p); norm_path = p; break
        except Exception:
            pass
    for p in head_paths:
        try:
            head = _get_nested(model, p); head_path = p; break
        except Exception:
            pass
    if norm is None or head is None:
        raise RuntimeError(f"Could not resolve final norm/lm_head (norm={norm_path}, head={head_path})")
    return norm, head, norm_path, head_path


def module_dtype_device(module: Any, fallback_device: torch.device) -> Tuple[torch.dtype, torch.device]:
    for p in module.parameters(recurse=True):
        return p.dtype, p.device
    return torch.float32, fallback_device


def logit_lens_option_logits(
    h_np: np.ndarray,
    final_norm: Any,
    lm_head: Any,
    option_token_map: Mapping[str, Sequence[int]],
    fallback_device: torch.device,
) -> Dict[str, float]:
    dtype, dev = module_dtype_device(final_norm, fallback_device)
    x = torch.as_tensor(h_np, dtype=dtype, device=dev).view(1, 1, -1)
    with torch.no_grad():
        y = final_norm(x)
        logits = lm_head(y)[0, 0].float()
    out: Dict[str, float] = {}
    for a in LETTERS:
        idx = torch.tensor(list(option_token_map[a]), device=logits.device, dtype=torch.long)
        out[a] = float(logits.index_select(0, idx).max().detach().cpu())
    return out


def softmax_abcd(logits: Mapping[str, float]) -> Dict[str, float]:
    x = np.array([float(logits[a]) for a in LETTERS], dtype=np.float64)
    x -= x.max()
    p = np.exp(x)
    p /= p.sum()
    return {a: float(p[i]) for i, a in enumerate(LETTERS)}


def rmse(xs: Sequence[float], ys: Sequence[float]) -> float:
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rows_by_sid(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[int, Dict[str, Any]]]:
    out: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(int(r["sid"]), {})[int(r["layer"])] = dict(r)
    return out


def traj_values(by: Mapping[int, Mapping[int, Mapping[str, Any]]], sid: int, layers: Sequence[int], key: str) -> List[float]:
    return [float(by[int(sid)][int(L)][key]) for L in layers]


def find_pairs(
    rows: Sequence[Mapping[str, Any]],
    target: str,
    opposite: str,
    mid_layers: Sequence[int],
    late_layers: Sequence[int],
    min_mid_target_prob_frac: float,
    max_mid_probgap_rmse: float,
    max_mid_targetlogit_rmse: float,
    require_late_sign_split: bool,
    min_late_probgap_separation: float,
    same_option_pair: bool,
    weight_mid_probgap: float,
    weight_mid_targetlogit: float,
    weight_late_separation: float,
) -> List[Dict[str, Any]]:
    by = rows_by_sid(rows)
    sample0 = {sid: vals[min(vals.keys())] for sid, vals in by.items()}

    correct = []
    wrong = []
    for sid, r in sample0.items():
        if str(r["gt"]) != target:
            continue
        pred_rel = str(r["final_prediction_relation"])
        if pred_rel == target:
            correct.append(sid)
        elif pred_rel == opposite:
            wrong.append(sid)

    pairs: List[Dict[str, Any]] = []
    for sc in correct:
        rc = sample0[sc]
        for sw in wrong:
            rw = sample0[sw]
            if same_option_pair:
                if str(rc["target_option"]) != str(rw["target_option"]) or str(rc["opposite_option"]) != str(rw["opposite_option"]):
                    continue

            c_mid_gap = traj_values(by, sc, mid_layers, "prob_gap")
            w_mid_gap = traj_values(by, sw, mid_layers, "prob_gap")
            c_mid_log = traj_values(by, sc, mid_layers, "target_logit")
            w_mid_log = traj_values(by, sw, mid_layers, "target_logit")

            pos_frac_c = float(np.mean(np.asarray(c_mid_gap) > 0))
            pos_frac_w = float(np.mean(np.asarray(w_mid_gap) > 0))
            if pos_frac_c < min_mid_target_prob_frac or pos_frac_w < min_mid_target_prob_frac:
                continue

            gap_rmse = rmse(c_mid_gap, w_mid_gap)
            log_rmse = rmse(c_mid_log, w_mid_log)
            if max_mid_probgap_rmse >= 0 and gap_rmse > max_mid_probgap_rmse:
                continue
            if max_mid_targetlogit_rmse >= 0 and log_rmse > max_mid_targetlogit_rmse:
                continue

            c_late = np.asarray(traj_values(by, sc, late_layers, "prob_gap"), dtype=np.float64)
            w_late = np.asarray(traj_values(by, sw, late_layers, "prob_gap"), dtype=np.float64)
            c_late_mean = float(c_late.mean())
            w_late_mean = float(w_late.mean())
            late_sep = float((c_late - w_late).mean())
            if require_late_sign_split and not (c_late_mean > 0 and w_late_mean < 0):
                continue
            if late_sep < min_late_probgap_separation:
                continue

            # Lower is better.  Matching dominates; late separation only breaks ties.
            score = (
                weight_mid_probgap * gap_rmse
                + weight_mid_targetlogit * log_rmse
                - weight_late_separation * late_sep
            )
            pairs.append({
                "sid_correct": int(sc),
                "sid_wrong": int(sw),
                "target_option": str(rc["target_option"]),
                "opposite_option": str(rc["opposite_option"]),
                "mid_probgap_rmse": gap_rmse,
                "mid_targetlogit_rmse": log_rmse,
                "correct_mid_positive_frac": pos_frac_c,
                "wrong_mid_positive_frac": pos_frac_w,
                "correct_late_mean": c_late_mean,
                "wrong_late_mean": w_late_mean,
                "late_probgap_separation": late_sep,
                "score": float(score),
            })
    pairs.sort(key=lambda x: x["score"])
    return pairs


def plot_prob_gap(path: Path, best: Mapping[str, Any], by: Mapping[int, Mapping[int, Mapping[str, Any]]], layers: Sequence[int], target: str, opposite: str) -> None:
    sc, sw = int(best["sid_correct"]), int(best["sid_wrong"])
    yc = traj_values(by, sc, layers, "prob_gap")
    yw = traj_values(by, sw, layers, "prob_gap")
    fig, ax = plt.subplots(figsize=(12.0, 6.1))
    ax.plot(layers, yc, marker="o", linewidth=3.0, label=f"Final correct (sid={sc})")
    ax.plot(layers, yw, marker="o", linewidth=3.0, linestyle="--", label=f"Final wrong→{disp_rel(opposite)} (sid={sw})")
    ax.axhline(0, color="0.4", linestyle="--", linewidth=1.8)
    ax.set_title("Layerwise mapped-option preference", fontsize=26)
    ax.set_xlabel("Decoder layer", fontsize=22)
    ax.set_ylabel(f"P({disp_rel(target)}) − P({disp_rel(opposite)})", fontsize=22)
    ax.tick_params(labelsize=17)
    ax.legend(fontsize=17, loc="best")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_target_logit(path: Path, best: Mapping[str, Any], by: Mapping[int, Mapping[int, Mapping[str, Any]]], layers: Sequence[int], target: str, opposite: str) -> None:
    sc, sw = int(best["sid_correct"]), int(best["sid_wrong"])
    yc = traj_values(by, sc, layers, "target_logit")
    yw = traj_values(by, sw, layers, "target_logit")
    fig, ax = plt.subplots(figsize=(12.0, 6.1))
    ax.plot(layers, yc, marker="o", linewidth=3.0, label=f"Final correct (sid={sc})")
    ax.plot(layers, yw, marker="o", linewidth=3.0, linestyle="--", label=f"Final wrong→{disp_rel(opposite)} (sid={sw})")
    ax.set_title(f"Layerwise mapped-{disp_rel(target)} logit", fontsize=26)
    ax.set_xlabel("Decoder layer", fontsize=22)
    ax.set_ylabel(f"Logit of option mapped to {disp_rel(target)}", fontsize=22)
    ax.tick_params(labelsize=17)
    ax.legend(fontsize=17, loc="best")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    a = parse_args()
    target = canonical_relation(a.target_relation)
    opposite = canonical_relation(a.opposite_relation)
    if target == opposite:
        raise ValueError("target and opposite must differ")
    if OPPOSITE.get(target) != opposite:
        print(f"[WARN] {target}/{opposite} are not canonical opposites; continuing.")

    layers = sorted(set(map(int, a.layers)))
    for name, vals in [("mid", a.mid_layers), ("late", a.late_layers)]:
        miss = sorted(set(map(int, vals)) - set(layers))
        if miss:
            raise ValueError(f"{name} layers missing from --layers: {miss}")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.output_dir)
    if a.overwrite and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)
    err_path = out / "errors.jsonl"

    two = base.import_two_object_module()
    prompts = base.load_standard_prompts(Path(a.prompt_jsonl))
    records, _audit = two.load_records("coco_two", Path(a.data_root), None)
    rec_by_sid = {int(r.sid): r for r in records}

    coco: List[Dict[str, Any]] = []
    for r in records:
        sid = int(r.sid)
        if sid not in prompts:
            continue
        p = prompts[sid]
        gt = traj.normalize_relation(base, p["answer_raw"])
        if gt not in REL:
            continue
        coco.append({"sid": sid, "gt": gt, "subject": str(p["subject"]), "reference": str(p["reference"])})
    coco = stratified_cap(coco, a.coco_max_samples, a.seed + 211)
    coco_map = assign_relation_balanced_mappings(coco, a.seed + 2003)

    specs = base.merged_model_specs(two)
    spec = specs[a.model]
    cls = getattr(transformers, spec.model_class)
    load_kw = dict(
        dtype=base.resolve_dtype(spec.dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
        device_map={"": a.device},
    )
    if a.attn_impl != "none":
        load_kw["attn_implementation"] = a.attn_impl

    model = processor = None
    try:
        print(f"[LOAD] {spec.repo_id}", flush=True)
        model = cls.from_pretrained(spec.repo_id, **load_kw)
        model.eval()
        processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        base.configure_processor(model, processor)
        device = torch.device(a.device)
        decoder_layers, decoder_path = base.resolve_decoder_layers(model)
        if min(layers) < 0 or max(layers) >= len(decoder_layers):
            raise ValueError(f"Requested {layers}; decoder has {len(decoder_layers)} layers")
        option_token_map = build_option_token_map(processor.tokenizer)
        final_norm, lm_head, norm_path, head_path = resolve_norm_and_head(model)
        print(f"[LOGIT LENS] norm={norm_path} | head={head_path}")

        per_rows: List[Dict[str, Any]] = []
        for m in tqdm(coco, desc="COCO hidden-state logit lens"):
            sid = int(m["sid"])
            if m["gt"] != target:
                continue
            mp = coco_map[sid]
            prompt = build_randmap_prompt(m["subject"], m["reference"], mp)
            image = batch = None
            try:
                image = base.record_image(rec_by_sid[sid])
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                batch = make_batch(processor, device, image, prompt)
                last_states, final_vocab_logits = capture_real_last_states(
                    model=model, decoder_layers=decoder_layers, batch=batch, layers=layers
                )
                inv = {letter: rel for rel, letter in mp.items()}
                # final prediction over A/B/C/D using the true final model logits
                final_letter_logits: Dict[str, float] = {}
                for A in LETTERS:
                    ids = torch.tensor(option_token_map[A], dtype=torch.long)
                    final_letter_logits[A] = float(final_vocab_logits.index_select(0, ids).max().item())
                final_pred_letter = max(LETTERS, key=lambda A: final_letter_logits[A])
                final_pred_relation = str(inv.get(final_pred_letter, "unknown"))
                target_option = str(mp[target])
                opposite_option = str(mp[opposite])

                for L in layers:
                    ll = logit_lens_option_logits(last_states[L], final_norm, lm_head, option_token_map, device)
                    pp = softmax_abcd(ll)
                    per_rows.append({
                        "sid": sid,
                        "layer": int(L),
                        "gt": str(m["gt"]),
                        "mapping": mapping_string(mp),
                        "target_relation": target,
                        "opposite_relation": opposite,
                        "target_option": target_option,
                        "opposite_option": opposite_option,
                        "final_prediction": final_pred_letter,
                        "final_prediction_relation": final_pred_relation,
                        "final_correct": int(final_pred_relation == m["gt"]),
                        "prob_gap": float(pp[target_option] - pp[opposite_option]),
                        "target_prob": float(pp[target_option]),
                        "opposite_prob": float(pp[opposite_option]),
                        "target_logit": float(ll[target_option]),
                        "opposite_logit": float(ll[opposite_option]),
                        "logit_gap": float(ll[target_option] - ll[opposite_option]),
                        **{f"logitlens_{A}": float(ll[A]) for A in LETTERS},
                    })
            except Exception as e:
                traj.append_jsonl(err_path, {
                    "phase": "coco", "sid": sid,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
                raise
            finally:
                if image is not None:
                    with contextlib.suppress(Exception):
                        image.close()
                del batch
                gc.collect()

        write_csv(out / "per_sample_layer.csv", per_rows)
        pairs = find_pairs(
            per_rows, target, opposite, a.mid_layers, a.late_layers,
            a.min_mid_target_prob_frac, a.max_mid_probgap_rmse, a.max_mid_targetlogit_rmse,
            a.require_late_sign_split, a.min_late_probgap_separation, a.same_option_pair,
            a.weight_mid_probgap, a.weight_mid_targetlogit, a.weight_late_separation,
        )
        if not pairs:
            raise RuntimeError(
                "No pair found. First try --allow-different-option-pair or relax "
                "--max-mid-probgap-rmse / --max-mid-targetlogit-rmse / --min-late-probgap-separation."
            )
        top = pairs[:max(1, int(a.topk))]
        write_csv(out / "top_pairs.csv", top)
        best = top[0]
        by = rows_by_sid(per_rows)

        plot_prob_gap(out / "intro_pair_prob_gap.png", best, by, layers, target, opposite)
        plot_target_logit(out / "intro_pair_target_logit.png", best, by, layers, target, opposite)

        best_rows: List[Dict[str, Any]] = []
        for role, sid in [("correct", int(best["sid_correct"])), ("wrong", int(best["sid_wrong"]))]:
            for L in layers:
                rr = dict(by[sid][L]); rr["role"] = role; best_rows.append(rr)
        write_csv(out / "best_pair_layer.csv", best_rows)

        meta = {
            "script_version": SCRIPT_VERSION,
            "model": a.model,
            "repo_id": spec.repo_id,
            "decoder_path": decoder_path,
            "norm_path": norm_path,
            "lm_head_path": head_path,
            "layers": layers,
            "mid_layers": list(map(int, a.mid_layers)),
            "late_layers": list(map(int, a.late_layers)),
            "target_relation": target,
            "opposite_relation": opposite,
            "metric": "REAL prompt-last hidden state -> final norm -> LM head; no learned reader",
            "best_pair": best,
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n[BEST PAIR]")
        for k, v in best.items():
            print(f"  {k}: {v}")
        print(f"[SAVED] {out / 'intro_pair_prob_gap.png'}")
        print(f"[SAVED] {out / 'intro_pair_target_logit.png'}")
        print(f"[SAVED] {out / 'top_pairs.csv'}")
        print(f"[SAVED] {out / 'best_pair_layer.csv'}")

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
