#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direction Evidence Strength Diagnostic + Causal Validation
==========================================================

Goal
----
Test whether generation-wrong samples fail because their spatial information is
actually weaker, rather than merely because the model "has the information but
does not use it."

This script deliberately separates three things at the REAL computation points
used in analyze_prepost_attention_direction_causality_v2.py:

    pre_attn  : decoder block input, before Attention_l
    post_attn : residual stream after Attention_l and before MLP_l

For every selected layer/stage, define the image-grounded subject-reference
residual:

    r = (h_sub - h_ref)^img - (h_sub - h_ref)^noimg

and center it with TRAIN statistics:

    q = r - center

A TRAIN-derived 2-D spatial subspace is:

    S = span(mu_right - mu_left, mu_above - mu_below)

with orthonormal basis B.

We then decompose q into:

    q_spatial = B B^T q
    q_other   = q - q_spatial

and measure:

1) Spatial magnitude / "how much spatial signal"
       S_abs  = ||q_spatial||
       S_frac = ||q_spatial|| / ||q||

2) Spatial orientation / "where the spatial signal points"
       A_GT = cos(q_spatial, projected GT prototype)

3) Relation-coordinate evidence
       s_left, s_right, s_above, s_below
       s_GT
       s_finalWrong
       max_nonGT
       GT-minus-maxNonGT

This separates:

    weak magnitude:
        wrong has lower S_abs / S_frac

    wrong orientation:
        wrong has similar S_abs but lower A_GT

    weak GT component:
        wrong has lower s_GT

    excessive competitor:
        wrong has higher s_finalWrong / max_nonGT

Attention transition
--------------------
For each sample and layer we explicitly compare POST_ATTN - PRE_ATTN:

    Delta S_abs
    Delta A_GT
    Delta s_GT
    Delta max_nonGT

so we can ask whether Attention accumulates less spatial signal, rotates the
signal away from GT, or amplifies competitors.

Causal validation
-----------------
Correlation is not enough. For selected layers/stages we perform four targeted
pair-preserving interventions on the REAL hidden state:

A) magnitude_only
   Keep the sample's own spatial direction fixed, but raise only the magnitude
   of q_spatial to the median generation-correct control magnitude for the same
   GT relation.

   This does NOT inject the oracle GT direction.

   If this improves wrong samples whose Direction prediction is already correct,
   it supports "correct spatial evidence exists but is too weak."

B) gt_boost_hold_foil
   Raise the GT prototype coordinate toward the correct-control median while
   holding the foil coordinate fixed (minimum-L2 edit).

C) foil_suppress_hold_gt
   Reduce the foil coordinate toward the correct-control median while holding
   the GT coordinate fixed.

D) both
   Apply GT boost + foil suppression simultaneously.

Every targeted edit can be compared with a norm-matched random direction
orthogonal to the spatial subspace.

Evaluation
----------
1) All selected problem layers:
       first-step four-relation margin and restricted argmax.

2) Optional top layer/stage candidates:
       fresh full model.generate(), reporting W->C / C->W.

Important interpretation
------------------------
The strongest evidence for "spatial information itself is weak" is NOT merely
that R or s_GT is lower.

The stronger causal pattern is:

    - wrong sample has a Direction-correct spatial state;
    - S_abs is below correct-control level;
    - magnitude_only strengthens the sample's EXISTING spatial component;
    - final relation margin / generation improves;
    - norm-matched non-spatial random edit does not.

That directly tests spatial-evidence strength without supplying a new GT
direction.

Dependency
----------
Place this script in the same directory as:

    analyze_prepost_attention_direction_causality_v2.py
    extract_two_object_relation_states.py
    analyze_layerwise_direction_failure_scan_v1.py

Recommended full run after your existing v2 experiment
--------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python analyze_direction_information_strength_causal_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --failure-dir output/qwen7b_fourway_layer_update_failure_v1 \
  --prepost-dir output/qwen7b_prepost_direction_causality_v2 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers auto \
  --eval-split test \
  --causal-layers all_selected \
  --causal-max-wrong 40 \
  --causal-max-correct 20 \
  --generation-top-k 2 \
  --generation-max-samples 30 \
  --output-dir output/qwen7b_direction_information_strength_causal_v1 \
  --overwrite

Smoke test
----------
CUDA_VISIBLE_DEVICES=0 python analyze_direction_information_strength_causal_v1.py \
  --direction-dir output/qwen7b_layer_direction_scan_v1 \
  --failure-dir output/qwen7b_fourway_layer_update_failure_v1 \
  --prepost-dir output/qwen7b_prepost_direction_causality_v2 \
  --dataset coco_two \
  --data-root data \
  --model qwen-7b \
  --device cuda:0 \
  --layers 14,16,19 \
  --max-eval 20 \
  --causal-layers 14,16,19 \
  --causal-max-wrong 8 \
  --causal-max-correct 4 \
  --generation-top-k 1 \
  --generation-max-samples 8 \
  --output-dir output/qwen7b_direction_information_strength_causal_smoke \
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
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor


# ---------------------------------------------------------------------------
# Embedded fallback for analyze_prepost_attention_direction_causality_v2
# ---------------------------------------------------------------------------
# This makes the script self-contained with respect to the v2 helper.  If the
# helper file exists next to this script, Python imports it normally.  If not,
# the exact helper implementation bundled below is loaded into an in-memory
# module named `core`.
try:
    import analyze_prepost_attention_direction_causality_v2 as core
except ModuleNotFoundError as _core_import_error:
    if getattr(_core_import_error, "name", None) != "analyze_prepost_attention_direction_causality_v2":
        raise
    import types as _types
    core = _types.ModuleType("analyze_prepost_attention_direction_causality_v2_embedded")
    _CORE_FALLBACK_SOURCE = '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\n"""\nPre-/post-Attention Semantic Direction causal scan.\n\nFix over the previous block-output experiment:\n  * semantic Direction codebooks are fitted directly at the ACTUAL computation\n    points, so there is no cache-layer indexing ambiguity;\n  * PRE_ATTN intervention is made at decoder-block input, before Attention reads\n    subject/reference tokens;\n  * POST_ATTN intervention is made by editing Attention output, which is exactly\n    equivalent to editing the post-Attention residual before both the MLP and\n    residual branches;\n  * gradient C is checked with very small +/- finite differences.\n\nFor stage s in {pre_attn, post_attn}:\n\n  r_l,s = (h_sub-h_ref)^img - (h_sub-h_ref)^noimg\n  d_l,s(g,c) = unit(mu_g - mu_c)\n\nRepresentation availability:\n\n  R = (r_l,s - center_l,s) dot d_l,s(g,c)\n\nLocal causal utilization:\n\n  C = d/d eps [logit_g-logit_c]\n\nunder the pair-preserving edit\n\n  h_sub += eps/2 * d ; h_ref -= eps/2 * d\n\nInterpretation:\n  R low, C normal/high -> information insufficiency candidate.\n  R high, C low       -> information exists but is weakly utilized.\n  R < 0, |C| high     -> wrong-side state on a causally relevant axis.\n  C < 0               -> semantic / causal direction mismatch.\n\nThe script uses cached ACTUAL model.generate() correct/wrong grouping, but does\nNOT reuse cached hidden states for the new codebooks.\n\nRecommended smoke test:\n\nCUDA_VISIBLE_DEVICES=0 python analyze_prepost_attention_direction_causality_v2.py \\\n  --direction-dir output/qwen7b_layer_direction_scan_v1 \\\n  --failure-dir output/qwen7b_fourway_layer_update_failure_v1 \\\n  --dataset coco_two --data-root data --model qwen-7b --device cuda:0 \\\n  --layers 14,16,19 --max-train 32 --max-eval 20 \\\n  --fd-layers 14,16,19 --fd-max-samples 8 \\\n  --output-dir output/qwen7b_prepost_direction_causality_smoke --overwrite\n\nRecommended full run:\n\nCUDA_VISIBLE_DEVICES=0 python analyze_prepost_attention_direction_causality_v2.py \\\n  --direction-dir output/qwen7b_layer_direction_scan_v1 \\\n  --failure-dir output/qwen7b_fourway_layer_update_failure_v1 \\\n  --dataset coco_two --data-root data --model qwen-7b --device cuda:0 \\\n  --layers auto --eval-split test \\\n  --fd-layers auto --fd-top-k 4 --fd-max-samples 30 \\\n  --output-dir output/qwen7b_prepost_direction_causality_v2 --overwrite\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport contextlib\nimport csv\nimport gc\nimport json\nimport math\nimport random\nimport shutil\nfrom collections import Counter, defaultdict\nfrom pathlib import Path\nfrom typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence\n\nimport numpy as np\nimport torch\nfrom PIL import Image\nfrom tqdm import tqdm\nimport transformers\nfrom transformers import AutoProcessor\n\nimport extract_two_object_relation_states as base\nimport analyze_layerwise_direction_failure_scan_v1 as direction\n\nRELATIONS = ("left", "right", "above", "below")\nREL2ID = {r: i for i, r in enumerate(RELATIONS)}\nSTAGES = ("pre_attn", "post_attn")\nEPS = 1e-10\n\n\ndef args_parser():\n    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)\n    p.add_argument("--direction-dir", required=True)\n    p.add_argument("--failure-dir", default=None)\n    p.add_argument("--dataset", default="coco_two")\n    p.add_argument("--data-root", default="data")\n    p.add_argument("--model", default="qwen-7b")\n    p.add_argument("--device", default="cuda:0")\n    p.add_argument("--attn-impl", default="eager",\n                   choices=["eager", "sdpa", "flash_attention_2", "none"])\n    p.add_argument("--prompt-template", default=(\n        "Determine the spatial relation of the {subject} to the {reference} "\n        "in the image. Answer with left, right, above, or below."))\n    p.add_argument("--layers", default="auto")\n    p.add_argument("--min-role-gap", type=float, default=0.5)\n    p.add_argument("--eval-split", default="test", choices=["train", "test", "all"])\n    p.add_argument("--max-train", type=int, default=None)\n    p.add_argument("--max-eval", type=int, default=None)\n    p.add_argument("--control-quantile", type=float, default=0.25)\n    p.add_argument("--random-controls", type=int, default=8)\n    p.add_argument("--bootstrap", type=int, default=3000)\n    p.add_argument("--seed", type=int, default=1)\n    p.add_argument("--fd-layers", default="auto")\n    p.add_argument("--fd-top-k", type=int, default=4)\n    p.add_argument("--fd-max-samples", type=int, default=30)\n    p.add_argument("--fd-eps-scales", default="0.01,0.025,0.05,0.1,0.25")\n    p.add_argument("--fd-random-control", action=argparse.BooleanOptionalAction, default=True)\n    p.add_argument("--save-every", type=int, default=10)\n    p.add_argument("--output-dir", required=True)\n    p.add_argument("--overwrite", action="store_true")\n    return p.parse_args()\n\n\ndef read_csv(path):\n    with Path(path).open("r", encoding="utf-8", newline="") as f:\n        return list(csv.DictReader(f))\n\n\ndef write_csv(path, rows):\n    rows = list(rows)\n    path = Path(path)\n    path.parent.mkdir(parents=True, exist_ok=True)\n    if not rows:\n        path.write_text("", encoding="utf-8")\n        return\n    fields, seen = [], set()\n    for r in rows:\n        for k in r:\n            if k not in seen:\n                seen.add(k); fields.append(k)\n    with path.open("w", encoding="utf-8", newline="") as f:\n        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)\n\n\ndef safe_mean(xs):\n    a = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)\n    return float(a.mean()) if len(a) else float("nan")\n\n\ndef safe_frac(xs):\n    x = list(xs)\n    return float(np.mean(x)) if x else float("nan")\n\n\ndef parse_layers(text, n):\n    t = str(text).strip().lower()\n    if t == "all": return list(range(n))\n    out = []\n    for z in str(text).split(","):\n        z = z.strip()\n        if not z: continue\n        if "-" in z:\n            a, b = map(int, z.split("-", 1)); step = 1 if b >= a else -1\n            out.extend(range(a, b + step, step))\n        else: out.append(int(z))\n    out = sorted(set(out))\n    bad = [x for x in out if x < 0 or x >= n]\n    if bad: raise ValueError(f"invalid layers {bad}; valid 0..{n-1}")\n    return out\n\n\ndef parse_floats(text):\n    x = [float(v.strip()) for v in str(text).split(",") if v.strip()]\n    if not x or any(v <= 0 for v in x): raise ValueError("eps scales must be positive")\n    return x\n\n\ndef load_meta(direction_dir):\n    dd = Path(direction_dir)\n    with np.load(dd / "vectors.npz", allow_pickle=True) as z:\n        sids = z["sample_index"].astype(np.int64)\n        labels = np.asarray([direction.norm_relation(x) for x in z["relation"]], dtype=object)\n    gt = {int(s): str(labels[i]) for i, s in enumerate(sids.tolist())}\n    split, gen = {}, {}\n    for r in read_csv(dd / "sample_split_and_generation.csv"):\n        sid = int(r["sample_index"]); split[sid] = str(r.get("split", "")).strip()\n        pred = direction.norm_relation(r.get("generation_pred", ""))\n        group = str(r.get("generation_group", "")).strip().lower()\n        if group not in ("correct", "wrong") and gt.get(sid) in REL2ID and pred in REL2ID:\n            group = "correct" if pred == gt[sid] else "wrong"\n        gen[sid] = {"generation_group": group, "generation_pred": pred,\n                    "generation_text": str(r.get("generation_text", ""))}\n    return {"sids": [int(x) for x in sids], "gt": gt, "split": split, "generation": gen}\n\n\ndef choose_layers(text, n_layers, failure_dir, min_gap):\n    if str(text).strip().lower() != "auto":\n        ls = parse_layers(text, n_layers)\n        return ls, [{"layer": l, "selected": 1, "reason": "explicit"} for l in ls]\n    if not failure_dir: raise ValueError("--failure-dir required for --layers auto")\n    rows = read_csv(Path(failure_dir) / "top_candidate_layers.csv")\n    out, audit = [], []\n    for r in rows:\n        l = int(r["layer"])\n        sig = int(float(r.get("pairwise_deficit_ci_positive", 0)))\n        gap = float(r["wrong_minus_correct_gt_minus_maxnonGT_gap"])\n        yes = sig == 1 and gap <= -float(min_gap)\n        audit.append({"layer": l, "pairwise_deficit_ci_positive": sig,\n                      "role_gap": gap, "selected": int(yes)})\n        if yes: out.append(l)\n    out = sorted(set(out))\n    if not out: raise RuntimeError("auto layer selection returned none")\n    return out, audit\n\n\ndef get_attr_path(obj, path):\n    for p in path.split("."): obj = getattr(obj, p)\n    return obj\n\n\ndef decoder_layers(model):\n    for path in ["model.language_model.layers", "language_model.layers",\n                 "model.model.layers", "model.layers", "language_model.model.layers"]:\n        try:\n            x = get_attr_path(model, path)\n            if len(x) and hasattr(x[0], "self_attn") and hasattr(x[0], "post_attention_layernorm"):\n                return x, path\n        except Exception: pass\n    raise RuntimeError("cannot resolve decoder layers")\n\n\ndef first_tensor(x):\n    if torch.is_tensor(x): return x\n    if isinstance(x, (tuple, list)):\n        for y in x:\n            if torch.is_tensor(y): return y\n    raise RuntimeError(f"no tensor in {type(x)}")\n\n\ndef replace_first_tensor(output, new):\n    if torch.is_tensor(output): return new\n    if isinstance(output, tuple):\n        q = list(output)\n        for i, x in enumerate(q):\n            if torch.is_tensor(x): q[i] = new; return tuple(q)\n    if isinstance(output, list):\n        q = list(output)\n        for i, x in enumerate(q):\n            if torch.is_tensor(x): q[i] = new; return q\n    raise RuntimeError(f"cannot replace tensor in {type(output)}")\n\n\ndef pool(x, pos):\n    idx = torch.as_tensor([int(p) for p in pos if 0 <= int(p) < x.shape[0]],\n                          device=x.device, dtype=torch.long)\n    if idx.numel() == 0: raise RuntimeError("empty phrase positions")\n    return x.index_select(0, idx).mean(0)\n\n\nclass StageCapture:\n    def __init__(self, layers, selected):\n        self.data = defaultdict(dict); self.handles = []\n        for l in selected:\n            b = layers[l]\n            def block_pre(_m, args, li=l):\n                if args: self.data[li]["pre_attn"] = first_tensor(args)\n            def post_pre(_m, args, li=l):\n                if args: self.data[li]["post_attn"] = first_tensor(args)\n            self.handles.append(b.register_forward_pre_hook(block_pre))\n            self.handles.append(b.post_attention_layernorm.register_forward_pre_hook(post_pre))\n    def validate(self, selected):\n        miss = [(l, s) for l in selected for s in STAGES if s not in self.data.get(l, {})]\n        if miss: raise RuntimeError(f"missing stage captures {miss}")\n    def close(self):\n        for h in self.handles:\n            with contextlib.suppress(Exception): h.remove()\n    def __enter__(self): return self\n    def __exit__(self, *_): self.close()\n\n\ndef build_batch(processor, rec, question, image, device):\n    prompt = direction.build_chat_prompt(processor, question, image is not None)\n    batch = direction.process_inputs(processor, prompt, image, device)\n    ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]\n    sp = direction.locate_phrase_positions(processor.tokenizer, ids, str(rec.subject))\n    rp = direction.locate_phrase_positions(processor.tokenizer, ids, str(rec.reference))\n    return batch, sp, rp\n\n\ndef pooled_capture(cap, selected, sp, rp):\n    out = {}\n    for l in selected:\n        out[l] = {}\n        for stage in STAGES:\n            x = cap.data[l][stage][0]\n            out[l][stage] = (pool(x, sp) - pool(x, rp)).detach().float().cpu().numpy().astype(np.float32)\n    return out\n\n\ndef capture_inference(model, processor, layers, selected, rec, image, device, prompt_template):\n    q = prompt_template.format(subject=rec.subject, reference=rec.reference)\n    batch, sp, rp = build_batch(processor, rec, q, image, device)\n    with StageCapture(layers, selected) as cap:\n        with torch.inference_mode():\n            model(**batch, output_attentions=False, output_hidden_states=False,\n                  use_cache=False, return_dict=True)\n        cap.validate(selected)\n        out = pooled_capture(cap, selected, sp, rp)\n    del batch\n    return out\n\n\ndef unit(v):\n    v = np.asarray(v, dtype=np.float64); n = np.linalg.norm(v)\n    return (v / max(float(n), EPS)).astype(np.float32)\n\n\ndef basis(vs):\n    A = np.stack(vs, 1).astype(np.float64); u, s, _ = np.linalg.svd(A, full_matrices=False)\n    keep = s > 1e-8 * max(float(s.max()), 1.0)\n    if not keep.any(): raise RuntimeError("degenerate spatial basis")\n    return u[:, keep].astype(np.float32)\n\n\ndef fit_cb(X, y):\n    center = X.mean(0).astype(np.float32); Xc = X - center\n    means, protos = {}, {}\n    for r in RELATIONS:\n        m = Xc[y == r].mean(0).astype(np.float32); means[r] = m; protos[r] = unit(m)\n    parr = np.stack([protos[r] for r in RELATIONS])\n    B = basis([means["right"] - means["left"], means["above"] - means["below"]])\n    std = {}\n    for g in RELATIONS:\n        for c in RELATIONS:\n            if c == g: continue\n            d = unit(protos[g] - protos[c]); std[(g, c)] = float(np.std(Xc @ d))\n    return {"center": center, "means": means, "protos": protos,\n            "proto_arr": parr, "basis": B, "axis_std": std}\n\n\ndef sem_axis(cb, g, c): return unit(cb["protos"][g] - cb["protos"][c])\n\n\ndef rep_metrics(v, cb, g, c):\n    q = v - cb["center"]; d = sem_axis(cb, g, c)\n    sc = q @ cb["proto_arr"].T\n    return {"R_raw": float(q @ d),\n            "R_cos": float(q @ d / max(float(np.linalg.norm(q)), EPS)),\n            "direction_pred": RELATIONS[int(np.argmax(sc))], "axis": d}\n\n\ndef fit_actual_point_codebooks(model, processor, layers, selected, train_sids,\n                                records, meta, device, prompt_template, outdir):\n    vecs = defaultdict(list); ys = []; kept = []; errors = []\n    for sid in tqdm(train_sids, desc="train actual-point codebooks"):\n        img = None\n        try:\n            rec = records[sid]; img = Image.open(rec.image_path).convert("RGB")\n            real = capture_inference(model, processor, layers, selected, rec, img,\n                                     device, prompt_template)\n            no = capture_inference(model, processor, layers, selected, rec, None,\n                                   device, prompt_template)\n            for l in selected:\n                for s in STAGES: vecs[(l, s)].append(real[l][s] - no[l][s])\n            ys.append(meta["gt"][sid]); kept.append(sid)\n        except Exception as e:\n            errors.append({"sid": sid, "error_type": type(e).__name__, "error": str(e)})\n        finally:\n            if img is not None: img.close()\n            gc.collect();\n            if torch.cuda.is_available(): torch.cuda.empty_cache()\n    y = np.asarray(ys, dtype=object)\n    cbs, diag = {}, []\n    arrays = {"sid": np.asarray(kept), "relation": y,\n              "layers": np.asarray(selected), "stages": np.asarray(STAGES, dtype=object)}\n    for l in selected:\n        for s in STAGES:\n            X = np.stack(vecs[(l, s)]).astype(np.float32); arrays[f"L{l}_{s}"] = X\n            cb = fit_cb(X, y); cbs[(l, s)] = cb\n            pred = np.asarray([RELATIONS[i] for i in np.argmax((X-cb["center"]) @ cb["proto_arr"].T, axis=1)])\n            diag.append({"layer": l, "stage": s, "n": len(X),\n                         "train_direction_acc": float(np.mean(pred == y)),\n                         "mean_norm": float(np.linalg.norm(X, axis=1).mean())})\n    np.savez_compressed(Path(outdir)/"train_stage_vectors.npz", **arrays)\n    write_csv(Path(outdir)/"stage_codebook_diagnostics.csv", diag)\n    Path(outdir, "train_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")\n    return cbs, diag\n\n\ndef tokenizer_ids(tok, text):\n    try: return [int(x) for x in tok.encode(text, add_special_tokens=False)]\n    except Exception:\n        o = tok(text, add_special_tokens=False); x = o["input_ids"] if isinstance(o, dict) else o.input_ids\n        if x and isinstance(x[0], (list, tuple)): x = x[0]\n        return [int(v) for v in x]\n\n\ndef relation_tokens(tok):\n    out = {}; unk = getattr(tok, "unk_token_id", None)\n    for r in RELATIONS:\n        ids = []\n        for t in (r, " "+r, "\\n"+r, r.capitalize(), " "+r.capitalize()):\n            x = tokenizer_ids(tok, t)\n            if len(x) == 1 and (unk is None or x[0] != unk): ids.append(x[0])\n        ids = list(dict.fromkeys(ids))\n        if not ids: raise RuntimeError(f"no one-token variant for {r}")\n        out[r] = ids\n    return out\n\n\ndef logits_of(out):\n    for x in [getattr(out, "logits", None),\n              getattr(getattr(out, "language_model_outputs", None), "logits", None),\n              getattr(getattr(out, "text_model_output", None), "logits", None)]:\n        if torch.is_tensor(x): return x\n    if isinstance(out, (tuple, list)):\n        for x in out:\n            if torch.is_tensor(x) and x.ndim == 3: return x\n    raise RuntimeError("cannot find logits")\n\n\ndef rel_scores(vec, token_map):\n    vals = []\n    for r in RELATIONS:\n        ids = torch.as_tensor(token_map[r], device=vec.device, dtype=torch.long)\n        vals.append(vec.index_select(0, ids).max())\n    return torch.stack(vals)\n\n\ndef rand_orth(dim, B, n, seed):\n    rng = np.random.default_rng(seed); B = B.astype(np.float64); out = []\n    for _ in range(max(20, 20*n)):\n        if len(out) >= n: break\n        v = rng.standard_normal(dim); v -= B @ (B.T @ v); nv = np.linalg.norm(v)\n        if nv > 1e-8: out.append((v/nv).astype(np.float32))\n    return out\n\n\ndef scan_sample(model, processor, token_map, layers, selected, cbs, rec, sid,\n                meta, device, prompt_template, random_controls, seed):\n    gt = meta["gt"][sid]; gen = meta["generation"][sid]; group = gen["generation_group"]\n    q = prompt_template.format(subject=rec.subject, reference=rec.reference)\n    no = capture_inference(model, processor, layers, selected, rec, None, device, prompt_template)\n    img = Image.open(rec.image_path).convert("RGB")\n    try:\n        batch, sp, rp = build_batch(processor, rec, q, img, device)\n        with StageCapture(layers, selected) as cap:\n            out = model(**batch, output_attentions=False, output_hidden_states=False,\n                        use_cache=False, return_dict=True)\n            cap.validate(selected)\n            rs = rel_scores(logits_of(out)[0, -1], token_map); rnp = rs.detach().float().cpu().numpy()\n            best = max([r for r in RELATIONS if r != gt], key=lambda r: rnp[REL2ID[r]])\n            primary = gen["generation_pred"] if group == "wrong" else best\n            inputs, keys = [], []\n            for l in selected:\n                for s in STAGES: inputs.append(cap.data[l][s]); keys.append((l, s))\n            grads = {}\n            for i, r in enumerate(RELATIONS):\n                gg = torch.autograd.grad(rs[i], inputs, retain_graph=i < 3, allow_unused=True)\n                grads[r] = {k: g for k, g in zip(keys, gg)}\n            real = pooled_capture(cap, selected, sp, rp)\n            rows = []\n            for l in selected:\n                for s in STAGES:\n                    rv = real[l][s] - no[l][s]; cb = cbs[(l, s)]\n                    for comp in RELATIONS:\n                        if comp == gt: continue\n                        rm = rep_metrics(rv, cb, gt, comp); d = rm["axis"]\n                        gg = grads[gt][(l,s)]; cg = grads[comp][(l,s)]\n                        if gg is None or cg is None: continue\n                        gp = .5*(pool(gg[0].float(), sp)-pool(gg[0].float(), rp))\n                        cp = .5*(pool(cg[0].float(), sp)-pool(cg[0].float(), rp))\n                        mg = (gp-cp).detach().cpu().numpy().astype(np.float32)\n                        C = float(mg @ d); B = cb["basis"]\n                        proj = B @ (B.T @ mg); pn = np.linalg.norm(proj); gn = np.linalg.norm(mg)\n                        align = float(proj @ d / pn) if pn > EPS else float("nan")\n                        rnd = rand_orth(len(d), B, random_controls,\n                                        seed + sid*100003 + l*1009 + (0 if s=="pre_attn" else 300007) + REL2ID[comp]*17)\n                        ra = safe_mean(abs(float(mg @ x)) for x in rnd)\n                        rows.append({"sid": sid, "layer": l, "stage": s, "gt": gt,\n                                     "competitor": comp, "generation_group": group,\n                                     "generation_pred": gen["generation_pred"],\n                                     "is_primary_foil": int(comp == primary),\n                                     "primary_foil": primary,\n                                     "baseline_firststep_pred": RELATIONS[int(np.argmax(rnp))],\n                                     "baseline_margin": float(rnp[REL2ID[gt]]-rnp[REL2ID[comp]]),\n                                     "R_raw": rm["R_raw"], "R_cos": rm["R_cos"],\n                                     "direction_pred": rm["direction_pred"],\n                                     "direction_correct": int(rm["direction_pred"] == gt),\n                                     "C_margin": C, "C_positive": int(C > 0),\n                                     "alignment": align,\n                                     "spatial_grad_fraction": float(pn/max(float(gn), EPS)),\n                                     "random_abs_mean": ra,\n                                     "specificity": float(abs(C)/ra) if math.isfinite(ra) and ra > EPS else float("nan")})\n        del batch\n        return rows\n    finally: img.close()\n\n\ndef boot_gap(a, b, n, rng):\n    a=np.asarray(a,float); b=np.asarray(b,float); a=a[np.isfinite(a)]; b=b[np.isfinite(b)]\n    if not len(a) or not len(b): return (float("nan"),)*3\n    obs=float(a.mean()-b.mean()); z=np.empty(n)\n    for i in range(n): z[i]=a[rng.integers(0,len(a),len(a))].mean()-b[rng.integers(0,len(b),len(b))].mean()\n    lo,hi=np.percentile(z,[2.5,97.5]); return obs,float(lo),float(hi)\n\n\ndef summarize(rows, selected, bootstrap, seed):\n    rng=np.random.default_rng(seed); prim=[r for r in rows if r["is_primary_foil"]==1]; out=[]\n    for l in selected:\n        for s in STAGES:\n            rr=[r for r in prim if r["layer"]==l and r["stage"]==s]\n            C=[r for r in rr if r["generation_group"]=="correct"]; W=[r for r in rr if r["generation_group"]=="wrong"]\n            z={"layer":l,"stage":s,"n_correct":len(C),"n_wrong":len(W)}\n            for m in ("R_raw","C_margin","alignment","spatial_grad_fraction","specificity"):\n                cv=[r[m] for r in C]; wv=[r[m] for r in W]; gap,lo,hi=boot_gap(cv,wv,bootstrap,rng)\n                z[f"{m}_correct"]=safe_mean(cv); z[f"{m}_wrong"]=safe_mean(wv)\n                z[f"{m}_gap_CminusW"]=gap; z[f"{m}_gap_ci95_lo"]=lo; z[f"{m}_gap_ci95_hi"]=hi\n            z["Cpos_correct"]=safe_frac(r["C_margin"]>0 for r in C); z["Cpos_wrong"]=safe_frac(r["C_margin"]>0 for r in W)\n            z["Rpos_Cnonpos_correct"]=safe_frac(r["R_raw"]>0 and r["C_margin"]<=0 for r in C)\n            z["Rpos_Cnonpos_wrong"]=safe_frac(r["R_raw"]>0 and r["C_margin"]<=0 for r in W)\n            out.append(z)\n    return out\n\n\ndef thresholds_and_failures(rows, q):\n    buckets=defaultdict(list)\n    for r in rows:\n        if r["generation_group"]=="correct": buckets[(r["layer"],r["stage"],r["gt"],r["competitor"])].append(r)\n    th={}; throws=[]\n    for k,rr in buckets.items():\n        R=np.asarray([r["R_raw"] for r in rr]); C=np.asarray([r["C_margin"] for r in rr])\n        th[k]={"Rq":float(np.quantile(R,q)),"Rm":float(np.median(R)),"Cq":float(np.quantile(C,q)),"Cm":float(np.median(C)),"n":len(rr)}\n        l,s,g,c=k; throws.append({"layer":l,"stage":s,"gt":g,"competitor":c,**th[k]})\n    fmap=[]\n    for r in rows:\n        if r["generation_group"]!="wrong" or not r["is_primary_foil"]: continue\n        k=(r["layer"],r["stage"],r["gt"],r["competitor"])\n        if k not in th: continue\n        t=th[k]; rd=r["R_raw"]<t["Rq"]; cd=r["C_margin"]<t["Cq"]\n        typ="both" if rd and cd else "representation_only" if rd else "utilization_only" if cd else "neither"\n        fmap.append({"sid":r["sid"],"layer":r["layer"],"stage":r["stage"],"gt":r["gt"],"final_wrong":r["competitor"],\n                     "R":r["R_raw"],"C":r["C_margin"],"representation_deficit":int(rd),"utilization_deficit":int(cd),\n                     "failure_type":typ,"R_deficit_from_median":t["Rm"]-r["R_raw"],"C_deficit_from_median":t["Cm"]-r["C_margin"]})\n    fsum=[]\n    for l in sorted(set(r["layer"] for r in fmap)):\n        for s in STAGES:\n            rr=[r for r in fmap if r["layer"]==l and r["stage"]==s]; n=len(rr); c=Counter(r["failure_type"] for r in rr)\n            fsum.append({"layer":l,"stage":s,"n_wrong":n,\n                         "RepDef":safe_frac(r["representation_deficit"] for r in rr),\n                         "UtilDef":safe_frac(r["utilization_deficit"] for r in rr),\n                         "Both":c["both"]/n if n else float("nan"),"RepOnly":c["representation_only"]/n if n else float("nan"),\n                         "UtilOnly":c["utilization_only"]/n if n else float("nan"),"Neither":c["neither"]/n if n else float("nan")})\n    return throws,fmap,fsum\n\n\ndef transition(summary):\n    d={(r["layer"],r["stage"]):r for r in summary}; out=[]\n    for l in sorted(set(r["layer"] for r in summary)):\n        a,b=d[(l,"pre_attn")],d[(l,"post_attn")]\n        out.append({"layer":l,"pre_Rgap":a["R_raw_gap_CminusW"],"post_Rgap":b["R_raw_gap_CminusW"],\n                    "attention_added_Rgap":b["R_raw_gap_CminusW"]-a["R_raw_gap_CminusW"],\n                    "pre_Cgap":a["C_margin_gap_CminusW"],"post_Cgap":b["C_margin_gap_CminusW"],\n                    "attention_changed_Cgap":b["C_margin_gap_CminusW"]-a["C_margin_gap_CminusW"]})\n    return out\n\n\nclass PreIntervention:\n    def __init__(self, block, sp, rp, delta):\n        self.sp=list(map(int,sp)); self.rp=list(map(int,rp)); self.delta=torch.from_numpy(delta.astype(np.float32)); self.applied=False\n        self.h=block.register_forward_pre_hook(self.hook)\n    def hook(self,_m,args):\n        if self.applied or not args:return None\n        x=first_tensor(args); y=x.clone(); half=.5*self.delta.to(y.device,y.dtype)\n        y[0,self.sp,:]+=half; y[0,self.rp,:]-=half; q=list(args)\n        for i,z in enumerate(q):\n            if torch.is_tensor(z): q[i]=y; break\n        self.applied=True; return tuple(q)\n    def close(self):\n        with contextlib.suppress(Exception): self.h.remove()\n\n\nclass PostIntervention:\n    def __init__(self, attn, sp, rp, delta):\n        self.sp=list(map(int,sp)); self.rp=list(map(int,rp)); self.delta=torch.from_numpy(delta.astype(np.float32)); self.applied=False\n        self.h=attn.register_forward_hook(self.hook)\n    def hook(self,_m,_a,out):\n        if self.applied:return out\n        x=first_tensor(out); y=x.clone(); half=.5*self.delta.to(y.device,y.dtype)\n        y[0,self.sp,:]+=half; y[0,self.rp,:]-=half; self.applied=True; return replace_first_tensor(out,y)\n    def close(self):\n        with contextlib.suppress(Exception): self.h.remove()\n\n\ndef score(model,batch,token_map):\n    with torch.inference_mode(): rs=rel_scores(logits_of(model(**batch,output_attentions=False,output_hidden_states=False,use_cache=False,return_dict=True))[0,-1],token_map)\n    a=rs.detach().float().cpu().numpy(); return {r:float(a[REL2ID[r]]) for r in RELATIONS}\n\n\ndef score_edit(model,batch,token_map,layers,l,stage,sp,rp,delta):\n    h=PreIntervention(layers[l],sp,rp,delta) if stage=="pre_attn" else PostIntervention(layers[l].self_attn,sp,rp,delta)\n    try:\n        x=score(model,batch,token_map)\n        if not h.applied: raise RuntimeError(f"{stage} hook not applied L{l}")\n        return x\n    finally:h.close()\n\n\ndef choose_fd(text, summary, n_layers, topk):\n    if text.lower()=="none":return []\n    if text.lower()!="auto":return parse_layers(text,n_layers)\n    by=defaultdict(dict)\n    for r in summary:by[r["layer"]][r["stage"]]=r\n    x=[]\n    for l,z in by.items():\n        if len(z)<2:continue\n        score=max(abs(z[s]["C_margin_gap_CminusW"]) for s in STAGES)\n        x.append((score,l))\n    return sorted([l for _,l in sorted(x,reverse=True)[:topk]])\n\n\ndef finite_diff(model,processor,token_map,layers,cbs,fd_layers,records,meta,gradrows,device,prompt_template,maxsamples,scales,random_control,seed):\n    prim=[r for r in gradrows if r["is_primary_foil"] and r["layer"] in fd_layers]\n    sids=sorted(set(r["sid"] for r in prim)); rng=random.Random(seed+99)\n    if maxsamples and len(sids)>maxsamples:\n        w=[s for s in sids if meta["generation"][s]["generation_group"]=="wrong"]; c=[s for s in sids if s not in set(w)]\n        rng.shuffle(w);rng.shuffle(c); sids=sorted(set(w[:maxsamples//2]+c[:maxsamples-maxsamples//2]))\n    look={(r["sid"],r["layer"],r["stage"]):r for r in prim}; rows=[]\n    for sid in tqdm(sids,desc="small-epsilon FD"):\n        rec=records[sid]; img=Image.open(rec.image_path).convert("RGB")\n        try:\n            q=prompt_template.format(subject=rec.subject,reference=rec.reference);batch,sp,rp=build_batch(processor,rec,q,img,device);base_sc=score(model,batch,token_map)\n            for l in fd_layers:\n                for stage in STAGES:\n                    k=(sid,l,stage)\n                    if k not in look:continue\n                    g=look[k];gt=g["gt"];comp=g["competitor"];cb=cbs[(l,stage)];d=sem_axis(cb,gt,comp);sigma=cb["axis_std"][(gt,comp)] or 1.0\n                    B=cb["basis"];rnd=rand_orth(len(d),B,1,seed+sid*911+l*17+(0 if stage=="pre_attn" else 3));rd=rnd[0] if rnd else None\n                    bm=base_sc[gt]-base_sc[comp]\n                    for scale in scales:\n                        eps=scale*sigma\n                        p=score_edit(model,batch,token_map,layers,l,stage,sp,rp,eps*d);m=score_edit(model,batch,token_map,layers,l,stage,sp,rp,-eps*d)\n                        pm=p[gt]-p[comp];mm=m[gt]-m[comp];fd=(pm-mm)/(2*eps)\n                        row={"sid":sid,"layer":l,"stage":stage,"group":g["generation_group"],"gt":gt,"competitor":comp,"eps_scale":scale,\n                             "gradient_C":g["C_margin"],"fd_slope":fd,"same_sign":int(np.sign(fd)==np.sign(g["C_margin"]) and fd!=0 and g["C_margin"]!=0),\n                             "monotonic":int(pm>bm>mm)}\n                        if random_control and rd is not None:\n                            rp1=score_edit(model,batch,token_map,layers,l,stage,sp,rp,eps*rd);rm1=score_edit(model,batch,token_map,layers,l,stage,sp,rp,-eps*rd)\n                            rfd=((rp1[gt]-rp1[comp])-(rm1[gt]-rm1[comp]))/(2*eps);row["random_fd_slope"]=rfd;row["semantic_gt_random"]=int(abs(fd)>abs(rfd))\n                        else:row["random_fd_slope"]=float("nan");row["semantic_gt_random"]=0\n                        rows.append(row)\n            del batch\n        finally:img.close();gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None\n    sm=[]\n    buckets=defaultdict(list)\n    for r in rows:buckets[(r["layer"],r["stage"],r["eps_scale"],r["group"])].append(r)\n    for k,rr in sorted(buckets.items()):\n        l,s,e,g=k;gv=np.asarray([r["gradient_C"] for r in rr]);fv=np.asarray([r["fd_slope"] for r in rr]);rv=np.asarray([r["random_fd_slope"] for r in rr])\n        corr=float(np.corrcoef(gv,fv)[0,1]) if len(rr)>1 and np.std(gv)>0 and np.std(fv)>0 else float("nan")\n        sm.append({"layer":l,"stage":s,"eps_scale":e,"group":g,"n":len(rr),"gradC":float(gv.mean()),"fdSlope":float(fv.mean()),"corr":corr,\n                   "sameSign":safe_frac(r["same_sign"] for r in rr),"monotonic":safe_frac(r["monotonic"] for r in rr),\n                   "semantic_gt_random":safe_frac(r["semantic_gt_random"] for r in rr),"absFD":float(np.mean(np.abs(fv))),"absRandom":float(np.nanmean(np.abs(rv)))})\n    return rows,sm\n\n\ndef print_results(summary,trans,fsum,fd):\n    print("\\n"+"="*160);print("PRE/POST-ATTENTION DIRECTION CAUSAL MAP");print("="*160)\n    print("layer stage      R cor/wr gap | C cor/wr gap [95%CI] | align cor/wr | C>0 cor/wr")\n    for r in summary:\n        print(f"L{r[\'layer\']:02d} {r[\'stage\']:9s} {r[\'R_raw_correct\']:+7.3f}/{r[\'R_raw_wrong\']:+7.3f} {r[\'R_raw_gap_CminusW\']:+7.3f} | "\n              f"{r[\'C_margin_correct\']:+8.5f}/{r[\'C_margin_wrong\']:+8.5f} {r[\'C_margin_gap_CminusW\']:+8.5f} "\n              f"[{r[\'C_margin_gap_ci95_lo\']:+7.5f},{r[\'C_margin_gap_ci95_hi\']:+7.5f}] | "\n              f"{r[\'alignment_correct\']:+6.3f}/{r[\'alignment_wrong\']:+6.3f} | {r[\'Cpos_correct\']:.3f}/{r[\'Cpos_wrong\']:.3f}")\n    print("\\nATTENTION TRANSITION (positive added_Rgap = attention enlarges representation gap)")\n    for r in trans:print(f"L{r[\'layer\']:02d}: Rgap {r[\'pre_Rgap\']:+.3f}->{r[\'post_Rgap\']:+.3f} add={r[\'attention_added_Rgap\']:+.3f} | Cgap {r[\'pre_Cgap\']:+.5f}->{r[\'post_Cgap\']:+.5f}")\n    print("\\nWRONG FAILURE TYPES")\n    for r in fsum:print(f"L{r[\'layer\']:02d} {r[\'stage\']:9s}: RepDef={r[\'RepDef\']:.3f} UtilDef={r[\'UtilDef\']:.3f} Both={r[\'Both\']:.3f} RepOnly={r[\'RepOnly\']:.3f} UtilOnly={r[\'UtilOnly\']:.3f}")\n    if fd:\n        print("\\nFINITE-DIFFERENCE SANITY")\n        for r in fd:print(f"L{r[\'layer\']:02d} {r[\'stage\']:9s} eps={r[\'eps_scale\']:.3f} {r[\'group\']:7s} N={r[\'n\']:2d} grad={r[\'gradC\']:+.5f} fd={r[\'fdSlope\']:+.5f} corr={r[\'corr\']:+.3f} sign={r[\'sameSign\']:.3f} mono={r[\'monotonic\']:.3f} sem>rand={r[\'semantic_gt_random\']:.3f}")\n\n\ndef main():\n    a=args_parser();out=Path(a.output_dir)\n    if a.overwrite and out.exists():shutil.rmtree(out)\n    out.mkdir(parents=True,exist_ok=True)\n    meta=load_meta(a.direction_dir);records,_=base.load_records(a.dataset,Path(a.data_root),None);records={int(r.sid):r for r in records}\n    spec=base.SPECS[a.model];cls=getattr(transformers,spec.model_class);dtype=base.resolve_dtype(spec.dtype_name)\n    kw={"low_cpu_mem_usage":True,"trust_remote_code":spec.trust_remote_code,"device_map":{"":a.device}}\n    if a.attn_impl!="none":kw["attn_implementation"]=a.attn_impl\n    try:model=cls.from_pretrained(spec.repo_id,dtype=dtype,**kw)\n    except TypeError:model=cls.from_pretrained(spec.repo_id,torch_dtype=dtype,**kw)\n    model.eval();processor=AutoProcessor.from_pretrained(spec.repo_id,trust_remote_code=spec.trust_remote_code);base.configure_processor(model,processor)\n    layers,path=decoder_layers(model);selected,audit=choose_layers(a.layers,len(layers),a.failure_dir,a.min_role_gap);write_csv(out/"selected_problem_layers.csv",audit)\n    print(f"[decoder] {path}; selected={selected}")\n    train=[s for s in meta["sids"] if meta["split"].get(s)=="train" and meta["gt"].get(s) in REL2ID and s in records]\n    if a.max_train and len(train)>a.max_train:\n        rng=random.Random(a.seed);by=defaultdict(list)\n        for s in train:by[meta["gt"][s]].append(s)\n        for r in by:rng.shuffle(by[r])\n        q=[]\n        while len(q)<a.max_train and any(by.values()):\n            for r in RELATIONS:\n                if by[r] and len(q)<a.max_train:q.append(by[r].pop())\n        train=sorted(q)\n    cbs,diag=fit_actual_point_codebooks(model,processor,layers,selected,train,records,meta,torch.device(a.device),a.prompt_template,out)\n    print("[codebooks]",[(r["layer"],r["stage"],round(r["train_direction_acc"],3)) for r in diag])\n    ev=[]\n    for s in meta["sids"]:\n        if a.eval_split!="all" and meta["split"].get(s)!=a.eval_split:continue\n        if s not in records or meta["gt"].get(s) not in REL2ID:continue\n        g=meta["generation"].get(s,{});group=g.get("generation_group");pred=g.get("generation_pred")\n        if group not in ("correct","wrong"):continue\n        if group=="wrong" and (pred not in REL2ID or pred==meta["gt"][s]):continue\n        ev.append(s)\n    if a.max_eval and len(ev)>a.max_eval:\n        rng=random.Random(a.seed+1);w=[s for s in ev if meta["generation"][s]["generation_group"]=="wrong"];c=[s for s in ev if s not in set(w)];rng.shuffle(w);rng.shuffle(c);ev=sorted(set(w[:a.max_eval//2]+c[:a.max_eval-a.max_eval//2]))\n    token_map=relation_tokens(processor.tokenizer);rows=[];errors=[]\n    for i,sid in enumerate(tqdm(ev,desc="pre/post causal scan"),1):\n        try:rows+=scan_sample(model,processor,token_map,layers,selected,cbs,records[sid],sid,meta,torch.device(a.device),a.prompt_template,a.random_controls,a.seed)\n        except Exception as e:errors.append({"sid":sid,"error_type":type(e).__name__,"error":str(e)});tqdm.write(f"[ERR {sid}] {e}")\n        finally:model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None\n        if a.save_every and i%a.save_every==0:write_csv(out/"per_sample_stage_axis.csv",rows)\n    write_csv(out/"per_sample_stage_axis.csv",rows);write_csv(out/"errors.csv",errors)\n    summ=summarize(rows,selected,a.bootstrap,a.seed);trans=transition(summ);th,fmap,fsum=thresholds_and_failures(rows,a.control_quantile)\n    write_csv(out/"primary_foil_stage_summary.csv",summ);write_csv(out/"attention_transition_summary.csv",trans);write_csv(out/"correct_stage_axis_thresholds.csv",th);write_csv(out/"wrong_stage_failure_map.csv",fmap);write_csv(out/"wrong_stage_failure_type_summary.csv",fsum)\n    fd_layers=choose_fd(a.fd_layers,summ,len(layers),a.fd_top_k);print("[fd layers]",fd_layers);fdrows,fd=[] ,[]\n    if fd_layers:\n        fdrows,fd=finite_diff(model,processor,token_map,layers,cbs,fd_layers,records,meta,rows,torch.device(a.device),a.prompt_template,a.fd_max_samples,parse_floats(a.fd_eps_scales),a.fd_random_control,a.seed)\n        write_csv(out/"finite_difference_per_sample.csv",fdrows);write_csv(out/"finite_difference_summary.csv",fd)\n    print_results(summ,trans,fsum,fd)\n    (out/"summary.json").write_text(json.dumps({"selected_layers":selected,"n_train":len(train),"n_eval":len(ev),"fd_layers":fd_layers,\n        "R":"actual-point img-noimg Direction availability","C":"local derivative of first-step GT-vs-competitor margin at actual pre/post Attention point",\n        "important":"accept C only when small-epsilon finite difference agrees"},indent=2),encoding="utf-8")\n\nif __name__=="__main__":main()\n'
    exec(
        compile(
            _CORE_FALLBACK_SOURCE,
            "<embedded analyze_prepost_attention_direction_causality_v2>",
            "exec",
        ),
        core.__dict__,
    )



RELATIONS = core.RELATIONS
REL2ID = core.REL2ID
STAGES = core.STAGES
EPS = 1e-10


# =============================================================================
# CLI / generic utilities
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--failure-dir", default=None)
    p.add_argument(
        "--prepost-dir",
        default=None,
        help=(
            "Existing v2 output containing train_stage_vectors.npz. "
            "Strongly recommended to avoid refitting TRAIN stage codebooks."
        ),
    )
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

    p.add_argument("--layers", default="auto")
    p.add_argument("--min-role-gap", type=float, default=0.5)
    p.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "test", "all"],
    )
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-eval", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--bootstrap", type=int, default=3000)

    p.add_argument(
        "--weak-quantile",
        type=float,
        default=0.25,
        help="Lower correct-control quantile used to call S / GT / alignment weak.",
    )
    p.add_argument(
        "--excess-quantile",
        type=float,
        default=0.75,
        help="Upper correct-control quantile used to call competitor evidence excessive.",
    )
    p.add_argument(
        "--target-stat",
        default="median",
        choices=["median", "mean"],
        help="Correct-control target for causal edits.",
    )

    p.add_argument(
        "--causal-layers",
        default="all_selected",
        help="none, all_selected, auto, or explicit layer list.",
    )
    p.add_argument(
        "--causal-stages",
        default="pre_attn,post_attn",
        help="Comma-separated subset of pre_attn,post_attn.",
    )
    p.add_argument("--causal-max-wrong", type=int, default=40)
    p.add_argument("--causal-max-correct", type=int, default=20)
    p.add_argument(
        "--causal-modes",
        default="magnitude_only,gt_boost_hold_foil,foil_suppress_hold_gt,both",
    )
    p.add_argument(
        "--random-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--max-edit-norm",
        type=float,
        default=0.0,
        help="Optional edit norm cap; <=0 disables clipping.",
    )

    p.add_argument(
        "--generation-top-k",
        type=int,
        default=2,
        help="Top layer-stage pairs from first-step causal scan to validate with generate(). 0 disables.",
    )
    p.add_argument(
        "--generation-max-samples",
        type=int,
        default=30,
    )
    p.add_argument(
        "--generation-modes",
        default="magnitude_only,gt_boost_hold_foil,foil_suppress_hold_gt,both",
    )
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument("--save-every", type=int, default=20)
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


def safe_mean(xs: Iterable[float]) -> float:
    vals = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(vals.mean()) if len(vals) else float("nan")


def safe_median(xs: Iterable[float]) -> float:
    vals = np.asarray(
        [float(x) for x in xs if math.isfinite(float(x))],
        dtype=np.float64,
    )
    return float(np.median(vals)) if len(vals) else float("nan")


def safe_frac(xs: Iterable[bool]) -> float:
    vals = list(xs)
    return float(np.mean(vals)) if vals else float("nan")


def parse_csv_words(text: str) -> List[str]:
    return [
        x.strip()
        for x in str(text).split(",")
        if x.strip()
    ]


def parse_stages(text: str) -> List[str]:
    vals = parse_csv_words(text)
    bad = [x for x in vals if x not in STAGES]
    if bad:
        raise ValueError(f"Unknown stages: {bad}")
    if not vals:
        raise ValueError("No stages selected.")
    return vals


def target_stat(vals: Sequence[float], which: str) -> float:
    x = np.asarray(vals, dtype=np.float64)
    if len(x) == 0:
        return float("nan")
    return float(np.median(x) if which == "median" else np.mean(x))


def bootstrap_gap(
    correct_vals: Sequence[float],
    wrong_vals: Sequence[float],
    n_boot: int,
    rng: np.random.Generator,
):
    a = np.asarray(correct_vals, dtype=np.float64)
    b = np.asarray(wrong_vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")

    obs = float(a.mean() - b.mean())
    boots = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        boots[i] = aa.mean() - bb.mean()

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


# =============================================================================
# Stage codebooks: reuse v2 TRAIN activations if possible
# =============================================================================

def load_codebooks_from_prepost(
    prepost_dir: Path,
    selected_layers: Sequence[int],
):
    path = prepost_dir / "train_stage_vectors.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as z:
        files = set(z.files)
        y = np.asarray(z["relation"], dtype=object)

        cbs = {}
        diagnostics = []

        for li in selected_layers:
            for stage in STAGES:
                key = f"L{li}_{stage}"
                if key not in files:
                    raise KeyError(
                        f"{path} missing {key}; rerun v2 with this layer "
                        "or omit --prepost-dir."
                    )

                X = np.asarray(z[key], dtype=np.float32)
                cb = core.fit_cb(X, y)
                cbs[(li, stage)] = cb

                pred_idx = np.argmax(
                    (X - cb["center"]) @ cb["proto_arr"].T,
                    axis=1,
                )
                pred = np.asarray(
                    [RELATIONS[int(i)] for i in pred_idx],
                    dtype=object,
                )

                diagnostics.append({
                    "layer": li,
                    "stage": stage,
                    "n_train": len(X),
                    "train_direction_acc":
                        float(np.mean(pred == y)),
                    "mean_residual_norm":
                        float(np.linalg.norm(X, axis=1).mean()),
                    "source": str(path),
                })

    return cbs, diagnostics


# =============================================================================
# Direction evidence decomposition
# =============================================================================

def project_spatial(q: np.ndarray, B: np.ndarray) -> np.ndarray:
    B64 = np.asarray(B, dtype=np.float64)
    q64 = np.asarray(q, dtype=np.float64)
    return (B64 @ (B64.T @ q64)).astype(np.float32)


def projected_relation_direction(cb, rel: str) -> np.ndarray:
    p = np.asarray(cb["protos"][rel], dtype=np.float32)
    p_sp = project_spatial(p, cb["basis"])
    return core.unit(p_sp)


def decompose_vector(
    residual_vec: np.ndarray,
    cb,
    gt: str,
):
    q = (
        np.asarray(residual_vec, dtype=np.float32)
        - np.asarray(cb["center"], dtype=np.float32)
    )
    q_sp = project_spatial(q, cb["basis"])

    q_norm = float(np.linalg.norm(q))
    S_abs = float(np.linalg.norm(q_sp))
    S_frac = S_abs / q_norm if q_norm > EPS else float("nan")

    sp_dirs = {
        rel: projected_relation_direction(cb, rel)
        for rel in RELATIONS
    }

    if S_abs > EPS:
        q_sp_unit = q_sp / S_abs
        align = {
            rel: float(q_sp_unit @ sp_dirs[rel])
            for rel in RELATIONS
        }
    else:
        align = {rel: float("nan") for rel in RELATIONS}

    # Keep the original normalized prototype coordinates too.
    proto_scores = {
        rel: float(q @ cb["protos"][rel])
        for rel in RELATIONS
    }

    spatial_scores = {
        rel: float(q_sp @ sp_dirs[rel])
        for rel in RELATIONS
    }

    non_gt = [r for r in RELATIONS if r != gt]
    max_non_gt = max(
        non_gt,
        key=lambda r: proto_scores[r],
    )

    direction_pred = max(
        RELATIONS,
        key=lambda r: proto_scores[r],
    )

    return {
        "q": q,
        "q_sp": q_sp,
        "q_norm": q_norm,
        "S_abs": S_abs,
        "S_frac": S_frac,
        "A_GT": align[gt],
        "A_max_nonGT": max(
            align[r] for r in non_gt
            if math.isfinite(align[r])
        ) if any(math.isfinite(align[r]) for r in non_gt)
        else float("nan"),
        "s_GT": proto_scores[gt],
        "s_max_nonGT": proto_scores[max_non_gt],
        "max_nonGT_relation": max_non_gt,
        "GT_minus_maxNonGT":
            proto_scores[gt] - proto_scores[max_non_gt],
        "direction_pred": direction_pred,
        "direction_correct": int(direction_pred == gt),
        "proto_scores": proto_scores,
        "spatial_scores": spatial_scores,
        "alignments": align,
    }


def select_eval_sids(
    meta,
    records,
    split,
    max_eval,
    seed,
):
    sids = []

    for sid in meta["idx_by_sid"]:
        if split != "all" and meta["split"].get(sid, "") != split:
            continue
        if sid not in records:
            continue

        gt = meta["gt"].get(sid, "")
        gen = meta["generation"].get(sid, {})
        group = gen.get("generation_group", "")
        pred = gen.get("generation_pred", "")

        if gt not in REL2ID:
            continue
        if group not in ("correct", "wrong"):
            continue
        if group == "wrong" and (
            pred not in REL2ID or pred == gt
        ):
            continue

        sids.append(int(sid))

    if max_eval is None or len(sids) <= max_eval:
        return sorted(sids)

    rng = random.Random(seed)

    wrong = [
        sid for sid in sids
        if meta["generation"][sid]["generation_group"] == "wrong"
    ]
    correct = [
        sid for sid in sids
        if meta["generation"][sid]["generation_group"] == "correct"
    ]

    rng.shuffle(wrong)
    rng.shuffle(correct)

    nw = min(len(wrong), max(1, max_eval // 2))
    nc = min(len(correct), max_eval - nw)

    chosen = wrong[:nw] + correct[:nc]
    if len(chosen) < max_eval:
        left = [s for s in sids if s not in set(chosen)]
        rng.shuffle(left)
        chosen += left[: max_eval - len(chosen)]

    return sorted(set(chosen))


def collect_eval_decomposition(
    *,
    model,
    processor,
    decoder_layers,
    selected_layers,
    codebooks,
    records,
    meta,
    eval_sids,
    device,
    prompt_template,
    out_dir,
    save_every,
):
    rows = []
    errors = []
    vectors: Dict[int, Dict[Tuple[int, str], np.ndarray]] = {}

    for i, sid in enumerate(
        tqdm(eval_sids, desc="Direction strength decomposition"),
        1,
    ):
        rec = records[sid]
        image = None

        try:
            image = Image.open(rec.image_path).convert("RGB")

            real = core.capture_inference(
                model,
                processor,
                decoder_layers,
                selected_layers,
                rec,
                image,
                device,
                prompt_template,
            )
            noimg = core.capture_inference(
                model,
                processor,
                decoder_layers,
                selected_layers,
                rec,
                None,
                device,
                prompt_template,
            )

            gt = meta["gt"][sid]
            gen = meta["generation"][sid]
            group = gen["generation_group"]
            gen_pred = gen["generation_pred"]

            vectors[sid] = {}

            for li in selected_layers:
                for stage in STAGES:
                    residual = (
                        real[li][stage] - noimg[li][stage]
                    ).astype(np.float32)
                    vectors[sid][(li, stage)] = residual

                    d = decompose_vector(
                        residual,
                        codebooks[(li, stage)],
                        gt,
                    )

                    row = {
                        "sid": sid,
                        "layer": li,
                        "stage": stage,
                        "gt": gt,
                        "generation_group": group,
                        "generation_pred": gen_pred,

                        "S_abs": d["S_abs"],
                        "S_frac": d["S_frac"],
                        "q_norm": d["q_norm"],

                        "A_GT": d["A_GT"],
                        "A_max_nonGT": d["A_max_nonGT"],

                        "s_GT": d["s_GT"],
                        "s_max_nonGT": d["s_max_nonGT"],
                        "max_nonGT_relation":
                            d["max_nonGT_relation"],
                        "GT_minus_maxNonGT":
                            d["GT_minus_maxNonGT"],

                        "direction_pred":
                            d["direction_pred"],
                        "direction_correct":
                            d["direction_correct"],
                    }

                    for rel in RELATIONS:
                        row[f"s_{rel}"] = d["proto_scores"][rel]
                        row[f"spatial_s_{rel}"] = d["spatial_scores"][rel]
                        row[f"align_{rel}"] = d["alignments"][rel]

                    if (
                        group == "wrong"
                        and gen_pred in REL2ID
                        and gen_pred != gt
                    ):
                        row["s_finalWrong"] = d["proto_scores"][gen_pred]
                        row["A_finalWrong"] = d["alignments"][gen_pred]
                        row["GT_minus_finalWrong"] = (
                            d["proto_scores"][gt]
                            - d["proto_scores"][gen_pred]
                        )
                    else:
                        row["s_finalWrong"] = float("nan")
                        row["A_finalWrong"] = float("nan")
                        row["GT_minus_finalWrong"] = float("nan")

                    rows.append(row)

            if save_every > 0 and i % save_every == 0:
                write_csv(
                    out_dir / "per_sample_direction_strength.csv",
                    rows,
                )

        except Exception as e:
            errors.append({
                "sid": sid,
                "error_type": type(e).__name__,
                "error": str(e),
            })
            tqdm.write(
                f"[ERROR sid={sid}] {type(e).__name__}: {e}"
            )

        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(
        out_dir / "per_sample_direction_strength.csv",
        rows,
    )
    write_csv(
        out_dir / "decomposition_errors.csv",
        errors,
    )

    # Save vectors for reproducibility / downstream intervention analysis.
    good_sids = sorted(vectors)
    arrays = {
        "sid": np.asarray(good_sids, dtype=np.int64),
        "layers": np.asarray(selected_layers, dtype=np.int64),
        "stages": np.asarray(STAGES, dtype=object),
    }
    for li in selected_layers:
        for stage in STAGES:
            vals = [
                vectors[sid][(li, stage)]
                for sid in good_sids
                if (li, stage) in vectors[sid]
            ]
            if len(vals) == len(good_sids):
                arrays[f"L{li}_{stage}"] = np.stack(vals)

    np.savez_compressed(
        out_dir / "eval_stage_residual_vectors.npz",
        **arrays,
    )

    return rows, vectors, errors


# =============================================================================
# Descriptive summaries and matched correct controls
# =============================================================================

def summarize_correct_wrong(
    rows,
    selected_layers,
    bootstrap,
    seed,
):
    rng = np.random.default_rng(seed)
    out = []

    metrics = (
        "S_abs",
        "S_frac",
        "A_GT",
        "s_GT",
        "s_max_nonGT",
        "GT_minus_maxNonGT",
    )

    for li in selected_layers:
        for stage in STAGES:
            rr = [
                r for r in rows
                if int(r["layer"]) == li
                and r["stage"] == stage
            ]
            cor = [
                r for r in rr
                if r["generation_group"] == "correct"
            ]
            wr = [
                r for r in rr
                if r["generation_group"] == "wrong"
            ]

            row = {
                "layer": li,
                "stage": stage,
                "n_correct": len(cor),
                "n_wrong": len(wr),
                "direction_acc_correct": safe_frac(
                    int(r["direction_correct"]) == 1
                    for r in cor
                ),
                "direction_acc_wrong": safe_frac(
                    int(r["direction_correct"]) == 1
                    for r in wr
                ),
            }

            for metric in metrics:
                cv = [float(r[metric]) for r in cor]
                wv = [float(r[metric]) for r in wr]
                gap, lo, hi = bootstrap_gap(
                    cv,
                    wv,
                    bootstrap,
                    rng,
                )

                row[f"{metric}_correct"] = safe_mean(cv)
                row[f"{metric}_wrong"] = safe_mean(wv)
                row[f"{metric}_gap_CminusW"] = gap
                row[f"{metric}_gap_ci95_lo"] = lo
                row[f"{metric}_gap_ci95_hi"] = hi

            out.append(row)

    return out


def build_control_targets(
    rows,
    weak_q,
    excess_q,
    which_stat,
):
    """
    Correct controls matched by layer/stage/GT.

    Foil-specific target is computed later from the stored per-relation columns,
    so the same correct control set can answer any wrong GT->foil pair.
    """
    buckets = defaultdict(list)

    for r in rows:
        if r["generation_group"] != "correct":
            continue
        key = (
            int(r["layer"]),
            str(r["stage"]),
            str(r["gt"]),
        )
        buckets[key].append(r)

    targets = {}
    output = []

    for key, rr in sorted(buckets.items()):
        li, stage, gt = key

        def vals(col):
            return np.asarray(
                [float(r[col]) for r in rr],
                dtype=np.float64,
            )

        S = vals("S_abs")
        Sf = vals("S_frac")
        A = vals("A_GT")
        G = vals("s_GT")

        t = {
            "n": len(rr),

            "S_target": target_stat(S, which_stat),
            "S_q_low": float(np.quantile(S, weak_q)),
            "Sfrac_target": target_stat(Sf, which_stat),
            "Sfrac_q_low": float(np.quantile(Sf, weak_q)),

            "A_target": target_stat(A, which_stat),
            "A_q_low": float(np.quantile(A, weak_q)),

            "GT_target": target_stat(G, which_stat),
            "GT_q_low": float(np.quantile(G, weak_q)),
        }

        for foil in RELATIONS:
            if foil == gt:
                continue
            F = vals(f"s_{foil}")
            P = G - F

            t[f"foil_{foil}_target"] = target_stat(F, which_stat)
            t[f"foil_{foil}_q_high"] = float(
                np.quantile(F, excess_q)
            )
            t[f"pair_{foil}_target"] = target_stat(P, which_stat)
            t[f"pair_{foil}_q_low"] = float(
                np.quantile(P, weak_q)
            )

        targets[key] = t

        row = {
            "layer": li,
            "stage": stage,
            "gt": gt,
            "weak_quantile": weak_q,
            "excess_quantile": excess_q,
            "target_stat": which_stat,
            **t,
        }
        output.append(row)

    return targets, output


def build_wrong_diagnosis(
    rows,
    targets,
):
    out = []

    for r in rows:
        if r["generation_group"] != "wrong":
            continue

        gt = str(r["gt"])
        foil = str(r["generation_pred"])

        if foil not in REL2ID or foil == gt:
            continue

        key = (
            int(r["layer"]),
            str(r["stage"]),
            gt,
        )
        if key not in targets:
            continue

        t = targets[key]

        S = float(r["S_abs"])
        A = float(r["A_GT"])
        sgt = float(r["s_GT"])
        sfoil = float(r[f"s_{foil}"])
        pair = sgt - sfoil

        magnitude_weak = S < float(t["S_q_low"])
        orientation_weak = A < float(t["A_q_low"])
        gt_weak = sgt < float(t["GT_q_low"])
        foil_excess = (
            sfoil > float(t[f"foil_{foil}_q_high"])
        )
        pair_weak = (
            pair < float(t[f"pair_{foil}_q_low"])
        )

        if magnitude_weak and orientation_weak:
            highlevel = "magnitude_and_orientation_weak"
        elif magnitude_weak:
            highlevel = "magnitude_weak_only"
        elif orientation_weak:
            highlevel = "orientation_weak_only"
        else:
            highlevel = "neither_magnitude_nor_orientation_weak"

        out.append({
            "sid": int(r["sid"]),
            "layer": int(r["layer"]),
            "stage": r["stage"],
            "gt": gt,
            "final_wrong": foil,

            "direction_correct":
                int(r["direction_correct"]),

            "S_abs": S,
            "S_target": t["S_target"],
            "S_deficit":
                float(t["S_target"]) - S,
            "magnitude_weak":
                int(magnitude_weak),

            "A_GT": A,
            "A_target": t["A_target"],
            "A_deficit":
                float(t["A_target"]) - A,
            "orientation_weak":
                int(orientation_weak),

            "s_GT": sgt,
            "GT_target": t["GT_target"],
            "GT_deficit":
                float(t["GT_target"]) - sgt,
            "GT_component_weak":
                int(gt_weak),

            "s_finalWrong": sfoil,
            "foil_target":
                t[f"foil_{foil}_target"],
            "foil_excess":
                sfoil - float(t[f"foil_{foil}_target"]),
            "foil_component_excess":
                int(foil_excess),

            "GT_minus_finalWrong": pair,
            "pair_target":
                t[f"pair_{foil}_target"],
            "pair_deficit":
                float(t[f"pair_{foil}_target"]) - pair,
            "pair_weak":
                int(pair_weak),

            "highlevel_strength_type":
                highlevel,
        })

    return out


def summarize_wrong_diagnosis(
    rows,
    selected_layers,
):
    out = []

    for li in selected_layers:
        for stage in STAGES:
            rr = [
                r for r in rows
                if int(r["layer"]) == li
                and r["stage"] == stage
            ]

            out.append({
                "layer": li,
                "stage": stage,
                "n_wrong": len(rr),

                "direction_correct_rate": safe_frac(
                    int(r["direction_correct"]) == 1
                    for r in rr
                ),

                "magnitude_weak_rate": safe_frac(
                    int(r["magnitude_weak"]) == 1
                    for r in rr
                ),
                "orientation_weak_rate": safe_frac(
                    int(r["orientation_weak"]) == 1
                    for r in rr
                ),
                "GT_component_weak_rate": safe_frac(
                    int(r["GT_component_weak"]) == 1
                    for r in rr
                ),
                "foil_component_excess_rate": safe_frac(
                    int(r["foil_component_excess"]) == 1
                    for r in rr
                ),
                "pair_weak_rate": safe_frac(
                    int(r["pair_weak"]) == 1
                    for r in rr
                ),

                "magnitude_and_orientation_weak_rate": safe_frac(
                    r["highlevel_strength_type"]
                    == "magnitude_and_orientation_weak"
                    for r in rr
                ),
                "magnitude_weak_only_rate": safe_frac(
                    r["highlevel_strength_type"]
                    == "magnitude_weak_only"
                    for r in rr
                ),
                "orientation_weak_only_rate": safe_frac(
                    r["highlevel_strength_type"]
                    == "orientation_weak_only"
                    for r in rr
                ),

                "mean_S_deficit": safe_mean(
                    r["S_deficit"] for r in rr
                ),
                "mean_A_deficit": safe_mean(
                    r["A_deficit"] for r in rr
                ),
                "mean_GT_deficit": safe_mean(
                    r["GT_deficit"] for r in rr
                ),
                "mean_foil_excess": safe_mean(
                    r["foil_excess"] for r in rr
                ),
                "mean_pair_deficit": safe_mean(
                    r["pair_deficit"] for r in rr
                ),
            })

    return out


def attention_transition_rows(rows):
    lookup = {
        (int(r["sid"]), int(r["layer"]), str(r["stage"])): r
        for r in rows
    }

    sids = sorted({int(r["sid"]) for r in rows})
    layers = sorted({int(r["layer"]) for r in rows})

    out = []

    for sid in sids:
        for li in layers:
            a = lookup.get((sid, li, "pre_attn"))
            b = lookup.get((sid, li, "post_attn"))
            if a is None or b is None:
                continue

            out.append({
                "sid": sid,
                "layer": li,
                "gt": a["gt"],
                "generation_group":
                    a["generation_group"],
                "generation_pred":
                    a["generation_pred"],
                "direction_correct_pre":
                    a["direction_correct"],
                "direction_correct_post":
                    b["direction_correct"],

                "delta_S_abs":
                    float(b["S_abs"]) - float(a["S_abs"]),
                "delta_S_frac":
                    float(b["S_frac"]) - float(a["S_frac"]),
                "delta_A_GT":
                    float(b["A_GT"]) - float(a["A_GT"]),
                "delta_s_GT":
                    float(b["s_GT"]) - float(a["s_GT"]),
                "delta_s_max_nonGT":
                    float(b["s_max_nonGT"])
                    - float(a["s_max_nonGT"]),
                "delta_GT_minus_maxNonGT":
                    float(b["GT_minus_maxNonGT"])
                    - float(a["GT_minus_maxNonGT"]),
            })

    return out


def summarize_attention_transition(
    rows,
    selected_layers,
    bootstrap,
    seed,
):
    rng = np.random.default_rng(seed + 500)
    out = []

    metrics = (
        "delta_S_abs",
        "delta_S_frac",
        "delta_A_GT",
        "delta_s_GT",
        "delta_s_max_nonGT",
        "delta_GT_minus_maxNonGT",
    )

    for li in selected_layers:
        rr = [r for r in rows if int(r["layer"]) == li]
        cor = [
            r for r in rr
            if r["generation_group"] == "correct"
        ]
        wr = [
            r for r in rr
            if r["generation_group"] == "wrong"
        ]

        row = {
            "layer": li,
            "n_correct": len(cor),
            "n_wrong": len(wr),
        }

        for metric in metrics:
            cv = [float(r[metric]) for r in cor]
            wv = [float(r[metric]) for r in wr]
            gap, lo, hi = bootstrap_gap(
                cv, wv, bootstrap, rng
            )

            row[f"{metric}_correct"] = safe_mean(cv)
            row[f"{metric}_wrong"] = safe_mean(wv)
            row[f"{metric}_gap_CminusW"] = gap
            row[f"{metric}_gap_ci95_lo"] = lo
            row[f"{metric}_gap_ci95_hi"] = hi

        out.append(row)

    return out


# =============================================================================
# Causal edit construction
# =============================================================================

def min_l2_two_constraint(
    v1: np.ndarray,
    v2: np.ndarray,
    c1: float,
    c2: float,
) -> np.ndarray:
    """
    Minimum L2 delta satisfying:
        delta dot v1 = c1
        delta dot v2 = c2
    using a stable pseudo-inverse in FP64.
    """
    A = np.stack(
        [
            np.asarray(v1, dtype=np.float64),
            np.asarray(v2, dtype=np.float64),
        ],
        axis=1,
    )  # [D,2]
    c = np.asarray([c1, c2], dtype=np.float64)
    gram = A.T @ A
    delta = A @ (np.linalg.pinv(gram, rcond=1e-10) @ c)
    return delta.astype(np.float32)


def clip_delta(delta: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return delta
    n = float(np.linalg.norm(delta))
    if n <= max_norm or n <= EPS:
        return delta
    return (delta * (max_norm / n)).astype(np.float32)


def random_orthogonal_delta(
    delta_norm: float,
    B: np.ndarray,
    dim: int,
    seed: int,
):
    if delta_norm <= EPS:
        return np.zeros(dim, dtype=np.float32)

    rng = np.random.default_rng(seed)
    B64 = np.asarray(B, dtype=np.float64)

    for _ in range(100):
        v = rng.standard_normal(dim)
        v = v - B64 @ (B64.T @ v)
        n = float(np.linalg.norm(v))
        if n > 1e-8:
            return (
                v / n * delta_norm
            ).astype(np.float32)

    raise RuntimeError("Could not sample random orthogonal delta.")


def build_edit(
    *,
    mode: str,
    decomposition: Mapping[str, Any],
    cb,
    gt: str,
    foil: str,
    target: Mapping[str, Any],
    max_edit_norm: float,
):
    q = np.asarray(decomposition["q"], dtype=np.float32)
    q_sp = np.asarray(
        decomposition["q_sp"],
        dtype=np.float32,
    )

    p_gt = np.asarray(cb["protos"][gt], dtype=np.float32)
    p_foil = np.asarray(cb["protos"][foil], dtype=np.float32)

    s_gt = float(q @ p_gt)
    s_foil = float(q @ p_foil)

    if mode == "magnitude_only":
        S = float(np.linalg.norm(q_sp))
        S_target = float(target["S_target"])

        if S <= EPS or S >= S_target:
            delta = np.zeros_like(q_sp)
        else:
            delta = q_sp * (S_target / S - 1.0)

    elif mode == "gt_boost_hold_foil":
        gt_def = max(
            0.0,
            float(target["GT_target"]) - s_gt,
        )
        delta = min_l2_two_constraint(
            p_gt,
            p_foil,
            gt_def,
            0.0,
        )

    elif mode == "foil_suppress_hold_gt":
        foil_excess = max(
            0.0,
            s_foil - float(target[f"foil_{foil}_target"]),
        )
        delta = min_l2_two_constraint(
            p_gt,
            p_foil,
            0.0,
            -foil_excess,
        )

    elif mode == "both":
        gt_def = max(
            0.0,
            float(target["GT_target"]) - s_gt,
        )
        foil_excess = max(
            0.0,
            s_foil - float(target[f"foil_{foil}_target"]),
        )
        delta = min_l2_two_constraint(
            p_gt,
            p_foil,
            gt_def,
            -foil_excess,
        )

    else:
        raise ValueError(f"Unknown causal mode: {mode}")

    delta = clip_delta(
        np.asarray(delta, dtype=np.float32),
        max_edit_norm,
    )

    return delta


# =============================================================================
# First-step causal evaluation
# =============================================================================

def select_causal_layers(
    text: str,
    selected_layers: Sequence[int],
    summary_rows,
    n_layers: int,
):
    t = str(text).strip().lower()

    if t == "none":
        return []
    if t == "all_selected":
        return list(selected_layers)
    if t != "auto":
        return core.parse_layers(text, n_layers)

    # Rank layers by representation gap in the diagnostic itself.
    by_layer = defaultdict(list)
    for r in summary_rows:
        by_layer[int(r["layer"])].append(r)

    scored = []
    for li, rr in by_layer.items():
        score = max(
            max(
                0.0,
                float(r["S_abs_gap_CminusW"]),
            )
            + max(
                0.0,
                float(r["GT_minus_maxNonGT_gap_CminusW"]),
            )
            for r in rr
        )
        scored.append((score, li))

    scored.sort(reverse=True)
    return sorted(
        li for score, li in scored if score > 0
    )


def choose_causal_sids(
    meta,
    eval_sids,
    max_wrong,
    max_correct,
    seed,
):
    rng = random.Random(seed + 707)

    wrong = [
        sid for sid in eval_sids
        if meta["generation"][sid]["generation_group"] == "wrong"
    ]
    correct = [
        sid for sid in eval_sids
        if meta["generation"][sid]["generation_group"] == "correct"
    ]

    rng.shuffle(wrong)
    rng.shuffle(correct)

    if max_wrong is not None:
        wrong = wrong[:max_wrong]
    if max_correct is not None:
        correct = correct[:max_correct]

    return sorted(wrong + correct)


def baseline_foil_from_scores(scores, gt):
    return max(
        [r for r in RELATIONS if r != gt],
        key=lambda r: scores[r],
    )


def restricted_pred(scores):
    return max(RELATIONS, key=lambda r: scores[r])


def causal_firststep_scan(
    *,
    model,
    processor,
    token_map,
    decoder_layers,
    causal_layers,
    causal_stages,
    causal_modes,
    causal_sids,
    records,
    meta,
    vectors,
    codebooks,
    targets,
    device,
    prompt_template,
    random_controls,
    max_edit_norm,
    seed,
):
    rows = []
    errors = []

    for sid in tqdm(
        causal_sids,
        desc="causal strength validation (first-step)",
    ):
        rec = records[sid]
        image = None

        try:
            image = Image.open(rec.image_path).convert("RGB")
            question = prompt_template.format(
                subject=rec.subject,
                reference=rec.reference,
            )
            batch, sp, rp = core.build_batch(
                processor,
                rec,
                question,
                image,
                device,
            )

            base_scores = core.score(
                model,
                batch,
                token_map,
            )
            base_pred = restricted_pred(base_scores)

            gt = meta["gt"][sid]
            group = meta["generation"][sid]["generation_group"]
            gen_pred = meta["generation"][sid]["generation_pred"]

            if (
                group == "wrong"
                and gen_pred in REL2ID
                and gen_pred != gt
            ):
                foil = gen_pred
                foil_kind = "actual_generated_wrong"
            else:
                foil = baseline_foil_from_scores(
                    base_scores,
                    gt,
                )
                foil_kind = "baseline_firststep_best_nonGT"

            base_margin = (
                base_scores[gt] - base_scores[foil]
            )
            base_correct = int(base_pred == gt)

            for li in causal_layers:
                for stage in causal_stages:
                    residual = vectors[sid][(li, stage)]
                    cb = codebooks[(li, stage)]

                    decomp = decompose_vector(
                        residual,
                        cb,
                        gt,
                    )

                    key = (li, stage, gt)
                    if key not in targets:
                        continue
                    t = targets[key]

                    subgroup = (
                        "wrong_direction_correct"
                        if (
                            group == "wrong"
                            and decomp["direction_correct"] == 1
                        )
                        else
                        "wrong_direction_incorrect"
                        if group == "wrong"
                        else "correct"
                    )

                    for mode in causal_modes:
                        delta = build_edit(
                            mode=mode,
                            decomposition=decomp,
                            cb=cb,
                            gt=gt,
                            foil=foil,
                            target=t,
                            max_edit_norm=max_edit_norm,
                        )
                        dnorm = float(np.linalg.norm(delta))
                        triggered = dnorm > EPS

                        edited = core.score_edit(
                            model,
                            batch,
                            token_map,
                            decoder_layers,
                            li,
                            stage,
                            sp,
                            rp,
                            delta,
                        )

                        pred = restricted_pred(edited)
                        margin = edited[gt] - edited[foil]

                        row = {
                            "sid": sid,
                            "layer": li,
                            "stage": stage,
                            "generation_group": group,
                            "subgroup": subgroup,
                            "gt": gt,
                            "foil": foil,
                            "foil_kind": foil_kind,
                            "mode": mode,
                            "is_random_control": 0,

                            "direction_pred":
                                decomp["direction_pred"],
                            "direction_correct":
                                decomp["direction_correct"],

                            "S_abs": decomp["S_abs"],
                            "A_GT": decomp["A_GT"],
                            "s_GT": decomp["s_GT"],
                            "s_foil":
                                decomp["proto_scores"][foil],

                            "triggered": int(triggered),
                            "edit_norm": dnorm,

                            "baseline_pred": base_pred,
                            "baseline_correct": base_correct,
                            "baseline_margin": base_margin,

                            "edited_pred": pred,
                            "edited_correct": int(pred == gt),
                            "edited_margin": margin,

                            "margin_gain":
                                margin - base_margin,
                            "W2C": int(
                                base_correct == 0
                                and pred == gt
                            ),
                            "C2W": int(
                                base_correct == 1
                                and pred != gt
                            ),
                        }
                        rows.append(row)

                        if random_controls and triggered:
                            rd = random_orthogonal_delta(
                                dnorm,
                                cb["basis"],
                                len(delta),
                                seed=(
                                    seed
                                    + sid * 100003
                                    + li * 1009
                                    + (0 if stage == "pre_attn" else 3001)
                                    + {"magnitude_only": 11, "gt_boost_hold_foil": 23, "foil_suppress_hold_gt": 37, "both": 53}[mode]
                                ),
                            )

                            random_scores = core.score_edit(
                                model,
                                batch,
                                token_map,
                                decoder_layers,
                                li,
                                stage,
                                sp,
                                rp,
                                rd,
                            )
                            rpred = restricted_pred(
                                random_scores
                            )
                            rmargin = (
                                random_scores[gt]
                                - random_scores[foil]
                            )

                            rr = dict(row)
                            rr.update({
                                "mode": f"random_{mode}",
                                "is_random_control": 1,
                                "edited_pred": rpred,
                                "edited_correct":
                                    int(rpred == gt),
                                "edited_margin": rmargin,
                                "margin_gain":
                                    rmargin - base_margin,
                                "W2C": int(
                                    base_correct == 0
                                    and rpred == gt
                                ),
                                "C2W": int(
                                    base_correct == 1
                                    and rpred != gt
                                ),
                            })
                            rows.append(rr)

            del batch

        except Exception as e:
            errors.append({
                "sid": sid,
                "error_type": type(e).__name__,
                "error": str(e),
            })
            tqdm.write(
                f"[CAUSAL ERROR sid={sid}] "
                f"{type(e).__name__}: {e}"
            )

        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return rows, errors


def summarize_causal_firststep(rows):
    out = []

    buckets = defaultdict(list)

    for r in rows:
        keys = [
            (
                int(r["layer"]),
                str(r["stage"]),
                str(r["mode"]),
                str(r["subgroup"]),
            ),
        ]
        # Also aggregate all wrong samples.
        if str(r["generation_group"]) == "wrong":
            keys.append(
                (
                    int(r["layer"]),
                    str(r["stage"]),
                    str(r["mode"]),
                    "wrong_all",
                )
            )

        for key in keys:
            buckets[key].append(r)

    for key, rr in sorted(buckets.items()):
        li, stage, mode, subgroup = key

        targeted = not mode.startswith("random_")

        # Specificity against matched random mode, if present.
        specific_gain = float("nan")
        base_mode = (
            mode[len("random_"):]
            if mode.startswith("random_")
            else mode
        )

        if targeted:
            rand_key = (
                li,
                stage,
                f"random_{base_mode}",
                subgroup,
            )
            rand_rr = buckets.get(rand_key, [])
            if rand_rr:
                specific_gain = (
                    safe_mean(r["margin_gain"] for r in rr)
                    - safe_mean(r["margin_gain"] for r in rand_rr)
                )

        out.append({
            "layer": li,
            "stage": stage,
            "mode": mode,
            "subgroup": subgroup,
            "n": len(rr),

            "trigger_rate": safe_frac(
                int(r["triggered"]) == 1
                for r in rr
            ),
            "mean_edit_norm": safe_mean(
                r["edit_norm"] for r in rr
            ),

            "baseline_acc": safe_mean(
                r["baseline_correct"] for r in rr
            ),
            "edited_acc": safe_mean(
                r["edited_correct"] for r in rr
            ),
            "acc_gain": (
                safe_mean(r["edited_correct"] for r in rr)
                - safe_mean(r["baseline_correct"] for r in rr)
            ),

            "mean_margin_gain": safe_mean(
                r["margin_gain"] for r in rr
            ),
            "specific_margin_gain_vs_random":
                specific_gain,

            "W2C": int(sum(int(r["W2C"]) for r in rr)),
            "C2W": int(sum(int(r["C2W"]) for r in rr)),
            "net": int(
                sum(int(r["W2C"]) for r in rr)
                - sum(int(r["C2W"]) for r in rr)
            ),
        })

    return out


# =============================================================================
# Full generation validation on top causal candidates
# =============================================================================

def parse_generated_relation(text: str) -> Optional[str]:
    s = text.strip().lower()
    pats = [
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("above", r"\babove\b"),
        ("below", r"\bbelow\b"),
        ("above", r"\bon top of\b"),
        ("below", r"\bunder(?:neath)?\b"),
    ]

    hits = []
    for rel, pat in pats:
        m = re.search(pat, s)
        if m:
            hits.append((m.start(), rel))

    if not hits:
        return None

    hits.sort()
    return hits[0][1]


def generate_from_batch(
    model,
    processor,
    batch,
    max_new_tokens,
):
    input_len = int(batch["input_ids"].shape[1])

    with torch.inference_mode():
        generated = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    suffix = generated[0, input_len:]
    text = processor.tokenizer.decode(
        suffix,
        skip_special_tokens=True,
    ).strip()

    del generated
    return text, parse_generated_relation(text)


def generate_with_stage_edit(
    *,
    model,
    processor,
    batch,
    decoder_layers,
    layer,
    stage,
    sp,
    rp,
    delta,
    max_new_tokens,
):
    if stage == "pre_attn":
        hook = core.PreIntervention(
            decoder_layers[layer],
            sp,
            rp,
            delta,
        )
    elif stage == "post_attn":
        hook = core.PostIntervention(
            decoder_layers[layer].self_attn,
            sp,
            rp,
            delta,
        )
    else:
        raise ValueError(stage)

    try:
        text, pred = generate_from_batch(
            model,
            processor,
            batch,
            max_new_tokens,
        )
        if not hook.applied:
            raise RuntimeError(
                f"generation hook not applied: L{layer} {stage}"
            )
        return text, pred
    finally:
        hook.close()


def choose_generation_candidates(
    causal_summary,
    top_k,
):
    if top_k <= 0:
        return []

    # Magnitude-only is the direct test of "spatial information strength".
    candidates = []

    for r in causal_summary:
        if r["mode"] != "magnitude_only":
            continue
        if r["subgroup"] != "wrong_direction_correct":
            continue

        spec = float(r["specific_margin_gain_vs_random"])
        gain = float(r["mean_margin_gain"])
        score = (
            spec if math.isfinite(spec)
            else gain
        )

        candidates.append(
            (
                score,
                int(r["layer"]),
                str(r["stage"]),
            )
        )

    candidates.sort(reverse=True)

    picked = []
    seen = set()

    for score, li, stage in candidates:
        key = (li, stage)
        if key in seen:
            continue
        seen.add(key)
        picked.append(key)
        if len(picked) >= top_k:
            break

    return picked


def generation_validation(
    *,
    model,
    processor,
    decoder_layers,
    candidates,
    modes,
    eval_sids,
    max_samples,
    records,
    meta,
    vectors,
    codebooks,
    targets,
    device,
    prompt_template,
    max_new_tokens,
    max_edit_norm,
    seed,
):
    if not candidates or max_samples <= 0:
        return [], []

    rng = random.Random(seed + 9001)

    wrong = [
        sid for sid in eval_sids
        if meta["generation"][sid]["generation_group"] == "wrong"
    ]
    correct = [
        sid for sid in eval_sids
        if meta["generation"][sid]["generation_group"] == "correct"
    ]

    rng.shuffle(wrong)
    rng.shuffle(correct)

    nw = min(len(wrong), max_samples // 2)
    nc = min(len(correct), max_samples - nw)

    selected = wrong[:nw] + correct[:nc]
    if len(selected) < max_samples:
        remain = [
            sid for sid in eval_sids
            if sid not in set(selected)
        ]
        rng.shuffle(remain)
        selected += remain[
            : max_samples - len(selected)
        ]

    selected = sorted(set(selected))

    rows = []

    for sid in tqdm(
        selected,
        desc="full generation causal validation",
    ):
        rec = records[sid]
        image = Image.open(rec.image_path).convert("RGB")

        try:
            question = prompt_template.format(
                subject=rec.subject,
                reference=rec.reference,
            )
            batch, sp, rp = core.build_batch(
                processor,
                rec,
                question,
                image,
                device,
            )

            gt = meta["gt"][sid]
            cached_group = (
                meta["generation"][sid]["generation_group"]
            )
            cached_pred = (
                meta["generation"][sid]["generation_pred"]
            )

            # Fresh baseline generation, so W2C/C2W compares exactly the same
            # runtime/model with and without intervention.
            baseline_text, baseline_pred = generate_from_batch(
                model,
                processor,
                batch,
                max_new_tokens,
            )
            baseline_correct = int(
                baseline_pred == gt
            )

            # Define foil for edit construction.
            if (
                cached_group == "wrong"
                and cached_pred in REL2ID
                and cached_pred != gt
            ):
                foil = cached_pred
            else:
                # Use Direction strongest competitor at the candidate point.
                li0, stage0 = candidates[0]
                d0 = decompose_vector(
                    vectors[sid][(li0, stage0)],
                    codebooks[(li0, stage0)],
                    gt,
                )
                foil = d0["max_nonGT_relation"]

            for li, stage in candidates:
                cb = codebooks[(li, stage)]
                decomp = decompose_vector(
                    vectors[sid][(li, stage)],
                    cb,
                    gt,
                )
                t = targets[(li, stage, gt)]

                subgroup = (
                    "wrong_direction_correct"
                    if (
                        cached_group == "wrong"
                        and decomp["direction_correct"] == 1
                    )
                    else
                    "wrong_direction_incorrect"
                    if cached_group == "wrong"
                    else "correct"
                )

                for mode in modes:
                    delta = build_edit(
                        mode=mode,
                        decomposition=decomp,
                        cb=cb,
                        gt=gt,
                        foil=foil,
                        target=t,
                        max_edit_norm=max_edit_norm,
                    )

                    text, pred = generate_with_stage_edit(
                        model=model,
                        processor=processor,
                        batch=batch,
                        decoder_layers=decoder_layers,
                        layer=li,
                        stage=stage,
                        sp=sp,
                        rp=rp,
                        delta=delta,
                        max_new_tokens=max_new_tokens,
                    )

                    edited_correct = int(pred == gt)

                    rows.append({
                        "sid": sid,
                        "layer": li,
                        "stage": stage,
                        "mode": mode,
                        "subgroup": subgroup,
                        "gt": gt,
                        "foil": foil,

                        "cached_generation_group":
                            cached_group,
                        "cached_generation_pred":
                            cached_pred,

                        "fresh_baseline_text":
                            baseline_text,
                        "fresh_baseline_pred":
                            baseline_pred,
                        "fresh_baseline_correct":
                            baseline_correct,

                        "edited_text": text,
                        "edited_pred": pred,
                        "edited_correct": edited_correct,

                        "edit_norm":
                            float(np.linalg.norm(delta)),
                        "triggered":
                            int(np.linalg.norm(delta) > EPS),

                        "W2C": int(
                            baseline_correct == 0
                            and edited_correct == 1
                        ),
                        "C2W": int(
                            baseline_correct == 1
                            and edited_correct == 0
                        ),
                    })

            del batch

        finally:
            image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    buckets = defaultdict(list)
    for r in rows:
        buckets[
            (
                int(r["layer"]),
                str(r["stage"]),
                str(r["mode"]),
                str(r["subgroup"]),
            )
        ].append(r)

        if str(r["cached_generation_group"]) == "wrong":
            buckets[
                (
                    int(r["layer"]),
                    str(r["stage"]),
                    str(r["mode"]),
                    "wrong_all",
                )
            ].append(r)

    summary = []

    for key, rr in sorted(buckets.items()):
        li, stage, mode, subgroup = key

        bacc = safe_mean(
            r["fresh_baseline_correct"] for r in rr
        )
        eacc = safe_mean(
            r["edited_correct"] for r in rr
        )

        summary.append({
            "layer": li,
            "stage": stage,
            "mode": mode,
            "subgroup": subgroup,
            "n": len(rr),

            "trigger_rate": safe_frac(
                int(r["triggered"]) == 1
                for r in rr
            ),
            "mean_edit_norm": safe_mean(
                r["edit_norm"] for r in rr
            ),

            "fresh_baseline_acc": bacc,
            "edited_generation_acc": eacc,
            "generation_acc_gain":
                eacc - bacc,

            "W2C":
                int(sum(int(r["W2C"]) for r in rr)),
            "C2W":
                int(sum(int(r["C2W"]) for r in rr)),
            "net": int(
                sum(int(r["W2C"]) for r in rr)
                - sum(int(r["C2W"]) for r in rr)
            ),
        })

    return rows, summary


# =============================================================================
# Console output
# =============================================================================

def print_decomposition_summary(rows):
    print("\n" + "=" * 178)
    print("DIRECTION EVIDENCE STRENGTH — CORRECT vs WRONG")
    print("=" * 178)
    print(
        "layer stage | Sabs cor/wr gap | Sfrac cor/wr gap | "
        "A_GT cor/wr gap | sGT cor/wr gap | maxNonGT cor/wr gap | pairGap"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} "
            f"{str(r['stage']):9s} | "
            f"{float(r['S_abs_correct']):+7.3f}/"
            f"{float(r['S_abs_wrong']):+7.3f} "
            f"{float(r['S_abs_gap_CminusW']):+7.3f} | "
            f"{float(r['S_frac_correct']):.3f}/"
            f"{float(r['S_frac_wrong']):.3f} "
            f"{float(r['S_frac_gap_CminusW']):+6.3f} | "
            f"{float(r['A_GT_correct']):+6.3f}/"
            f"{float(r['A_GT_wrong']):+6.3f} "
            f"{float(r['A_GT_gap_CminusW']):+6.3f} | "
            f"{float(r['s_GT_correct']):+7.3f}/"
            f"{float(r['s_GT_wrong']):+7.3f} "
            f"{float(r['s_GT_gap_CminusW']):+7.3f} | "
            f"{float(r['s_max_nonGT_correct']):+7.3f}/"
            f"{float(r['s_max_nonGT_wrong']):+7.3f} "
            f"{float(r['s_max_nonGT_gap_CminusW']):+7.3f} | "
            f"{float(r['GT_minus_maxNonGT_gap_CminusW']):+7.3f}"
        )


def print_wrong_diagnosis(rows):
    print("\n" + "=" * 158)
    print("GENERATION-WRONG: WHAT KIND OF SPATIAL WEAKNESS?")
    print("=" * 158)
    print(
        "layer stage | dirCorrect magWeak orientWeak GTweak foilEx pairWeak | "
        "mean Sdef Adef GTdef foilEx pairDef"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} "
            f"{str(r['stage']):9s} | "
            f"{float(r['direction_correct_rate']):.3f} "
            f"{float(r['magnitude_weak_rate']):.3f} "
            f"{float(r['orientation_weak_rate']):.3f} "
            f"{float(r['GT_component_weak_rate']):.3f} "
            f"{float(r['foil_component_excess_rate']):.3f} "
            f"{float(r['pair_weak_rate']):.3f} | "
            f"{float(r['mean_S_deficit']):+7.3f} "
            f"{float(r['mean_A_deficit']):+6.3f} "
            f"{float(r['mean_GT_deficit']):+7.3f} "
            f"{float(r['mean_foil_excess']):+7.3f} "
            f"{float(r['mean_pair_deficit']):+7.3f}"
        )


def print_attention_summary(rows):
    print("\n" + "=" * 158)
    print("ATTENTION ACCUMULATION — POST minus PRE")
    print("=" * 158)
    print(
        "layer | dS cor/wr gap | dA cor/wr gap | dsGT cor/wr gap | "
        "dMaxNonGT cor/wr gap | dPair cor/wr gap"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} | "
            f"{float(r['delta_S_abs_correct']):+7.3f}/"
            f"{float(r['delta_S_abs_wrong']):+7.3f} "
            f"{float(r['delta_S_abs_gap_CminusW']):+7.3f} | "
            f"{float(r['delta_A_GT_correct']):+6.3f}/"
            f"{float(r['delta_A_GT_wrong']):+6.3f} "
            f"{float(r['delta_A_GT_gap_CminusW']):+6.3f} | "
            f"{float(r['delta_s_GT_correct']):+7.3f}/"
            f"{float(r['delta_s_GT_wrong']):+7.3f} "
            f"{float(r['delta_s_GT_gap_CminusW']):+7.3f} | "
            f"{float(r['delta_s_max_nonGT_correct']):+7.3f}/"
            f"{float(r['delta_s_max_nonGT_wrong']):+7.3f} "
            f"{float(r['delta_s_max_nonGT_gap_CminusW']):+7.3f} | "
            f"{float(r['delta_GT_minus_maxNonGT_correct']):+7.3f}/"
            f"{float(r['delta_GT_minus_maxNonGT_wrong']):+7.3f} "
            f"{float(r['delta_GT_minus_maxNonGT_gap_CminusW']):+7.3f}"
        )


def print_causal_summary(rows):
    print("\n" + "=" * 178)
    print("CAUSAL VALIDATION — FIRST-STEP RELATION DECISION")
    print("=" * 178)
    print(
        "layer stage mode subgroup N | trigger editNorm | "
        "acc base->edit gain | marginGain specificVsRandom | W2C C2W net"
    )

    for r in rows:
        if r["mode"].startswith("random_"):
            continue
        print(
            f"L{int(r['layer']):02d} "
            f"{str(r['stage']):9s} "
            f"{str(r['mode']):25s} "
            f"{str(r['subgroup']):24s} "
            f"{int(r['n']):3d} | "
            f"{float(r['trigger_rate']):.3f} "
            f"{float(r['mean_edit_norm']):6.3f} | "
            f"{float(r['baseline_acc']):.3f}->"
            f"{float(r['edited_acc']):.3f} "
            f"{float(r['acc_gain']):+6.3f} | "
            f"{float(r['mean_margin_gain']):+8.4f} "
            f"{float(r['specific_margin_gain_vs_random']):+8.4f} | "
            f"{int(r['W2C']):3d} "
            f"{int(r['C2W']):3d} "
            f"{int(r['net']):+4d}"
        )


def print_generation_summary(rows):
    if not rows:
        return

    print("\n" + "=" * 150)
    print("FULL model.generate() VALIDATION")
    print("=" * 150)
    print(
        "layer stage mode subgroup N | trigger | "
        "generation acc base->edit gain | W2C C2W net"
    )

    for r in rows:
        print(
            f"L{int(r['layer']):02d} "
            f"{str(r['stage']):9s} "
            f"{str(r['mode']):25s} "
            f"{str(r['subgroup']):24s} "
            f"{int(r['n']):3d} | "
            f"{float(r['trigger_rate']):.3f} | "
            f"{float(r['fresh_baseline_acc']):.3f}->"
            f"{float(r['edited_generation_acc']):.3f} "
            f"{float(r['generation_acc_gain']):+6.3f} | "
            f"{int(r['W2C']):3d} "
            f"{int(r['C2W']):3d} "
            f"{int(r['net']):+4d}"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    direction_dir = Path(args.direction_dir)
    failure_dir = (
        Path(args.failure_dir)
        if args.failure_dir is not None
        else None
    )

    meta = core.load_meta(direction_dir)

    records_list, _audit = core.base.load_records(
        args.dataset,
        Path(args.data_root),
        None,
    )
    records = {
        int(r.sid): r
        for r in records_list
    }

    # Model.
    spec = core.base.SPECS[args.model]
    cls = getattr(transformers, spec.model_class)
    dtype = core.base.resolve_dtype(spec.dtype_name)

    kw: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    print(f"[model] loading {spec.repo_id} on {args.device}")

    try:
        model = cls.from_pretrained(
            spec.repo_id,
            dtype=dtype,
            **kw,
        )
    except TypeError:
        model = cls.from_pretrained(
            spec.repo_id,
            torch_dtype=dtype,
            **kw,
        )

    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
    )
    core.base.configure_processor(
        model,
        processor,
    )

    device = torch.device(args.device)

    decoder_layers, layer_path = core.decoder_layers(
        model
    )
    n_layers = len(decoder_layers)

    print(
        f"[decoder] {layer_path}; n_layers={n_layers}"
    )

    selected_layers, selection_audit = core.choose_layers(
        args.layers,
        n_layers,
        failure_dir,
        args.min_role_gap,
    )
    print("[selected problem layers]", selected_layers)

    write_csv(
        out_dir / "selected_problem_layers.csv",
        selection_audit,
    )

    # -------------------------------------------------------------------------
    # TRAIN stage codebooks.
    # -------------------------------------------------------------------------
    if args.prepost_dir is not None:
        codebooks, codebook_diag = load_codebooks_from_prepost(
            Path(args.prepost_dir),
            selected_layers,
        )
        print(
            "[codebook] reused",
            Path(args.prepost_dir) / "train_stage_vectors.npz",
        )
    else:
        train_sids = [
            sid
            for sid in meta["idx_by_sid"]
            if meta["split"].get(sid, "") == "train"
            and sid in records
            and meta["gt"].get(sid, "") in REL2ID
        ]

        if args.max_train is not None:
            rng = random.Random(args.seed)
            rng.shuffle(train_sids)
            train_sids = train_sids[: args.max_train]

        codebooks, codebook_diag = (
            core.fit_actual_point_codebooks(
                model,
                processor,
                decoder_layers,
                selected_layers,
                train_sids,
                records,
                meta,
                device,
                args.prompt_template,
                out_dir,
            )
        )

    write_csv(
        out_dir / "stage_codebook_diagnostics.csv",
        codebook_diag,
    )

    # -------------------------------------------------------------------------
    # Eval decomposition.
    # -------------------------------------------------------------------------
    eval_sids = select_eval_sids(
        meta,
        records,
        args.eval_split,
        args.max_eval,
        args.seed,
    )

    counts = Counter(
        meta["generation"][sid]["generation_group"]
        for sid in eval_sids
    )
    print(
        f"[eval] N={len(eval_sids)} groups={dict(counts)}"
    )

    decomposition_rows, vectors, errors = (
        collect_eval_decomposition(
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            selected_layers=selected_layers,
            codebooks=codebooks,
            records=records,
            meta=meta,
            eval_sids=eval_sids,
            device=device,
            prompt_template=args.prompt_template,
            out_dir=out_dir,
            save_every=args.save_every,
        )
    )

    if not decomposition_rows:
        raise RuntimeError(
            "No decomposition rows were produced."
        )

    correct_wrong_summary = summarize_correct_wrong(
        decomposition_rows,
        selected_layers,
        args.bootstrap,
        args.seed,
    )
    write_csv(
        out_dir / "correct_wrong_strength_summary.csv",
        correct_wrong_summary,
    )

    targets, target_rows = build_control_targets(
        decomposition_rows,
        args.weak_quantile,
        args.excess_quantile,
        args.target_stat,
    )
    write_csv(
        out_dir / "correct_strength_targets.csv",
        target_rows,
    )

    wrong_diag = build_wrong_diagnosis(
        decomposition_rows,
        targets,
    )
    write_csv(
        out_dir / "wrong_strength_diagnosis.csv",
        wrong_diag,
    )

    wrong_diag_summary = summarize_wrong_diagnosis(
        wrong_diag,
        selected_layers,
    )
    write_csv(
        out_dir / "wrong_strength_diagnosis_summary.csv",
        wrong_diag_summary,
    )

    trans_rows = attention_transition_rows(
        decomposition_rows
    )
    write_csv(
        out_dir / "per_sample_attention_strength_transition.csv",
        trans_rows,
    )

    trans_summary = summarize_attention_transition(
        trans_rows,
        selected_layers,
        args.bootstrap,
        args.seed,
    )
    write_csv(
        out_dir / "attention_strength_transition_summary.csv",
        trans_summary,
    )

    print_decomposition_summary(
        correct_wrong_summary
    )
    print_wrong_diagnosis(
        wrong_diag_summary
    )
    print_attention_summary(
        trans_summary
    )

    # -------------------------------------------------------------------------
    # First-step causal validation.
    # -------------------------------------------------------------------------
    causal_layers = select_causal_layers(
        args.causal_layers,
        selected_layers,
        correct_wrong_summary,
        n_layers,
    )
    causal_stages = parse_stages(
        args.causal_stages
    )
    causal_modes = parse_csv_words(
        args.causal_modes
    )

    valid_modes = {
        "magnitude_only",
        "gt_boost_hold_foil",
        "foil_suppress_hold_gt",
        "both",
    }
    bad = [
        x for x in causal_modes
        if x not in valid_modes
    ]
    if bad:
        raise ValueError(
            f"Unknown causal modes: {bad}"
        )

    print(
        f"\n[causal] layers={causal_layers}, "
        f"stages={causal_stages}, modes={causal_modes}"
    )

    causal_rows = []
    causal_summary = []

    if causal_layers:
        causal_sids = choose_causal_sids(
            meta,
            eval_sids,
            args.causal_max_wrong,
            args.causal_max_correct,
            args.seed,
        )

        token_map = core.relation_tokens(
            processor.tokenizer
        )

        causal_rows, causal_errors = causal_firststep_scan(
            model=model,
            processor=processor,
            token_map=token_map,
            decoder_layers=decoder_layers,
            causal_layers=causal_layers,
            causal_stages=causal_stages,
            causal_modes=causal_modes,
            causal_sids=causal_sids,
            records=records,
            meta=meta,
            vectors=vectors,
            codebooks=codebooks,
            targets=targets,
            device=device,
            prompt_template=args.prompt_template,
            random_controls=args.random_controls,
            max_edit_norm=args.max_edit_norm,
            seed=args.seed,
        )

        write_csv(
            out_dir / "causal_firststep_per_sample.csv",
            causal_rows,
        )
        write_csv(
            out_dir / "causal_errors.csv",
            causal_errors,
        )

        causal_summary = summarize_causal_firststep(
            causal_rows
        )
        write_csv(
            out_dir / "causal_firststep_summary.csv",
            causal_summary,
        )

        print_causal_summary(
            causal_summary
        )

    # -------------------------------------------------------------------------
    # Full generation validation on top layer-stage pairs.
    # -------------------------------------------------------------------------
    generation_candidates = choose_generation_candidates(
        causal_summary,
        args.generation_top_k,
    )

    print(
        "\n[generation candidates]",
        generation_candidates,
    )

    generation_rows = []
    generation_summary = []

    if generation_candidates:
        generation_modes = parse_csv_words(
            args.generation_modes
        )
        bad = [
            x for x in generation_modes
            if x not in valid_modes
        ]
        if bad:
            raise ValueError(
                f"Unknown generation modes: {bad}"
            )

        generation_rows, generation_summary = generation_validation(
            model=model,
            processor=processor,
            decoder_layers=decoder_layers,
            candidates=generation_candidates,
            modes=generation_modes,
            eval_sids=eval_sids,
            max_samples=args.generation_max_samples,
            records=records,
            meta=meta,
            vectors=vectors,
            codebooks=codebooks,
            targets=targets,
            device=device,
            prompt_template=args.prompt_template,
            max_new_tokens=args.max_new_tokens,
            max_edit_norm=args.max_edit_norm,
            seed=args.seed,
        )

        write_csv(
            out_dir / "generation_validation_per_sample.csv",
            generation_rows,
        )
        write_csv(
            out_dir / "generation_validation_summary.csv",
            generation_summary,
        )

        print_generation_summary(
            generation_summary
        )

    meta_out = {
        "experiment":
            "Direction evidence strength decomposition + causal validation",

        "selected_layers": selected_layers,
        "causal_layers": causal_layers,
        "causal_stages": causal_stages,
        "generation_candidates":
            generation_candidates,

        "n_eval": len(eval_sids),
        "eval_generation_groups": dict(counts),

        "definitions": {
            "S_abs":
                "norm of centered residual projected into TRAIN-derived "
                "2D spatial subspace",
            "S_frac":
                "S_abs / norm(centered residual)",
            "A_GT":
                "cosine between projected spatial component and projected "
                "GT relation prototype",
            "s_GT":
                "centered residual dot normalized GT prototype",
            "magnitude_only":
                "scale only the sample's existing spatial projection to "
                "correct-control target magnitude; orientation unchanged",
            "gt_boost_hold_foil":
                "minimum-L2 edit raising GT prototype score while holding "
                "foil prototype score fixed",
            "foil_suppress_hold_gt":
                "minimum-L2 edit reducing foil prototype score while holding "
                "GT prototype score fixed",
        },

        "strongest_information_weakness_test":
            "magnitude_only improves wrong_direction_correct samples more "
            "than norm-matched non-spatial random controls",

        "warning":
            "GT/foil coordinate interventions are oracle diagnostic tests. "
            "Magnitude-only is less oracle because it preserves each sample's "
            "own spatial direction, but the target magnitude still comes from "
            "same-GT correct controls.",
    }

    (out_dir / "summary.json").write_text(
        json.dumps(
            meta_out,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    for name in [
        "selected_problem_layers.csv",
        "stage_codebook_diagnostics.csv",
        "per_sample_direction_strength.csv",
        "eval_stage_residual_vectors.npz",
        "correct_wrong_strength_summary.csv",
        "correct_strength_targets.csv",
        "wrong_strength_diagnosis.csv",
        "wrong_strength_diagnosis_summary.csv",
        "per_sample_attention_strength_transition.csv",
        "attention_strength_transition_summary.csv",
        "causal_firststep_per_sample.csv",
        "causal_firststep_summary.csv",
        "generation_validation_per_sample.csv",
        "generation_validation_summary.csv",
        "summary.json",
    ]:
        print(" ", out_dir / name)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
