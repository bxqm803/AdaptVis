#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunPod AdaptVis: Synthetic400 -> all COCO-two 440, margin decomposition.

Question: For middle-readout-correct cases, how much of the final margin is due
  to the gray-reference baseline, the image-dependent (Real minus Gray) change,
  and the causal response to signed GT-direction spatial intervention? Can the
  final-wrong/correct contrast be explained by different baselines instead of
  weaker transfer? These are distinct measurements, not equivalent measures
  of a natural encoding-to-decision causal transfer.

Mechanism (same as AdaptVis llava16 v6): fit Real-Gray object-pair residual
centroids on Synthetic400 alone, derive each decoder layer's horizontal and
vertical axes, and apply a constant-norm intervention to subject/reference
residual positions at L20-26. No COCO labels fit the geometry or edit size.

IMPORTANT: This is an ORACLE direction-targeted *mechanism* diagnostic because
COCO GT chooses one of four frozen source directions at test time. Not a
non-oracle inference result. At alpha=0 there is no edit. The Real-Gray
logit-margin change captures ALL image-dependent changes (not just spatial
transfer); the edited-minus-clean margin captures intervention response,
NOT the amount of naturally occurring transfer.

Primary causal metric: at a single, answer-free original question prompt,
  record GT-vs-baseline-best-wrong scores for clean-real, clean-gray and each alpha.
(One-token alternative case; fall back to mean sequence logP if needed.)
This is NOT the probability of arbitrary free-form generation; separately
report real greedy generation and its last spatial relation.

Requires existing AdaptVis llava16 repo modules:
  eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6.py
  eval_spatial_to_causaltoken_mediation_v2.py  (for geometry cache subsets only)
  analyze_generated_writer_causal_trajectory_v3.py  (last relation parser)
  scan_synthetic_frozen_direction_heads_7models_v2.py

Example CPU:
  cd /workspace/AdaptVis && source .venv/bin/activate
  python eval_syn400_coco440_transfer_baseline_v2.py --phase selftest
  python eval_syn400_coco440_transfer_baseline_v2.py --phase preflight
Run on RunPod GPU:
  python -u eval_syn400_coco440_transfer_baseline_v2.py --phase run --resume --fail-fast
Then:
  python eval_syn400_coco440_transfer_baseline_v2.py --phase summarize
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

REL = ("left", "right", "above", "below")
SCRIPT = "synthetic400-qwen3b-cocotwo-transfer-baseline-v2"


def floats(s):
    xs = [float(q.strip()) for q in str(s).split(",") if q.strip()]
    if not xs or 0.0 not in xs or len(xs) != len(set(xs)) or any(not math.isfinite(x) for x in xs):
        raise ValueError("--alphas must be distinct, finite, comma separated, and contain 0")
    return sorted(xs)


def layers_from(s):
    out = set()
    for part in str(s).split(","):
        q = part.strip().lstrip("Ll")
        if not q:
            continue
        if "-" in q:
            lo, hi = map(int, q.split("-", 1))
            out.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            out.add(int(q))
    if not out:
        raise ValueError("Empty layers")
    return sorted(out)


def sha_file(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def args_parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--phase", default="preflight", choices=("preflight", "run", "summarize", "selftest"))
    p.add_argument("--model", default="qwen-3b", choices=("qwen-3b",))
    p.add_argument("--data-root", default="data")
    p.add_argument("--coco-prompt-jsonl", default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument("--synthetic-labels", default=None)
    p.add_argument("--geometry-cache", default="", help="Optional explicit Synthetic400 source NPZ. Empty = create/reuse source cache in output-dir/geometry_cache. Never fit on COCO.")
    p.add_argument("--rebuild-geometry", action="store_true", help="Ignore old geometry cache; extract Synthetic400 from scratch")
    p.add_argument("--require-source-400", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-target-440", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--layers", default="20-26")
    p.add_argument("--readout-layer", type=int, default=23, help="Frozen Synthetic400 four-direction reader, NOT a causal endpoint")
    p.add_argument("--v3-cache-dir", default="", help="Optional legacy cached reader. Empty (recommended) recomputes matching Real-Gray reader from this prompt.")
    p.add_argument("--max-eval-samples", type=int, default=0, help="0=all 440; positive number for pilot")
    p.add_argument("--alphas", default="-5,-2,0,2,5", help="Signed L2 norm of source-derived direction PER LAYER; positive=toward GT")
    p.add_argument("--gen-check", choices=("all", "endpoints", "none"), default="endpoints", help="Re-run greedy free generation for each alpha, only 0/min/max, or only 0. Scores always measured for all alpha.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", choices=("eager", "sdpa", "none"), default="eager")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--revision", default="main")
    p.add_argument("--control", default="gray")
    p.add_argument("--pool", default="mean")
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-dir", default="output/qwen3b_syn400_coco440_transfer_baseline_v2")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--print-every", type=int, default=5)
    a = p.parse_args(argv)
    a.alphas = floats(a.alphas)
    a.layers = layers_from(a.layers)
    if a.readout_layer not in a.layers:
        raise ValueError("--readout-layer must be within --layers for this protocol")
    if a.max_new_tokens < 16:
        raise ValueError("--max-new-tokens must be >=16 for natural generation")
    if a.max_eval_samples != 0:
        raise ValueError("This protocol requires ALL 440 COCO-two examples: set --max-eval-samples 0")
    if not 0 <= a.gray_value <= 255:
        raise ValueError("--gray-value must be in [0,255]")
    # Compatible v6 arguments (even when certain paths are unused for Qwen3b).
    a.source_max_samples = None
    a.target_max_samples = None
    a.overwrite = False
    a.keep_fp32 = True
    a.geometry_control = "gray"
    a.internvl_input_size = 448
    a.internvl_max_num_tiles = 12
    a.internvl_use_thumbnail = True
    return a


def plan(a):
    import eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6 as ctrl
    source = ctrl.hs.load_synthetic(a)
    target = ctrl.hs.load_target("coco", a)
    if len(source) != 400:
        raise RuntimeError(f"Expected exactly 400 Synthetic source rows, got {len(source)}")
    if len(target) != 440:
        raise RuntimeError(f"Expected exactly 440 COCO-two target rows, got {len(target)}")
    source_counts = collections.Counter(str(r['relation']) for r in source)
    if any(source_counts[rel] != 100 for rel in REL):
        raise RuntimeError(f"Synthetic400 must contain 100 per relation; got {source_counts}")
    if any(not Path(r['image_path']).is_file() for r in source + target):
        raise RuntimeError("Some Synthetic400/COCO-two images do not exist")
    if len({r['sid'] for r in source}) != len(source) or len({r['sid'] for r in target}) != len(target):
        raise RuntimeError("Duplicate source/target sid")
    for lab in REL:
        if sum(str(r['relation']) == lab for r in source) == 0:
            raise RuntimeError(f"Source lacks {lab}")
    cache = (Path(a.geometry_cache) if a.geometry_cache else
             Path(a.output_dir) / "geometry_cache" / "synthetic400_residual_real_minus_gray.npz").resolve()
    if a.rebuild_geometry and a.geometry_cache:
        raise ValueError("--rebuild-geometry cannot be combined with external --geometry-cache")
    if cache.is_file():
        with np.load(cache, allow_pickle=True) as z:
            keys = set(z.files)
            need = {"decoder_block_index", "relation_vectors", "relation"}
            if not need.issubset(keys):
                raise RuntimeError(f"Geometry cache missing keys: {need - keys}")
            x = np.asarray(z["relation_vectors"])
            cached_layers = list(map(int, z["decoder_block_index"].tolist()))
            missing = [L for L in a.layers if L not in cached_layers]
            if missing:
                raise RuntimeError(f"Geometry cache {cache} lacks layers {missing}; available={cached_layers}")
            if x.shape[0] != 400:
                raise RuntimeError(f"Existing geometry has {x.shape[0]} samples, expected 400")
            if 'sample_index' in keys:
                ids = [int(v) for v in z['sample_index'].tolist()]
                if ids != [int(r['sid']) for r in source]:
                    raise RuntimeError("Geometry cache source sample IDs differ from Synthetic400. Use a new --output-dir")
            cached_rels = [str(v) for v in z['relation'].tolist()]
            if cached_rels != [str(r['relation']) for r in source]:
                raise RuntimeError("Geometry cache relation labels/order differ from Synthetic400")
            if not a.geometry_cache and 'sample_index' not in keys:
                raise RuntimeError("Native geometry cache lacks source sample IDs; cannot validate provenance")
        print(f"[GEOMETRY] existing={cache} shape={x.shape} layers={cached_layers}", flush=True)
    else:
        print(f"[GEOMETRY] cache absent. GPU phase will build frozen source directions from 400 Synthetic images: {cache}", flush=True)
    print(f"[DATA] Synthetic={len(source)} COCO-two_requested={len(target)} alphas={a.alphas} layers={a.layers}", flush=True)
    return source, target, cache


def read_geometry(a, source, cache, model, processor, decoder, ctrl):
    if cache.is_file() and not a.rebuild_geometry:
        with np.load(cache, allow_pickle=True) as z:
            c_layers = list(map(int, z["decoder_block_index"].tolist()))
            ix = [c_layers.index(L) for L in a.layers]
            X = np.asarray(z["relation_vectors"], dtype=np.float32)[:, ix, :]
            y = np.asarray([ctrl.norm_rel(v) for v in z["relation"].tolist()], dtype=object)
        if len(y) != 400:
            raise RuntimeError("Cached source has wrong N")
        if any(int(np.count_nonzero(y == r)) != 100 for r in REL):
            raise RuntimeError("Cached source must contain 100 vectors per relation")
    else:
        # v6's builder will reuse an existing file unless overwrite is set;
        # rebuilding must really regenerate all 400 Real-Gray vectors.
        original_overwrite = a.overwrite
        try:
            a.overwrite = bool(a.rebuild_geometry)
            X, y = ctrl.build_synthetic_geometry_cache(
                model_alias="qwen-3b", model=model, processor_or_backend=processor,
                decoder_layers=decoder, layers=a.layers, source_rows=source,
                cache_path=cache, args=a,
            )
        finally:
            a.overwrite = original_overwrite
    if X.shape[0] != 400 or any(int(np.count_nonzero(y == r)) != 100 for r in REL):
        raise RuntimeError("Source geometry is not exactly Synthetic400 (100 per relation)")
    geom, stats = ctrl.fit_geometry(X, y, a.layers)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out / "synthetic400_axes.csv", index=False)
    if not all(np.max(np.abs(g['control_basis'].T @ g['control_basis'] - np.eye(2))) < 5e-4 for g in geom.values()):
        raise RuntimeError("Non-orthonormal geometry")
    return geom


def get_middle_vector(a, sid, row, model, processor, decoder, ctrl, image):
    """Use v3 observed L23 Reader if valid; else recompute REAL-GRAY pair."""
    cp = Path(a.v3_cache_dir) / "cache" / f"sid_{sid:06d}.npz" if a.v3_cache_dir else None
    mpath = Path(a.v3_cache_dir) / "metadata.json" if a.v3_cache_dir else None
    if cp is not None and cp.is_file() and mpath.is_file():
        meta = json.loads(mpath.read_text())
        if (meta.get('model') == 'qwen-3b' and meta.get('dataset') == 'coco'
                and int(meta.get('readout_layer', -1)) == a.readout_layer):
            with np.load(cp, allow_pickle=False) as z:
                if 'reader' in z.files and 'fingerprint' in z.files and str(z['fingerprint'].item()) == meta.get('fingerprint'):
                    return np.asarray(z['reader'], dtype=np.float32), 'v3_cache'
    from PIL import Image
    kw = dict(model=model, processor=processor, decoder_layers=decoder,
              question=str(row['question_text']), subject=str(row['subject']),
              reference=str(row['reference']), selected_layers=[a.readout_layer],
              device=__import__('torch').device(a.device))
    real = ctrl.hidden_pair_from_standard(image=image, **kw)
    gray_im = Image.new("RGB", image.size, (a.gray_value,) * 3)
    try:
        gray = ctrl.hidden_pair_from_standard(image=gray_im, **kw)
    finally:
        gray_im.close()
    return (real - gray)[0].astype(np.float32), 'computed'


def reader_pred(x, geom, L):
    g = geom[L]
    center = np.asarray(g['center'], np.float64)
    v = np.asarray(x, np.float64) - center
    vn = float(np.linalg.norm(v))
    scores = {}
    for rel in REL:
        direction = np.asarray(g['class_means'][rel], np.float64) - center
        denom = max(1e-12, vn * float(np.linalg.norm(direction)))
        scores[rel] = float(np.dot(v, direction) / denom)
    return max(scores, key=scores.get), scores


def oracle_coords(a, geom, gt, alpha):
    import eval_spatial_to_causaltoken_mediation_v2 as med
    return np.stack([float(alpha) * med.relation_axis_coords(geom[L], gt)
                     for L in a.layers], axis=0).astype(np.float32)


def logits_from_prompt(model, prepared, decoder, ctrl, geom, coords, one_token_map, fallback_ids):
    import torch
    with ctrl.SpatialPatch(decoder_layers=decoder, layers=list(geom), geom=geom,
                           sub_pos=prepared.sub_pos, ref_pos=prepared.ref_pos,
                           coords=torch.as_tensor(coords, device=prepared.batch['input_ids'].device, dtype=torch.float32)):
        if one_token_map is not None:
            with torch.inference_mode():
                z = model(**prepared.batch, use_cache=False, return_dict=True).logits[0,-1].float()
            return {r:float(z[list(map(int,one_token_map[r]))].max().item()) for r in REL}, 'logit_single_token_variants'
    if fallback_ids is None:
        raise RuntimeError("No valid fast one-token map and no fallback candidate ids")
    scores = ctrl.all_scores(model=model, decoder_layers=decoder, prepared=prepared,
                             candidate_ids=fallback_ids, layers=list(geom), geom=geom,
                             coords=coords, reduction='mean')
    return scores, 'mean_sequence_logp_fallback'


def get_gray_unedited_scores(a, row, img, model, processor, decoder, ctrl, geom,
                             fast, fallback):
    """Score the SAME question with a same-size gray image and zero edit.

    Returns fixed-candidate relation scores for the gray-reference condition.
    Using the identical prompt and tokenizer is essential for paired margins.
    """
    from PIL import Image
    gray_im = Image.new('RGB', img.size, (a.gray_value,) * 3)
    prep = None
    try:
        prep = ctrl.prepare_standard(model=model, processor=processor,
                                     decoder_layers=decoder, row=row,
                                     image=gray_im, args=a)
        if not prep.sub_pos or not prep.ref_pos:
            raise RuntimeError(f"SID={row['sid']}: gray prompt lost object positions")
        coords = np.zeros((len(a.layers), 2), dtype=np.float32)
        gray_scores, score_type = logits_from_prompt(
            model, prep, decoder, ctrl, geom, coords, fast, fallback)
        return gray_scores, score_type
    finally:
        if prep is not None:
            prep.close()
        gray_im.close()


def score_decomposition(real_scores, gray_scores, gt, fixed_comp):
    """Exactly paired numerical decomposition; no mechanism is assumed."""
    real_margin = float(real_scores[gt] - real_scores[fixed_comp])
    gray_margin = float(gray_scores[gt] - gray_scores[fixed_comp])
    image_delta = real_margin - gray_margin
    gray_best = float(gray_scores[gt] - max(gray_scores[r] for r in REL if r != gt))
    real_best = float(real_scores[gt] - max(real_scores[r] for r in REL if r != gt))
    if not np.isclose(gray_margin + image_delta, real_margin, rtol=1e-6, atol=1e-6):
        raise AssertionError("Real-Gray final-margin identity failed")
    return dict(real_base_margin=real_margin, gray_fixed_margin=gray_margin,
                image_dependent_delta=image_delta,
                real_best_margin=real_best, gray_best_margin=gray_best,
                gray_fixed_competitor=fixed_comp)


def generated(model, prepared, decoder, ctrl, geom, coords, max_new_tokens):
    import torch
    import analyze_generated_writer_causal_trajectory_v3 as parser
    prompt_n = int(prepared.raw_prompt_len)
    c = torch.as_tensor(coords, device=prepared.batch['input_ids'].device, dtype=torch.float32)
    with ctrl.SpatialPatch(decoder_layers=decoder, layers=list(geom), geom=geom,
                           sub_pos=prepared.sub_pos, ref_pos=prepared.ref_pos, coords=c):
        with torch.inference_mode():
            seq = model.generate(**prepared.batch, do_sample=False, num_beams=1,
                                 max_new_tokens=max_new_tokens, use_cache=True)
    ids = seq[0,prompt_n:].detach().cpu().tolist()
    n,eos,cap = parser.generation_completion_info(ids, model, prepared.tokenizer, max_new_tokens)
    text = prepared.tokenizer.decode(ids, skip_special_tokens=True).strip()
    # Never interpret a truncated generated explanation as a completed relation.
    return (parser.parse_final_relation(text) if not cap else None), text, n, bool(cap)


def gen_alphas(a):
    if a.gen_check == 'none':
        return {0.0}
    if a.gen_check == 'endpoints':
        return {min(a.alphas), 0.0, max(a.alphas)}
    return set(a.alphas)


def empty_or_float(v):
    return None if v is None else float(v)


def run(a, ctx):
    import torch
    from PIL import Image
    import eval_multimodel_targetselected_head_spatial_control_genflip_trust_v6 as ctrl
    source,target,cache = ctx
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for --phase run")
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'samples').mkdir(exist_ok=True)
    if a.rebuild_geometry and any((out/'samples').glob('sid_*.json')):
        raise RuntimeError('Cannot rebuild Synthetic400 geometry while saved COCO results exist. Use a NEW --output-dir.')
    # Fingerprint prevents incorrectly resuming files from another experiment.
    setting = dict(script=SCRIPT, model=a.model, layers=a.layers, readout_layer=a.readout_layer,
                    gray_logit_decomposition=True, reader_cache_dir=str(a.v3_cache_dir),
                   n_source=len(source), n_target=len(target), source_sids=[int(r['sid']) for r in source],
                   target_sids=[int(r['sid']) for r in target], alphas=a.alphas, gen_check=a.gen_check,
                   gray_value=a.gray_value, max_new_tokens=a.max_new_tokens, attn_impl=a.attn_impl,
                   geometry_cache_path=str(cache), seed=a.seed,
                   prompt_hash=sha_file(a.coco_prompt_jsonl),
                   source_hash=sha_file(a.synthetic_labels or str(Path(a.synthetic_dir)/'labels.jsonl')))
    fp = hashlib.sha256(json.dumps(setting, sort_keys=True).encode()).hexdigest()
    meta_path = out/'settings.json'
    if meta_path.is_file():
        prev = json.loads(meta_path.read_text())
        if prev.get('fingerprint') != fp:
            raise RuntimeError('Output dir contains DIFFERENT settings; choose a new --output-dir')
        if not a.resume:
            raise RuntimeError('Output exists; pass --resume')
    else:
        atomic_json(meta_path, {**setting,'fingerprint':fp})
    torch.manual_seed(a.seed)
    model, processor, decoder, *_ = ctrl.hs.load_model_bundle(a.model, a)
    for p in model.parameters():
        p.requires_grad_(False)
    if max(a.layers) >= len(decoder):
        raise ValueError(f"Model has only {len(decoder)} layers")
    geom = read_geometry(a, source, cache, model, processor, decoder, ctrl)
    hidden_size = getattr(getattr(model.config, 'text_config', model.config), 'hidden_size', None)
    if hidden_size is not None and any(g['control_basis'].shape[0] != int(hidden_size) for g in geom.values()):
        raise RuntimeError(f'Geometry cache hidden dimension mismatch: expected {hidden_size}; check Qwen3B source cache')
    if a.readout_layer not in geom:
        raise RuntimeError('Readout layer missing in geometry')
    fast, tokmeta = ctrl.encode_fast_relation_tokens(processor.tokenizer, 'coco')
    fallback = ctrl.encode_candidates(processor.tokenizer, 'coco') if fast is None else None
    atomic_json(out/'token_score_protocol.json', dict(mode='one_token_max_variants' if fast else 'mean_sequence_logp', detail=tokmeta))
    gen_set = gen_alphas(a)
    # Preserve the user's previous 352-sample COCO-only middle-readout split
    # as a SECONDARY subgroup; it never fits/source-routes any intervention.
    old_middle = {}
    old_held = Path(a.v3_cache_dir) / 'heldout_samples.csv' if a.v3_cache_dir else None
    if old_held is not None and old_held.is_file():
        old_h = pd.read_csv(old_held)
        if {'sid','middle_correct'}.issubset(old_h.columns):
            old_middle = {int(r.sid): str(r.middle_correct).lower() in ('1','true','yes')
                          for r in old_h.itertuples()}
            print(f'[V3 MIDDLE] Found previous heldout flags for {len(old_middle)} SIDs; reporting separately',flush=True)
    done = skipped = failed = 0
    for i, row in enumerate(target,1):
        sid = int(row['sid'])
        f = out/'samples'/f'sid_{sid:06d}.json'
        if f.is_file():
            data = json.loads(f.read_text())
            if data.get('fingerprint') != fp or len(data.get('alpha_rows', [])) != len(a.alphas):
                raise RuntimeError(f'Invalid resume artifact: {f}')
            skipped += 1
            continue
        prepared = None
        img = None
        try:
            img = Image.open(row['image_path']).convert('RGB')
            prepared = ctrl.prepare_standard(model=model, processor=processor, decoder_layers=decoder,
                                             row=row, image=img, args=a)
            if not prepared.sub_pos or not prepared.ref_pos:
                raise RuntimeError(f'SID={sid} missing subject/reference token positions')
            gt = str(row['relation'])
            middle, middle_origin = get_middle_vector(a,sid,row,model,processor,decoder,ctrl,img)
            mr, middle_scores = reader_pred(middle,geom,a.readout_layer)
            scores_by_alpha = {}
            alpha_rows = []
            for alpha in a.alphas:
                coords = oracle_coords(a,geom,gt,alpha)
                logits, typ = logits_from_prompt(model,prepared,decoder,ctrl,geom,coords,fast,fallback)
                scores_by_alpha[alpha] = logits
                rec = dict(alpha=float(alpha), score_type=typ,
                           winner=max(logits,key=logits.get),
                           **{f'score_{rel}':float(logits[rel]) for rel in REL})
                if alpha in gen_set:
                    pred, text, nnew, cap = generated(model,prepared,decoder,ctrl,geom,coords,a.max_new_tokens)
                    rec.update(generated_relation=pred,generated_text=text,generated_n_tokens=nnew,generation_hit_cap=cap)
                else:
                    rec.update(generated_relation=None,generated_text=None,generated_n_tokens=None,generation_hit_cap=None)
                alpha_rows.append(rec)

            # Gray-reference final scores measured with the same no-answer prompt.
            gray_scores, gray_score_type = get_gray_unedited_scores(
                a, row, img, model, processor, decoder, ctrl, geom, fast, fallback)
            if gray_score_type != typ:
                raise RuntimeError(f"Score type mismatch Real={typ}, Gray={gray_score_type}")
            base = scores_by_alpha[0.0]
            neg_rel = max((r for r in REL if r != gt), key=lambda r:base[r])
            decomposition = score_decomposition(base, gray_scores, gt, neg_rel)
            # Use a SINGLE FIXED competitor for all alpha so causal margins are paired.
            for rec in alpha_rows:
                alpha = float(rec['alpha'])
                this = scores_by_alpha[alpha]
                rec['gt_minus_fixed_comp'] = float(this[gt]-this[neg_rel])
                rec['gt_minus_best_other'] = float(this[gt]-max(this[r] for r in REL if r!=gt))
                rec['gt_minus_fixed_change'] = float((this[gt]-this[neg_rel])-(base[gt]-base[neg_rel]))
                rec['gt_logit_change'] = float(this[gt]-base[gt])
                rec['margin_gt_baseline_comp'] = float(this[gt]-this[neg_rel])
            b_gen = next(q for q in alpha_rows if q['alpha']==0)['generated_relation']
            b_cap = next(q for q in alpha_rows if q['alpha']==0)['generation_hit_cap']
            if b_cap or b_gen not in REL:
                print(f'[WARN] SID {sid}: baseline generation not complete/parseable; excluded from correctness groups',flush=True)
            output = dict(fingerprint=fp,sid=sid,gt=gt,subject=row['subject'],reference=row['reference'],
                           image_path=str(row['image_path']),**decomposition,
                           gray_relation_scores=gray_scores,real_relation_scores=base,
                           gray_score_type=gray_score_type,middle_pred=mr,middle_correct=bool(mr==gt),
                          middle_origin=middle_origin,middle_scores=middle_scores,
                          middle_v3_correct=old_middle.get(sid),
                          baseline_gen=b_gen,baseline_correct=(bool(b_gen==gt) if b_gen in REL else None),
                          baseline_competitor=neg_rel,alpha_rows=alpha_rows)
            atomic_json(f,output)
            done += 1
            if a.print_every and (done%a.print_every==0 or i==len(target)):
                print(f'[SAMPLE] {i}/{len(target)} done={done} cached={skipped} failures={failed} sid={sid} '
                      f'GT={gt} base_gen={b_gen} readout={mr} '
                      f'dM(+max)={alpha_rows[-1]["gt_minus_fixed_change"]:+.4f}',flush=True)
        except Exception as exc:
            failed += 1
            with (out/'errors.jsonl').open('a',encoding='utf-8') as ef:
                ef.write(json.dumps(dict(sid=sid,error=f'{type(exc).__name__}: {exc}',
                                         traceback=traceback.format_exc().splitlines()[-25:]))+'\n')
            print(f'[ERROR] sid={sid}: {type(exc).__name__}: {exc}',flush=True)
            if a.fail_fast:
                raise
        finally:
            if prepared is not None:prepared.close()
            if img is not None:img.close()
    print(f'[RUN END] success_new={done} reused={skipped} errors={failed} requested={len(target)}',flush=True)
    summarize(a)
    if done + skipped != 440 or failed:
        raise RuntimeError(f'COCO-two run incomplete: completed={done + skipped}/440, failures={failed}. Rerun with --resume.')


def mean_ci(x, seed=17, nboot=1000):
    q = np.asarray(list(x),dtype=np.float64)
    q = q[np.isfinite(q)]
    if not len(q):return float('nan'),float('nan'),float('nan')
    rng = np.random.default_rng(seed)
    means = np.mean(q[rng.integers(len(q),size=(nboot,len(q)))],axis=1)
    return float(q.mean()),float(np.percentile(means,2.5)),float(np.percentile(means,97.5))


def summarize(a):
    out = Path(a.output_dir)
    files = sorted((out/'samples').glob('sid_*.json')) if (out/'samples').exists() else []
    if not files:
        print('[SUMMARY] no sample files',flush=True)
        return
    known_settings = json.loads((out/'settings.json').read_text())
    expected=set(map(int,known_settings['target_sids']))
    per=[]
    for p in files:
        d = json.loads(p.read_text())
        if d.get('fingerprint') != known_settings['fingerprint']:
            raise RuntimeError(f'Foreign sample file: {p}')
        if int(d['sid']) not in expected:
            raise RuntimeError(f'Unexpected sid {d["sid"]}')
        for r in d['alpha_rows']:
            record={k:v for k,v in d.items() if k not in ('middle_scores','alpha_rows')}
            record.update({f'middle_score_{rel}':d['middle_scores'][rel] for rel in REL})
            record.update(r)
            record['generated_correct'] = (None if r['generated_relation'] not in REL else int(r['generated_relation']==d['gt']))
            record['w2c'] = int(d['baseline_correct'] is False and r['generated_relation']==d['gt'])
            record['c2w'] = int(d['baseline_correct'] is True and r['generated_relation'] in REL and r['generated_relation']!=d['gt'])
            record['baseline_correct'] = d['baseline_correct']
            record['middle_gt_margin'] = float(d['middle_scores'][d['gt']] - max(d['middle_scores'][other] for other in REL if other!=d['gt']))
            per.append(record)
    df = pd.DataFrame(per)
    if df.duplicated(['sid','alpha']).any():
        raise RuntimeError('Duplicate SID / alpha')
    if not set(df.alpha.unique()) == set(a.alphas):
        print('[WARN] recorded alpha grid differs from current --alphas; summary uses recorded rows',flush=True)
    df.to_csv(out/'per_sample_alpha.csv',index=False)
    info = dict(n_expected=len(expected),n_completed=int(df.sid.nunique()),n_missing=len(expected-set(df.sid)),
                missing_sids=sorted(expected-set(df.sid)),source_n=known_settings['n_source'],
                alpha_grid=known_settings['alphas'], middle_reader='Synthetic400 source-only L23 centered cosine, independent of GT',
                target_routing='Oracle GT for diagnostic, never claim non-oracle method',
                 baseline_decomposition='real = gray + (real-gray), fixed baseline-real competitor; not isolated spatial transfer')
    atomic_json(out/'coverage.json',info)
    print(f'[COVERAGE] {info["n_completed"]}/{info["n_expected"]} COCO-two samples',flush=True)
    if len(expected - set(df.sid)) > 0:
        print('[INCOMPLETE] Some COCO-two samples not processed yet. Resume before using summary.', flush=True)
    # Relation-stratified results guard against different GT composition in Correct/Wrong groups.
    rel_summary=[]
    for (rel, middle, correct, alpha),g in df[df.baseline_correct.notna()].groupby(['gt','middle_correct','baseline_correct','alpha'],dropna=False):
        est,low,high=mean_ci(g.gt_minus_fixed_change,seed=a.seed)
        rel_summary.append(dict(gt=rel,middle_correct=bool(middle),baseline_correct=bool(correct),
                                alpha=float(alpha),n=int(g.sid.nunique()),
                                mean_margin_change=est,ci95_low=low,ci95_high=high,
                                positive_delta_frac=float((g.gt_minus_fixed_change>1e-6).mean()),
                                w2c=int(g.w2c.sum()),c2w=int(g.c2w.sum())))
    pd.DataFrame(rel_summary).to_csv(out/'summary_by_relation.csv',index=False)
    if 'middle_v3_correct' not in df.columns:
        df['middle_v3_correct'] = None
    summary=[]
    for group_name,mask in [
        ('all',pd.Series(True,index=df.index)),
        ('final_correct',df.baseline_correct.eq(True)),
        ('final_wrong',df.baseline_correct.eq(False)),
        ('middle_correct',df.middle_correct.eq(True)),
        ('middle_correct_final_correct',df.middle_correct.eq(True)&df.baseline_correct.eq(True)),
        ('middle_correct_final_wrong',df.middle_correct.eq(True)&df.baseline_correct.eq(False)),
        ('middle_wrong_final_correct',df.middle_correct.eq(False)&df.baseline_correct.eq(True)),
        ('middle_wrong_final_wrong',df.middle_correct.eq(False)&df.baseline_correct.eq(False)),
        ('v3_middle_correct_final_correct',df.middle_v3_correct.eq(True)&df.baseline_correct.eq(True)),
        ('v3_middle_correct_final_wrong',df.middle_v3_correct.eq(True)&df.baseline_correct.eq(False)),
    ]:
        data=df.loc[mask]
        for alpha,g in data.groupby('alpha',sort=True):
            good=g['gt_minus_fixed_change'].astype(float)
            est,low,high=mean_ci(good,seed=a.seed)
            free=g[g['generated_correct'].notna()]
            row=dict(group=group_name,alpha=float(alpha),n_samples=int(g.sid.nunique()),
                     mean_base_margin=float((g.gt_minus_fixed_comp-g.gt_minus_fixed_change).mean()),
                     mean_margin=float(g.gt_minus_fixed_comp.mean()),
                     mean_margin_change=est,margin_change_ci95_low=low,margin_change_ci95_high=high,
                     positive_delta_frac=float((good>1e-6).mean()),
                     negative_delta_frac=float((good< -1e-6).mean()),
                     mean_gt_logit_change=float(g.gt_logit_change.mean()),
                     gt_is_fourway_winner_frac=float(g.apply(lambda r:r['winner']==r['gt'],axis=1).mean()),
                     frac_gt_margin_over_zero=float((g.gt_minus_best_other>0).mean()),
                     generation_checked=int(len(free)),
                     gen_correct_frac=float(free.generated_correct.mean()) if len(free) else None,
                     w2c=int(g.w2c.sum()),c2w=int(g.c2w.sum()))
            summary.append(row)
    res=pd.DataFrame(summary).sort_values(['group','alpha'])
    res.to_csv(out/'summary_by_alpha.csv',index=False)

    # Baseline / natural image-dependent / edited-perturbation decomposition.
    # De-duplicate alpha rows so each sample contributes ONCE to baseline stats.
    at0_decomp = df[df.alpha == 0.0].drop_duplicates('sid').copy()
    decomp_groups = [
        ('all', at0_decomp),
        ('middle_correct_final_correct', at0_decomp[(at0_decomp.middle_correct == True) & (at0_decomp.baseline_correct == True)]),
        ('middle_correct_final_wrong', at0_decomp[(at0_decomp.middle_correct == True) & (at0_decomp.baseline_correct == False)]),
    ]
    decomp_rows = []
    for group_name, grp in decomp_groups:
        if grp.empty:
            continue
        for field in ('gray_fixed_margin', 'image_dependent_delta', 'real_base_margin'):
            mean, lo, hi = mean_ci(grp[field], seed=a.seed)
            decomp_rows.append(dict(group=group_name, quantity=field, n=len(grp),
                                    mean=mean, ci95_low=lo, ci95_high=hi,
                                    median=float(grp[field].median()),
                                    positive_frac=float((grp[field] > 0).mean()),
                                    negative_frac=float((grp[field] < 0).mean())))
    pd.DataFrame(decomp_rows).to_csv(out/'summary_baseline_decomposition.csv', index=False)
    print('\n[BASELINE / NATURAL IMAGE-DEPENDENT DELTA / CLEAN REAL MARGIN]', flush=True)
    for group_name, grp in decomp_groups:
        if grp.empty: continue
        print(f'{group_name}: N={len(grp)} '
              f'gray={grp.gray_fixed_margin.mean():+.4f}, '
              f'real-gray={grp.image_dependent_delta.mean():+.4f}, '
              f'real={grp.real_base_margin.mean():+.4f}', flush=True)
    print('\n[MIDDLE CORRECT / FINAL CORRECT vs WRONG: GT-ORACLE SOURCE-FROZEN CONTROL]',flush=True)
    mid=res[res.group.isin(('middle_correct_final_correct','middle_correct_final_wrong'))]
    print(mid[['group','alpha','n_samples','mean_base_margin','mean_margin_change','margin_change_ci95_low','margin_change_ci95_high','positive_delta_frac','w2c','c2w','generation_checked']].to_string(index=False,float_format=lambda v:f'{v:.4f}'),flush=True)
    # Paired strength dependence per sample: slope through zero and threshold crossing.
    sub=df[['sid','gt','baseline_correct','middle_correct','alpha','gt_minus_fixed_change','gt_minus_fixed_comp','generated_relation','generated_correct']].copy()
    x=[]
    for sid,g in sub.groupby('sid'):
        z=g[g.alpha>=0].sort_values('alpha')
        vals=z.gt_minus_fixed_change.to_numpy(float)
        aa=z.alpha.to_numpy(float)
        slope=float((aa@vals)/(aa@aa)) if (aa@aa)>0 else None
        clean_margin = float(z.loc[z.alpha == 0, 'gt_minus_fixed_comp'].iloc[0])
        first_cross = (next((float(r.alpha) for r in z.itertuples()
                             if r.alpha > 0 and r.gt_minus_fixed_comp > 0), None)
                       if clean_margin <= 0 else None)
        x.append(dict(sid=int(sid),gt=str(g['gt'].iloc[0]),baseline_correct=g.baseline_correct.iloc[0],
                      middle_correct=g.middle_correct.iloc[0],positive_strength_slope=slope,
                      clean_margin_already_positive=bool(clean_margin > 0),
                      first_negative_to_positive_margin_alpha=first_cross))
    pd.DataFrame(x).to_csv(out/'per_sample_strength_slope.csv',index=False)
    # Matched analysis: for each MIDDLE-CORRECT/FINAL-WRONG sample select the
    # closest unused FINAL-CORRECT sample with identical GT relation, matching
    # on SOURCE-only reader confidence (no outcome-derived score matching).
    at0=df[df.alpha==0].drop_duplicates('sid')
    wrong=at0[(at0.middle_correct==True)&(at0.baseline_correct==False)].sort_values('sid')
    corr=at0[(at0.middle_correct==True)&(at0.baseline_correct==True)].sort_values('sid')
    used=set();matched=[]
    for w in wrong.itertuples():
        pool=corr[(corr['gt']==w.gt)&(~corr.sid.isin(used))]
        if pool.empty:continue
        candidates=pool.assign(distance=(pool.middle_gt_margin-float(w.middle_gt_margin)).abs())
        chosen=candidates.sort_values(['distance','sid']).iloc[0]
        used.add(int(chosen.sid))
        matched.append(dict(wrong_sid=int(w.sid),correct_sid=int(chosen.sid),gt=w.gt,
                            wrong_middle_margin=float(w.middle_gt_margin),
                            correct_middle_margin=float(chosen.middle_gt_margin),
                             abs_reader_margin_difference=float(chosen.distance),
                             wrong_gray_margin=float(w.gray_fixed_margin),
                             correct_gray_margin=float(chosen.gray_fixed_margin),
                             wrong_image_delta=float(w.image_dependent_delta),
                             correct_image_delta=float(chosen.image_dependent_delta),
                             wrong_real_margin=float(w.real_base_margin),
                             correct_real_margin=float(chosen.real_base_margin)))
    pd.DataFrame(matched).to_csv(out/'matched_pairs.csv',index=False)
    matched_summary=[]
    if matched:
        for alpha in sorted(df.alpha.unique()):
            left=df[df.alpha==alpha].set_index('sid')
            wd=np.array([float(left.loc[m['wrong_sid'],'gt_minus_fixed_change']) for m in matched])
            cd=np.array([float(left.loc[m['correct_sid'],'gt_minus_fixed_change']) for m in matched])
            diff=cd-wd
            est,low,high=mean_ci(diff,seed=a.seed)
            matched_summary.append(dict(alpha=float(alpha),n_pairs=len(matched),
                                        wrong_mean_change=float(wd.mean()),correct_mean_change=float(cd.mean()),
                                        correct_minus_wrong=est,diff_ci95_low=low,diff_ci95_high=high))
    pd.DataFrame(matched_summary).to_csv(out/'matched_effect_by_alpha.csv',index=False)
    matched_decomp = []
    baseline_by_sid = at0.set_index('sid')
    for field, pfx in [('gray_fixed_margin','gray_margin'),
                       ('image_dependent_delta','image_delta'),
                       ('real_base_margin','real_margin')]:
        if not matched: continue
        ww = np.array([float(baseline_by_sid.loc[m['wrong_sid'], field]) for m in matched])
        cc = np.array([float(baseline_by_sid.loc[m['correct_sid'], field]) for m in matched])
        mean, lo, hi = mean_ci(cc - ww, seed=a.seed)
        matched_decomp.append(dict(quantity=pfx,n_pairs=len(matched),
                                   wrong_mean=float(ww.mean()), correct_mean=float(cc.mean()),
                                   correct_minus_wrong=mean, diff_ci95_low=lo, diff_ci95_high=hi))
    pd.DataFrame(matched_decomp, columns=['quantity', 'n_pairs', 'wrong_mean', 'correct_mean', 'correct_minus_wrong', 'diff_ci95_low', 'diff_ci95_high']).to_csv(out/'matched_baseline_decomposition.csv', index=False)

    print(f'[MATCHED] middle-correct wrong={len(wrong)} paired={len(matched)} '
          f'correct_pool={len(corr)}; exact GT / nearest frozen-reader margin, without replacement',flush=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7.2,4.5))
        for group in ('middle_correct_final_correct','middle_correct_final_wrong'):
            g=mid[mid.group==group].sort_values('alpha')
            if not g.empty:
                ax.plot(g.alpha,g.mean_margin_change,marker='o',label=f'{group} (N={int(g.n_samples.iloc[0])})')
                ax.fill_between(g.alpha.to_numpy(float),g.margin_change_ci95_low.to_numpy(float),g.margin_change_ci95_high.to_numpy(float),alpha=.15)
        ax.axhline(0,color='gray',linewidth=.8)
        ax.axvline(0,color='gray',linewidth=.8)
        ax.set(xlabel='Signed Synthetic400 spatial control magnitude per layer',ylabel='GT vs baseline competitor logit-margin change',title='Source-frozen spatial amplification: Qwen3B COCO-two')
        ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/'strength_response.png',dpi=170);plt.close(fig)
    except Exception as e:
        print('[PLOT] skipped:',type(e).__name__,e,flush=True)
    print('[FILES] per_sample_alpha.csv summary_by_alpha.csv summary_baseline_decomposition.csv matched_baseline_decomposition.csv summary_by_relation.csv matched_pairs.csv matched_effect_by_alpha.csv per_sample_strength_slope.csv coverage.json strength_response.png',flush=True)
    print('[INTERPRETATION] alpha>0 is GT-oracle strength of Synthetic400 direction. Endpoints are free generation; '
          'logit margins are from ORIGINAL prompt (not leaked generated prefix).'
          '\n[CAUTION] Real-Gray contains all image-dependent effects; edited response does NOT measure natural transfer. '
          'GT-selected direction is ORACLE diagnostic, not deployment inference.',flush=True)


def selftest():
    assert floats('-5,-2,0,2,5')==[-5.,-2.,0.,2.,5.]
    assert layers_from('20-26')==list(range(20,27))
    assert layers_from('20,21,24-26')==[20,21,24,25,26]
    assert gen_alphas(argparse.Namespace(gen_check='endpoints',alphas=[-5.,-2.,0.,2.,5.]))=={-5.,0.,5.}
    x=np.asarray([-1.,2.,3.,-4.]);y=np.asarray([-.5,2.,2.5,-4.])
    before=x[0]-x[2];after=y[0]-y[2]
    assert np.isclose(after-before,1.)
    # Test source-only reader arithmetic is independent of target GT.
    d=np.eye(4,dtype=np.float32)
    geom={23:dict(center=np.zeros(4),class_means={r:d[i] for i,r in enumerate(REL)})}
    pr,s=reader_pred(d[2],geom,23)
    assert pr=='above' and s['above']>s['below']
    example = score_decomposition(
        {'left': 3., 'right': 1., 'above': -2., 'below': -3.},
        {'left': 0.5, 'right': 2.5, 'above': -2., 'below': -3.},
        'left', 'right')
    assert np.isclose(example['real_base_margin'], 2.0)
    assert np.isclose(example['gray_fixed_margin'], -2.0)
    assert np.isclose(example['image_dependent_delta'], 4.0)
    print('[SELFTEST PASS] grids, GT-fixed margins, exact Real=Gray+Image decomposition, source-only reader',flush=True)


def main():
    a=args_parse()
    if a.phase=='selftest':
        selftest();return
    if a.phase=='summarize':
        summarize(a);return
    ctx=plan(a)
    if a.phase=='preflight':
        return
    run(a,ctx)


if __name__=='__main__':main()
