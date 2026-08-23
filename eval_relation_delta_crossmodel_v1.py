#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-model diagnostic for consensus-guided relation-specific last-token repair.

This script reuses the relation-specific last-token repair from
eval_consensus_spatial_redeployment_v1.py, but uses model-depth-relative windows
so Qwen, LLaVA, InternVL, etc. can be compared fairly.

It evaluates:

  A) a single-layer sweep (default: every decoder layer)
  B) model-relative multi-layer windows:
       last_4       = final 4 decoder blocks
       last_6       = final 6 decoder blocks
       last_8       = final 8 decoder blocks
       last_quarter = final 25% of decoder blocks
       last_half    = final 50% of decoder blocks
       all_layers   = every decoder block

For every target layer l, the repair is layer-specific:

  residual_l      = h_img(l,last) - h_noimg(l,last)
  current_coord   = <residual_l, u_l(relation)>
  delta_l         = alpha * (target_l(relation) - current_coord) * u_l(relation)
  h'_l,last       = h_l,last + delta_l

u_l and target_l are fit from TRAIN only, exactly as in the prior working
redeployment script. The guide relation comes from the existing multi-head
Image-NoImage Direction consensus. Test GT is never used to choose whether,
where, or in which direction to repair; it is used only for final scoring.

Default policy is conflict_only because it exactly matches the previous +4.89pp
result and remains GT-free: repair iff high-confidence internal consensus and
baseline generation disagree. --policy all_covered is also supported.

Interpretation:
  * If single L26 is positive but neighboring single layers are negative and
    windows get worse -> layer placement / repeated edits are the problem.
  * If several single layers and/or relation-delta windows are positive while
    direct Direction-head-mean windows were negative -> head-write averaging is
    the problem.
  * If all relation-delta windows are also poor -> multi-layer preservation via
    this coordinate correction is not a good mechanism.

Dependency:
  Keep eval_consensus_spatial_redeployment_v1.py in the same repository root.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

import extract_two_object_relation_states as base
import analyze_coco_head_object_residual_direction_probe_v1 as direction_base
import validate_grounded_spatial_consensus_v1 as feas
import eval_consensus_spatial_redeployment_v1 as rd

SCRIPT_VERSION = "eval-relation-delta-crossmodel-v1"
RELATIONS = ("left", "right", "above", "below")
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--feasibility-dir", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2", "none"])
    p.add_argument(
        "--prompt-template",
        default=(
            "Determine the spatial relation of the {subject} to the {reference} "
            "in the image. Answer with left, right, above, or below."
        ),
    )
    p.add_argument("--baseline-generation-jsonl", default=None)
    p.add_argument(
        "--single-layers",
        default="all",
        help=(
            "Single-layer diagnostic sweep. Supports 'all', 'last8', 'last6', "
            "or explicit ranges such as 18-31 / 0,5,10."
        ),
    )
    p.add_argument(
        "--skip-single-sweep", action="store_true",
        help="Skip single-layer sweep and run only the relative multi-layer windows."
    )
    p.add_argument("--alpha", type=float, default=1.0, help="Fixed alpha; no DEV tuning.")
    p.add_argument(
        "--max-delta-ratio",
        type=float,
        default=0.15,
        help="Match the previous successful redeployment experiment; <=0 disables clipping.",
    )
    p.add_argument(
        "--policy",
        choices=["conflict_only", "all_covered"],
        default="conflict_only",
        help="GT-free trigger policy used for all variants.",
    )
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-test-samples", type=int, default=None, help="Smoke test only")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--reuse-noimage-cache", action="store_true")
    return p.parse_args()


def parse_layers(text: str) -> List[int]:
    out: List[int] = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(piece))
    return list(dict.fromkeys(out))


def resolve_layer_spec(text: str, n_layers: int) -> List[int]:
    raw = str(text).strip().lower()
    if raw == "all":
        return list(range(n_layers))
    if raw.startswith("last") and raw[4:].isdigit():
        k = max(1, int(raw[4:]))
        return list(range(max(0, n_layers-k), n_layers))
    return [L for L in parse_layers(text) if 0 <= L < n_layers]


def relative_windows(n_layers: int) -> Dict[str, List[int]]:
    def last_k(k: int) -> List[int]:
        return list(range(max(0, n_layers-k), n_layers))
    q = max(1, int(math.ceil(n_layers * 0.25)))
    h = max(1, int(math.ceil(n_layers * 0.50)))
    windows = {
        "last_4": last_k(4),
        "last_6": last_k(6),
        "last_8": last_k(8),
        "last_quarter": last_k(q),
        "last_half": last_k(h),
        "all_layers": list(range(n_layers)),
    }
    # Remove exact duplicate layer sets while preserving the first, more interpretable name.
    out: Dict[str, List[int]] = {}
    seen = set()
    for name, layers in windows.items():
        key = tuple(layers)
        if key not in seen:
            seen.add(key)
            out[name] = layers
    return out


def safe_mean(xs: Iterable[float]) -> float:
    ys = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(ys)) if ys else float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def subset_sids(sids: Sequence[int], limit: Optional[int], seed: int) -> List[int]:
    out = list(map(int, sids))
    if limit is None or int(limit) >= len(out):
        return out
    rng = random.Random(seed)
    rng.shuffle(out)
    return out[: int(limit)]


class MultiLayerSpatialCoordinatePatch:
    """Apply the prior relation-specific coordinate correction at multiple layers."""

    def __init__(
        self,
        *,
        decoder_layers: Sequence[torch.nn.Module],
        target_layers: Sequence[int],
        noimage_states: Mapping[int, torch.Tensor],
        axes: Mapping[int, Mapping[str, torch.Tensor]],
        targets: Mapping[int, Mapping[str, float]],
        guide: str,
        alpha: float,
        max_delta_ratio: float,
    ) -> None:
        self.target_layers = [int(x) for x in target_layers]
        self.noimage_states = noimage_states
        self.axes = axes
        self.targets = targets
        self.guide = str(guide)
        self.alpha = float(alpha)
        self.max_delta_ratio = float(max_delta_ratio)
        self.handles = []
        self.applied: Dict[int, int] = {L: 0 for L in self.target_layers}
        self.meta: Dict[int, Dict[str, Any]] = {}

        for L in self.target_layers:
            module = decoder_layers[L]

            def make_hook(layer_id: int):
                def hook(_module: Any, _args: Any, output: Any) -> Any:
                    tensor = rd.first_tensor(output)
                    if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
                        raise RuntimeError(f"Unexpected L{layer_id} output {tuple(tensor.shape)}")
                    # prefill only; cached decode normally has sequence length 1
                    if self.applied[layer_id] > 0 or int(tensor.shape[1]) <= 1:
                        return output

                    current = tensor[0, -1].float()
                    noimg = self.noimage_states[layer_id].to(device=current.device, dtype=torch.float32)
                    axis = self.axes[layer_id][self.guide].to(device=current.device, dtype=torch.float32)
                    target_coord = float(self.targets[layer_id][self.guide])
                    residual = current - noimg
                    current_coord = float(torch.dot(residual, axis).item())
                    raw_scalar = self.alpha * (target_coord - current_coord)
                    delta = raw_scalar * axis

                    current_norm = float(current.norm().item())
                    delta_norm = float(delta.norm().item())
                    clipped = False
                    if self.max_delta_ratio > 0 and current_norm > EPS:
                        max_norm = self.max_delta_ratio * current_norm
                        if delta_norm > max_norm and delta_norm > EPS:
                            delta = delta * (max_norm / delta_norm)
                            delta_norm = float(delta.norm().item())
                            clipped = True

                    modified = tensor.clone()
                    modified[0, -1] = modified[0, -1] + delta.to(modified.dtype)
                    self.applied[layer_id] += 1
                    self.meta[layer_id] = {
                        "layer": layer_id,
                        "guide": self.guide,
                        "current_coord": current_coord,
                        "target_coord": target_coord,
                        "raw_scalar": float(raw_scalar),
                        "delta_norm": delta_norm,
                        "hidden_norm": current_norm,
                        "delta_ratio": delta_norm / max(current_norm, EPS),
                        "clipped": bool(clipped),
                    }
                    return rd.replace_first_tensor(output, modified)

                return hook

            self.handles.append(module.register_forward_hook(make_hook(L)))

    def validate(self) -> None:
        bad = {L: n for L, n in self.applied.items() if n != 1}
        if bad:
            raise RuntimeError(f"Expected exactly one prefill patch per target layer, got {bad}")

    def close(self) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


def generate_multilayer(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    decoder_layers: Sequence[torch.nn.Module],
    question: str,
    image: Image.Image,
    target_layers: Sequence[int],
    noimage_states: Mapping[int, torch.Tensor],
    axes: Mapping[int, Mapping[str, torch.Tensor]],
    targets: Mapping[int, Mapping[str, float]],
    guide: str,
    alpha: float,
    max_delta_ratio: float,
    max_new_tokens: int,
) -> Dict[str, Any]:
    rendered = direction_base.build_chat_prompt(processor, question, True)
    batch = direction_base.process_inputs(processor, rendered, image, device)
    prompt_length = int(batch["input_ids"].shape[1])
    tok = processor.tokenizer
    eos = getattr(tok, "eos_token_id", None)
    pad = getattr(tok, "pad_token_id", None)
    if pad is None:
        pad = eos
    gen_kwargs: Dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": int(max_new_tokens),
        "use_cache": True,
    }
    if pad is not None:
        gen_kwargs["pad_token_id"] = int(pad)

    patch = MultiLayerSpatialCoordinatePatch(
        decoder_layers=decoder_layers,
        target_layers=target_layers,
        noimage_states=noimage_states,
        axes=axes,
        targets=targets,
        guide=guide,
        alpha=alpha,
        max_delta_ratio=max_delta_ratio,
    )
    try:
        with torch.inference_mode():
            seq = model.generate(**batch, **gen_kwargs)
        patch.validate()
        result = rd.decode_generated(tok, seq, prompt_length)
        result["patches"] = {str(L): patch.meta[L] for L in sorted(patch.meta)}
        return result
    finally:
        patch.close()
        del batch


def make_trigger_map(
    *,
    sids: Sequence[int],
    baseline: Mapping[int, Mapping[str, Any]],
    guide_by_sid: Mapping[int, str],
    covered_by_sid: Mapping[int, bool],
    policy: str,
) -> Dict[int, bool]:
    out: Dict[int, bool] = {}
    for sid in sids:
        covered = bool(covered_by_sid[sid])
        basep = feas.normalize_relation(baseline[sid].get("prediction"))
        guide = guide_by_sid[sid]
        if policy == "all_covered":
            trig = covered
        elif policy == "conflict_only":
            trig = covered and basep != guide
        else:
            raise ValueError(policy)
        out[int(sid)] = bool(trig)
    return out


def run_variant(
    *,
    name: str,
    target_layers: Sequence[int],
    test_sids: Sequence[int],
    trigger_by_sid: Mapping[int, bool],
    guide_by_sid: Mapping[int, str],
    model: Any,
    processor: Any,
    device: torch.device,
    decoder_layers: Sequence[torch.nn.Module],
    record_by_sid: Mapping[int, Any],
    noimage_states: Mapping[int, Mapping[int, torch.Tensor]],
    axes: Mapping[int, Mapping[str, torch.Tensor]],
    targets: Mapping[int, Mapping[str, float]],
    prompt_template: str,
    alpha: float,
    max_delta_ratio: float,
    max_new_tokens: int,
    print_every: int,
) -> Dict[int, Dict[str, Any]]:
    run_sids = [int(s) for s in test_sids if trigger_by_sid[int(s)]]
    results: Dict[int, Dict[str, Any]] = {}
    for pos, sid in enumerate(tqdm(run_sids, desc=name), 1):
        rec = record_by_sid[sid]
        question = rd.make_question(rec, prompt_template)
        image = None
        try:
            image = Image.open(rec.image_path).convert("RGB")
            results[sid] = generate_multilayer(
                model=model,
                processor=processor,
                device=device,
                decoder_layers=decoder_layers,
                question=question,
                image=image,
                target_layers=target_layers,
                noimage_states=noimage_states[sid],
                axes=axes,
                targets=targets,
                guide=guide_by_sid[sid],
                alpha=alpha,
                max_delta_ratio=max_delta_ratio,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            results[sid] = {
                "prediction": None,
                "text": "",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
            }
            tqdm.write(f"[{name} ERROR] sid={sid}: {type(exc).__name__}: {exc}")
        finally:
            if image is not None:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if print_every > 0 and pos % print_every == 0:
            tqdm.write(f"[{name}] {pos}/{len(run_sids)}")
    return results


def evaluate_variant(
    *,
    name: str,
    target_layers: Sequence[int],
    test_sids: Sequence[int],
    label_by_sid: Mapping[int, str],
    baseline: Mapping[int, Mapping[str, Any]],
    repaired: Mapping[int, Mapping[str, Any]],
    trigger_by_sid: Mapping[int, bool],
    guide_by_sid: Mapping[int, str],
    covered_by_sid: Mapping[int, bool],
    policy: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    gt = [label_by_sid[s] for s in test_sids]
    basep = [feas.normalize_relation(baseline[s].get("prediction")) for s in test_sids]
    rep_final: List[Optional[str]] = []
    patch_rows: List[Dict[str, Any]] = []

    for i, sid in enumerate(test_sids):
        if trigger_by_sid[sid]:
            rr = repaired.get(sid, {})
            rp = feas.normalize_relation(rr.get("prediction"))
            if rp not in RELATIONS:
                # Failed repair does not get to count as a free error; fall back to baseline.
                rp = basep[i]
            rep_final.append(rp)
            for Ls, meta in (rr.get("patches") or {}).items():
                patch_rows.append({"variant": name, "sid": sid, **meta})
        else:
            rep_final.append(basep[i])

    summ = rd.summarize_policy(
        gt=gt,
        baseline_pred=basep,
        repaired_pred=rep_final,
        guide=[guide_by_sid[s] for s in test_sids],
        covered=[covered_by_sid[s] for s in test_sids],
        policy=policy,
    )
    # summarize_policy recomputes trigger from covered/base/guide, which matches trigger_by_sid.
    summ.update({
        "variant": name,
        "layers": ",".join(map(str, target_layers)),
        "n_layers": len(target_layers),
        "alpha": None,
        "mean_delta_ratio": safe_mean(r.get("delta_ratio", float("nan")) for r in patch_rows),
        "mean_delta_norm": safe_mean(r.get("delta_norm", float("nan")) for r in patch_rows),
        "clip_rate": safe_mean(float(bool(r.get("clipped", False))) for r in patch_rows),
    })
    return summ, patch_rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    out = Path(args.output_dir)
    if args.overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    feasibility_dir = Path(args.feasibility_dir)
    fsum_path = feasibility_dir / "summary.json"
    if not fsum_path.exists():
        raise FileNotFoundError(fsum_path)
    fsum = json.loads(fsum_path.read_text(encoding="utf-8"))
    dataset = args.dataset or str(fsum["dataset"])
    model_alias = args.model or str(fsum["model"])
    direction_dir = Path(fsum["direction_dir"])
    npz_path = direction_dir / "relation_vectors.npz"
    split_csv = feasibility_dir / "split.csv"
    baseline_path = Path(args.baseline_generation_jsonl) if args.baseline_generation_jsonl else Path(fsum["generation_jsonl"])
    for pth in (npz_path, split_csv, baseline_path):
        if not pth.exists():
            raise FileNotFoundError(pth)

    consensus = rd.rebuild_consensus(feasibility_summary=fsum, npz_path=npz_path, split_csv=split_csv)
    sids = consensus["sids"]
    labels = consensus["labels"]
    train_idx = consensus["train_idx"]
    test_idx_full = consensus["test_idx"]

    # consensus maps in official TEST order
    result = consensus["result_test"]
    test_guide: Dict[int, str] = {}
    test_cov: Dict[int, bool] = {}
    for local, global_i in enumerate(test_idx_full.tolist()):
        sid = int(sids[global_i])
        test_guide[sid] = RELATIONS[int(result["prediction"][local])]
        test_cov[sid] = bool(result["covered"][local])

    label_by_sid = {int(sid): str(label) for sid, label in zip(sids.tolist(), labels.tolist())}
    train_sids = [int(sids[i]) for i in train_idx.tolist()]
    test_sids_full = [int(sids[i]) for i in test_idx_full.tolist()]
    test_sids = subset_sids(test_sids_full, args.max_test_samples, args.seed + 202)
    if len(test_sids) != len(test_sids_full):
        print("[SMOKE MODE] TEST capped; do not report these numbers as final")

    baseline = rd.load_baseline_generation(baseline_path)
    for sid in test_sids:
        if sid not in baseline:
            raise RuntimeError(f"Missing baseline sid={sid}")

    spec = base.SPECS[model_alias]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers=={transformers.__version__} has no {spec.model_class}")
    kwargs: Dict[str, Any] = {
        "torch_dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kwargs["attn_implementation"] = args.attn_impl
    print(f"[model] loading {model_alias} from {spec.repo_id}")
    model = model_cls.from_pretrained(spec.repo_id, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    base.configure_processor(model, processor)
    device = torch.device(args.device)
    decoder_layers, decoder_path = direction_base.resolve_decoder_layers(model)
    n_layers = len(decoder_layers)
    print(f"[model] decoder={decoder_path} n_layers={n_layers}")

    single_layers = [] if args.skip_single_sweep else resolve_layer_spec(args.single_layers, n_layers)
    if not args.skip_single_sweep and not single_layers:
        raise ValueError("No valid --single-layers")
    windows = relative_windows(n_layers)
    required = set(single_layers)
    for v in windows.values():
        required.update(v)
    required_layers = sorted(required)
    print("[windows] " + "; ".join(f"{k}=L{v[0]}-L{v[-1]} ({len(v)})" for k,v in windows.items()))

    records, _audit = base.load_records(dataset, Path(args.data_root), None)
    record_by_sid = {int(r.sid): r for r in records}
    missing = [sid for sid in [*train_sids, *test_sids] if sid not in record_by_sid]
    if missing:
        raise RuntimeError(f"Dataset loader missing SIDs: {missing[:10]}")

    print("\nStage 1/4: fit TRAIN relation-specific last-token residual coordinates for all required layers")
    centroids = rd.fit_train_last_residual_centroids(
        model=model,
        processor=processor,
        device=device,
        decoder_layers=decoder_layers,
        candidate_layers=required_layers,
        train_sids=train_sids,
        label_by_sid=label_by_sid,
        record_by_sid=record_by_sid,
        prompt_template=args.prompt_template,
    )
    axes, targets = rd.relation_axis_and_targets(centroids)

    proto_arrays: Dict[str, np.ndarray] = {}
    for L in required_layers:
        for rel in RELATIONS:
            proto_arrays[f"centroid_L{L}_{rel}"] = centroids[L][rel].numpy().astype(np.float32)
            proto_arrays[f"axis_L{L}_{rel}"] = axes[L][rel].numpy().astype(np.float32)
            proto_arrays[f"target_L{L}_{rel}"] = np.asarray(targets[L][rel], dtype=np.float32)
    np.savez_compressed(out / "train_relation_coordinates.npz", **proto_arrays)

    trigger_by_sid = make_trigger_map(
        sids=test_sids,
        baseline=baseline,
        guide_by_sid=test_guide,
        covered_by_sid=test_cov,
        policy=args.policy,
    )
    trigger_sids = [sid for sid in test_sids if trigger_by_sid[sid]]
    print(f"[trigger] policy={args.policy} n={len(trigger_sids)}/{len(test_sids)}")

    print("\nStage 2/4: capture no-image last-token references for triggered TEST samples")
    sig = rd.noimage_cache_signature(
        model=model_alias,
        dataset=dataset,
        layers=required_layers,
        sids=trigger_sids,
        prompt_template=args.prompt_template,
    )
    noimage_states = rd.build_noimage_cache(
        cache_path=out / "noimage_last_states.pt",
        reuse=args.reuse_noimage_cache,
        signature=sig,
        model=model,
        processor=processor,
        device=device,
        decoder_layers=decoder_layers,
        candidate_layers=required_layers,
        sids=trigger_sids,
        record_by_sid=record_by_sid,
        prompt_template=args.prompt_template,
    )

    all_summaries: List[Dict[str, Any]] = []
    all_patch_rows: List[Dict[str, Any]] = []
    sample_path = out / "variant_samples.jsonl"
    if sample_path.exists():
        sample_path.unlink()

    print("\nStage 3/4: single-layer sweep")
    if args.skip_single_sweep:
        print("[single-layer sweep skipped]")
    for L in single_layers:
        name = f"single_L{L}"
        repaired = run_variant(
            name=name,
            target_layers=[L],
            test_sids=test_sids,
            trigger_by_sid=trigger_by_sid,
            guide_by_sid=test_guide,
            model=model,
            processor=processor,
            device=device,
            decoder_layers=decoder_layers,
            record_by_sid=record_by_sid,
            noimage_states=noimage_states,
            axes=axes,
            targets=targets,
            prompt_template=args.prompt_template,
            alpha=args.alpha,
            max_delta_ratio=args.max_delta_ratio,
            max_new_tokens=args.max_new_tokens,
            print_every=0,
        )
        summ, patch_rows = evaluate_variant(
            name=name,
            target_layers=[L],
            test_sids=test_sids,
            label_by_sid=label_by_sid,
            baseline=baseline,
            repaired=repaired,
            trigger_by_sid=trigger_by_sid,
            guide_by_sid=test_guide,
            covered_by_sid=test_cov,
            policy=args.policy,
        )
        summ["alpha"] = float(args.alpha)
        all_summaries.append(summ)
        all_patch_rows.extend(patch_rows)
        print(
            f"{name:12s} ACC={summ['intervention_accuracy']:.4f} "
            f"delta={summ['accuracy_change']:+.4f} "
            f"W->C={summ['repaired_wrong']} C->W={summ['damaged_correct']}"
        )
        for sid, rr in repaired.items():
            append_jsonl(sample_path, {
                "variant": name,
                "sid": sid,
                "gt": label_by_sid[sid],
                "baseline_prediction": feas.normalize_relation(baseline[sid].get("prediction")),
                "guide": test_guide[sid],
                "repaired_prediction": feas.normalize_relation(rr.get("prediction")),
                "patches": rr.get("patches"),
                "error": rr.get("error"),
            })
        del repaired
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nStage 4/4: fixed multi-layer windows")
    for name, layers in windows.items():
        repaired = run_variant(
            name=name,
            target_layers=layers,
            test_sids=test_sids,
            trigger_by_sid=trigger_by_sid,
            guide_by_sid=test_guide,
            model=model,
            processor=processor,
            device=device,
            decoder_layers=decoder_layers,
            record_by_sid=record_by_sid,
            noimage_states=noimage_states,
            axes=axes,
            targets=targets,
            prompt_template=args.prompt_template,
            alpha=args.alpha,
            max_delta_ratio=args.max_delta_ratio,
            max_new_tokens=args.max_new_tokens,
            print_every=args.print_every,
        )
        summ, patch_rows = evaluate_variant(
            name=name,
            target_layers=layers,
            test_sids=test_sids,
            label_by_sid=label_by_sid,
            baseline=baseline,
            repaired=repaired,
            trigger_by_sid=trigger_by_sid,
            guide_by_sid=test_guide,
            covered_by_sid=test_cov,
            policy=args.policy,
        )
        summ["alpha"] = float(args.alpha)
        all_summaries.append(summ)
        all_patch_rows.extend(patch_rows)
        print(
            f"{name:16s} ACC={summ['intervention_accuracy']:.4f} "
            f"delta={summ['accuracy_change']:+.4f} "
            f"W->C={summ['repaired_wrong']} C->W={summ['damaged_correct']} "
            f"mean_delta/h={summ['mean_delta_ratio']:.6f}"
        )
        for sid, rr in repaired.items():
            append_jsonl(sample_path, {
                "variant": name,
                "sid": sid,
                "gt": label_by_sid[sid],
                "baseline_prediction": feas.normalize_relation(baseline[sid].get("prediction")),
                "guide": test_guide[sid],
                "repaired_prediction": feas.normalize_relation(rr.get("prediction")),
                "patches": rr.get("patches"),
                "error": rr.get("error"),
            })
        del repaired
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    single_rows = [r for r in all_summaries if str(r["variant"]).startswith("single_L")]
    window_rows = [r for r in all_summaries if not str(r["variant"]).startswith("single_L")]
    write_csv(out / "single_layer_sweep.csv", single_rows)
    write_csv(out / "window_results.csv", window_rows)
    write_csv(out / "all_results.csv", all_summaries)
    write_csv(out / "patch_diagnostics.csv", all_patch_rows)

    base_acc = float(np.mean([
        feas.normalize_relation(baseline[s].get("prediction")) == label_by_sid[s]
        for s in test_sids
    ]))
    best_single = max(single_rows, key=lambda r: float(r["intervention_accuracy"])) if single_rows else None
    summary = {
        "script_version": SCRIPT_VERSION,
        "model": model_alias,
        "dataset": dataset,
        "decoder_path": decoder_path,
        "n_layers": n_layers,
        "test_n": len(test_sids),
        "trigger_policy": args.policy,
        "trigger_n": len(trigger_sids),
        "baseline_accuracy": base_acc,
        "alpha": float(args.alpha),
        "max_delta_ratio": float(args.max_delta_ratio),
        "single_layers": single_layers,
        "windows": windows,
        "best_single": best_single,
        "window_results": window_rows,
        "interpretation_rule": {
            "relation_delta_windows_positive_head_mean_windows_negative": "direct Direction-head write averaging is likely the main problem",
            "only_narrow_single_layers_positive_windows_worse": "layer placement / repeated multi-layer edits are likely harmful",
            "broad_single_region_positive_and_windows_more_positive": "multi-layer spatial preservation is supported",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("CROSS-MODEL RELATION-SPECIFIC DELTA-H DIAGNOSTIC")
    print("=" * 100)
    print(f"TEST N / trigger policy          : {len(test_sids)} / {args.policy}")
    print(f"triggered                        : {len(trigger_sids)}")
    print(f"baseline generation ACC          : {base_acc:.4f}")
    print(f"alpha / max-delta-ratio          : {args.alpha:g} / {args.max_delta_ratio:g}")
    if best_single:
        print("-")
        print(f"best single layer                : {best_single['variant']}")
        print(f"best single repaired ACC         : {best_single['intervention_accuracy']:.4f}")
        print(f"best single ACC change           : {best_single['accuracy_change']:+.4f}")
        print(f"best single W->C / C->W          : {best_single['repaired_wrong']} / {best_single['damaged_correct']}")
    print("-")
    for r in window_rows:
        print(
            f"{r['variant']:16s} ACC={r['intervention_accuracy']:.4f} "
            f"delta={r['accuracy_change']:+.4f} "
            f"W->C={r['repaired_wrong']} C->W={r['damaged_correct']}"
        )
    print("\nSaved:")
    for fn in ["summary.json", "single_layer_sweep.csv", "window_results.csv", "all_results.csv", "patch_diagnostics.csv", "variant_samples.jsonl", "train_relation_coordinates.npz", "noimage_last_states.pt"]:
        print(f"  {out / fn}")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
