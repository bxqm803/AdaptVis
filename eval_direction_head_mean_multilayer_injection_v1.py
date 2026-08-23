#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training-free test of direct multi-head Direction-vector injection.

Goal
----
Assume a set of Direction heads has already been identified offline. For each
sample, reuse that sample's Image-NoImage object-relation vector from each
selected Direction head:

    r_{l,h} = [(z_img,A - z_img,B) - (z_noimg,A - z_noimg,B)]

The stored vectors are pre-W_O and have head_dim dimensions, so they cannot be
added directly to the residual stream. For each selected head, map its vector
through ONLY that head's slice of the layer output projection W_O:

    w_{l,h} = W_O^{(l,h)} r_{l,h}

(no output-projection bias is added). Then compute the sample-specific mean:

    g = mean_h w_{l,h}

During image generation, add the SAME g to the prompt-last block output at each
layer in one of three fixed windows:

    late_26_31     : L26-L31
    midlate_18_31  : L18-L31
    all_layers     : every decoder layer

No relation prototype, GT label, consensus label, learned spatial axis, DEV
layer selection, or alpha tuning is used by default. Ground truth is used only
for final evaluation. This script therefore tests the user's simplest proposal:
"average the selected Direction-head result and repeatedly add it to the last
prompt token".

Important
---------
* Head discovery itself is still an offline prior step. By default the script
  reads selected_heads from validate_grounded_spatial_consensus_v1.py output.
* relation_vectors.npz must contain `residual` vectors. Those were produced by
  an image + no-image extraction pass, but no additional training is required.
* Default alpha=1 and scale_mode=per_layer implement the literal proposal.
  If repeated injection is too strong, --scale-mode fixed_budget is available
  as a control, dividing the vector by the number of target layers.

Example
-------
CUDA_VISIBLE_DEVICES=0 python eval_direction_head_mean_multilayer_injection_v1.py \
  --feasibility-dir output/qwen3b_coco_grounded_consensus_v1 \
  --device cuda:0 \
  --output-dir output/qwen3b_direction_mean_multilayer_v1 \
  --overwrite

Smoke test:
  add --max-test-samples 24
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


SCRIPT_VERSION = "eval-direction-head-mean-multilayer-injection-v1"
RELATIONS = ("left", "right", "above", "below")
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--feasibility-dir",
        required=True,
        help="Directory produced by validate_grounded_spatial_consensus_v1.py; used only for selected_heads, split, model/dataset paths.",
    )
    p.add_argument("--dataset", default=None, help="Override dataset from feasibility summary")
    p.add_argument("--model", default=None, help="Override model alias from feasibility summary")
    p.add_argument("--data-root", default="data")
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
        help="Must match the prompt used when relation_vectors.npz and baseline generation were produced.",
    )
    p.add_argument(
        "--baseline-generation-jsonl",
        default=None,
        help="Override baseline generation cache; default uses feasibility summary generation_jsonl.",
    )
    p.add_argument(
        "--heads",
        default=None,
        help="Optional comma-separated head names, e.g. L27H03,L21H11. Default: selected_heads from feasibility summary.",
    )
    p.add_argument(
        "--vector-source",
        default="residual",
        choices=["residual", "img", "no_image"],
        help="Which stored pre-W_O relation vector to project and average. residual = Image-NoImage.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Fixed multiplier; not tuned on DEV.",
    )
    p.add_argument(
        "--scale-mode",
        choices=["per_layer", "fixed_budget"],
        default="per_layer",
        help="per_layer adds alpha*g at every target layer. fixed_budget adds alpha*g/num_layers at each layer.",
    )
    p.add_argument(
        "--max-delta-ratio",
        type=float,
        default=0.0,
        help="Optional per-layer clip: ||delta|| <= ratio*||current last hidden||. <=0 means no clipping (literal test).",
    )
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-test-samples", type=int, default=None, help="Smoke test only")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_relation(x: Any) -> str:
    return feas.normalize_relation(x)


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        f.flush()


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Cannot find tensor in output type {type(output)}")


def replace_first_tensor(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item):
                items[i] = replacement
                return tuple(items)
    if isinstance(output, list):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item):
                items[i] = replacement
                return items
    raise TypeError(f"Cannot replace tensor in output type {type(output)}")


def parse_head_name(name: str) -> Tuple[int, int]:
    m = re.fullmatch(r"L(\d+)H(\d+)", str(name).strip())
    if not m:
        raise ValueError(f"Bad head name: {name!r}")
    return int(m.group(1)), int(m.group(2))


def parse_heads(text: str) -> List[str]:
    names = [x.strip() for x in str(text).split(",") if x.strip()]
    for x in names:
        parse_head_name(x)
    return list(dict.fromkeys(names))


def load_test_sids(split_csv: Path) -> List[int]:
    out: List[int] = []
    with split_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("split", "")).strip() == "test":
                out.append(int(row["sid"]))
    if not out:
        raise RuntimeError(f"No test rows in {split_csv}")
    return out


def subset_sids(sids: Sequence[int], limit: Optional[int], seed: int) -> List[int]:
    sids = list(map(int, sids))
    if limit is None or int(limit) >= len(sids):
        return sids
    rng = random.Random(seed)
    rng.shuffle(sids)
    return sids[: int(limit)]


def make_question(rec: Any, template: str) -> str:
    return template.format(subject=rec.subject, reference=rec.reference)


def decode_generated(tokenizer: Any, sequences: torch.Tensor, prompt_length: int) -> Dict[str, Any]:
    text, token_ids = feas.decode_new_tokens(tokenizer, sequences, prompt_length)
    pred = feas.parse_generated_relation(text)
    return {"prediction": pred, "text": text, "token_ids": token_ids}


class MultiLayerLastVectorPatch:
    """Add one sample-specific residual-space vector at prompt-last across many layers."""

    def __init__(
        self,
        *,
        decoder_layers: Sequence[torch.nn.Module],
        target_layers: Sequence[int],
        guide_vector: torch.Tensor,
        alpha: float,
        scale_mode: str,
        max_delta_ratio: float,
    ) -> None:
        self.target_layers = [int(x) for x in target_layers]
        self.guide_vector = guide_vector.detach().float().cpu()
        self.alpha = float(alpha)
        self.scale_mode = str(scale_mode)
        self.max_delta_ratio = float(max_delta_ratio)
        self.applied: Dict[int, int] = {L: 0 for L in self.target_layers}
        self.meta: Dict[int, Dict[str, Any]] = {}
        self.handles = []

        if self.scale_mode == "fixed_budget":
            self.per_layer_scale = self.alpha / max(len(self.target_layers), 1)
        elif self.scale_mode == "per_layer":
            self.per_layer_scale = self.alpha
        else:
            raise ValueError(self.scale_mode)

        for L in self.target_layers:
            module = decoder_layers[L]

            def make_hook(layer_id: int):
                def hook(_module: Any, _args: Any, output: Any) -> Any:
                    tensor = first_tensor(output)
                    if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
                        raise RuntimeError(
                            f"Unexpected L{layer_id} output shape {tuple(tensor.shape)}"
                        )

                    # Only patch prefill. Cached autoregressive decode normally has sequence length 1.
                    if self.applied[layer_id] > 0 or int(tensor.shape[1]) <= 1:
                        return output

                    current = tensor[0, -1].float()
                    delta = self.guide_vector.to(
                        device=current.device, dtype=torch.float32
                    ) * self.per_layer_scale

                    raw_delta_norm = float(delta.norm().item())
                    current_norm = float(current.norm().item())
                    clipped = False
                    if self.max_delta_ratio > 0 and current_norm > EPS:
                        max_norm = self.max_delta_ratio * current_norm
                        if raw_delta_norm > max_norm and raw_delta_norm > EPS:
                            delta = delta * (max_norm / raw_delta_norm)
                            clipped = True

                    delta_norm = float(delta.norm().item())
                    modified = tensor.clone()
                    modified[0, -1] = modified[0, -1] + delta.to(modified.dtype)
                    self.applied[layer_id] += 1
                    self.meta[layer_id] = {
                        "layer": int(layer_id),
                        "current_hidden_norm": current_norm,
                        "raw_delta_norm": raw_delta_norm,
                        "delta_norm": delta_norm,
                        "delta_ratio": delta_norm / max(current_norm, EPS),
                        "clipped": bool(clipped),
                    }
                    return replace_first_tensor(output, modified)

                return hook

            self.handles.append(module.register_forward_hook(make_hook(L)))

    def validate(self) -> None:
        bad = {L: n for L, n in self.applied.items() if n != 1}
        if bad:
            raise RuntimeError(f"Patch application count mismatch: {bad}")

    def close(self) -> None:
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()
        self.handles = []


def generate_with_vector_patch(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    decoder_layers: Sequence[torch.nn.Module],
    question: str,
    image: Image.Image,
    target_layers: Sequence[int],
    guide_vector: torch.Tensor,
    alpha: float,
    scale_mode: str,
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
    kwargs: Dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": int(max_new_tokens),
        "use_cache": True,
    }
    if pad is not None:
        kwargs["pad_token_id"] = int(pad)

    patch = MultiLayerLastVectorPatch(
        decoder_layers=decoder_layers,
        target_layers=target_layers,
        guide_vector=guide_vector,
        alpha=alpha,
        scale_mode=scale_mode,
        max_delta_ratio=max_delta_ratio,
    )
    try:
        with torch.inference_mode():
            seq = model.generate(**batch, **kwargs)
        patch.validate()
        result = decode_generated(tok, seq, prompt_length)
        metas = list(patch.meta.values())
        result["patch"] = {
            "target_layers": list(map(int, target_layers)),
            "n_layers": int(len(target_layers)),
            "per_layer_scale": float(patch.per_layer_scale),
            "mean_delta_ratio": safe_mean(x["delta_ratio"] for x in metas),
            "max_delta_ratio_actual": max((x["delta_ratio"] for x in metas), default=float("nan")),
            "clipped_layers": int(sum(bool(x["clipped"]) for x in metas)),
        }
        return result
    finally:
        patch.close()
        del batch


def build_head_lookup(
    *,
    layer_ids: np.ndarray,
    head_ids: np.ndarray,
) -> Dict[str, Tuple[int, int]]:
    lookup: Dict[str, Tuple[int, int]] = {}
    for lp, L in enumerate(layer_ids.tolist()):
        for hp, H in enumerate(head_ids.tolist()):
            lookup[f"L{int(L)}H{int(H):02d}"] = (int(lp), int(hp))
    return lookup


def project_selected_head_vectors(
    *,
    vectors4: np.ndarray,
    sample_sids: np.ndarray,
    selected_head_names: Sequence[str],
    layer_ids: np.ndarray,
    head_ids: np.ndarray,
    decoder_layers: Sequence[torch.nn.Module],
    model: Any,
    device: torch.device,
) -> Tuple[Dict[int, torch.Tensor], List[Dict[str, Any]]]:
    """
    Convert selected pre-W_O head relation vectors into residual-space writes,
    then average across selected heads for each sample.

    Returns:
        guide_by_sid[sid] -> float16 CPU hidden-size vector
        diagnostic rows per sample
    """
    lookup = build_head_lookup(layer_ids=layer_ids, head_ids=head_ids)
    missing = [x for x in selected_head_names if x not in lookup]
    if missing:
        raise RuntimeError(f"Selected heads absent from NPZ: {missing}")

    N, _, _, head_dim = vectors4.shape
    n_selected = len(selected_head_names)
    if n_selected == 0:
        raise RuntimeError("No selected heads")

    # Infer hidden size and verify head geometry using repository helper.
    n_heads_model, head_dim_model = direction_base.scan_shape(model, decoder_layers)
    if int(head_dim_model) != int(head_dim):
        raise RuntimeError(
            f"NPZ head_dim={head_dim}, model head_dim={head_dim_model}"
        )

    # [N, hidden] accumulator on GPU. This is small for the current datasets.
    hidden = None
    acc = None
    indiv_norms: List[torch.Tensor] = []
    normalized_writes: List[torch.Tensor] = []

    with torch.inference_mode():
        for name in selected_head_names:
            lp, hp = lookup[name]
            L = int(layer_ids[lp])
            H = int(head_ids[hp])
            if L < 0 or L >= len(decoder_layers):
                raise RuntimeError(f"Head {name}: layer out of model range")

            attn = direction_base.resolve_self_attention(decoder_layers[L])
            o_proj = direction_base.resolve_o_proj(attn)
            W = o_proj.weight.detach()
            if W.ndim != 2:
                raise RuntimeError(f"{name}: o_proj.weight shape={tuple(W.shape)}")
            start = H * int(head_dim)
            end = start + int(head_dim)
            if end > int(W.shape[1]):
                raise RuntimeError(
                    f"{name}: slice [{start}:{end}] exceeds o_proj input {W.shape[1]}"
                )
            Wslice = W[:, start:end].to(device=device)
            if hidden is None:
                hidden = int(W.shape[0])
                acc = torch.zeros((N, hidden), device=device, dtype=torch.float32)

            x = torch.as_tensor(
                np.asarray(vectors4[:, lp, hp, :], dtype=np.float32),
                device=device,
                dtype=torch.float32,
            )
            # Ignore o_proj bias: bias belongs to the full concatenated attention
            # output, not to an isolated head contribution.
            write = torch.matmul(x, Wslice.float().transpose(0, 1))
            assert acc is not None
            acc.add_(write)
            norms = write.norm(dim=1)
            indiv_norms.append(norms.detach().cpu())
            normalized_writes.append(
                (write / norms.clamp_min(EPS).unsqueeze(1)).detach().cpu().half()
            )
            del x, write, Wslice

        assert acc is not None
        guide = acc / float(n_selected)
        guide_norm = guide.norm(dim=1).detach().cpu().float()
        guide_cpu = guide.detach().cpu().half()
        del guide, acc

    norm_stack = torch.stack(indiv_norms, dim=1).float()  # [N,K]
    mean_indiv_norm = norm_stack.mean(dim=1)
    cancellation_ratio = guide_norm / mean_indiv_norm.clamp_min(EPS)

    # Mean pairwise cosine between post-W_O head writes, per sample.
    # K is small (typically 10), so this is cheap.
    U = torch.stack(normalized_writes, dim=1).float()  # [N,K,H]
    gram = torch.matmul(U, U.transpose(1, 2))
    K = U.shape[1]
    if K > 1:
        offdiag_sum = gram.sum(dim=(1, 2)) - float(K)
        pair_cos = offdiag_sum / float(K * (K - 1))
    else:
        pair_cos = torch.ones(U.shape[0])

    guide_by_sid: Dict[int, torch.Tensor] = {}
    diag_rows: List[Dict[str, Any]] = []
    for i, sid in enumerate(sample_sids.tolist()):
        guide_by_sid[int(sid)] = guide_cpu[i].clone()
        diag_rows.append({
            "sid": int(sid),
            "guide_norm": float(guide_norm[i].item()),
            "mean_individual_head_write_norm": float(mean_indiv_norm[i].item()),
            "cancellation_ratio": float(cancellation_ratio[i].item()),
            "mean_pairwise_head_write_cosine": float(pair_cos[i].item()),
        })
    return guide_by_sid, diag_rows


def summarize_variant(
    *,
    name: str,
    sids: Sequence[int],
    gt_by_sid: Mapping[int, str],
    baseline: Mapping[int, Mapping[str, Any]],
    repaired: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    base_correct = []
    rep_correct = []
    changed = []
    parsed = []
    delta_ratios = []
    for sid in sids:
        sid = int(sid)
        gt = gt_by_sid[sid]
        bp = baseline[sid].get("prediction")
        rp = repaired[sid].get("prediction")
        bc = bp == gt
        rc = rp == gt
        base_correct.append(bool(bc))
        rep_correct.append(bool(rc))
        changed.append(bp != rp)
        parsed.append(rp in RELATIONS)
        patch = repaired[sid].get("patch", {})
        v = patch.get("mean_delta_ratio")
        if v is not None:
            delta_ratios.append(float(v))

    b = np.asarray(base_correct, dtype=bool)
    r = np.asarray(rep_correct, dtype=bool)
    c = np.asarray(changed, dtype=bool)
    wrong_to_correct = (~b) & r
    correct_to_wrong = b & (~r)
    return {
        "variant": name,
        "n": int(len(sids)),
        "baseline_accuracy": float(b.mean()),
        "repaired_accuracy": float(r.mean()),
        "accuracy_change": float(r.mean() - b.mean()),
        "wrong_to_correct": int(wrong_to_correct.sum()),
        "correct_to_wrong": int(correct_to_wrong.sum()),
        "net_repair": int(wrong_to_correct.sum() - correct_to_wrong.sum()),
        "prediction_changed": int(c.sum()),
        "parse_rate": float(np.mean(parsed)),
        "wrong_repair_rate": float(wrong_to_correct[~b].mean()) if (~b).any() else float("nan"),
        "correct_damage_rate": float(correct_to_wrong[b].mean()) if b.any() else float("nan"),
        "mean_per_layer_delta_ratio": safe_mean(delta_ratios),
    }


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
    split_csv = feasibility_dir / "split.csv"
    if not fsum_path.exists():
        raise FileNotFoundError(fsum_path)
    if not split_csv.exists():
        raise FileNotFoundError(split_csv)
    fsum = json.loads(fsum_path.read_text(encoding="utf-8"))

    dataset = args.dataset or str(fsum["dataset"])
    model_alias = args.model or str(fsum["model"])
    direction_dir = Path(fsum["direction_dir"])
    npz_path = direction_dir / "relation_vectors.npz"
    baseline_path = (
        Path(args.baseline_generation_jsonl)
        if args.baseline_generation_jsonl
        else Path(fsum["generation_jsonl"])
    )
    for path in (npz_path, baseline_path):
        if not path.exists():
            raise FileNotFoundError(path)

    selected_heads = (
        parse_heads(args.heads)
        if args.heads is not None
        else [str(x) for x in fsum["selected_heads"]]
    )
    if not selected_heads:
        raise RuntimeError("No Direction heads selected")

    with np.load(npz_path, allow_pickle=True) as data:
        if args.vector_source not in data.files:
            raise RuntimeError(
                f"{npz_path} has {data.files}, missing --vector-source={args.vector_source}"
            )
        sample_sids = np.asarray(data["sample_index"], dtype=np.int64)
        labels = np.asarray([normalize_relation(x) for x in data["relation"]], dtype=object)
        vectors4 = np.asarray(data[args.vector_source], dtype=np.float32)
        layer_ids = np.asarray(
            data["decoder_block_index"]
            if "decoder_block_index" in data.files
            else np.arange(vectors4.shape[1]),
            dtype=np.int64,
        )
        head_ids = np.asarray(
            data["head_index"]
            if "head_index" in data.files
            else np.arange(vectors4.shape[2]),
            dtype=np.int64,
        )

    sid_to_npz_row = {int(sid): i for i, sid in enumerate(sample_sids.tolist())}
    gt_by_sid = {int(sid): str(rel) for sid, rel in zip(sample_sids.tolist(), labels.tolist())}

    test_sids = load_test_sids(split_csv)
    missing_test = [sid for sid in test_sids if sid not in sid_to_npz_row]
    if missing_test:
        raise RuntimeError(f"Test SIDs absent from direction NPZ: {missing_test[:10]}")
    test_sids = subset_sids(test_sids, args.max_test_samples, args.seed + 202)
    if args.max_test_samples is not None:
        print("[SMOKE MODE] TEST capped; do not report as final")

    baseline = feas.load_generation_cache(baseline_path)
    for sid in test_sids:
        if sid not in baseline:
            raise RuntimeError(f"Baseline generation missing sid={sid}")

    # Load dataset records using repository loader.
    records, _audit = base.load_records(dataset, Path(args.data_root), None)
    record_by_sid = {int(r.sid): r for r in records}
    for sid in test_sids:
        if sid not in record_by_sid:
            raise RuntimeError(f"Dataset loader missing sid={sid}")

    # Load model using repository specs.
    spec = base.SPECS[model_alias]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers=={transformers.__version__} has no {spec.model_class}"
        )
    kwargs: Dict[str, Any] = {
        "torch_dtype": base.resolve_dtype(spec.dtype_name),
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        kwargs["attn_implementation"] = args.attn_impl
    model = model_cls.from_pretrained(spec.repo_id, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        spec.repo_id, trust_remote_code=spec.trust_remote_code
    )
    base.configure_processor(model, processor)
    device = torch.device(args.device)

    decoder_layers, decoder_path = direction_base.resolve_decoder_layers(model)
    n_layers = len(decoder_layers)
    print(
        f"model={model_alias} dataset={dataset} decoder={decoder_path} "
        f"layers={n_layers} selected_heads={len(selected_heads)}"
    )
    print("selected heads:", ", ".join(selected_heads))

    # Fixed windows requested by the user. No DEV selection.
    requested_windows: Dict[str, List[int]] = {
        "late_26_31": list(range(26, 32)),
        "midlate_18_31": list(range(18, 32)),
        "all_layers": list(range(n_layers)),
    }
    for name, layers in requested_windows.items():
        bad = [L for L in layers if L < 0 or L >= n_layers]
        if bad:
            raise RuntimeError(
                f"Window {name} contains invalid layers {bad}; model has {n_layers} blocks"
            )

    # Build one sample-specific mean Direction-head write vector per NPZ sample.
    print(f"[guide] projecting selected heads through their own W_O slices; source={args.vector_source}")
    guide_by_sid, guide_diag = project_selected_head_vectors(
        vectors4=vectors4,
        sample_sids=sample_sids,
        selected_head_names=selected_heads,
        layer_ids=layer_ids,
        head_ids=head_ids,
        decoder_layers=decoder_layers,
        model=model,
        device=device,
    )
    write_csv(out / "guide_diagnostics.csv", guide_diag)

    test_diag = [x for x in guide_diag if int(x["sid"]) in set(test_sids)]
    print(
        "[guide diagnostics TEST] "
        f"mean ||g||={safe_mean(x['guide_norm'] for x in test_diag):.6f}  "
        f"cancellation={safe_mean(x['cancellation_ratio'] for x in test_diag):.4f}  "
        f"pairwise_cos={safe_mean(x['mean_pairwise_head_write_cosine'] for x in test_diag):.4f}"
    )

    all_variant_results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    sample_rows: Dict[int, Dict[str, Any]] = {
        sid: {
            "sid": int(sid),
            "gt": gt_by_sid[int(sid)],
            "baseline_prediction": baseline[int(sid)].get("prediction"),
            "baseline_correct": int(baseline[int(sid)].get("prediction") == gt_by_sid[int(sid)]),
            "guide_norm": float(next(x["guide_norm"] for x in guide_diag if int(x["sid"]) == int(sid))),
        }
        for sid in test_sids
    }

    details_path = out / "generation_details.jsonl"
    if details_path.exists():
        details_path.unlink()

    for variant_name, target_layers in requested_windows.items():
        print("\n" + "=" * 100)
        print(
            f"RUN {variant_name}: layers={target_layers[0]}..{target_layers[-1]} "
            f"(n={len(target_layers)}) alpha={args.alpha} scale_mode={args.scale_mode}"
        )
        print("=" * 100)
        results: Dict[int, Dict[str, Any]] = {}

        for pos, sid in enumerate(tqdm(test_sids, desc=variant_name), 1):
            sid = int(sid)
            rec = record_by_sid[sid]
            question = make_question(rec, args.prompt_template)
            image = None
            try:
                image = Image.open(rec.image_path).convert("RGB")
                result = generate_with_vector_patch(
                    model=model,
                    processor=processor,
                    device=device,
                    decoder_layers=decoder_layers,
                    question=question,
                    image=image,
                    target_layers=target_layers,
                    guide_vector=guide_by_sid[sid],
                    alpha=float(args.alpha),
                    scale_mode=args.scale_mode,
                    max_delta_ratio=float(args.max_delta_ratio),
                    max_new_tokens=int(args.max_new_tokens),
                )
            except Exception as exc:
                result = {
                    "prediction": None,
                    "text": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-12:],
                }
                tqdm.write(f"[{variant_name} ERROR] sid={sid}: {type(exc).__name__}: {exc}")
            finally:
                if image is not None:
                    image.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results[sid] = result
            sample_rows[sid][f"{variant_name}_prediction"] = result.get("prediction")
            sample_rows[sid][f"{variant_name}_correct"] = int(
                result.get("prediction") == gt_by_sid[sid]
            )
            sample_rows[sid][f"{variant_name}_mean_delta_ratio"] = (
                result.get("patch", {}).get("mean_delta_ratio", float("nan"))
            )
            append_jsonl(details_path, {
                "variant": variant_name,
                "sid": sid,
                "gt": gt_by_sid[sid],
                "baseline_prediction": baseline[sid].get("prediction"),
                "repaired_prediction": result.get("prediction"),
                "generation_text": result.get("text", ""),
                "patch": result.get("patch", {}),
                "error": result.get("error"),
            })

            if args.print_every > 0 and pos % args.print_every == 0:
                partial = summarize_variant(
                    name=variant_name,
                    sids=test_sids[:pos],
                    gt_by_sid=gt_by_sid,
                    baseline=baseline,
                    repaired=results,
                )
                tqdm.write(
                    f"[{variant_name}] {pos}/{len(test_sids)} "
                    f"ACC={partial['repaired_accuracy']:.4f} "
                    f"delta={partial['accuracy_change']:+.4f} "
                    f"W->C={partial['wrong_to_correct']} C->W={partial['correct_to_wrong']}"
                )

        all_variant_results[variant_name] = results

    summaries = []
    for name in requested_windows:
        summaries.append(
            summarize_variant(
                name=name,
                sids=test_sids,
                gt_by_sid=gt_by_sid,
                baseline=baseline,
                repaired=all_variant_results[name],
            )
        )

    write_csv(out / "test_samples.csv", list(sample_rows.values()))
    write_csv(out / "variant_summary.csv", summaries)

    baseline_acc = float(np.mean([
        baseline[int(sid)].get("prediction") == gt_by_sid[int(sid)] for sid in test_sids
    ]))
    summary_json = {
        "script_version": SCRIPT_VERSION,
        "dataset": dataset,
        "model": model_alias,
        "n_test": int(len(test_sids)),
        "vector_source": args.vector_source,
        "selected_heads": selected_heads,
        "alpha": float(args.alpha),
        "scale_mode": args.scale_mode,
        "max_delta_ratio": float(args.max_delta_ratio),
        "baseline_accuracy": baseline_acc,
        "windows": requested_windows,
        "variants": summaries,
        "guide_diagnostics_test": {
            "mean_guide_norm": safe_mean(x["guide_norm"] for x in test_diag),
            "mean_individual_write_norm": safe_mean(x["mean_individual_head_write_norm"] for x in test_diag),
            "mean_cancellation_ratio": safe_mean(x["cancellation_ratio"] for x in test_diag),
            "mean_pairwise_write_cosine": safe_mean(x["mean_pairwise_head_write_cosine"] for x in test_diag),
        },
        "protocol_note": (
            "No GT/prototype/consensus label/DEV tuning is used for repair. The only offline prior is the selected Direction-head set. "
            "Each sample uses its own stored head relation residuals, mapped through each head's W_O slice and averaged."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 100)
    print("DIRECT DIRECTION-HEAD MEAN MULTI-LAYER INJECTION")
    print("=" * 100)
    print(f"TEST N                  : {len(test_sids)}")
    print(f"baseline generation ACC : {baseline_acc:.4f}")
    print(f"vector source           : {args.vector_source}")
    print(f"alpha / scale mode      : {args.alpha:g} / {args.scale_mode}")
    print(f"selected heads          : {', '.join(selected_heads)}")
    print("-")
    for row in summaries:
        print(f"variant                  : {row['variant']}")
        print(f"repaired generation ACC  : {row['repaired_accuracy']:.4f}")
        print(f"ACC change               : {row['accuracy_change']:+.4f}")
        print(f"wrong -> correct         : {row['wrong_to_correct']}")
        print(f"correct -> wrong         : {row['correct_to_wrong']}")
        print(f"net repair               : {row['net_repair']:+d}")
        print(f"prediction changed       : {row['prediction_changed']}")
        print(f"wrong repair rate        : {row['wrong_repair_rate']:.4f}")
        print(f"correct damage rate      : {row['correct_damage_rate']:.4f}")
        print(f"mean per-layer delta/h   : {row['mean_per_layer_delta_ratio']:.6f}")
        print("-")
    print("Saved:")
    print(f"  {out / 'summary.json'}")
    print(f"  {out / 'variant_summary.csv'}")
    print(f"  {out / 'test_samples.csv'}")
    print(f"  {out / 'guide_diagnostics.csv'}")
    print(f"  {details_path}")


if __name__ == "__main__":
    main()
