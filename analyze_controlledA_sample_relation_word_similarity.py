#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample-level similarity between Controlled A object-difference vectors and direction word tokens.

For each sample and layer, compare:

  raw:
      cos( r_i, h_i(word_c) )

  centered:
      cos( r_i - mean_j r_j, h_i(word_c) - mean_{w in {left,right,on,under}} h_i(w) )

where r_i = h(subject_i) - h(reference_i) is loaded from an existing Controlled A states.npz.
The direction word vectors are extracted from a text-only LLaVA prompt, so no image is fed.

Outputs per layer:
  - mean 4x4 similarity matrices, rows=true relation, cols=word candidate
  - sample-level argmax accuracy over the four word candidates
  - per-sample CSV scores
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor, AutoTokenizer
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")

RELATIONS = ["left", "right", "on", "under"]
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
REL_MAP = {
    "left": "left",
    "right": "right",
    "on": "on",
    "top": "on",
    "above": "on",
    "under": "under",
    "below": "under",
    "bottom": "under",
    "underneath": "under",
}


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo_id: str
    model_class: str
    dtype: torch.dtype
    trust_remote_code: bool = False


SPECS = {
    "llava-7b": ModelSpec(
        alias="llava-7b",
        repo_id="llava-hf/llava-1.5-7b-hf",
        model_class="LlavaForConditionalGeneration",
        dtype=torch.float16,
    ),
    "llava-13b": ModelSpec(
        alias="llava-13b",
        repo_id="llava-hf/llava-1.5-13b-hf",
        model_class="LlavaForConditionalGeneration",
        dtype=torch.float16,
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, help="Controlled A states.npz with relation_vectors")
    p.add_argument("--model", default="llava-7b", choices=sorted(SPECS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", choices=["sdpa", "eager", "flash_attention_2", "none"], default="sdpa")
    p.add_argument("--prompt-path", default="prompts/Controlled_Images_A_with_answer_four_options.jsonl")
    p.add_argument("--layers", default="auto", help="auto or comma-separated decoder block indices")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--save-plots", action="store_true", default=True)
    return p.parse_args()


def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def normalize_relation(x: Any) -> str:
    if isinstance(x, (list, tuple, np.ndarray)):
        x = x[0] if len(x) else ""
    return REL_MAP.get(str(x).strip().lower(), str(x).strip().lower())


def load_prompt_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for expected_id, line in enumerate(f):
            row = json.loads(line)
            if int(row.get("id", expected_id)) != expected_id:
                raise ValueError(f"Prompt IDs not contiguous: expected {expected_id}, got {row.get('id')}")
            rows.append(row)
    return rows


def clean_question_text(q: str) -> str:
    text = str(q).strip()
    text = text.replace("<image>", " ")
    text = re.sub(r"\bUSER\s*:\s*", " ", text, flags=re.IGNORECASE)
    text = re.split(r"\bASSISTANT\s*:\s*", text, flags=re.IGNORECASE)[0]
    text = re.sub(r"\s+", " ", text).strip()
    lower = text.lower()
    if not all(re.search(rf"\b{re.escape(w)}\b", lower) for w in RELATIONS):
        text = text.rstrip(" .") + ". Answer with left, right, on or under."
    return text


def build_text_only_prompt(question: str) -> str:
    q = clean_question_text(question)
    return f"USER: {q}\nASSISTANT:"


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    if not needle:
        return []
    n = len(needle)
    h = list(haystack)
    nd = list(needle)
    return [i for i in range(len(h) - n + 1) if h[i:i + n] == nd]


def find_word_last_token(tokenizer: Any, input_ids: Sequence[int], word: str) -> int:
    matches: List[Tuple[int, int, str, List[int]]] = []
    seen = set()
    variants = [word, " " + word, ", " + word, ": " + word, "(" + word, "\n" + word]
    for v in variants:
        ids = list(tokenizer(v, add_special_tokens=False).input_ids)
        key = tuple(int(x) for x in ids)
        if not key or key in seen:
            continue
        seen.add(key)
        for start in find_subsequence(input_ids, ids):
            matches.append((start, start + len(ids) - 1, v, ids))
    if not matches:
        decoded = tokenizer.decode(list(input_ids), skip_special_tokens=False)
        raise ValueError(f"Could not find relation word token for {word!r}. prompt={decoded!r}")
    return max(matches, key=lambda x: x[0])[1]


def hidden_tuple(outputs: Any) -> Tuple[torch.Tensor, ...]:
    candidates = [
        getattr(outputs, "hidden_states", None),
        getattr(getattr(outputs, "language_model_outputs", None), "hidden_states", None),
        getattr(getattr(outputs, "text_model_output", None), "hidden_states", None),
    ]
    for states in candidates:
        if isinstance(states, (tuple, list)) and states and torch.is_tensor(states[-1]):
            return tuple(states)
    raise RuntimeError("No hidden_states returned")


def parse_layers(raw: str, available: Sequence[int]) -> List[int]:
    available = [int(x) for x in available]
    if raw == "auto":
        return available
    wanted = [int(x.strip().lstrip("L")) for x in raw.split(",") if x.strip()]
    missing = [x for x in wanted if x not in available]
    if missing:
        raise ValueError(f"Requested layers {missing} not in npz available layers {available}")
    return wanted


def load_relation_vectors(npz_path: Path, layers: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    with np.load(npz_path, allow_pickle=True) as z:
        all_layers = [int(v) for v in z["decoder_block_index"].tolist()]
        layer_to_col = {L: i for i, L in enumerate(all_layers)}
        cols = [layer_to_col[L] for L in layers]
        X = z["relation_vectors"][:, cols, :].astype(np.float32)
        y = np.asarray([normalize_relation(v) for v in z["relation"].tolist()], dtype=object)
        if "sample_index" in z:
            sid = z["sample_index"].astype(np.int64)
        elif "sid" in z:
            sid = z["sid"].astype(np.int64)
        else:
            sid = np.arange(X.shape[0], dtype=np.int64)
    bad = sorted(set(y.tolist()) - set(RELATIONS))
    if bad:
        raise ValueError(f"Unexpected relation labels: {bad}")
    return X, y, sid, layers


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def write_matrix_csv(path: Path, matrix: np.ndarray, row_labels: List[str], col_labels: List[str]) -> None:
    lines = ["," + ",".join(col_labels)]
    for r, vals in zip(row_labels, matrix):
        lines.append(r + "," + ",".join(f"{float(v):.6f}" for v in vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_heatmap(path: Path, matrix: np.ndarray, title: str, row_labels: List[str], col_labels: List[str]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=200)
    im = ax.imshow(matrix, vmin=-0.15, vmax=0.15)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("candidate direction word token")
    ax.set_ylabel("true relation of object difference")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def safe_max_offdiag(row: np.ndarray, true_idx: int) -> float:
    tmp = row.copy()
    tmp[true_idx] = -np.inf
    return float(np.max(tmp))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = Path(args.npz)
    with np.load(npz_path, allow_pickle=True) as z:
        available_layers = [int(v) for v in z["decoder_block_index"].tolist()]
    layers = parse_layers(args.layers, available_layers)
    X, y, sids, layers = load_relation_vectors(npz_path, layers)

    if args.max_samples is not None:
        X = X[: args.max_samples]
        y = y[: args.max_samples]
        sids = sids[: args.max_samples]

    # Raw and globally-centered object-difference vectors.
    X_raw = l2norm(X, axis=2)
    X_centered = l2norm(X - X.mean(axis=0, keepdims=True), axis=2)

    prompt_rows = load_prompt_rows(Path(args.prompt_path))
    prompts: List[str] = []
    used_row_indices: List[int] = []
    used_sids: List[int] = []
    for local_i, sid in enumerate(sids):
        sid_int = int(sid)
        if 0 <= sid_int < len(prompt_rows):
            prompts.append(build_text_only_prompt(prompt_rows[sid_int]["question"]))
            used_row_indices.append(local_i)
            used_sids.append(sid_int)

    if not prompts:
        raise RuntimeError("No prompt rows matched npz sample_index/sid")

    # Keep X/y in the exact prompt extraction order.
    X_raw = X_raw[used_row_indices]
    X_centered = X_centered[used_row_indices]
    y_used = y[used_row_indices]

    spec = SPECS[args.model]
    model_cls = getattr(transformers, spec.model_class, None)
    if model_cls is None:
        raise RuntimeError(f"transformers=={transformers.__version__} has no {spec.model_class}")

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": spec.dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": {"": args.device},
    }
    if args.attn_impl != "none":
        load_kwargs["attn_implementation"] = args.attn_impl

    print(f"Input npz: {npz_path}")
    print(f"Model: {args.model} ({spec.repo_id})")
    print(f"Layers: {layers}")
    print(f"Relations/words: {RELATIONS}")
    print(f"Samples: {len(prompts)}")
    print("Modes:")
    print("  raw      = cos(r_i, h_i(word_c))")
    print("  centered = cos(r_i-mean(r), h_i(word_c)-mean_words(h_i))")

    model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
    model.eval()

    try:
        processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        tokenizer = processor.tokenizer
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(args.device)
    H = int(X_raw.shape[-1])
    Lnum = len(layers)

    # Per-layer, per-mode accumulators.
    modes = ["raw", "centered"]
    score_sum = {m: np.zeros((Lnum, len(RELATIONS), len(RELATIONS)), dtype=np.float64) for m in modes}
    score_count = {m: np.zeros((Lnum, len(RELATIONS), len(RELATIONS)), dtype=np.int64) for m in modes}
    correct = {m: np.zeros(Lnum, dtype=np.int64) for m in modes}
    total = {m: np.zeros(Lnum, dtype=np.int64) for m in modes}
    diag_sum = {m: np.zeros(Lnum, dtype=np.float64) for m in modes}
    margin_sum = {m: np.zeros(Lnum, dtype=np.float64) for m in modes}
    per_sample_rows: Dict[Tuple[int, str], List[Dict[str, Any]]] = {(L, m): [] for L in layers for m in modes}
    errors: List[Dict[str, Any]] = []

    for idx, (sid, prompt) in enumerate(tqdm(list(zip(used_sids, prompts)), desc="Scoring object-diff vs word tokens")):
        try:
            enc = tokenizer([prompt], return_tensors="pt", add_special_tokens=True)
            enc = move_batch(dict(enc), device)
            input_ids = enc["input_ids"][0].detach().cpu().tolist()
            word_pos = [find_word_last_token(tokenizer, input_ids, w) for w in RELATIONS]

            with torch.inference_mode():
                outputs = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
            states = hidden_tuple(outputs)

            true_rel = str(y_used[idx])
            true_idx = REL_TO_ID[true_rel]

            for li, block in enumerate(layers):
                W = torch.stack([states[block + 1][0, p] for p in word_pos], dim=0).detach().float().cpu().numpy()
                if W.shape != (len(RELATIONS), H):
                    raise RuntimeError(f"Unexpected word hidden shape {W.shape}, expected {(len(RELATIONS), H)}")

                W_raw = l2norm(W.astype(np.float32), axis=1)
                W_centered = l2norm((W - W.mean(axis=0, keepdims=True)).astype(np.float32), axis=1)

                for mode_name, rv, wv in [
                    ("raw", X_raw[idx, li], W_raw),
                    ("centered", X_centered[idx, li], W_centered),
                ]:
                    s = (rv[None, :] @ wv.T).reshape(-1).astype(np.float64)
                    pred_idx = int(np.argmax(s))
                    pred_rel = RELATIONS[pred_idx]
                    margin = float(s[true_idx] - safe_max_offdiag(s, true_idx))

                    score_sum[mode_name][li, true_idx, :] += s
                    score_count[mode_name][li, true_idx, :] += 1
                    correct[mode_name][li] += int(pred_idx == true_idx)
                    total[mode_name][li] += 1
                    diag_sum[mode_name][li] += float(s[true_idx])
                    margin_sum[mode_name][li] += margin

                    per_sample_rows[(block, mode_name)].append({
                        "sample_index": int(sid),
                        "true_relation": true_rel,
                        "pred_word": pred_rel,
                        "correct": int(pred_idx == true_idx),
                        "margin": margin,
                        **{f"score_{r}": float(s[j]) for j, r in enumerate(RELATIONS)},
                    })

            del outputs, states, enc
            if torch.cuda.is_available() and (idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
        except Exception as exc:
            errors.append({
                "sid": int(sid),
                "prompt": prompt,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary: Dict[str, Any] = {
        "npz": str(npz_path),
        "model": args.model,
        "repo_id": spec.repo_id,
        "layers_order": layers,
        "relations": RELATIONS,
        "comparison": {
            "raw": "cos(r_i, h_i(word_c))",
            "centered": "cos(r_i - global_mean_r, h_i(word_c) - mean_words_i)",
        },
        "n_samples_requested": int(len(prompts)),
        "n_errors": int(len(errors)),
        "layers": {},
        "results": {},
    }

    for li, L in enumerate(layers):
        layer_info: Dict[str, Any] = {}
        for mode_name in modes:
            mat = score_sum[mode_name][li] / np.maximum(score_count[mode_name][li], 1)
            acc = float(correct[mode_name][li] / max(int(total[mode_name][li]), 1))
            diag_mean = float(diag_sum[mode_name][li] / max(int(total[mode_name][li]), 1))
            margin_mean = float(margin_sum[mode_name][li] / max(int(total[mode_name][li]), 1))
            row_argmax = [RELATIONS[int(j)] for j in mat.argmax(axis=1)]
            row_match_acc = float(np.mean([row_argmax[i] == RELATIONS[i] for i in range(len(RELATIONS))]))
            diag_by_class = {r: float(mat[i, i]) for i, r in enumerate(RELATIONS)}
            info = {
                "matrix": mat.tolist(),
                "sample_argmax_acc": acc,
                "diag_mean": diag_mean,
                "sample_margin_mean": margin_mean,
                "row_argmax": row_argmax,
                "row_match_acc": row_match_acc,
                "diag_by_class": diag_by_class,
                "n_scored": int(total[mode_name][li]),
            }
            layer_info[mode_name] = info
            summary["results"][f"L{L}_{mode_name}"] = info

            write_matrix_csv(out_dir / f"sample_relation_word_mean_matrix_L{L}_{mode_name}.csv", mat, RELATIONS, RELATIONS)
            with (out_dir / f"sample_relation_word_scores_L{L}_{mode_name}.csv").open("w", newline="", encoding="utf-8") as f:
                fieldnames = ["sample_index", "true_relation", "pred_word", "correct", "margin"] + [f"score_{r}" for r in RELATIONS]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_sample_rows[(L, mode_name)])
            if args.save_plots:
                plot_heatmap(
                    out_dir / f"sample_relation_word_mean_matrix_L{L}_{mode_name}.png",
                    mat,
                    f"{args.model} Controlled A L{L} object-diff vs word ({mode_name})",
                    RELATIONS,
                    RELATIONS,
                )

        summary["layers"][str(L)] = layer_info
        raw = layer_info["raw"]
        cen = layer_info["centered"]
        print(
            f"L{L}: "
            f"raw acc={raw['sample_argmax_acc']:.3f} diag={raw['diag_mean']:.3f} margin={raw['sample_margin_mean']:.3f} rowmatch={raw['row_match_acc']:.2f} | "
            f"centered acc={cen['sample_argmax_acc']:.3f} diag={cen['diag_mean']:.3f} margin={cen['sample_margin_mean']:.3f} rowmatch={cen['row_match_acc']:.2f}"
        )

    (out_dir / "sample_relation_word_similarity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "sample_relation_word_similarity_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved summary: {out_dir / 'sample_relation_word_similarity_summary.json'}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
