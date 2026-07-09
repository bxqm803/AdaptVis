#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Controlled A relation directions with textual left/right/on/under vectors.

Standalone LLaVA-1.5 script.

It does NOT feed images. This avoids LLaVA image-token mismatch and measures the
language-side option-word vectors in the same question text context:

  USER: Where is A in relation to B? Answer with left, right, on or under.
  ASSISTANT:

For each layer:
  1) relation directions from existing states.npz:
       d_c = normalize(mean_{y=c}(r_i - mean(r)))
  2) textual word vectors from the model hidden states:
       raw:             mean_i h_i(word_c)
       prompt_centered: mean_i [h_i(word_c) - mean_c h_i(word_c)]
  3) cosine matrix D @ W^T.
"""
from __future__ import annotations

import argparse
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
    if isinstance(x, (list, tuple)):
        x = x[0] if x else ""
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
    """Remove image/chat wrappers but keep the actual question/options text."""
    text = str(q).strip()
    text = text.replace("<image>", " ")
    text = re.sub(r"\bUSER\s*:\s*", " ", text, flags=re.IGNORECASE)
    # Drop everything after ASSISTANT: because we only want prompt-side option words.
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
    return [i for i in range(len(h) - n + 1) if h[i:i+n] == nd]


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
    # Last occurrence targets the explicit option-list word.
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


def compute_relation_directions(npz_path: Path, layers: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
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

    D = []
    for li, _L in enumerate(layers):
        Xl = X[:, li, :]
        center = Xl.mean(axis=0, keepdims=True)
        Xc = Xl - center
        dirs = []
        for rel in RELATIONS:
            mask = y == rel
            if mask.sum() == 0:
                raise ValueError(f"No samples for relation {rel}")
            dirs.append(Xc[mask].mean(axis=0))
        D.append(l2norm(np.stack(dirs, axis=0), axis=1))
    return np.stack(D, axis=0), y, sid, layers


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def write_csv(path: Path, matrix: np.ndarray, row_labels: List[str], col_labels: List[str]) -> None:
    lines = ["," + ",".join(col_labels)]
    for r, vals in zip(row_labels, matrix):
        lines.append(r + "," + ",".join(f"{float(v):.6f}" for v in vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_heatmap(path: Path, matrix: np.ndarray, title: str, row_labels: List[str], col_labels: List[str]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=200)
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("text relation word vector")
    ax.set_ylabel("relation direction")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = Path(args.npz)
    with np.load(npz_path, allow_pickle=True) as z:
        available_layers = [int(v) for v in z["decoder_block_index"].tolist()]
    layers = parse_layers(args.layers, available_layers)
    D, y, sids, layers = compute_relation_directions(npz_path, layers)
    if args.max_samples is not None:
        sids = sids[: args.max_samples]

    prompt_rows = load_prompt_rows(Path(args.prompt_path))
    prompts = []
    used_sids = []
    for sid in sids:
        sid_int = int(sid)
        if 0 <= sid_int < len(prompt_rows):
            prompts.append(build_text_only_prompt(prompt_rows[sid_int]["question"]))
            used_sids.append(sid_int)

    if not prompts:
        raise RuntimeError("No prompt rows matched npz sample_index/sid")

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
    print(f"Text-only prompts: {len(prompts)}")

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
    H = int(D.shape[-1])
    Lnum = len(layers)
    sum_raw = np.zeros((Lnum, len(RELATIONS), H), dtype=np.float64)
    sum_centered = np.zeros_like(sum_raw)
    count = 0
    errors: List[Dict[str, Any]] = []

    for sid, prompt in tqdm(list(zip(used_sids, prompts)), desc="Extracting text-only word states"):
        try:
            enc = tokenizer([prompt], return_tensors="pt", add_special_tokens=True)
            enc = move_batch(dict(enc), device)
            input_ids = enc["input_ids"][0].detach().cpu().tolist()
            word_pos = [find_word_last_token(tokenizer, input_ids, w) for w in RELATIONS]

            with torch.inference_mode():
                outputs = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
            states = hidden_tuple(outputs)

            per_layer_words = []
            for block in layers:
                # hidden_states[0] is embedding output; block index L maps to hidden_states[L+1]
                hw = torch.stack([states[block + 1][0, p] for p in word_pos], dim=0).detach().float().cpu().numpy()
                if hw.shape != (len(RELATIONS), H):
                    raise RuntimeError(f"Unexpected word hidden shape {hw.shape}, expected {(len(RELATIONS), H)}")
                per_layer_words.append(hw)
            W = np.stack(per_layer_words, axis=0)
            sum_raw += W
            sum_centered += W - W.mean(axis=1, keepdims=True)
            count += 1

            del outputs, states, enc
            if torch.cuda.is_available() and count % 50 == 0:
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

    if count == 0:
        raise RuntimeError(f"No word states extracted. First errors: {errors[:3]}")

    W_raw = l2norm((sum_raw / count).astype(np.float32), axis=2)
    W_centered = l2norm((sum_centered / count).astype(np.float32), axis=2)

    summary: Dict[str, Any] = {
        "npz": str(npz_path),
        "model": args.model,
        "repo_id": spec.repo_id,
        "layers_order": layers,
        "relations": RELATIONS,
        "word_context": "text_only_prompt",
        "n_word_records_used": int(count),
        "n_errors": int(len(errors)),
        "layers": {},
        "results": {},
    }

    for li, L in enumerate(layers):
        layer_dict: Dict[str, Any] = {}
        for mode_name, Wmode in [("raw", W_raw), ("prompt_centered", W_centered)]:
            S = D[li] @ Wmode[li].T
            diag = np.diag(S)
            off = S.copy()
            np.fill_diagonal(off, -np.inf)
            margin = diag - off.max(axis=1)
            row_argmax = [RELATIONS[int(j)] for j in S.argmax(axis=1)]
            row_match = [row_argmax[i] == RELATIONS[i] for i in range(len(RELATIONS))]
            info = {
                "matrix": S.tolist(),
                "diag_mean": float(diag.mean()),
                "diag_min": float(diag.min()),
                "diag_margin_mean": float(margin.mean()),
                "mean_margin": float(margin.mean()),
                "row_argmax": row_argmax,
                "row_match_acc": float(np.mean(row_match)),
            }
            layer_dict[mode_name] = info
            summary["results"][f"L{L}_{mode_name}"] = info

            csv_path = out_dir / f"direction_word_similarity_L{L}_{mode_name}.csv"
            png_path = out_dir / f"direction_word_similarity_L{L}_{mode_name}.png"
            write_csv(csv_path, S, RELATIONS, RELATIONS)
            if args.save_plots:
                plot_heatmap(png_path, S, f"{args.model} Controlled A L{L} ({mode_name})", RELATIONS, RELATIONS)

        summary["layers"][str(L)] = layer_dict
        pc = layer_dict["prompt_centered"]
        raw = layer_dict["raw"]
        print(
            f"L{L}: "
            f"raw diag={raw['diag_mean']:.3f} margin={raw['diag_margin_mean']:.3f} match={raw['row_match_acc']:.2f} | "
            f"centered diag={pc['diag_mean']:.3f} margin={pc['diag_margin_mean']:.3f} match={pc['row_match_acc']:.2f}"
        )

    (out_dir / "direction_word_similarity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "direction_word_similarity_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved summary: {out_dir / 'direction_word_similarity_summary.json'}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
