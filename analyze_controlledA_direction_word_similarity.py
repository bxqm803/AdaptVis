#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Controlled A four relation directions with textual relation-word vectors.

This script is standalone: it does not import previous extractor/analyzer scripts.

For LLaVA-1.5 + Controlled A:
  1) load relation_vectors from an existing Controlled A states.npz;
  2) compute four full-data relation directions d_left/d_right/d_on/d_under;
  3) run the same model on Controlled A prompts and extract hidden states of
     the textual words left/right/on/under in the question/options;
  4) compare cosine similarities between relation directions and word vectors.

The main diagnostic is prompt-centered word vectors:
    w_c = mean_i [ h_i(word_c) - mean_{c' in words} h_i(word_{c'}) ]
This removes common prompt/image context shared by all four option words.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    raise SystemExit(f"Unable to import transformers: {exc}")

try:
    from dataset_zoo import get_dataset
except Exception as exc:
    raise SystemExit(f"Unable to import dataset_zoo.get_dataset. Run from AdaptVis repo root. Error: {exc}")

try:
    from misc import _default_collate as repository_default_collate
except Exception:
    repository_default_collate = None

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

QUESTION_PATTERNS = [
    re.compile(
        r"Where\s+(?P<verb>is|are)\s+(?:the\s+)?(?P<subject>.+?)\s+"
        r"in\s+relation\s+to\s+(?:the\s+)?(?P<reference>.+?)\?\s*Answer",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"Where\s+(?P<verb>is|are)\s+(?:the\s+)?(?P<subject>.+?)\s+"
        r"relative\s+to\s+(?:the\s+)?(?P<reference>.+?)\?",
        flags=re.IGNORECASE | re.DOTALL,
    ),
]


@dataclass(frozen=True)
class Record:
    sid: int
    relation: str
    subject: str
    reference: str
    question: str
    image: Image.Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, help="Controlled A states.npz with relation_vectors")
    p.add_argument("--model", default="llava-7b", choices=sorted(SPECS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", choices=["sdpa", "eager", "flash_attention_2", "none"], default="sdpa")
    p.add_argument("--prompt-path", default="prompts/Controlled_Images_A_with_answer_four_options.jsonl")
    p.add_argument("--download", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--layers", default="auto", help="auto or comma-separated decoder block indices")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--append-options-if-missing", action="store_true", default=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def canonical_object(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[\s\.,;:!?]+$", "", text)
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_relation(x: Any) -> str:
    if isinstance(x, (list, tuple)):
        x = x[0] if x else ""
    return REL_MAP.get(str(x).strip().lower(), str(x).strip().lower())


def parse_prompt_objects(question: str) -> Tuple[str, str]:
    for pat in QUESTION_PATTERNS:
        m = pat.search(str(question))
        if m is not None:
            return canonical_object(m.group("subject")), canonical_object(m.group("reference"))
    return "", ""


def load_prompt_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for expected_id, line in enumerate(f):
            row = json.loads(line)
            if int(row.get("id", expected_id)) != expected_id:
                raise ValueError(f"Prompt IDs not contiguous: expected {expected_id}, got {row.get('id')}")
            rows.append(row)
    return rows


def extract_images_from_batch(batch: Mapping[str, Any]) -> Iterable[Any]:
    if "image_options" in batch:
        for image_option in batch["image_options"]:
            for image in image_option:
                yield image
    elif "images" in batch:
        for image in batch["images"]:
            yield image
    elif "image" in batch:
        vals = batch["image"]
        if isinstance(vals, (list, tuple)):
            yield from vals
        else:
            yield vals
    else:
        raise KeyError(f"Cannot find images in batch keys={list(batch.keys())}")


def ensure_pil_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 3:
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)} shape={getattr(arr, 'shape', None)}")


def load_records(prompt_path: Path, download: bool, num_workers: int) -> Dict[int, Record]:
    prompt_rows = load_prompt_rows(prompt_path)
    dataset = get_dataset("Controlled_Images_A", image_preprocess=None, download=download)
    total = min(len(dataset), len(prompt_rows))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=repository_default_collate)

    records: Dict[int, Record] = {}
    sid = 0
    pbar = tqdm(total=total, desc="Loading Controlled_A")
    for batch in loader:
        for raw_img in extract_images_from_batch(batch):
            if sid >= total:
                break
            row = prompt_rows[sid]
            try:
                relation = normalize_relation(row.get("answer", ""))
                if relation not in RELATIONS:
                    sid += 1
                    pbar.update(1)
                    continue
                subject, reference = parse_prompt_objects(str(row["question"]))
                records[sid] = Record(
                    sid=sid,
                    relation=relation,
                    subject=subject,
                    reference=reference,
                    question=str(row["question"]),
                    image=ensure_pil_rgb(raw_img),
                )
            finally:
                sid += 1
                pbar.update(1)
        if sid >= total:
            break
    pbar.close()
    return records


def configure_processor(model: Any, processor: Any) -> None:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None and hasattr(processor, "patch_size") and hasattr(vision_config, "patch_size"):
        processor.patch_size = int(vision_config.patch_size)
    strategy = getattr(config, "vision_feature_select_strategy", None)
    if strategy is not None and hasattr(processor, "vision_feature_select_strategy"):
        processor.vision_feature_select_strategy = str(strategy)
    if getattr(config, "model_type", "") == "llava" and hasattr(processor, "num_additional_image_tokens"):
        processor.num_additional_image_tokens = 1


def build_chat_prompt(processor: Any, question: str, append_options_if_missing: bool = True) -> str:
    text = str(question).strip()
    if append_options_if_missing:
        lower = text.lower()
        if not all(re.search(rf"\b{re.escape(w)}\b", lower) for w in RELATIONS):
            text = text.rstrip() + " Options: left, right, on, under."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    if not needle:
        return []
    n = len(needle)
    return [i for i in range(len(haystack) - n + 1) if list(haystack[i:i+n]) == list(needle)]


def find_word_last_token(tokenizer: Any, input_ids: Sequence[int], word: str) -> int:
    matches: List[Tuple[int, int]] = []
    seen = set()
    variants = [word, " " + word, ", " + word, ": " + word, "(" + word, "\n" + word]
    for v in variants:
        ids = list(tokenizer(v, add_special_tokens=False).input_ids)
        key = tuple(int(x) for x in ids)
        if not key or key in seen:
            continue
        seen.add(key)
        for start in find_subsequence(input_ids, ids):
            matches.append((start, start + len(ids) - 1))
    if not matches:
        raise ValueError(f"Could not find relation word token for {word!r}")
    # Last occurrence is usually the options/list word, not earlier text.
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


def compute_relation_directions(npz_path: Path, layers: List[int]) -> Tuple[np.ndarray, np.ndarray, List[int], np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as z:
        all_layers = [int(v) for v in z["decoder_block_index"].tolist()]
        layer_to_col = {L: i for i, L in enumerate(all_layers)}
        cols = [layer_to_col[L] for L in layers]
        X = z["relation_vectors"][:, cols, :].astype(np.float32)
        y = np.asarray([str(v) for v in z["relation"].tolist()], dtype=object)
        sid = z["sample_index"].astype(np.int64) if "sample_index" in z else z["sid"].astype(np.int64)

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
    D_arr = np.stack(D, axis=0)  # [L, 4, H]
    return D_arr, y, layers, sid, X


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
    D, y, layers, sids, _X = compute_relation_directions(npz_path, layers)
    if args.max_samples is not None:
        sids = sids[:args.max_samples]

    print(f"Input npz: {npz_path}")
    print(f"Model: {args.model}")
    print(f"Layers: {layers}")
    print(f"Relations/words: {RELATIONS}")

    records_by_sid = load_records(Path(args.prompt_path), download=args.download, num_workers=args.num_workers)
    records = [records_by_sid[int(sid)] for sid in sids if int(sid) in records_by_sid]
    if not records:
        raise RuntimeError("No records matched npz sample_index/sid")
    print(f"Matched records: {len(records)}")

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

    model = model_cls.from_pretrained(spec.repo_id, **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    configure_processor(model, processor)
    device = torch.device(args.device)

    H = int(D.shape[-1])
    Lnum = len(layers)
    sum_raw = np.zeros((Lnum, len(RELATIONS), H), dtype=np.float64)
    sum_centered = np.zeros_like(sum_raw)
    count = 0
    errors: List[Dict[str, Any]] = []

    for rec in tqdm(records, desc="Extracting relation-word hidden states"):
        try:
            rendered = build_chat_prompt(processor, rec.question, append_options_if_missing=args.append_options_if_missing)
            batch = processor(text=[rendered], images=[rec.image], return_tensors="pt")
            batch = move_batch(batch, device)
            input_ids = batch["input_ids"][0].detach().cpu().tolist()
            word_pos = [find_word_last_token(processor.tokenizer, input_ids, w) for w in RELATIONS]

            with torch.inference_mode():
                outputs = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
            states = hidden_tuple(outputs)
            if int(states[-1].shape[1]) != len(input_ids):
                raise RuntimeError(f"hidden/input length mismatch: hidden={states[-1].shape[1]}, input={len(input_ids)}")

            per_layer_words = []
            for li, block in enumerate(layers):
                hw = torch.stack([states[block + 1][0, p] for p in word_pos], dim=0).detach().float().cpu().numpy()
                if hw.shape != (len(RELATIONS), H):
                    raise RuntimeError(f"Unexpected word hidden shape {hw.shape}, expected {(len(RELATIONS), H)}")
                per_layer_words.append(hw)
            W = np.stack(per_layer_words, axis=0)  # [L, 4, H]
            sum_raw += W
            sum_centered += W - W.mean(axis=1, keepdims=True)
            count += 1

            del outputs, states, batch
            if torch.cuda.is_available() and count % 50 == 0:
                torch.cuda.empty_cache()
        except Exception as exc:
            errors.append({
                "sid": rec.sid,
                "question": rec.question,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    if count == 0:
        raise RuntimeError(f"No word states extracted. First errors: {errors[:3]}")

    W_raw = l2norm((sum_raw / count).astype(np.float32), axis=2)
    W_centered = l2norm((sum_centered / count).astype(np.float32), axis=2)

    summary: Dict[str, Any] = {
        "npz": str(npz_path),
        "model": args.model,
        "repo_id": spec.repo_id,
        "layers": layers,
        "relations": RELATIONS,
        "n_word_records_used": count,
        "n_errors": len(errors),
        "results": {},
    }

    for li, L in enumerate(layers):
        for mode_name, Wmode in [("raw", W_raw), ("prompt_centered", W_centered)]:
            S = D[li] @ Wmode[li].T
            diag = np.diag(S)
            off = S.copy()
            np.fill_diagonal(off, -np.inf)
            margin = diag - off.max(axis=1)
            row_argmax = [RELATIONS[int(j)] for j in S.argmax(axis=1)]
            key = f"L{L}_{mode_name}"
            summary["results"][key] = {
                "matrix": S.tolist(),
                "diag_mean": float(diag.mean()),
                "diag_min": float(diag.min()),
                "diag_margin_mean": float(margin.mean()),
                "row_argmax": row_argmax,
                "row_match_acc": float(np.mean([row_argmax[i] == RELATIONS[i] for i in range(len(RELATIONS))])),
            }
            csv_path = out_dir / f"direction_word_similarity_L{L}_{mode_name}.csv"
            png_path = out_dir / f"direction_word_similarity_L{L}_{mode_name}.png"
            write_csv(csv_path, S, RELATIONS, RELATIONS)
            plot_heatmap(png_path, S, f"{args.model} Controlled A L{L} ({mode_name})", RELATIONS, RELATIONS)

        pc = summary["results"][f"L{L}_prompt_centered"]
        raw = summary["results"][f"L{L}_raw"]
        print(
            f"L{L}: prompt_centered diag_mean={pc['diag_mean']:.3f} "
            f"margin={pc['diag_margin_mean']:.3f} row_match={pc['row_match_acc']:.2f} | "
            f"raw diag_mean={raw['diag_mean']:.3f} margin={raw['diag_margin_mean']:.3f} row_match={raw['row_match_acc']:.2f}"
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
