#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test whether pure horizontal image flip (with the SAME prompt) produces the
expected left<->right answer change on COCO-two samples.

Purpose
=======
This is the first sanity check before doing any mechanistic scan.

Compared with the previous query-swap setup

    "Where is A relative to B?"
        vs
    "Where is B relative to A?"

this script keeps the TEXT fixed and changes only the IMAGE:

    original image
        vs
    horizontally flipped image

restricted to gold relation in {left, right}.

So the counterfactual changes:

    visual horizontal geometry

while keeping fixed:

    object identities
    subject/reference roles in the question
    prompt text
    answer vocabulary

Main outputs
============
summary.txt
flip_generation_summary.json
flip_generation_pairs.csv
flip_generation_pairs.jsonl

Key metrics
===========
original_acc
flip_aligned_acc
    Accuracy on the horizontally flipped image after mapping
        left -> right
        right -> left

both_correct
    original correct AND flipped correct (aligned to opposite label)

prediction_opposition_rate
    whether model prediction flips exactly between left/right

clean_correct_to_opposite_rate
    among clean-correct examples, fraction whose flipped prediction becomes the
    opposite gold label

Requirements
============
This script reuses the same repo helpers and model bundle as your earlier runs.

Recommended
===========
CUDA_VISIBLE_DEVICES=0 python -u eval_coco_horizontal_flip_generation_v1_fix.py \
  --model qwen-3b \
  --source-output-dir output/spatial_storage_transport_utilization/coco/qwen-3b \
  --sample-scope das_eval \
  --pair-status all \
  --restrict-relations left,right \
  --device cuda:0 \
  --output-dir output/qwen3b_coco_horizontal_flip_generation_v1 \
  --overwrite
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

RELATIONS = ("left", "right", "above", "below")
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}
LR = ("left", "right")


def import_file(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: Mapping[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs: Iterable[float]) -> float:
    arr = np.asarray(list(xs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def parse_relations(text: str) -> List[str]:
    vals = []
    for s in str(text).split(","):
        s = s.strip().lower()
        if not s:
            continue
        if s not in RELATIONS:
            raise ValueError(f"unknown relation={s}")
        if s not in vals:
            vals.append(s)
    if not vals:
        raise ValueError("empty relation set")
    return vals


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--dataset", default="coco_two")
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--object-state", default="mean")
    p.add_argument(
        "--source-output-dir",
        default="output/spatial_storage_transport_utilization/coco/qwen-3b",
    )
    p.add_argument(
        "--sample-scope",
        default="das_eval",
        choices=("das_eval", "all"),
    )
    p.add_argument(
        "--pair-status",
        default="all",
        choices=("all", "both_correct", "original_only", "swapped_only", "both_wrong"),
        help="Optional filter using existing query-swap status in extraction rows.",
    )
    p.add_argument(
        "--restrict-relations",
        default="left,right",
        help="Usually left,right for this experiment.",
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=17)
    p.add_argument("--cache-flipped-images", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


@dataclass
class FlipEvalItem:
    sid: int
    gt: str
    opposite_gt: str
    clean_prediction: str
    flip_prediction: str
    clean_correct: bool
    flip_correct_aligned: bool
    prediction_opposes: bool
    clean_to_opposite: bool
    clean_margin: float
    flip_margin_aligned: float
    image_path: str
    prompt_text: str


def try_get_attr(obj: Any, names: Sequence[str]) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return None


def try_get_from_mapping(mp: Mapping[str, Any], names: Sequence[str]) -> Any:
    for n in names:
        if n in mp and mp[n] is not None:
            return mp[n]
    return None


def infer_image_path(row: Mapping[str, Any], records_by_sid: Mapping[int, Mapping[str, Any]], pair: Any) -> str:
    sid = int(row["sid"])
    rec = records_by_sid.get(sid, {})
    # Search common names on pair, row, record.
    cand = try_get_attr(pair, [
        "image_path", "original_image_path", "img_path", "image_file", "image",
    ])
    if cand is None:
        cand = try_get_from_mapping(row, [
            "image_path", "img_path", "image", "file_name", "filename",
        ])
    if cand is None:
        cand = try_get_from_mapping(rec, [
            "image_path", "img_path", "image", "file_name", "filename",
        ])
    if cand is None:
        raise KeyError(
            f"Could not infer image path for sid={sid}. "
            f"pair attrs={list(getattr(pair, '__dict__', {}).keys())} "
            f"row keys={list(row.keys())} rec keys={list(rec.keys())}"
        )
    return str(cand)


def infer_prompt_text(row: Mapping[str, Any], prompt_rows: Mapping[int, Mapping[str, Any]], pair: Any) -> str:
    sid = int(row["sid"])
    prow = prompt_rows.get(sid, {})
    cand = try_get_attr(pair, [
        "original_prompt", "prompt_text", "question", "original_question",
        "clean_prompt", "original_text", "text",
    ])
    if cand is None:
        cand = try_get_from_mapping(row, [
            "prompt", "question", "text", "input_text",
        ])
    if cand is None:
        cand = try_get_from_mapping(prow, [
            "prompt", "question", "text", "input_text",
        ])
    if cand is None:
        raise KeyError(
            f"Could not infer prompt text for sid={sid}. "
            f"pair attrs={list(getattr(pair, '__dict__', {}).keys())} "
            f"row keys={list(row.keys())} prompt_row keys={list(prow.keys())}"
        )
    return str(cand)


def open_image(path: str) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    img = Image.open(p).convert("RGB")
    return img


def horizontal_flip_pil(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def encode_single(processor, prompt_text: str, image: Image.Image, device: torch.device):
    # Try a few common call patterns across VLM processors.
    errs = []

    call_patterns = [
        lambda: processor(text=[prompt_text], images=[image], return_tensors="pt"),
        lambda: processor(text=prompt_text, images=image, return_tensors="pt"),
        lambda: processor(images=[image], text=[prompt_text], return_tensors="pt"),
        lambda: processor(images=image, text=prompt_text, return_tensors="pt"),
    ]

    out = None
    for fn in call_patterns:
        try:
            out = fn()
            break
        except Exception as e:
            errs.append(f"{type(e).__name__}: {e}")

    if out is None:
        raise RuntimeError("Processor encoding failed:\n" + "\n".join(errs))

    batch = {}
    for k, v in out.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
        else:
            batch[k] = v
    return batch


def relation_scores(logits_last: torch.Tensor, relation_token_map: Mapping[str, Sequence[int]]) -> Tuple[str, np.ndarray]:
    scores = []
    for rel in RELATIONS:
        ids = relation_token_map[rel]
        vals = logits_last[ids].detach().float().cpu().numpy()
        scores.append(float(np.max(vals)))
    scores = np.asarray(scores, dtype=np.float32)
    pred = RELATIONS[int(np.argmax(scores))]
    return pred, scores


def parse_summary_counts(items: Sequence[FlipEvalItem]) -> Dict[str, Any]:
    n = len(items)
    clean_acc = sum(int(x.clean_correct) for x in items) / n if n else float("nan")
    flip_acc = sum(int(x.flip_correct_aligned) for x in items) / n if n else float("nan")
    both_correct = sum(int(x.clean_correct and x.flip_correct_aligned) for x in items) / n if n else float("nan")
    oppose = sum(int(x.prediction_opposes) for x in items) / n if n else float("nan")

    cc = [x for x in items if x.clean_correct]
    clean_to_opp = (
        sum(int(x.clean_to_opposite) for x in cc) / len(cc)
        if cc else float("nan")
    )

    return {
        "N": n,
        "original_acc": clean_acc,
        "flip_aligned_acc": flip_acc,
        "both_correct": both_correct,
        "prediction_opposition_rate": oppose,
        "clean_correct_to_opposite_rate": clean_to_opp,
        "mean_clean_margin": safe_mean(x.clean_margin for x in items),
        "mean_flip_margin_aligned": safe_mean(x.flip_margin_aligned for x in items),
    }


def format_pct(x: float) -> str:
    return f"{100*x:.2f}%" if np.isfinite(x) else "n/a"


def main():
    args = parse_args()
    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)

    keep_rel = parse_relations(args.restrict_relations)
    outdir = Path(args.output_dir)

    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.cache_flipped_images:
        (outdir / "flipped_images").mkdir(parents=True, exist_ok=True)

    error_path = outdir / "errors.jsonl"

    base = import_file(Path("analyze_coco_centroid_generation_step1_v4.py"), "_flip_base")
    ioi = import_file(Path("analyze_coco_ioi_backward_circuit_v1.py"), "_flip_ioi")
    producer = import_file(Path("analyze_coco_producer_qk_ov_v1.py"), "_flip_prod")
    receiver = import_file(Path("analyze_coco_receiver_qkv_v1.py"), "_flip_recv")
    v3 = import_file(
        Path("analyze_spatial_storage_transport_utilization_v3.py"),
        "_flip_v3",
    )
    writer = import_file(Path("validate_writer_to_das_u_v1.py"), "_flip_writer")

    rows = [
        r for r in writer.read_jsonl(Path(args.source_output_dir) / "extraction.jsonl")
        if str(r.get("gt")) in keep_rel
    ]

    if args.sample_scope == "das_eval":
        # Same convention as DAS scripts.
        das_cfg_path = Path("output/qwen3b_relation_das_full/config.json")
        if das_cfg_path.exists():
            das_cfg = json.loads(das_cfg_path.read_text(encoding="utf-8"))
            eval_sids = set(map(int, das_cfg.get("eval_sids", [])))
            if eval_sids:
                rows = [r for r in rows if int(r["sid"]) in eval_sids]

    if args.pair_status != "all":
        rows = [
            r for r in rows
            if str(r.get("generation_pair_status", "")) == args.pair_status
        ]

    rows = writer.stratified_subset(rows=rows, limit=args.max_samples, seed=args.sample_seed)
    if not rows:
        raise RuntimeError("No rows after filtering.")

    device = torch.device(args.device)
    model = processor = None
    items: List[FlipEvalItem] = []

    try:
        model, processor, spec, decoder_layers, decoder_path, relation_token_map = producer.load_model_bundle(
            args=args,
            base=base,
        )
        model.eval()

        saved = args.max_samples
        args.max_samples = None
        try:
            records_by_sid, prompt_rows, audit = ioi.prepare_data_helpers(args, base)
        finally:
            args.max_samples = saved

        print("\n" + "=" * 112)
        print("HORIZONTAL FLIP GENERATION CHECK")
        print("=" * 112)
        print("model          :", args.model)
        print("N samples      :", len(rows))
        print("relations      :", keep_rel)
        print("sample scope   :", args.sample_scope)
        print("pair status    :", args.pair_status)
        print("prompt         : fixed")
        print("counterfactual : horizontal image flip only")
        print("=" * 112, flush=True)

        for row in tqdm(rows, desc="flip-generation"):
            sid = int(row["sid"])
            pair = None
            try:
                pair = receiver.prepare_pair(
                    args=args,
                    row=row,
                    records_by_sid=records_by_sid,
                    prompt_rows=prompt_rows,
                    base=base,
                    v3=v3,
                    processor=processor,
                    device=device,
                )

                prompt_text = infer_prompt_text(row, prompt_rows, pair)
                image_path = infer_image_path(row, records_by_sid, pair)
                img = open_image(image_path)
                img_flip = horizontal_flip_pil(img)

                if args.cache_flipped_images:
                    ext = Path(image_path).suffix or ".jpg"
                    flip_path = outdir / "flipped_images" / f"{sid}{ext}"
                    img_flip.save(flip_path)

                clean_batch = pair.original_batch
                with torch.no_grad():
                    out_clean = model(
                        **clean_batch,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                pred_clean, scores_clean = relation_scores(out_clean.logits[0, -1], relation_token_map)
                del out_clean

                flip_batch = encode_single(processor, prompt_text, img_flip, device)
                with torch.no_grad():
                    out_flip = model(
                        **flip_batch,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                pred_flip, scores_flip = relation_scores(out_flip.logits[0, -1], relation_token_map)
                del out_flip

                gt = str(row["gt"])
                opp = OPPOSITE[gt]
                clean_correct = (pred_clean == gt)
                flip_correct = (pred_flip == opp)
                prediction_opposes = (
                    (pred_clean in LR) and (pred_flip in LR) and (pred_flip == OPPOSITE[pred_clean])
                )
                clean_to_opposite = clean_correct and (pred_flip == opp)

                rel_to_idx = {r: i for i, r in enumerate(RELATIONS)}
                clean_margin = float(scores_clean[rel_to_idx[gt]] - scores_clean[rel_to_idx[opp]])
                flip_margin = float(scores_flip[rel_to_idx[opp]] - scores_flip[rel_to_idx[gt]])

                item = FlipEvalItem(
                    sid=sid,
                    gt=gt,
                    opposite_gt=opp,
                    clean_prediction=pred_clean,
                    flip_prediction=pred_flip,
                    clean_correct=clean_correct,
                    flip_correct_aligned=flip_correct,
                    prediction_opposes=prediction_opposes,
                    clean_to_opposite=clean_to_opposite,
                    clean_margin=clean_margin,
                    flip_margin_aligned=flip_margin,
                    image_path=image_path,
                    prompt_text=prompt_text,
                )
                items.append(item)

                append_jsonl(outdir / "flip_generation_pairs.jsonl", item.__dict__)

            except Exception as e:
                append_jsonl(error_path, {
                    "sid": sid,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                if args.fail_fast:
                    raise
            finally:
                if pair is not None:
                    try:
                        receiver.release_pair(pair)
                    except Exception:
                        pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        summary = parse_summary_counts(items)
        write_json(outdir / "flip_generation_summary.json", summary)
        write_csv(outdir / "flip_generation_pairs.csv", [x.__dict__ for x in items])

        lines = []
        lines.append("=" * 96)
        lines.append("HORIZONTAL FLIP GENERATION SUMMARY")
        lines.append("=" * 96)
        lines.append(f"N                         : {summary['N']}")
        lines.append(f"original_acc              : {format_pct(summary['original_acc'])}")
        lines.append(f"flip_aligned_acc          : {format_pct(summary['flip_aligned_acc'])}")
        lines.append(f"both_correct              : {format_pct(summary['both_correct'])}")
        lines.append(f"prediction_opposition     : {format_pct(summary['prediction_opposition_rate'])}")
        lines.append(f"clean_correct_to_opposite : {format_pct(summary['clean_correct_to_opposite_rate'])}")
        lines.append(f"mean_clean_margin         : {summary['mean_clean_margin']:+.4f}")
        lines.append(f"mean_flip_margin_aligned  : {summary['mean_flip_margin_aligned']:+.4f}")
        lines.append("")
        lines.append("Interpretation:")
        lines.append("  If flip_aligned_acc and prediction_opposition_rate are high,")
        lines.append("  horizontal flip is a viable pure-visual counterfactual for left/right.")
        lines.append("  Then you can build a mechanistic scan using original vs flipped image")
        lines.append("  instead of query-swap.")
        (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n" + "\n".join(lines))

        print("\nSaved:")
        for name in (
            "summary.txt",
            "flip_generation_summary.json",
            "flip_generation_pairs.csv",
            "flip_generation_pairs.jsonl",
        ):
            print(" ", outdir / name)

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
