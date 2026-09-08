#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_qwen3b_synthetic_to_coco_oracle_audit_v1.py

Freshly re-extract Qwen3B late directions from synthetic_shapes_4dir_400,
then evaluate baseline + GT-oracle on COCO_two.

Important:
- does NOT load any previously saved direction .npz
- source directions are fit ONLY from the synthetic shape images
- COCO hidden states are never used for direction fitting
- source prompt wording is automatically changed to the dominant COCO
  question template before extracting the synthetic directions
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

import eval_synthetic_shapes_to_coco_late_direction_qwen3b_v1 as exp
import eval_synthetic_shapes_to_coco_multiselector_multimodel_v2 as mm


RELATIONS = ("left", "right", "above", "below")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-dir", default="synthetic_shapes_4dir_400")
    p.add_argument(
        "--prompt-jsonl",
        default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl",
    )
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--actuator-layers", default="32-35")
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--audit-examples", type=int, default=8)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    # fields expected by imported experiment helpers
    p.add_argument("--synthetic-labels", default=None)
    p.add_argument("--source-max-samples", type=int, default=None)
    p.add_argument("--target-max-samples", type=int, default=None)
    p.add_argument(
        "--template-filter",
        default="all",
        choices=["all", "real_correct", "real_correct_gray_wrong"],
    )

    args = p.parse_args()
    args.model = "qwen-3b"
    return args


def parse_layers(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            out.extend(range(a, b + 1))
        elif part:
            out.append(int(part))
    return list(dict.fromkeys(out))


def normalize_template(question, subject, reference):
    text = str(question)

    pairs = [
        (str(subject), "__SUBJECT__"),
        (str(reference), "__REFERENCE__"),
    ]
    pairs.sort(key=lambda x: len(x[0]), reverse=True)

    for phrase, token in pairs:
        pat = re.compile(re.escape(phrase), re.IGNORECASE)
        text, n = pat.subn(token, text, count=1)
        if n != 1:
            return None

    return (
        text.replace("__SUBJECT__", "{subject}")
            .replace("__REFERENCE__", "{reference}")
    )


def infer_dominant_coco_template(records):
    counts = Counter()

    for r in records:
        t = normalize_template(
            r["question_text"],
            r["subject"],
            r["reference"],
        )
        if t is not None:
            counts[t] += 1

    if not counts:
        raise RuntimeError("Could not infer COCO question template.")

    return counts.most_common(1)[0], counts


def sha256_vec(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return hashlib.sha256(x.tobytes()).hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    outdir = Path(args.output_dir)

    if args.overwrite and outdir.exists():
        import shutil
        shutil.rmtree(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    actuator_layers = parse_layers(args.actuator_layers)

    # ------------------------------------------------------------------
    # Load COCO metadata/prompts ONLY. No model forward / hidden states.
    # ------------------------------------------------------------------
    target_records, target_audit = exp.load_coco_records(args)

    (coco_template, template_count), all_templates = (
        infer_dominant_coco_template(target_records)
    )

    # ------------------------------------------------------------------
    # Load synthetic source, then FORCE source questions to use the exact
    # dominant COCO question template.
    # ------------------------------------------------------------------
    source_records = exp.load_synthetic_records(args)

    for r in source_records:
        r["question_text"] = coco_template.format(
            subject=r["subject"],
            reference=r["reference"],
        )

    print("\n" + "=" * 150)
    print("PROMPT / PROVENANCE AUDIT")
    print("=" * 150)

    print(
        f"Dominant COCO template: {template_count}/{len(target_records)}"
    )
    print(f"  {coco_template}")
    print(f"Unique normalized COCO templates: {len(all_templates)}")

    print("\nSynthetic relation counts:")
    print(dict(Counter(r["relation"] for r in source_records)))

    print("\nCOCO relation counts:")
    print(dict(Counter(r["relation"] for r in target_records)))

    n = min(
        args.audit_examples,
        len(source_records),
        len(target_records),
    )

    print("\nSynthetic questions actually used for direction extraction:")
    for r in source_records[:n]:
        print(
            f"  SYN  sid={r['sid']:3d} GT={r['relation']:>5s} | "
            f"{r['question_text']}"
        )

    print("\nCOCO questions actually used for evaluation:")
    for r in target_records[:n]:
        print(
            f"  COCO sid={r['sid']:3d} GT={r['relation']:>5s} | "
            f"{r['question_text']}"
        )

    print("\nLabel mapping:")
    print("  synthetic left  -> left")
    print("  synthetic right -> right")
    print("  synthetic on    -> above")
    print("  synthetic under -> below")
    print("=" * 150)

    # ------------------------------------------------------------------
    # Explicit-GPU loader from v2. No Accelerate device_map.
    # ------------------------------------------------------------------
    model, processor, layers, decoder_path, spec = mm.load_model(args)

    for layer in actuator_layers:
        if not 0 <= layer < len(layers):
            raise RuntimeError(
                f"Invalid actuator L{layer}; model has {len(layers)} layers."
            )

    # ------------------------------------------------------------------
    # FRESH extraction from the 400 synthetic images.
    #
    # No cached direction file is loaded.
    # No COCO model forward has happened before this point.
    # ------------------------------------------------------------------
    source_rows, delta_cache = exp.collect_synthetic_source(
        model,
        processor,
        layers,
        source_records,
        actuator_layers,
        args,
    )

    templates, template_counts = exp.fit_synthetic_templates(
        source_records,
        source_rows,
        delta_cache,
        actuator_layers,
        args.template_filter,
    )

    exp.write_csv(
        outdir / "synthetic_source_generation.csv",
        source_rows,
    )

    # Save fresh direction bank + fingerprints.
    arrays = {
        "relation_order": np.asarray(RELATIONS, dtype=object),
        "actuator_layers": np.asarray(actuator_layers, dtype=np.int32),
    }

    manifest = []

    print("\n" + "=" * 150)
    print("FRESH SYNTHETIC DIRECTION BANK")
    print("=" * 150)
    print(f"fit counts={template_counts}")

    for layer in actuator_layers:
        for relation in RELATIONS:
            vec = templates[layer]["shared"][relation]
            fp = sha256_vec(vec)
            norm = float(np.linalg.norm(vec))

            arrays[f"L{layer}_{relation}"] = vec

            manifest.append({
                "source_dataset": str(args.synthetic_dir),
                "layer": layer,
                "relation": relation,
                "norm": norm,
                "sha256": fp,
                "fit_uses_coco_hidden_states": False,
            })

            print(
                f"L{layer:02d} {relation:>5s} | "
                f"norm={norm:.6f} | sha256={fp[:16]}..."
            )

    np.savez_compressed(
        outdir / "fresh_synthetic_direction_bank.npz",
        **arrays,
    )

    write_csv(
        outdir / "direction_manifest.csv",
        manifest,
    )

    print("\nDirection source = synthetic shapes ONLY")
    print("COCO hidden states used to fit directions = FALSE")
    print("=" * 150)

    # ------------------------------------------------------------------
    # Only NOW touch COCO with the model.
    # baseline + oracle selected frozen synthetic writer
    # ------------------------------------------------------------------
    target_rows = exp.evaluate_coco_target(
        model,
        processor,
        layers,
        target_records,
        templates,
        actuator_layers,
        args,
    )

    exp.write_csv(
        outdir / "target_details.csv",
        target_rows,
    )

    summary, per_relation = exp.build_target_summary(
        target_rows
    )

    summary.update({
        "model": "qwen-3b",
        "synthetic_source_N": len(source_records),
        "direction_source": str(args.synthetic_dir),
        "direction_fit_uses_coco_hidden_states": False,
        "source_prompt_template": coco_template,
        "source_prompt_template_coco_coverage": (
            f"{template_count}/{len(target_records)}"
        ),
        "actuator_layers": ",".join(map(str, actuator_layers)),
        "scale": args.scale,
    })

    exp.write_csv(
        outdir / "summary.csv",
        [summary],
    )

    exp.write_csv(
        outdir / "per_relation.csv",
        per_relation,
    )

    metadata = {
        "model": "qwen-3b",
        "repo_id": spec.repo_id,
        "decoder_path": decoder_path,
        "synthetic_source": str(args.synthetic_dir),
        "synthetic_source_N": len(source_records),
        "direction_source": "fresh synthetic extraction",
        "loads_previous_direction_cache": False,
        "direction_fit_uses_coco_hidden_states": False,
        "coco_gt_used_for_direction_fit": False,
        "coco_gt_used_for_oracle_selection": True,
        "coco_gt_used_for_metrics": True,
        "coco_prompt_template": coco_template,
        "coco_prompt_template_count": template_count,
        "coco_N": len(target_records),
        "synthetic_label_mapping": {
            "left": "left",
            "right": "right",
            "on": "above",
            "under": "below",
        },
        "actuator_layers": actuator_layers,
        "scale": args.scale,
        "gray_value": args.gray_value,
        "target_audit": target_audit,
    }

    (outdir / "audit_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\n" + "=" * 150)
    print("AUDITED ACTUAL GREEDY GENERATION: FRESH SYNTHETIC DIRECTIONS -> COCO")
    print("=" * 150)

    print(
        f"N_TARGET={summary['N']} | "
        f"N_SYNTHETIC={len(source_records)} | "
        f"layers={actuator_layers}"
    )

    print(
        f"COCO baseline              : "
        f"{summary['baseline_acc']:.4f}"
    )

    print(
        f"fresh synthetic oracle     : "
        f"{summary['synthetic_oracle_acc']:.4f} "
        f"({summary['gain']:+.4f}) | "
        f"W2C={summary['W2C']} "
        f"C2W={summary['C2W']} "
        f"net={summary['net']:+d}"
    )

    print(
        f"repair(base-wrong)         : "
        f"{summary['repair_rate_on_baseline_wrong']:.4f}"
    )

    print(
        f"preserve(base-correct)     : "
        f"{summary['preserve_rate_on_baseline_correct']:.4f}"
    )

    print("\nPer relation:")

    for r in per_relation:
        print(
            f"{r['relation']:>5s} | "
            f"N={r['N']:3d} | "
            f"base={r['baseline_acc']:.4f} | "
            f"syn_oracle={r['synthetic_oracle_acc']:.4f} | "
            f"W2C/C2W={r['W2C']}/{r['C2W']}"
        )

    print("=" * 150)

    print("\nPROVENANCE:")
    print(f"  source = {args.synthetic_dir}")
    print("  previous direction cache loaded = FALSE")
    print("  COCO hidden states used in fit = FALSE")
    print(f"  source prompt = {coco_template}")
    print(
        f"  prompt coverage = "
        f"{template_count}/{len(target_records)} COCO samples"
    )


if __name__ == "__main__":
    main()
