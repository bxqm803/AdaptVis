#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Count GQA two-object basic spatial-relation candidates.

Expected official GQA files:
  train_sceneGraphs.json
  train_balanced_questions.json

The script streams large JSON dictionaries with ijson.

It reports several stages:
  1. relation questions with exactly one relate operation
  2. two grounded objects
  3. positive (yes) direct questions
  4. both object boxes >= min-area-ratio of the image
  5. scene-graph relation agrees with the question program
  6. strict unique-name subset (automatic approximation of manual ambiguity filtering)

Canonical labels:
  left, right, above, below, on, under, front, behind

It does not merge above->on or below->under. Both raw and canonical
relation counts are retained so you can decide the final benchmark taxonomy.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


TARGET_RELATIONS = (
    "left",
    "right",
    "above",
    "below",
    "on",
    "under",
    "front",
    "behind",
)

REL_ALIASES = {
    # horizontal
    "left": "left",
    "left of": "left",
    "to left of": "left",
    "to the left of": "left",
    "on the left of": "left",

    "right": "right",
    "right of": "right",
    "to right of": "right",
    "to the right of": "right",
    "on the right of": "right",

    # vertical: keep geometric and support-like predicates separate
    "above": "above",
    "over": "above",

    "below": "below",
    "beneath": "below",

    "on": "on",
    "on top of": "on",

    "under": "under",
    "underneath": "under",

    # depth
    "in front of": "front",
    "in front": "front",
    "front of": "front",
    "front": "front",

    "behind": "behind",
    "in back of": "behind",
    "in the back of": "behind",
}

INVERSE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "on": "under",
    "under": "on",
    "front": "behind",
    "behind": "front",
}

ID_RE = re.compile(r"\((\d+)\)")


def norm_text(x: Any) -> str:
    s = str(x or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def norm_relation(x: Any) -> Optional[str]:
    s = norm_text(x)
    return REL_ALIASES.get(s)


def require_ijson():
    try:
        import ijson  # type: ignore
        return ijson
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'ijson'. Install it with:\n"
            "  pip install ijson\n"
        ) from exc


def iter_json_dict(path: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    ijson = require_ijson()
    with path.open("rb") as f:
        for key, value in ijson.kvitems(f, ""):
            if isinstance(value, dict):
                yield str(key), value


def extract_ids_from_value(value: Any) -> List[str]:
    out: List[str] = []

    if value is None:
        return out
    if isinstance(value, str):
        # Annotation values are often bare object IDs.
        if value.isdigit():
            out.append(value)
        out.extend(ID_RE.findall(value))
        return out
    if isinstance(value, (int, float)):
        if int(value) == value:
            out.append(str(int(value)))
        return out
    if isinstance(value, dict):
        for v in value.values():
            out.extend(extract_ids_from_value(v))
        return out
    if isinstance(value, (list, tuple)):
        for v in value:
            out.extend(extract_ids_from_value(v))
        return out

    return out


def semantic_object_ids(question: Dict[str, Any]) -> List[str]:
    ids: List[str] = []

    annotations = question.get("annotations", {})
    if isinstance(annotations, dict):
        ids.extend(
            extract_ids_from_value(annotations.get("question", {}))
        )

    for step in question.get("semantic", []) or []:
        if isinstance(step, dict):
            ids.extend(ID_RE.findall(str(step.get("argument", ""))))

    return list(dict.fromkeys(ids))


def parse_relate_argument(argument: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Typical GQA form:
      "on, subject, apple (271881)"
      "left of, object, table (279472)"

    Returns:
      canonical_relation, role(subject/object/None), related_object_id
    """
    text = str(argument or "").strip()
    parts = [p.strip() for p in text.split(",")]

    raw_rel = parts[0] if parts else text
    relation = norm_relation(raw_rel)

    role: Optional[str] = None
    for part in parts[1:]:
        p = norm_text(part)
        if p in {"subject", "object"}:
            role = p
            break

    ids = ID_RE.findall(text)
    related_id = ids[-1] if ids else None
    return relation, role, related_id


def step_primary_id(step: Dict[str, Any]) -> Optional[str]:
    ids = ID_RE.findall(str(step.get("argument", "")))
    return ids[-1] if ids else None


def infer_direct_relation(
    question: Dict[str, Any],
) -> Optional[Tuple[str, str, str, str]]:
    """
    Infer (relation, subject_id, reference_id, raw_relate_argument)
    for a direct single-relate program.

    Returns None when orientation cannot be recovered reliably.
    """
    semantic = question.get("semantic", []) or []
    if not isinstance(semantic, list):
        return None

    relate_positions = [
        i
        for i, step in enumerate(semantic)
        if isinstance(step, dict)
        and norm_text(step.get("operation")) == "relate"
    ]
    if len(relate_positions) != 1:
        return None

    rel_pos = relate_positions[0]
    rel_step = semantic[rel_pos]
    raw_argument = str(rel_step.get("argument", ""))
    relation, role, related_id = parse_relate_argument(raw_argument)

    if relation not in TARGET_RELATIONS or related_id is None:
        return None

    # Resolve the selected/base object through dependencies first.
    base_id: Optional[str] = None
    dependencies = rel_step.get("dependencies", [])
    if isinstance(dependencies, list):
        for dep in reversed(dependencies):
            try:
                dep_i = int(dep)
            except Exception:
                continue
            if 0 <= dep_i < len(semantic):
                candidate = step_primary_id(semantic[dep_i])
                if candidate and candidate != related_id:
                    base_id = candidate
                    break

    # Some GQA programs omit dependencies; use the nearest prior grounded step.
    if base_id is None:
        for prior in reversed(semantic[:rel_pos]):
            if not isinstance(prior, dict):
                continue
            candidate = step_primary_id(prior)
            if candidate and candidate != related_id:
                base_id = candidate
                break

    # Final fallback: exactly two grounded object IDs in question/program.
    if base_id is None:
        object_ids = semantic_object_ids(question)
        alternatives = [x for x in object_ids if x != related_id]
        if len(alternatives) == 1:
            base_id = alternatives[0]

    if base_id is None or base_id == related_id:
        return None

    if role == "subject":
        subject_id, reference_id = related_id, base_id
    elif role == "object":
        subject_id, reference_id = base_id, related_id
    else:
        # Cannot safely orient the relation.
        return None

    return relation, subject_id, reference_id, raw_argument


def iter_relations(obj: Dict[str, Any]) -> Iterator[Tuple[str, str]]:
    relations = obj.get("relations", [])
    if isinstance(relations, dict):
        iterable = relations.values()
    elif isinstance(relations, list):
        iterable = relations
    else:
        iterable = []

    for rel in iterable:
        if not isinstance(rel, dict):
            continue
        name = str(rel.get("name", ""))
        target = rel.get("object")
        if target is None:
            continue
        yield name, str(target)


def scene_relation_matches(
    objects: Dict[str, Any],
    subject_id: str,
    reference_id: str,
    relation: str,
) -> bool:
    subject = objects.get(subject_id)
    reference = objects.get(reference_id)
    if not isinstance(subject, dict) or not isinstance(reference, dict):
        return False

    # Direct edge subject -> reference.
    for raw_name, target_id in iter_relations(subject):
        if target_id == reference_id and norm_relation(raw_name) == relation:
            return True

    # Equivalent inverse edge reference -> subject.
    inverse = INVERSE.get(relation)
    if inverse is not None:
        for raw_name, target_id in iter_relations(reference):
            if target_id == subject_id and norm_relation(raw_name) == inverse:
                return True

    return False


def object_area_ratio(obj: Dict[str, Any], width: float, height: float) -> float:
    try:
        w = float(obj["w"])
        h = float(obj["h"])
    except Exception:
        return 0.0

    image_area = max(width * height, 1.0)
    return max(w, 0.0) * max(h, 0.0) / image_area


def normalized_name(obj: Dict[str, Any]) -> str:
    return norm_text(obj.get("name", ""))


def add_count(stage_counts: Dict[str, Counter], stage: str, relation: str) -> None:
    stage_counts[stage][relation] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-graphs", required=True)
    parser.add_argument(
        "--questions",
        required=True,
        nargs="+",
        help="One or more question JSON files.",
    )
    parser.add_argument("--min-area-ratio", type=float, default=0.03)
    parser.add_argument(
        "--require-balanced",
        action="store_true",
        help="Require question['isBalanced'] == true.",
    )
    parser.add_argument(
        "--output-report",
        required=True,
    )
    parser.add_argument(
        "--output-candidates",
        default=None,
        help="Optional JSON containing strict unique-name candidates.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200000,
    )
    args = parser.parse_args()

    scene_path = Path(args.scene_graphs)
    question_paths = [Path(p) for p in args.questions]
    report_path = Path(args.output_report)

    for path in [scene_path, *question_paths]:
        if not path.exists():
            raise FileNotFoundError(path)

    funnel = Counter()
    question_stage_counts: Dict[str, Counter] = defaultdict(Counter)
    candidates_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    total_questions = 0

    for question_path in question_paths:
        print(f"[INFO] scanning questions: {question_path}", flush=True)

        for qid, q in iter_json_dict(question_path):
            total_questions += 1
            funnel["questions_total"] += 1

            if args.require_balanced and not bool(q.get("isBalanced", False)):
                continue
            funnel["balanced_or_not_required"] += 1

            types = q.get("types", {})
            if isinstance(types, dict):
                if norm_text(types.get("semantic")) != "relation":
                    continue
            else:
                continue
            funnel["semantic_relation"] += 1

            semantic = q.get("semantic", []) or []
            relate_count = sum(
                1
                for step in semantic
                if isinstance(step, dict)
                and norm_text(step.get("operation")) == "relate"
            )
            if relate_count != 1:
                continue
            funnel["single_relate"] += 1

            inferred = infer_direct_relation(q)
            if inferred is None:
                continue

            relation, subject_id, reference_id, raw_argument = inferred
            add_count(
                question_stage_counts,
                "direct_oriented",
                relation,
            )
            funnel["direct_oriented"] += 1

            grounded_ids = semantic_object_ids(q)
            if len(set(grounded_ids)) != 2:
                continue
            funnel["exactly_two_grounded_objects"] += 1
            add_count(
                question_stage_counts,
                "exactly_two_grounded_objects",
                relation,
            )

            answer = norm_text(q.get("answer", ""))
            if answer not in {"yes", "true"}:
                continue
            funnel["positive_yes"] += 1
            add_count(
                question_stage_counts,
                "positive_yes",
                relation,
            )

            image_id = str(q.get("imageId", ""))
            if not image_id:
                continue

            candidates_by_image[image_id].append(
                {
                    "question_id": qid,
                    "image_id": image_id,
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "relation": relation,
                    "subject_id": subject_id,
                    "reference_id": reference_id,
                    "raw_relate_argument": raw_argument,
                }
            )

            if (
                args.progress_every > 0
                and total_questions % args.progress_every == 0
            ):
                print(
                    f"[INFO] questions={total_questions:,} "
                    f"positive_candidates={funnel['positive_yes']:,}",
                    flush=True,
                )

    print(
        f"[INFO] positive direct questions={sum(len(v) for v in candidates_by_image.values()):,} "
        f"across images={len(candidates_by_image):,}",
        flush=True,
    )

    graph_stage_counts: Dict[str, Counter] = defaultdict(Counter)
    strict_records: List[Dict[str, Any]] = []
    strict_keys = set()
    seen_candidate_images = set()

    scanned_graphs = 0
    print(f"[INFO] scanning scene graphs: {scene_path}", flush=True)

    for image_id, graph in iter_json_dict(scene_path):
        scanned_graphs += 1
        image_candidates = candidates_by_image.get(image_id)
        if not image_candidates:
            continue

        seen_candidate_images.add(image_id)

        objects = graph.get("objects", {})
        if not isinstance(objects, dict):
            continue

        try:
            width = float(graph["width"])
            height = float(graph["height"])
        except Exception:
            continue

        name_counts = Counter(
            normalized_name(obj)
            for obj in objects.values()
            if isinstance(obj, dict) and normalized_name(obj)
        )

        for rec in image_candidates:
            relation = rec["relation"]
            subject_id = rec["subject_id"]
            reference_id = rec["reference_id"]

            subject = objects.get(subject_id)
            reference = objects.get(reference_id)
            if not isinstance(subject, dict) or not isinstance(reference, dict):
                continue

            add_count(
                graph_stage_counts,
                "objects_exist",
                relation,
            )

            subject_area = object_area_ratio(subject, width, height)
            reference_area = object_area_ratio(reference, width, height)

            if (
                subject_area < args.min_area_ratio
                or reference_area < args.min_area_ratio
            ):
                continue

            add_count(
                graph_stage_counts,
                "area_filtered",
                relation,
            )

            if not scene_relation_matches(
                objects,
                subject_id,
                reference_id,
                relation,
            ):
                continue

            add_count(
                graph_stage_counts,
                "scene_relation_agrees",
                relation,
            )

            subject_name = normalized_name(subject)
            reference_name = normalized_name(reference)

            unique_names = (
                subject_name
                and reference_name
                and subject_name != reference_name
                and name_counts[subject_name] == 1
                and name_counts[reference_name] == 1
            )

            if not unique_names:
                continue

            add_count(
                graph_stage_counts,
                "strict_unique_names",
                relation,
            )

            dedupe_key = (
                image_id,
                subject_id,
                reference_id,
                relation,
            )
            if dedupe_key in strict_keys:
                continue
            strict_keys.add(dedupe_key)

            record = dict(rec)
            record.update(
                {
                    "subject": subject_name,
                    "reference": reference_name,
                    "subject_box": [
                        subject.get("x"),
                        subject.get("y"),
                        subject.get("w"),
                        subject.get("h"),
                    ],
                    "reference_box": [
                        reference.get("x"),
                        reference.get("y"),
                        reference.get("w"),
                        reference.get("h"),
                    ],
                    "subject_area_ratio": subject_area,
                    "reference_area_ratio": reference_area,
                }
            )
            strict_records.append(record)
            add_count(
                graph_stage_counts,
                "strict_unique_deduplicated",
                relation,
            )

    missing_graph_images = set(candidates_by_image).difference(
        seen_candidate_images
    )

    report = {
        "inputs": {
            "scene_graphs": str(scene_path),
            "questions": [str(p) for p in question_paths],
            "min_area_ratio": args.min_area_ratio,
            "require_balanced": bool(args.require_balanced),
        },
        "target_relations": list(TARGET_RELATIONS),
        "funnel": dict(funnel),
        "question_stage_counts": {
            stage: dict(counter)
            for stage, counter in question_stage_counts.items()
        },
        "graph_stage_counts": {
            stage: dict(counter)
            for stage, counter in graph_stage_counts.items()
        },
        "strict_unique_deduplicated_total": len(strict_records),
        "strict_unique_deduplicated_counts": dict(
            Counter(r["relation"] for r in strict_records)
        ),
        "unique_images_in_strict_set": len(
            set(r["image_id"] for r in strict_records)
        ),
        "candidate_images_missing_from_scene_graphs": len(
            missing_graph_images
        ),
        "notes": {
            "strict_unique_names": (
                "Automatic conservative approximation of the paper's manual "
                "ambiguity filtering. It requires each target object name to "
                "occur exactly once in the image."
            ),
            "positive_yes": (
                "Negative yes/no questions are excluded because a false "
                "stated relation does not uniquely determine the opposite."
            ),
            "vertical_labels": (
                "above/below and on/under are counted separately."
            ),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.output_candidates:
        candidate_path = Path(args.output_candidates)
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(
            json.dumps(strict_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] strict candidates: {candidate_path}")

    print("\n=== STRICT UNIQUE-NAME DEDUPLICATED COUNTS ===")
    strict_counts = Counter(r["relation"] for r in strict_records)
    for relation in TARGET_RELATIONS:
        print(f"{relation:>8}: {strict_counts[relation]:,}")
    print(f"{'total':>8}: {len(strict_records):,}")
    print(f"{'images':>8}: {report['unique_images_in_strict_set']:,}")
    print(f"\n[OK] report: {report_path}")


if __name__ == "__main__":
    main()
