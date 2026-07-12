#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import ijson


REL_PATTERNS = [
    (r"\bto the left of\b", "left"),
    (r"\bon the left of\b", "left"),
    (r"\bleft of\b", "left"),
    (r"\bto the right of\b", "right"),
    (r"\bon the right of\b", "right"),
    (r"\bright of\b", "right"),
    (r"\bin front of\b", "front"),
    (r"\bon top of\b", "on"),
    (r"\bunderneath\b", "under"),
    (r"\bbeneath\b", "under"),
    (r"\bbehind\b", "behind"),
    (r"\babove\b", "above"),
    (r"\bbelow\b", "below"),
    (r"\bunder\b", "under"),
]

TARGETS = ("left", "right", "above", "below", "on", "under", "front", "behind")

SG_REL = {
    "left": "left", "left of": "left", "to the left of": "left",
    "right": "right", "right of": "right", "to the right of": "right",
    "above": "above", "below": "below",
    "on": "on", "on top of": "on",
    "under": "under", "underneath": "under", "beneath": "under",
    "in front of": "front", "front of": "front",
    "behind": "behind",
}

INVERSE = {
    "left": "right", "right": "left",
    "above": "below", "below": "above",
    "on": "under", "under": "on",
    "front": "behind", "behind": "front",
}


def norm(x):
    x = str(x or "").strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", x)


def iter_dict(path):
    with open(path, "rb") as f:
        yield from ijson.kvitems(f, "")


def find_relation(text):
    text = str(text)
    for pattern, label in REL_PATTERNS:
        m = re.search(pattern, text, flags=re.I)
        if m:
            return label, m.group(0)
    return None


def span_start(x):
    m = re.match(r"(\d+)", str(x))
    return int(m.group(1)) if m else 10**9


def grounded_ids(question):
    annotations = question.get("annotations", {})
    qann = annotations.get("question", {}) if isinstance(annotations, dict) else {}
    if not isinstance(qann, dict):
        return []

    rows = []
    for span, value in qann.items():
        ids = re.findall(r"\d+", str(value))
        if ids:
            rows.append((span_start(span), str(span), ids[0]))

    rows.sort()
    out, seen = [], set()
    for _, span, oid in rows:
        if oid not in seen:
            out.append((span, oid))
            seen.add(oid)
    return out


def iter_relations(obj):
    rels = obj.get("relations", [])
    if isinstance(rels, dict):
        rels = rels.values()
    for item in rels:
        if isinstance(item, dict) and item.get("object") is not None:
            yield SG_REL.get(norm(item.get("name"))), str(item["object"])


def relation_agrees(objects, sid, rid, relation):
    s = objects.get(sid)
    r = objects.get(rid)
    if not isinstance(s, dict) or not isinstance(r, dict):
        return False

    for rel, target in iter_relations(s):
        if rel == relation and target == rid:
            return True

    inv = INVERSE[relation]
    for rel, target in iter_relations(r):
        if rel == inv and target == sid:
            return True

    return False


def area_ratio(obj, width, height):
    try:
        return float(obj["w"]) * float(obj["h"]) / max(float(width) * float(height), 1.0)
    except Exception:
        return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-graphs", required=True)
    p.add_argument("--questions", required=True, nargs="+")
    p.add_argument("--min-area-ratio", type=float, default=0.03)
    p.add_argument("--require-balanced", action="store_true")
    p.add_argument("--output-report", required=True)
    p.add_argument("--output-candidates")
    p.add_argument("--progress-every", type=int, default=200000)
    a = p.parse_args()

    semantic_types = Counter()
    structural_types = Counter()
    funnel = Counter()
    phrase_counts = Counter()
    candidates_by_image = defaultdict(list)
    examples = defaultdict(list)

    for qpath in a.questions:
        print(f"[INFO] scanning questions: {qpath}", flush=True)

        for qid, q in iter_dict(qpath):
            funnel["total"] += 1

            types = q.get("types", {})
            semantic = norm(types.get("semantic"))
            structural = norm(types.get("structural"))
            semantic_types[semantic] += 1
            structural_types[structural] += 1

            if a.require_balanced and not bool(q.get("isBalanced", False)):
                continue
            funnel["balanced"] += 1

            # Correct GQA value: "rel", not "relation".
            if semantic != "rel":
                continue
            funnel["semantic_rel"] += 1

            # Original VG2-style questions are binary verification questions.
            if structural != "verify":
                continue
            funnel["structural_verify"] += 1

            if norm(q.get("answer")) not in {"yes", "true"}:
                continue
            funnel["positive_yes"] += 1

            found = find_relation(q.get("question", ""))
            if found is None:
                continue
            relation, surface = found
            phrase_counts[relation] += 1
            funnel["target_relation_phrase"] += 1

            ids = grounded_ids(q)
            if len(ids) != 2:
                continue
            funnel["two_grounded_objects"] += 1

            image_id = str(q.get("imageId", ""))
            if not image_id:
                continue

            # For direct "Is A REL B?" questions, annotation order follows text order.
            rec = {
                "question_id": str(qid),
                "image_id": image_id,
                "question": q.get("question", ""),
                "answer": q.get("answer", ""),
                "relation": relation,
                "surface_relation": surface,
                "subject_id": ids[0][1],
                "reference_id": ids[1][1],
                "subject_span": ids[0][0],
                "reference_span": ids[1][0],
                "semanticStr": q.get("semanticStr", ""),
                "types": types,
            }
            candidates_by_image[image_id].append(rec)

            if len(examples[relation]) < 5:
                examples[relation].append(rec)

            if a.progress_every and funnel["total"] % a.progress_every == 0:
                print(
                    f"[INFO] total={funnel['total']:,} "
                    f"question_candidates={sum(len(v) for v in candidates_by_image.values()):,}",
                    flush=True,
                )

    print(
        f"[INFO] question candidates={sum(len(v) for v in candidates_by_image.values()):,} "
        f"across images={len(candidates_by_image):,}",
        flush=True,
    )

    graph_counts = Counter()
    strict = []
    seen = set()

    print(f"[INFO] scanning scene graphs: {a.scene_graphs}", flush=True)

    for image_id, graph in iter_dict(a.scene_graphs):
        rows = candidates_by_image.get(str(image_id))
        if not rows:
            continue

        objects = graph.get("objects", {})
        if not isinstance(objects, dict):
            continue

        try:
            width = float(graph["width"])
            height = float(graph["height"])
        except Exception:
            continue

        name_counts = Counter(
            norm(obj.get("name"))
            for obj in objects.values()
            if isinstance(obj, dict) and norm(obj.get("name"))
        )

        for rec in rows:
            sid = rec["subject_id"]
            rid = rec["reference_id"]
            relation = rec["relation"]

            sobj = objects.get(sid)
            robj = objects.get(rid)
            if not isinstance(sobj, dict) or not isinstance(robj, dict):
                continue
            graph_counts["objects_exist"] += 1

            sa = area_ratio(sobj, width, height)
            ra = area_ratio(robj, width, height)
            if sa < a.min_area_ratio or ra < a.min_area_ratio:
                continue
            graph_counts["area_pass"] += 1

            if not relation_agrees(objects, sid, rid, relation):
                continue
            graph_counts["scene_relation_agrees"] += 1

            sname = norm(sobj.get("name"))
            rname = norm(robj.get("name"))
            if not sname or not rname or sname == rname:
                continue
            if name_counts[sname] != 1 or name_counts[rname] != 1:
                continue
            graph_counts["unique_names"] += 1

            key = (str(image_id), sid, rid, relation)
            if key in seen:
                continue
            seen.add(key)

            out = dict(rec)
            out.update({
                "subject": sname,
                "reference": rname,
                "subject_box": [sobj.get("x"), sobj.get("y"), sobj.get("w"), sobj.get("h")],
                "reference_box": [robj.get("x"), robj.get("y"), robj.get("w"), robj.get("h")],
                "subject_area_ratio": sa,
                "reference_area_ratio": ra,
            })
            strict.append(out)

    final_counts = Counter(x["relation"] for x in strict)

    report = {
        "inputs": {
            "scene_graphs": a.scene_graphs,
            "questions": a.questions,
            "min_area_ratio": a.min_area_ratio,
            "require_balanced": a.require_balanced,
        },
        "funnel": dict(funnel),
        "semantic_types_top20": semantic_types.most_common(20),
        "structural_types": dict(structural_types),
        "question_phrase_counts": dict(phrase_counts),
        "graph_counts": dict(graph_counts),
        "strict_counts": dict(final_counts),
        "strict_total": len(strict),
        "strict_images": len(set(x["image_id"] for x in strict)),
        "examples": dict(examples),
    }

    report_path = Path(a.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if a.output_candidates:
        candidate_path = Path(a.output_candidates)
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(
            json.dumps(strict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] candidates: {candidate_path}")

    print("\n=== QUESTION PHRASE COUNTS ===")
    for relation in TARGETS:
        print(f"{relation:>8}: {phrase_counts[relation]:,}")

    print("\n=== STRICT COUNTS ===")
    for relation in TARGETS:
        print(f"{relation:>8}: {final_counts[relation]:,}")
    print(f"{'total':>8}: {len(strict):,}")
    print(f"{'images':>8}: {len(set(x['image_id'] for x in strict)):,}")
    print(f"\n[OK] report: {report_path}")


if __name__ == "__main__":
    main()
