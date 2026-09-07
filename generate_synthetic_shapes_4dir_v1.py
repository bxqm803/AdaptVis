#!/usr/bin/env python3
"""
generate_synthetic_shapes_4dir_v1.py

Generate a simple white-background synthetic spatial-relation dataset.
Each image contains two different shapes. Labels are one of:
    left, right, on, under
where:
    on    = subject is above reference
    under = subject is below reference

Default output:
    400 images total
    100 per relation

Shape set:
    circle, square, triangle, diamond, pentagon, hexagon, star

Key design choices:
- white background only
- not perfectly horizontal/vertical
- relation is still unambiguous via dominant dx/dy
- shape usage is approximately balanced across the whole dataset
- saves images, CSV labels, JSONL labels, metadata, and an optional preview grid

Example:
    python generate_synthetic_shapes_4dir_v1.py \
        --output-dir synthetic_shapes_4dir_400 \
        --n-per-relation 100 \
        --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw


DEFAULT_SHAPES = [
    "circle",
    "square",
    "triangle",
    "diamond",
    "pentagon",
    "hexagon",
    "star",
]

DEFAULT_RELATIONS = ["left", "right", "on", "under"]

DEFAULT_PALETTE = [
    (231, 76, 60),    # red
    (52, 152, 219),   # blue
    (46, 204, 113),   # green
    (241, 196, 15),   # yellow
    (155, 89, 182),   # purple
    (230, 126, 34),   # orange
    (26, 188, 156),   # teal
    (127, 140, 141),  # gray
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="synthetic_shapes_4dir_400")
    p.add_argument(
        "--n-per-relation",
        type=int,
        default=100,
        help="Number of images for each of left/right/on/under.",
    )
    p.add_argument("--canvas-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-shape-size", type=int, default=60)
    p.add_argument("--max-shape-size", type=int, default=105)
    p.add_argument("--margin", type=int, default=70)
    p.add_argument(
        "--min-axis-margin",
        type=float,
        default=0.22,
        help="Minimum dominance margin between |dx| and |dy|.",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        default=True,
        help="Save a preview grid.",
    )
    p.add_argument("--no-preview", dest="preview", action="store_false")
    return p.parse_args()


def regular_polygon(
    center: Tuple[float, float],
    radius: float,
    n_sides: int,
    rotation_deg: float = 0.0,
):
    cx, cy = center
    pts = []
    for i in range(n_sides):
        ang = math.radians(rotation_deg + 360.0 * i / n_sides - 90.0)
        x = cx + radius * math.cos(ang)
        y = cy + radius * math.sin(ang)
        pts.append((x, y))
    return pts


def star_polygon(
    center: Tuple[float, float],
    outer_radius: float,
    inner_radius: float,
    rotation_deg: float = 0.0,
):
    cx, cy = center
    pts = []
    for i in range(10):
        r = outer_radius if i % 2 == 0 else inner_radius
        ang = math.radians(rotation_deg + 36.0 * i - 90.0)
        x = cx + r * math.cos(ang)
        y = cy + r * math.sin(ang)
        pts.append((x, y))
    return pts


def draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    center: Tuple[float, float],
    size: int,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int] = (0, 0, 0),
    width: int = 3,
    rotation: float = 0.0,
) -> None:
    cx, cy = center
    r = size / 2.0

    if shape == "circle":
        draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=fill,
            outline=outline,
            width=width,
        )
    elif shape == "square":
        pts = regular_polygon(center, r * 1.05, 4, rotation_deg=45.0 + rotation)
        draw.polygon(pts, fill=fill, outline=outline)
    elif shape == "diamond":
        pts = regular_polygon(center, r * 1.05, 4, rotation_deg=rotation)
        draw.polygon(pts, fill=fill, outline=outline)
    elif shape == "triangle":
        pts = regular_polygon(center, r * 1.15, 3, rotation_deg=rotation)
        draw.polygon(pts, fill=fill, outline=outline)
    elif shape == "pentagon":
        pts = regular_polygon(center, r * 1.05, 5, rotation_deg=rotation)
        draw.polygon(pts, fill=fill, outline=outline)
    elif shape == "hexagon":
        pts = regular_polygon(center, r * 1.05, 6, rotation_deg=rotation)
        draw.polygon(pts, fill=fill, outline=outline)
    elif shape == "star":
        pts = star_polygon(center, r * 1.15, r * 0.50, rotation_deg=rotation)
        draw.polygon(pts, fill=fill, outline=outline)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def choose_balanced_pairs(
    shape_names: Sequence[str],
    total_samples: int,
    rng: random.Random,
):
    """
    Build total_samples ordered pairs (subject, reference), with:
    - subject != reference
    - approximate global balance over all object slots
    """
    total_slots = total_samples * 2
    base = total_slots // len(shape_names)
    extra = total_slots % len(shape_names)

    target = {
        shape: base + (1 if i < extra else 0)
        for i, shape in enumerate(shape_names)
    }
    remaining = dict(target)

    def choose_shape(candidates: List[str]) -> str:
        ranked = sorted(candidates, key=lambda s: (-remaining[s], s))
        top_count = remaining[ranked[0]]
        top = [s for s in ranked if remaining[s] == top_count]
        return rng.choice(top[: min(4, len(top))])

    pairs = []
    for _ in range(total_samples):
        valid_first = [s for s in shape_names if remaining[s] > 0]
        s1 = choose_shape(valid_first)
        remaining[s1] -= 1

        valid_second = [s for s in shape_names if remaining[s] > 0 and s != s1]
        if not valid_second:
            remaining[s1] += 1
            valid_first = sorted(
                [s for s in shape_names if remaining[s] > 0],
                key=lambda s: (-remaining[s], s),
            )
            s1 = valid_first[0]
            remaining[s1] -= 1
            valid_second = [s for s in shape_names if remaining[s] > 0 and s != s1]

        s2 = choose_shape(valid_second)
        remaining[s2] -= 1
        pairs.append((s1, s2))

    if sum(remaining.values()) != 0:
        raise RuntimeError(f"Unassigned shape slots remain: {remaining}")

    return pairs, target


def sample_layout(
    relation: str,
    size_sub: int,
    size_ref: int,
    canvas_size: int,
    margin: int,
    min_axis_margin: float,
    rng: random.Random,
):
    """
    Sample two centers with jitter.

    relation meanings:
        left  : subject is left of reference
        right : subject is right of reference
        on    : subject is above reference
        under : subject is below reference
    """
    W = canvas_size
    H = canvas_size

    for _ in range(5000):
        rx = rng.uniform(margin + size_ref / 2.0, W - margin - size_ref / 2.0)
        ry = rng.uniform(margin + size_ref / 2.0, H - margin - size_ref / 2.0)

        if relation == "left":
            dx = -rng.uniform(140.0, 230.0)
            dy = rng.uniform(-70.0, 70.0)
        elif relation == "right":
            dx = rng.uniform(140.0, 230.0)
            dy = rng.uniform(-70.0, 70.0)
        elif relation == "on":
            dx = rng.uniform(-70.0, 70.0)
            dy = -rng.uniform(140.0, 230.0)
        elif relation == "under":
            dx = rng.uniform(-70.0, 70.0)
            dy = rng.uniform(140.0, 230.0)
        else:
            raise ValueError(f"Unknown relation: {relation}")

        sx = rx + dx
        sy = ry + dy

        if not (margin + size_sub / 2.0 <= sx <= W - margin - size_sub / 2.0):
            continue
        if not (margin + size_sub / 2.0 <= sy <= H - margin - size_sub / 2.0):
            continue

        if relation in ("left", "right"):
            if abs(dx) <= abs(dy) + 35.0:
                continue
            if abs(dy) < 10.0:
                continue
        else:
            if abs(dy) <= abs(dx) + 35.0:
                continue
            if abs(dx) < 10.0:
                continue

        center_dist = math.dist((sx, sy), (rx, ry))
        if center_dist < (size_sub + size_ref) * 0.62 + 25.0:
            continue

        axis_margin = abs(abs(dx) - abs(dy)) / (abs(dx) + abs(dy) + 1e-8)
        if axis_margin < min_axis_margin:
            continue

        return (sx, sy), (rx, ry), dx / W, dy / H, axis_margin

    raise RuntimeError(f"Could not sample layout for relation={relation}")


def save_jsonl(path: Path, records: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_csv(path: Path, records: Sequence[Dict]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def build_preview(
    out_path: Path,
    image_dir: Path,
    records: Sequence[Dict],
    rng: random.Random,
) -> None:
    preview_ids = rng.sample(range(len(records)), min(16, len(records)))
    thumb_w = 160
    thumb_h = 160
    grid = Image.new("RGB", (4 * thumb_w, 4 * (thumb_h + 20)), (255, 255, 255))

    for i, sample_id in enumerate(preview_ids):
        row = i // 4
        col = i % 4
        im = Image.open(image_dir / f"{sample_id:04d}.png").resize((thumb_w, thumb_h))
        grid.paste(im, (col * thumb_w, row * (thumb_h + 20)))
        label = records[sample_id]["relation"]
        tag = Image.new("RGB", (thumb_w, 20), (255, 255, 255))
        tag_draw = ImageDraw.Draw(tag)
        tag_draw.text((5, 2), f"{sample_id:04d} | {label}", fill=(0, 0, 0))
        grid.paste(tag, (col * thumb_w, row * (thumb_h + 20) + thumb_h))

    grid.save(out_path)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    shape_names = list(DEFAULT_SHAPES)
    relations = list(DEFAULT_RELATIONS)
    total_samples = args.n_per_relation * len(relations)

    out_dir = Path(args.output_dir)
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    pairs, target_counts = choose_balanced_pairs(shape_names, total_samples, rng)

    relation_list = []
    for relation in relations:
        relation_list.extend([relation] * args.n_per_relation)
    rng.shuffle(relation_list)

    records: List[Dict] = []
    shape_counter = Counter()
    relation_counter = Counter()

    for idx in range(total_samples):
        subject, reference = pairs[idx]
        relation = relation_list[idx]

        size_sub = rng.randint(args.min_shape_size, args.max_shape_size)
        size_ref = rng.randint(args.min_shape_size, args.max_shape_size)

        (sx, sy), (rx, ry), dx_norm, dy_norm, axis_margin = sample_layout(
            relation=relation,
            size_sub=size_sub,
            size_ref=size_ref,
            canvas_size=args.canvas_size,
            margin=args.margin,
            min_axis_margin=args.min_axis_margin,
            rng=rng,
        )

        color_sub, color_ref = rng.sample(DEFAULT_PALETTE, 2)
        rot_sub = rng.uniform(-18.0, 18.0)
        rot_ref = rng.uniform(-18.0, 18.0)

        img = Image.new("RGB", (args.canvas_size, args.canvas_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        draw_shape(draw, reference, (rx, ry), size_ref, fill=color_ref, rotation=rot_ref)
        draw_shape(draw, subject, (sx, sy), size_sub, fill=color_sub, rotation=rot_sub)

        filename = f"{idx:04d}.png"
        img.save(image_dir / filename)

        records.append(
            {
                "id": idx,
                "image": f"images/{filename}",
                "subject": subject,
                "reference": reference,
                "relation": relation,
                "answer": relation,
                "question": f"Where is the {subject} relative to the {reference}? Answer with left, right, on, or under.",
                "subject_center_x": round(sx / args.canvas_size, 4),
                "subject_center_y": round(sy / args.canvas_size, 4),
                "reference_center_x": round(rx / args.canvas_size, 4),
                "reference_center_y": round(ry / args.canvas_size, 4),
                "dx": round(dx_norm, 4),
                "dy": round(dy_norm, 4),
                "axis_margin": round(axis_margin, 4),
                "subject_size": size_sub,
                "reference_size": size_ref,
                "subject_rotation": round(rot_sub, 2),
                "reference_rotation": round(rot_ref, 2),
                "subject_color": color_sub,
                "reference_color": color_ref,
            }
        )

        shape_counter[subject] += 1
        shape_counter[reference] += 1
        relation_counter[relation] += 1

    save_jsonl(out_dir / "labels.jsonl", records)
    save_csv(out_dir / "labels.csv", records)

    metadata = {
        "dataset_name": out_dir.name,
        "canvas_size": args.canvas_size,
        "background": "white",
        "shape_set": shape_names,
        "relations": relations,
        "relation_counts": dict(relation_counter),
        "target_shape_counts_total_slots": target_counts,
        "actual_shape_counts_total_slots": dict(shape_counter),
        "n_per_relation": args.n_per_relation,
        "seed": args.seed,
        "notes": [
            "Each image contains two different shapes.",
            "No background variation in this version.",
            "Relation labels are left/right/on/under.",
            "on means the subject is above the reference.",
            "Layouts are jittered and not perfectly axis-aligned.",
        ],
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.preview:
        build_preview(out_dir / "preview_grid.png", image_dir, records, rng)

    print("Done.")
    print(f"Output directory : {out_dir}")
    print(f"Images directory : {image_dir}")
    print(f"JSONL labels     : {out_dir / 'labels.jsonl'}")
    print(f"CSV labels       : {out_dir / 'labels.csv'}")
    print(f"Metadata         : {out_dir / 'metadata.json'}")
    if args.preview:
        print(f"Preview grid     : {out_dir / 'preview_grid.png'}")

    print("\nRelation counts:")
    for r in relations:
        print(f"  {r:>5s}: {relation_counter[r]}")

    print("\nShape usage counts (all subject/reference slots):")
    for s in shape_names:
        print(f"  {s:>9s}: {shape_counter[s]}")


if __name__ == "__main__":
    main()
