import os
import json
import csv
import glob
from pathlib import Path
from PIL import Image

SRC_DIR = Path("data/val2017")
OUT_RESIZE = Path("data/vsr_resize336")
OUT_PAD = Path("data/vsr_pad336")
MANIFEST = Path("data/vsr_preprocess_manifest.csv")
MISSING_TXT = Path("data/vsr_missing_images.txt")

SIZE = 336
CLIP_MEAN_FILL = (122, 116, 104)

OUT_RESIZE.mkdir(parents=True, exist_ok=True)
OUT_PAD.mkdir(parents=True, exist_ok=True)


def collect_image_filenames():
    files = []
    for pat in [
        "data/**/*vsr*.json",
        "data/**/*vsr*.jsonl",
        "data/**/*vsr*.csv",
        "data/**/*VSR*.json",
        "data/**/*VSR*.jsonl",
        "data/**/*VSR*.csv",
        "dataset_zoo/**/*vsr*.json",
        "dataset_zoo/**/*vsr*.jsonl",
        "dataset_zoo/**/*vsr*.csv",
        "dataset_zoo/**/*VSR*.json",
        "dataset_zoo/**/*VSR*.jsonl",
        "dataset_zoo/**/*VSR*.csv",
        "**/*vsr*.json",
        "**/*vsr*.jsonl",
        "**/*vsr*.csv",
        "**/*VSR*.json",
        "**/*VSR*.jsonl",
        "**/*VSR*.csv",
    ]:
        files.extend(glob.glob(pat, recursive=True))

    files = sorted(set(files))
    names = set()

    def add_value(x):
        if not isinstance(x, str):
            return
        s = x.strip()
        if s.lower().endswith((".jpg", ".jpeg", ".png")):
            names.add(os.path.basename(s))

    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        else:
            add_value(obj)

    for fp in files:
        try:
            if fp.endswith(".json"):
                with open(fp, "r", encoding="utf-8") as f:
                    walk(json.load(f))
            elif fp.endswith(".jsonl"):
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            walk(json.loads(line))
            elif fp.endswith(".csv"):
                with open(fp, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        walk(row)
        except Exception:
            pass

    return sorted(names)


def resize336(img):
    return img.convert("RGB").resize((SIZE, SIZE), resample=Image.BICUBIC)


def pad336(img):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(SIZE / float(w), SIZE / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    resized = img.resize((nw, nh), resample=Image.BICUBIC)
    canvas = Image.new("RGB", (SIZE, SIZE), CLIP_MEAN_FILL)
    left = (SIZE - nw) // 2
    top = (SIZE - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas


def main():
    names = collect_image_filenames()
    print("VSR referenced images:", len(names))

    missing = []
    ok = 0

    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "src",
                "resize_path",
                "pad_path",
                "status",
                "orig_w",
                "orig_h",
            ],
        )
        writer.writeheader()

        for name in names:
            src = SRC_DIR / name
            resize_path = OUT_RESIZE / name
            pad_path = OUT_PAD / name

            if not src.exists():
                missing.append(name)
                writer.writerow({
                    "filename": name,
                    "src": str(src),
                    "resize_path": str(resize_path),
                    "pad_path": str(pad_path),
                    "status": "missing",
                    "orig_w": "",
                    "orig_h": "",
                })
                continue

            try:
                with Image.open(src) as img:
                    img = img.convert("RGB")
                    orig_w, orig_h = img.size

                    if not resize_path.exists():
                        resize336(img).save(resize_path, quality=95)

                    if not pad_path.exists():
                        pad336(img).save(pad_path, quality=95)

                ok += 1
                writer.writerow({
                    "filename": name,
                    "src": str(src),
                    "resize_path": str(resize_path),
                    "pad_path": str(pad_path),
                    "status": "ok",
                    "orig_w": orig_w,
                    "orig_h": orig_h,
                })

            except Exception as e:
                missing.append(name)
                writer.writerow({
                    "filename": name,
                    "src": str(src),
                    "resize_path": str(resize_path),
                    "pad_path": str(pad_path),
                    "status": f"error:{e}",
                    "orig_w": "",
                    "orig_h": "",
                })

    with open(MISSING_TXT, "w", encoding="utf-8") as f:
        for name in missing:
            f.write(name + "\n")

    print("processed:", ok)
    print("missing:", len(missing))
    print("resize dir:", OUT_RESIZE)
    print("pad dir:", OUT_PAD)
    print("manifest:", MANIFEST)
    print("missing list:", MISSING_TXT)

    if missing:
        print("\nFirst 30 missing:")
        for name in missing[:30]:
            print(name)


if __name__ == "__main__":
    main()
