import os
import re
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional

import torch


def _patch_torchvision_nms_import_error():
    """
    Work around torch/torchvision version mismatch where importing torchvision
    fails at fake registration with:
        RuntimeError: operator torchvision::nms does not exist

    This defines the missing operator schema before torchvision is imported.
    It is only for import-time compatibility; this script does not call NMS.
    """
    try:
        from torch.library import Library
        try:
            lib = Library("torchvision", "DEF")
            lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        except Exception:
            # Already defined, or torch version does not support this path.
            pass
    except Exception:
        pass


_patch_torchvision_nms_import_error()

from tqdm import tqdm
from torch.utils.data import DataLoader

from model_zoo import get_model
from dataset_zoo import get_dataset

try:
    from misc import _default_collate
except Exception:
    _default_collate = None

import save_llava_hidden_similarity_features as sf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Controlled_Images_A")
    p.add_argument("--option", default="four")
    p.add_argument("--model-name", default="llava1.5", help="Only used to get the same dataset/image loader style.")
    p.add_argument("--method", default="adapt_vis")
    p.add_argument("--root-dir", default="data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dino-device", default=None)
    p.add_argument("--dino-model-id", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--box-threshold", type=float, default=0.25)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--patch-side", type=int, default=24, help="LLaVA-1.5 usually has 24x24 image tokens.")
    p.add_argument("--box-expand", type=float, default=0.03, help="Expand box by this fraction of width/height before mapping to patches.")
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--out-json", default="output/groundingdino_object_patch_masks.json")
    p.add_argument("--debug-json", default="")
    return p.parse_args()


def normalize_phrase(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(a|an|the)\s+", "", s, flags=re.I)
    return s.strip(" .?\n\t")


def extract_two_objects(prompt: str) -> Tuple[str, str]:
    """Robustly parse two object phrases from ARO-style questions.

    Handles both "What is ..." and "What are ..." forms. The most common
    pattern is relation/relationship between X and Y.
    """
    p = " ".join(str(prompt).replace("\n", " ").split())
    p_clean = p.strip().rstrip("?")

    patterns = [
        # What is/are the relationship/relation between X and Y?
        r"what\s+(?:is|are)\s+(?:the\s+)?(?:spatial\s+)?(?:relationship|relation|position)\s+between\s+(.+?)\s+and\s+(.+)$",
        # What is/are the position of X relative to Y?
        r"what\s+(?:is|are)\s+(?:the\s+)?(?:spatial\s+)?(?:relationship|relation|position)\s+of\s+(.+?)\s+(?:relative\s+to|with\s+respect\s+to|to)\s+(.+)$",
        # Where is/are X relative to Y?
        r"where\s+(?:is|are)\s+(.+?)\s+(?:relative\s+to|with\s+respect\s+to|to)\s+(.+)$",
        # Is/Are X left/right/on/under Y? fallback
        r"(?:is|are)\s+(.+?)\s+(?:on|under|above|below|left\s+of|right\s+of|in\s+front\s+of|behind)\s+(.+)$",
    ]

    for pat in patterns:
        m = re.search(pat, p_clean, flags=re.I)
        if m:
            obj1 = normalize_phrase(m.group(1))
            obj2 = normalize_phrase(m.group(2))
            # Remove trailing answer-choice fragments if present.
            obj2 = re.split(r"\s+(?:Options?|Choices?)\s*:", obj2, flags=re.I)[0]
            return obj1, obj2

    raise ValueError(f"Could not parse two objects from prompt: {prompt}")


def iter_samples(loader):
    sid = 0
    for batch in loader:
        for i_option in batch["image_options"]:
            for image in i_option:
                yield sid, image
                sid += 1


def load_grounding_dino(model_id: str, device: str):
    try:
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
    except Exception as e:
        raise RuntimeError(
            "This script uses HuggingFace GroundingDINO. A compatible setup is usually:\n"
            "  transformers==4.47.1 with a torch/torchvision pair that can import torchvision.\n"
            "This joint script also patches the common torchvision::nms import-time error."
        ) from e

    processor = AutoProcessor.from_pretrained(model_id)
    model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model


@torch.no_grad()
def detect_best_box(processor, model, image, phrase: str, device: str, box_threshold: float, text_threshold: float):
    """Run GroundingDINO for one object phrase and return best xyxy box in original image coordinates."""
    text = normalize_phrase(phrase)
    if not text.endswith("."):
        text = text + "."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)  # [h, w]

    # Transformers versions changed this API name/signature several times.
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids", None),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]

    boxes = results.get("boxes", None)
    scores = results.get("scores", None)
    labels = results.get("labels", results.get("text_labels", []))

    if boxes is None or scores is None or len(boxes) == 0:
        return None

    best = int(torch.argmax(scores).item())
    box = boxes[best].detach().float().cpu().tolist()
    score = float(scores[best].detach().float().cpu().item())
    label = labels[best] if isinstance(labels, (list, tuple)) and len(labels) > best else normalize_phrase(phrase)
    return {"box_xyxy": box, "score": score, "label": str(label)}


def box_to_patch_ids(box_xyxy: List[float], image_size: Tuple[int, int], patch_side: int = 24, expand: float = 0.03) -> List[int]:
    """Map an original-image xyxy box to patch ids on a patch_side x patch_side grid.

    This uses normalized coordinates. For square Controlled Images this matches the
    LLaVA 24x24 image-token grid well. If your preprocessing uses non-square center
    crop, replace this with crop-aware mapping.
    """
    W, H = image_size
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    x1 -= expand * bw
    x2 += expand * bw
    y1 -= expand * bh
    y2 += expand * bh

    x1 = max(0.0, min(float(W), x1))
    x2 = max(0.0, min(float(W), x2))
    y1 = max(0.0, min(float(H), y1))
    y2 = max(0.0, min(float(H), y2))

    if x2 <= x1 or y2 <= y1:
        return []

    c0 = int(torch.floor(torch.tensor(x1 / W * patch_side)).item())
    c1 = int(torch.ceil(torch.tensor(x2 / W * patch_side)).item()) - 1
    r0 = int(torch.floor(torch.tensor(y1 / H * patch_side)).item())
    r1 = int(torch.ceil(torch.tensor(y2 / H * patch_side)).item()) - 1

    c0 = max(0, min(patch_side - 1, c0))
    c1 = max(0, min(patch_side - 1, c1))
    r0 = max(0, min(patch_side - 1, r0))
    r1 = max(0, min(patch_side - 1, r1))

    ids = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            ids.append(r * patch_side + c)
    return sorted(set(ids))


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    dino_device = args.dino_device or args.device

    print("[LOAD LLaVA DATASET WRAPPER]", args.model_name, args.method)
    # We only need image_preprocess compatibility for dataset loading; no generation here.
    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    wrapper.model = None  # release reference as much as possible
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=False)
    collate_fn = _default_collate if image_preprocess is None else None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    prompts, answers = sf.load_prompts(args.dataset, args.option)

    print("[LOAD GROUNDINGDINO]", args.dino_model_id, "device=", dino_device)
    processor, dino = load_grounding_dino(args.dino_model_id, dino_device)

    output: Dict[str, Any] = {}
    failures = []

    for sid, image in tqdm(iter_samples(loader), desc="grounding dino object boxes"):
        if args.limit > 0 and sid >= args.limit:
            break

        prompt = prompts[sid]
        try:
            obj1, obj2 = extract_two_objects(prompt)
        except Exception as e:
            output[str(sid)] = {
                "sample_id": sid,
                "prompt": prompt,
                "error": str(e),
                "objects": [],
                "boxes": [],
                "patch_ids": [],
            }
            failures.append((sid, str(e)))
            continue

        dets = []
        patch_ids = set()
        for obj in [obj1, obj2]:
            det = detect_best_box(
                processor=processor,
                model=dino,
                image=image,
                phrase=obj,
                device=dino_device,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            if det is not None:
                ids = box_to_patch_ids(
                    det["box_xyxy"],
                    image.size,
                    patch_side=args.patch_side,
                    expand=args.box_expand,
                )
                det["object"] = obj
                det["patch_ids"] = ids
                patch_ids.update(ids)
                dets.append(det)
            else:
                dets.append({"object": obj, "box_xyxy": None, "score": None, "label": None, "patch_ids": []})

        output[str(sid)] = {
            "sample_id": sid,
            "prompt": prompt,
            "objects": [obj1, obj2],
            "image_size": list(image.size),
            "boxes": dets,
            "patch_ids": sorted(int(x) for x in patch_ids),
            "num_patch_ids": len(patch_ids),
        }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("[SAVED]", args.out_json)
    print("[NUM RECORDS]", len(output))
    print("[NUM PARSE FAILURES]", len(failures))
    if failures[:10]:
        print("[FIRST FAILURES]", failures[:10])


if __name__ == "__main__":
    main()
