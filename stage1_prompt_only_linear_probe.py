import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import LlavaForConditionalGeneration as LlavaModel
except ImportError:
    from transformers import AutoModelForVision2Seq as LlavaModel

from dataset_zoo import get_dataset


LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LLAVA_REVISION = "a272c74"

RELATIONS = ["left", "right", "on", "under"]
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
ID2REL = {i: r for r, i in REL2ID.items()}


# ============================================================
# Dataset / prompt helpers
# ============================================================

def load_prompt_rows(dataset_name: str, option: str) -> List[dict]:
    path = Path(f"prompts/{dataset_name}_with_answer_{option}_options.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def normalize_relation(x: str) -> str:
    s = str(x).strip().lower()

    if "under" in s:
        return "under"
    if re.search(r"\bon\b", s) and "front" not in s:
        return "on"
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"

    return "unknown"


def get_gold_from_prompt_row(row: dict) -> str:
    ans = row.get("answer", "")
    if isinstance(ans, list):
        ans = ans[0] if ans else ""
    return normalize_relation(ans)


def clean_question(q: str) -> str:
    q = str(q)
    q = q.replace("<image>", " ")
    q = q.replace("USER:", " ")
    q = q.replace("ASSISTANT:", " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def get_raw_pil_from_dataset(dataset, idx: int) -> Image.Image:
    item = dataset[idx]

    if not isinstance(item, dict):
        raise TypeError(f"Expected dataset item to be dict, got {type(item)}")

    if "image_options" in item:
        image = item["image_options"][0]
    elif "image" in item:
        image = item["image"]
    else:
        raise KeyError(f"Cannot find image in dataset item keys: {list(item.keys())}")

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL image, got {type(image)}")

    return image.convert("RGB")


def load_dataset_any_signature(dataset_name: str, root_dir: str, download: bool):
    try:
        return get_dataset(
            dataset_name,
            root_dir=root_dir,
            image_preprocess=None,
            download=download,
        )
    except TypeError:
        return get_dataset(
            dataset_name,
            image_preprocess=None,
            download=download,
        )


# ============================================================
# LLaVA-like manual image preprocessing
# ============================================================

def get_size_value(size_obj, key: str, default: int) -> int:
    if isinstance(size_obj, dict):
        return int(size_obj.get(key, default))
    if isinstance(size_obj, int):
        return int(size_obj)
    return int(default)


def get_resample_from_processor(image_processor):
    resample = getattr(image_processor, "resample", None)
    if resample is not None:
        return resample
    return Image.BICUBIC


def infer_llava_geometry(image_processor, fallback_size: int = 336) -> Dict:
    do_resize = bool(getattr(image_processor, "do_resize", True))
    do_center_crop = bool(getattr(image_processor, "do_center_crop", True))
    do_pad = bool(getattr(image_processor, "do_pad", False))

    size = getattr(image_processor, "size", None) or {}
    crop_size = getattr(image_processor, "crop_size", None) or {}

    shortest_edge = None
    resize_h = None
    resize_w = None

    if isinstance(size, dict):
        if "shortest_edge" in size:
            shortest_edge = int(size["shortest_edge"])
        if "height" in size and "width" in size:
            resize_h = int(size["height"])
            resize_w = int(size["width"])
    elif isinstance(size, int):
        shortest_edge = int(size)

    crop_h = get_size_value(crop_size, "height", fallback_size)
    crop_w = get_size_value(crop_size, "width", fallback_size)

    if shortest_edge is None and resize_h is None:
        shortest_edge = fallback_size

    return {
        "do_resize": do_resize,
        "do_center_crop": do_center_crop,
        "do_pad": do_pad,
        "shortest_edge": shortest_edge,
        "resize_h": resize_h,
        "resize_w": resize_w,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "resample": get_resample_from_processor(image_processor),
    }


def expand2square(img: Image.Image, background_color: Tuple[int, int, int]):
    w, h = img.size
    if w == h:
        return img, 0, 0, w

    size = max(w, h)
    out = Image.new("RGB", (size, size), background_color)

    pad_x = (size - w) // 2
    pad_y = (size - h) // 2

    out.paste(img, (pad_x, pad_y))
    return out, pad_x, pad_y, size


def make_processed_pil_like_llava(
    raw: Image.Image,
    image_processor,
    force_mode: str = "auto",
) -> Tuple[Image.Image, Dict]:
    raw = raw.convert("RGB")
    geom = infer_llava_geometry(image_processor)

    if force_mode not in ["auto", "crop", "pad"]:
        raise ValueError(f"Unknown force_mode={force_mode}")

    mode = force_mode
    if mode == "auto":
        mode = "pad" if geom["do_pad"] else "crop"

    if mode == "pad":
        mean = getattr(
            image_processor,
            "image_mean",
            [0.48145466, 0.4578275, 0.40821073],
        )
        bg = tuple(int(float(x) * 255) for x in mean)

        square, pad_x, pad_y, square_size = expand2square(raw, bg)

        target_h = geom["crop_h"]
        target_w = geom["crop_w"]
        processed = square.resize((target_w, target_h), geom["resample"])

        meta = {
            "mode": "pad",
            "raw_size": raw.size,
            "square_size": square_size,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "processed_size": processed.size,
            "grid": target_h // 14,
        }
        return processed, meta

    w, h = raw.size

    if geom["resize_h"] is not None and geom["resize_w"] is not None:
        resized = raw.resize((geom["resize_w"], geom["resize_h"]), geom["resample"])
    else:
        shortest = geom["shortest_edge"]
        scale = shortest / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = raw.resize((new_w, new_h), geom["resample"])

    rw, rh = resized.size
    crop_w = geom["crop_w"]
    crop_h = geom["crop_h"]

    left = max(0, int(round((rw - crop_w) / 2)))
    top = max(0, int(round((rh - crop_h) / 2)))

    processed = resized.crop((left, top, left + crop_w, top + crop_h))

    meta = {
        "mode": "crop",
        "raw_size": raw.size,
        "resized_size": resized.size,
        "crop_left": left,
        "crop_top": top,
        "processed_size": processed.size,
        "grid": crop_h // 14,
    }
    return processed, meta


# ============================================================
# LLaVA loading
# ============================================================

def load_llava_hf(
    model_id: str,
    revision: str,
    cache_dir: str,
    device: str,
    dtype: str,
):
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
    )

    model = LlavaModel.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    patch_size = getattr(getattr(model.config, "vision_config", None), "patch_size", 14)
    vision_feature_select_strategy = getattr(
        model.config,
        "vision_feature_select_strategy",
        "default",
    )

    processor.patch_size = patch_size
    processor.vision_feature_select_strategy = vision_feature_select_strategy

    model = model.to(device).eval()

    print("[INFO] loaded official HF LLaVA")
    print(f"  model_id={model_id}")
    print(f"  revision={revision}")
    print(f"  model={type(model)}")
    print(f"  processor={type(processor)}")
    print(f"  patch_size={processor.patch_size}")
    print(f"  vision_feature_select_strategy={processor.vision_feature_select_strategy}")

    return processor, model


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_prompt_only_features(
    model,
    processor,
    dataset,
    prompt_rows: List[dict],
    indices: List[int],
    image_source: str,
    preprocess_mode: str,
    device: str,
    out_dir: Path,
    cache_name: str = "prompt_only_features.pt",
):
    cache_path = out_dir / cache_name

    if cache_path.exists():
        print(f"[INFO] loading cached features: {cache_path}")
        obj = torch.load(cache_path, map_location="cpu")
        return obj

    image_processor = processor.image_processor

    all_features = []
    all_labels = []
    all_sample_ids = []
    all_prompts = []
    all_clean_prompts = []

    print("[INFO] extracting prompt-only hidden states")
    print("[INFO] feature position: last non-padding prompt token")

    for sample_id in tqdm(indices):
        prompt = prompt_rows[sample_id].get("question", "")
        gold = get_gold_from_prompt_row(prompt_rows[sample_id])

        if gold not in RELATIONS:
            print(f"[SKIP] sample_id={sample_id}, invalid gold={gold}")
            continue

        raw = get_raw_pil_from_dataset(dataset, sample_id)
        processed, _ = make_processed_pil_like_llava(
            raw=raw,
            image_processor=image_processor,
            force_mode=preprocess_mode,
        )

        image = processed if image_source == "processed" else raw

        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(device)

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("outputs.hidden_states is None.")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        nonpad_positions = torch.nonzero(
            attention_mask[0],
            as_tuple=False,
        ).squeeze(-1)

        last_pos = int(nonpad_positions[-1].item())

        # shape: [num_layers, hidden_dim]
        feat_layers = []
        for h in hidden_states:
            feat = h[0, last_pos, :].detach().float().cpu()
            feat_layers.append(feat)

        feat_layers = torch.stack(feat_layers, dim=0)

        all_features.append(feat_layers)
        all_labels.append(REL2ID[gold])
        all_sample_ids.append(sample_id)
        all_prompts.append(prompt)
        all_clean_prompts.append(clean_question(prompt))

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    features = torch.stack(all_features, dim=0)  # [N, L, H]
    labels = torch.tensor(all_labels, dtype=torch.long)

    obj = {
        "features": features,
        "labels": labels,
        "sample_ids": all_sample_ids,
        "prompts": all_prompts,
        "clean_prompts": all_clean_prompts,
        "relations": RELATIONS,
        "image_source": image_source,
        "preprocess_mode": preprocess_mode,
        "feature_position": "last_prompt_token",
    }

    torch.save(obj, cache_path)

    print("[INFO] saved features:", cache_path)
    print("[INFO] features shape:", tuple(features.shape))
    print("[INFO] labels shape:", tuple(labels.shape))

    return obj


# ============================================================
# Probe training
# ============================================================

def stratified_split(labels: torch.Tensor, val_ratio: float, seed: int):
    rng = random.Random(seed)

    labels_np = labels.cpu().numpy().tolist()
    train_idx = []
    val_idx = []

    for c in range(len(RELATIONS)):
        idxs = [i for i, y in enumerate(labels_np) if y == c]
        rng.shuffle(idxs)

        n_val = max(1, int(round(len(idxs) * val_ratio)))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long)


def compute_metrics(logits: torch.Tensor, y: torch.Tensor):
    pred = logits.argmax(dim=-1)
    correct = pred.eq(y)

    acc = float(correct.float().mean().item())

    per_class = {}
    for c, rel in enumerate(RELATIONS):
        mask = y.eq(c)
        if mask.sum().item() == 0:
            per_class[f"acc_{rel}"] = None
            per_class[f"n_{rel}"] = 0
        else:
            per_class[f"acc_{rel}"] = float(correct[mask].float().mean().item())
            per_class[f"n_{rel}"] = int(mask.sum().item())

    return acc, pred, per_class


def train_one_layer_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    layer_idx: int,
    seed: int,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
):
    torch.manual_seed(seed + layer_idx)

    X_train = X[train_idx].float()
    y_train = y[train_idx].long()
    X_val = X[val_idx].float()
    y_val = y[val_idx].long()

    # Standardize using train statistics only.
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    probe = nn.Linear(X_train.shape[-1], len(RELATIONS)).to(device)

    opt = torch.optim.AdamW(
        probe.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    ce = nn.CrossEntropyLoss()

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1

    for ep in range(epochs):
        probe.train()
        opt.zero_grad(set_to_none=True)

        logits = probe(X_train)
        loss = ce(logits, y_train)

        loss.backward()
        opt.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val)
            val_loss = float(ce(val_logits, y_val).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in probe.state_dict().items()
            }

    probe.load_state_dict(best_state)
    probe.eval()

    with torch.no_grad():
        train_logits = probe(X_train)
        val_logits = probe(X_val)

        train_loss = float(ce(train_logits, y_train).item())
        val_loss = float(ce(val_logits, y_val).item())

        train_acc, train_pred, train_per_class = compute_metrics(train_logits, y_train)
        val_acc, val_pred, val_per_class = compute_metrics(val_logits, y_val)

    return {
        "layer_idx": layer_idx,
        "best_epoch": best_epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_per_class": train_per_class,
        "val_per_class": val_per_class,
        "val_pred": val_pred.detach().cpu(),
        "val_y": y_val.detach().cpu(),
    }


def run_probe_experiment(
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: List[int],
    out_dir: Path,
    device: str,
    val_ratio: float,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    shuffle_labels: bool = False,
):
    if shuffle_labels:
        g = torch.Generator().manual_seed(seed + 999)
        labels_used = labels[torch.randperm(len(labels), generator=g)].clone()
        tag = "shuffle"
    else:
        labels_used = labels.clone()
        tag = "real"

    train_idx, val_idx = stratified_split(labels_used, val_ratio=val_ratio, seed=seed)

    N, num_layers, hidden_dim = features.shape

    print(f"\n[INFO] probe experiment: {tag}")
    print(f"  N={N}, num_layers={num_layers}, hidden_dim={hidden_dim}")
    print(f"  train={len(train_idx)}, val={len(val_idx)}")
    print(f"  epochs={epochs}, lr={lr}, weight_decay={weight_decay}")

    summary_rows = []
    pred_rows = []

    for layer_idx in tqdm(range(num_layers), desc=f"training probes ({tag})"):
        result = train_one_layer_probe(
            X=features[:, layer_idx, :],
            y=labels_used,
            train_idx=train_idx,
            val_idx=val_idx,
            layer_idx=layer_idx,
            seed=seed,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )

        layer_name = "emb" if layer_idx == 0 else ("final" if layer_idx == num_layers - 1 else f"layer_{layer_idx}")

        row = {
            "experiment": tag,
            "layer_idx": layer_idx,
            "layer_name": layer_name,
            "best_epoch": result["best_epoch"],
            "train_loss": result["train_loss"],
            "val_loss": result["val_loss"],
            "train_acc": result["train_acc"],
            "val_acc": result["val_acc"],
        }

        for rel in RELATIONS:
            row[f"train_acc_{rel}"] = result["train_per_class"][f"acc_{rel}"]
            row[f"train_n_{rel}"] = result["train_per_class"][f"n_{rel}"]
            row[f"val_acc_{rel}"] = result["val_per_class"][f"acc_{rel}"]
            row[f"val_n_{rel}"] = result["val_per_class"][f"n_{rel}"]

        summary_rows.append(row)

        val_pred = result["val_pred"].tolist()
        val_y = result["val_y"].tolist()

        for local_i, global_i in enumerate(val_idx.tolist()):
            pred_rows.append(
                {
                    "experiment": tag,
                    "layer_idx": layer_idx,
                    "layer_name": layer_name,
                    "sample_id": sample_ids[global_i],
                    "gold_id": int(val_y[local_i]),
                    "gold": ID2REL[int(val_y[local_i])],
                    "pred_id": int(val_pred[local_i]),
                    "pred": ID2REL[int(val_pred[local_i])],
                    "correct": bool(int(val_pred[local_i]) == int(val_y[local_i])),
                }
            )

        print(
            f"[{tag}] layer {layer_idx:02d} ({layer_name}) "
            f"val_acc={result['val_acc']:.4f} "
            f"train_acc={result['train_acc']:.4f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    pred_df = pd.DataFrame(pred_rows)

    summary_path = out_dir / f"probe_summary_{tag}.csv"
    pred_path = out_dir / f"probe_predictions_{tag}.csv"

    summary_df.to_csv(summary_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    print("[SAVED]", summary_path)
    print("[SAVED]", pred_path)

    return summary_df, pred_df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--option", default="four")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Comma-separated sample ids. If set, only these samples are processed.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--llava-model-id", default=LLAVA_MODEL_ID)
    parser.add_argument("--llava-revision", default=LLAVA_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--preprocess-mode", default="auto", choices=["auto", "crop", "pad"])
    parser.add_argument(
        "--image-source",
        default="processed",
        choices=["processed", "raw"],
    )

    parser.add_argument("--val-ratio", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)

    parser.add_argument(
        "--run-shuffle-control",
        action="store_true",
        help="Also train probes on shuffled labels as a sanity control.",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        args.dtype = "float32"

    out_dir = Path(args.out_dir or f"output/stage1_prompt_only_probe_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[INFO] dataset={args.dataset}")
    print(f"[INFO] device={device}, dtype={args.dtype}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] image_source={args.image_source}")

    print("[INFO] loading LLaVA")
    processor, model = load_llava_hf(
        model_id=args.llava_model_id,
        revision=args.llava_revision,
        cache_dir=args.root_dir,
        device=device,
        dtype=args.dtype,
    )

    print("[INFO] loading dataset")
    dataset = load_dataset_any_signature(
        dataset_name=args.dataset,
        root_dir=args.root_dir,
        download=args.download,
    )

    prompt_rows = load_prompt_rows(args.dataset, args.option)
    n_total = min(len(dataset), len(prompt_rows))

    if args.sample_ids.strip():
        indices = []
        for x in args.sample_ids.split(","):
            x = x.strip()
            if not x:
                continue
            sid = int(x)
            if 0 <= sid < n_total:
                indices.append(sid)
            else:
                print(f"[WARN] sample_id out of range and skipped: {sid}")
    else:
        indices = list(range(n_total))
        if args.max_samples > 0:
            random.seed(args.seed)
            indices = random.sample(indices, min(args.max_samples, len(indices)))

    print(f"[INFO] total samples to extract: {len(indices)}")

    obj = extract_prompt_only_features(
        model=model,
        processor=processor,
        dataset=dataset,
        prompt_rows=prompt_rows,
        indices=indices,
        image_source=args.image_source,
        preprocess_mode=args.preprocess_mode,
        device=device,
        out_dir=out_dir,
    )

    features = obj["features"]
    labels = obj["labels"]
    sample_ids = obj["sample_ids"]

    metadata_path = out_dir / "feature_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_ids": sample_ids,
                "labels": labels.tolist(),
                "relations": RELATIONS,
                "image_source": args.image_source,
                "preprocess_mode": args.preprocess_mode,
                "feature_position": obj["feature_position"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("[SAVED]", metadata_path)

    real_summary, real_pred = run_probe_experiment(
        features=features,
        labels=labels,
        sample_ids=sample_ids,
        out_dir=out_dir,
        device=device,
        val_ratio=args.val_ratio,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        shuffle_labels=False,
    )

    if args.run_shuffle_control:
        shuffle_summary, shuffle_pred = run_probe_experiment(
            features=features,
            labels=labels,
            sample_ids=sample_ids,
            out_dir=out_dir,
            device=device,
            val_ratio=args.val_ratio,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            shuffle_labels=True,
        )

    print("\n================ BEST REAL PROBE LAYERS ================")
    best = real_summary.sort_values("val_acc", ascending=False).head(10)
    print(best[["layer_idx", "layer_name", "val_acc", "train_acc", "val_loss"]].to_string(index=False))

    print("\n================ REAL PROBE BY LAYER ================")
    print(real_summary[["layer_idx", "layer_name", "val_acc", "train_acc", "val_loss"]].to_string(index=False))

    print("\n[DONE]")


if __name__ == "__main__":
    main()
