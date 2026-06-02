import os
import csv
import json
import argparse
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from model_zoo import get_model
from dataset_zoo import get_dataset

try:
    from misc import _default_collate
except Exception:
    _default_collate = None

import save_llava_hidden_similarity_features as sf


class ProcessedManifestDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["processed_image_path"]).convert("RGB")
        img.filename = r["processed_image_path"]
        return {"image_options": [[img]]}


def load_processed_manifest(path, fresh_limit=-1):
    data = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(data, dict):
        records = list(data.values())
    else:
        records = data

    records = sorted(records, key=lambda r: int(r.get("sample_idx", r.get("sample_id", 0))))

    if fresh_limit > 0:
        records = records[:fresh_limit]

    prompts, answers = [], []
    for r in records:
        prompt = str(r.get("prompt", "")).strip()
        gold = str(r.get("gold", r.get("answer", ""))).strip()

        if not prompt:
            obj1 = str(r.get("obj1", "")).strip()
            obj2 = str(r.get("obj2", "")).strip()
            if obj1 and obj2:
                prompt = (
                    "<image>\n"
                    f"USER: Where is the {obj1} in relation to the {obj2}? "
                    "Answer with left, right, above, or below.\n"
                    "ASSISTANT:"
                )

        prompts.append(prompt)
        answers.append(gold)

    return records, prompts, answers


def norm_gold(x):
    if isinstance(x, list):
        return str(x[0]).strip() if x else ""
    return str(x).strip()


def iter_samples(loader):
    sid = 0
    for batch in loader:
        for i_option in batch["image_options"]:
            for image in i_option:
                yield sid, image
                sid += 1


def get_image_span(single_input, model, num_image_tokens=576):
    image_token_index = getattr(getattr(model, "config", None), "image_token_index", None)
    if image_token_index is None:
        image_token_index = 32001

    ids = single_input["input_ids"][0]
    pos = torch.where(ids == int(image_token_index))[0]

    if len(pos) == 0:
        return None, None

    image_start = int(pos[0].detach().cpu())
    image_end = image_start + int(num_image_tokens)
    return image_start, image_end


def block_ids_to_patch_ids(blocks, patch_grid=4, patch_side=24):
    if not blocks:
        return None

    block_ids = [int(x) for x in str(blocks).split(",") if x.strip()]
    block_h = patch_side // patch_grid
    block_w = patch_side // patch_grid

    patch_ids = []
    for bid in block_ids:
        br = bid // patch_grid
        bc = bid % patch_grid

        r0 = br * block_h
        r1 = (br + 1) * block_h
        c0 = bc * block_w
        c1 = (bc + 1) * block_w

        for r in range(r0, r1):
            for c in range(c0, c1):
                patch_ids.append(r * patch_side + c)

    return sorted(set(patch_ids))


class LogitStatsCollector:
    def __init__(self, num_layers, num_image_tokens=576, patch_ids=None):
        self.num_layers = int(num_layers)
        self.num_image_tokens = int(num_image_tokens)
        self.patch_ids = patch_ids
        self.reset_global()
        self.reset_sample()

    def reset_global(self):
        self.stats = {}
        for layer in range(self.num_layers):
            self.stats[layer] = {
                "num_calls": 0,
                "count": 0,
                "sum": 0.0,
                "neg_count": 0,
                "neg_sum": 0.0,
                "pos_count": 0,
                "pos_sum": 0.0,
                "zero_count": 0,
                "min": None,
                "max": None,
            }

    def reset_sample(self):
        self.call_idx = 0
        self.image_start = None
        self.image_end = None

    def set_image_span(self, image_start, image_end):
        self.image_start = image_start
        self.image_end = image_end

    def update_from_softmax_input(self, x):
        # We only want attention logits: [bsz, heads, q_len, kv_len]
        if not torch.is_tensor(x):
            return
        if x.dim() != 4:
            return
        if self.image_start is None or self.image_end is None:
            return
        if x.shape[-1] < self.image_end:
            return

        layer = self.call_idx % self.num_layers
        self.call_idx += 1

        # last query token -> image tokens
        if self.patch_ids is None:
            vals = x[:, :, -1, self.image_start:self.image_end]
        else:
            abs_ids = [self.image_start + int(pid) for pid in self.patch_ids]
            abs_ids = [p for p in abs_ids if p < x.shape[-1]]
            if not abs_ids:
                return
            idx = torch.tensor(abs_ids, device=x.device, dtype=torch.long)
            vals = x[:, :, -1, idx]

        vals = vals.detach().float()
        vals = vals[torch.isfinite(vals)]
        if vals.numel() == 0:
            return

        neg = vals < 0
        pos = vals > 0
        zero = vals == 0

        s = self.stats[layer]
        s["num_calls"] += 1
        s["count"] += int(vals.numel())
        s["sum"] += float(vals.sum().cpu())

        if neg.any():
            neg_vals = vals[neg]
            s["neg_count"] += int(neg_vals.numel())
            s["neg_sum"] += float(neg_vals.sum().cpu())

        if pos.any():
            pos_vals = vals[pos]
            s["pos_count"] += int(pos_vals.numel())
            s["pos_sum"] += float(pos_vals.sum().cpu())

        if zero.any():
            s["zero_count"] += int(zero.sum().cpu())

        vmin = float(vals.min().cpu())
        vmax = float(vals.max().cpu())
        s["min"] = vmin if s["min"] is None else min(s["min"], vmin)
        s["max"] = vmax if s["max"] is None else max(s["max"], vmax)


@contextmanager
def patch_softmax_for_logits(collector):
    old_f_softmax = F.softmax
    old_torch_softmax = torch.softmax

    def wrapped_f_softmax(input, *args, **kwargs):
        collector.update_from_softmax_input(input)
        return old_f_softmax(input, *args, **kwargs)

    def wrapped_torch_softmax(input, *args, **kwargs):
        collector.update_from_softmax_input(input)
        return old_torch_softmax(input, *args, **kwargs)

    F.softmax = wrapped_f_softmax
    torch.softmax = wrapped_torch_softmax

    try:
        yield
    finally:
        F.softmax = old_f_softmax
        torch.softmax = old_torch_softmax


def write_stats_csv(path, collector, dataset_name, region_name, num_samples):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "dataset",
        "region",
        "num_samples",
        "layer",
        "num_calls",
        "count",
        "mean_all",
        "neg_count",
        "neg_frac",
        "mean_neg",
        "pos_count",
        "pos_frac",
        "mean_pos",
        "zero_count",
        "zero_frac",
        "min",
        "max",
    ]

    rows = []
    for layer in range(collector.num_layers):
        s = collector.stats[layer]
        count = max(s["count"], 1)
        neg_count = s["neg_count"]
        pos_count = s["pos_count"]
        zero_count = s["zero_count"]

        row = {
            "dataset": dataset_name,
            "region": region_name,
            "num_samples": num_samples,
            "layer": layer,
            "num_calls": s["num_calls"],
            "count": s["count"],
            "mean_all": s["sum"] / count,
            "neg_count": neg_count,
            "neg_frac": neg_count / count,
            "mean_neg": s["neg_sum"] / max(neg_count, 1),
            "pos_count": pos_count,
            "pos_frac": pos_count / count,
            "mean_pos": s["pos_sum"] / max(pos_count, 1),
            "zero_count": zero_count,
            "zero_frac": zero_count / count,
            "min": s["min"],
            "max": s["max"],
        }
        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("[CSV SAVED]", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Controlled_Images_A")
    parser.add_argument("--option", default="four")
    parser.add_argument("--processed-manifest-json", default="")
    parser.add_argument("--model-name", default="llava1.5")
    parser.add_argument("--method", default="adapt_vis")
    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-image-tokens", type=int, default=576)
    parser.add_argument("--patch-side", type=int, default=24)
    parser.add_argument("--patch-grid", type=int, default=4)
    parser.add_argument("--blocks", default="", help="Optional block ids, e.g. 13,14 or 9,10,13,14. Empty = all image patches.")

    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fresh-limit", type=int, default=-1)
    parser.add_argument("--out-csv", default="output/base_attention_logit_stats.csv")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    patch_ids = block_ids_to_patch_ids(
        args.blocks,
        patch_grid=args.patch_grid,
        patch_side=args.patch_side,
    )
    region_name = "all_image_patches" if patch_ids is None else f"blocks_{args.blocks.replace(',', '_')}"
    if patch_ids is None:
        print("[REGION] all image patches")
    else:
        print("[REGION]", region_name, "num_patch_ids=", len(patch_ids), "head=", patch_ids[:20])

    print("[LOAD MODEL]", args.model_name, args.method)
    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method, root_dir=args.root_dir)
    wrapper.model.eval()

    if args.processed_manifest_json:
        print("[LOAD PROCESSED MANIFEST]", args.processed_manifest_json)
        records, prompts, answers = load_processed_manifest(args.processed_manifest_json, args.fresh_limit)
        dataset = ProcessedManifestDataset(records)
        collate_fn = lambda xs: xs[0]
        args.fresh_limit = -1
    else:
        print("[LOAD DATASET]", args.dataset)
        dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=False)
        collate_fn = _default_collate if image_preprocess is None else None
        prompts, answers = sf.load_prompts(args.dataset, args.option)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    collector = LogitStatsCollector(
        num_layers=args.num_layers,
        num_image_tokens=args.num_image_tokens,
        patch_ids=patch_ids,
    )

    num_seen = 0

    with patch_softmax_for_logits(collector):
        with torch.no_grad():
            for sid, image in tqdm(iter_samples(loader), desc=f"collect logits {args.dataset}"):
                if args.fresh_limit > 0 and sid >= args.fresh_limit:
                    break
                if sid >= len(prompts):
                    break

                collector.reset_sample()

                prompt = prompts[sid]
                single_input = wrapper.processor(
                    text=prompt,
                    images=image,
                    padding="max_length",
                    return_tensors="pt",
                    max_length=args.max_length,
                ).to(args.device)

                image_start, image_end = get_image_span(
                    single_input,
                    wrapper.model,
                    num_image_tokens=args.num_image_tokens,
                )

                if image_start is None:
                    print("[WARN] no image token at sid", sid)
                    continue

                collector.set_image_span(image_start, image_end)

                if sid < 3:
                    print(
                        f"[DEBUG] sid={sid} image_start={image_start} image_end={image_end} "
                        f"prompt={str(prompt)[:80]}"
                    )

                wrapper.model.generate(
                    **single_input,
                    max_new_tokens=args.max_new_tokens,
                    output_scores=False,
                    return_dict_in_generate=True,
                )

                num_seen += 1

    write_stats_csv(
        args.out_csv,
        collector,
        dataset_name=args.dataset,
        region_name=region_name,
        num_samples=num_seen,
    )

    print("[DONE]")
    print("num_seen:", num_seen)


if __name__ == "__main__":
    main()
