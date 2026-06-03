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
    records = list(data.values()) if isinstance(data, dict) else data
    records = sorted(records, key=lambda r: int(r.get("sample_idx", r.get("sample_id", 0))))

    if fresh_limit > 0:
        records = records[:fresh_limit]

    prompts = []
    answers = []

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


def iter_samples(loader):
    sid = 0
    for batch in loader:
        for i_option in batch["image_options"]:
            for image in i_option:
                yield sid, image
                sid += 1


def get_image_and_text_indices(single_input, model, num_image_tokens=576):
    """
    LLaVA expands one <image> token into 576 visual tokens.
    We map original text token positions to expanded sequence positions.
    """
    image_token_index = getattr(getattr(model, "config", None), "image_token_index", None)
    if image_token_index is None:
        image_token_index = 32001

    ids = single_input["input_ids"][0]
    attn_mask = single_input.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask[0]
    else:
        attn_mask = torch.ones_like(ids)

    pos = torch.where(ids == int(image_token_index))[0]
    if len(pos) == 0:
        return None, None, []

    image_pos = int(pos[0].detach().cpu())
    image_start = image_pos
    image_end = image_start + int(num_image_tokens)

    text_indices = []
    shift = int(num_image_tokens) - 1

    for i in range(ids.shape[0]):
        if int(attn_mask[i].detach().cpu()) == 0:
            continue
        if i == image_pos:
            continue

        if i < image_pos:
            expanded_i = i
        else:
            expanded_i = i + shift

        text_indices.append(int(expanded_i))

    return image_start, image_end, text_indices


class AttentionMassCollector:
    def __init__(self, num_layers):
        self.num_layers = int(num_layers)
        self.reset_global()
        self.reset_sample()

    def reset_global(self):
        self.stats = {}
        for layer in range(self.num_layers):
            self.stats[layer] = {
                "num_calls": 0,
                "num_rows": 0,

                "sum_row_mass": 0.0,

                "sum_image_mass": 0.0,
                "sum_text_mass": 0.0,
                "sum_other_mass": 0.0,

                "sum_image_token_mean": 0.0,
                "sum_text_token_mean": 0.0,

                "sum_image_max": 0.0,
                "sum_text_max": 0.0,

                "image_token_count_sum": 0,
                "text_token_count_sum": 0,
            }

    def reset_sample(self):
        self.call_idx = 0
        self.image_start = None
        self.image_end = None
        self.text_indices = []

    def set_indices(self, image_start, image_end, text_indices):
        self.image_start = image_start
        self.image_end = image_end
        self.text_indices = text_indices

    def update_from_softmax_output(self, probs):
        # Attention probability tensor should be [bsz, heads, q_len, kv_len]
        if not torch.is_tensor(probs):
            return
        if probs.dim() != 4:
            return
        if self.image_start is None or self.image_end is None:
            return
        if probs.shape[-1] < self.image_end:
            return

        layer = self.call_idx % self.num_layers
        self.call_idx += 1

        # last query token -> all key tokens
        row = probs[:, :, -1, :].detach().float()
        row = row[torch.isfinite(row).all(dim=-1)]

        if row.numel() == 0:
            return

        kv_len = probs.shape[-1]

        image_abs = list(range(self.image_start, min(self.image_end, kv_len)))
        text_abs = [i for i in self.text_indices if 0 <= i < kv_len]

        if len(image_abs) == 0:
            return

        image_idx = torch.tensor(image_abs, device=probs.device, dtype=torch.long)
        image_probs = row.index_select(dim=-1, index=image_idx)

        image_mass = image_probs.sum(dim=-1)
        row_mass = row.sum(dim=-1)

        if len(text_abs) > 0:
            text_idx = torch.tensor(text_abs, device=probs.device, dtype=torch.long)
            text_probs = row.index_select(dim=-1, index=text_idx)
            text_mass = text_probs.sum(dim=-1)
            text_token_mean = text_mass / float(len(text_abs))
            text_max = text_probs.max(dim=-1).values
        else:
            text_mass = torch.zeros_like(image_mass)
            text_token_mean = torch.zeros_like(image_mass)
            text_max = torch.zeros_like(image_mass)

        other_mass = row_mass - image_mass - text_mass

        image_token_mean = image_mass / float(len(image_abs))
        image_max = image_probs.max(dim=-1).values

        nrows = int(image_mass.numel())

        s = self.stats[layer]
        s["num_calls"] += 1
        s["num_rows"] += nrows

        s["sum_row_mass"] += float(row_mass.sum().cpu())

        s["sum_image_mass"] += float(image_mass.sum().cpu())
        s["sum_text_mass"] += float(text_mass.sum().cpu())
        s["sum_other_mass"] += float(other_mass.sum().cpu())

        s["sum_image_token_mean"] += float(image_token_mean.sum().cpu())
        s["sum_text_token_mean"] += float(text_token_mean.sum().cpu())

        s["sum_image_max"] += float(image_max.sum().cpu())
        s["sum_text_max"] += float(text_max.sum().cpu())

        s["image_token_count_sum"] += int(len(image_abs)) * nrows
        s["text_token_count_sum"] += int(len(text_abs)) * nrows


@contextmanager
def patch_softmax_for_attention_mass(collector):
    old_f_softmax = F.softmax
    old_torch_softmax = torch.softmax

    def wrapped_f_softmax(input, *args, **kwargs):
        out = old_f_softmax(input, *args, **kwargs)
        collector.update_from_softmax_output(out)
        return out

    def wrapped_torch_softmax(input, *args, **kwargs):
        out = old_torch_softmax(input, *args, **kwargs)
        collector.update_from_softmax_output(out)
        return out

    F.softmax = wrapped_f_softmax
    torch.softmax = wrapped_torch_softmax

    try:
        yield
    finally:
        F.softmax = old_f_softmax
        torch.softmax = old_torch_softmax


def write_csv(path, collector, dataset_name, num_samples):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "dataset",
        "num_samples",
        "layer",
        "num_calls",
        "num_rows",
        "row_mass_mean",

        "image_mass_mean",
        "text_mass_mean",
        "other_mass_mean",

        "image_over_text_mass_ratio",
        "image_mass_frac_over_image_plus_text",
        "text_mass_frac_over_image_plus_text",

        "image_token_prob_mean",
        "text_token_prob_mean",
        "image_token_max_mean",
        "text_token_max_mean",

        "image_token_count_avg",
        "text_token_count_avg",
    ]

    rows = []

    for layer in range(collector.num_layers):
        s = collector.stats[layer]
        n = max(int(s["num_rows"]), 1)

        image_mass = s["sum_image_mass"] / n
        text_mass = s["sum_text_mass"] / n
        other_mass = s["sum_other_mass"] / n
        denom = image_mass + text_mass

        rows.append({
            "dataset": dataset_name,
            "num_samples": num_samples,
            "layer": layer,
            "num_calls": s["num_calls"],
            "num_rows": s["num_rows"],
            "row_mass_mean": s["sum_row_mass"] / n,

            "image_mass_mean": image_mass,
            "text_mass_mean": text_mass,
            "other_mass_mean": other_mass,

            "image_over_text_mass_ratio": image_mass / max(text_mass, 1e-12),
            "image_mass_frac_over_image_plus_text": image_mass / max(denom, 1e-12),
            "text_mass_frac_over_image_plus_text": text_mass / max(denom, 1e-12),

            "image_token_prob_mean": s["sum_image_token_mean"] / n,
            "text_token_prob_mean": s["sum_text_token_mean"] / n,
            "image_token_max_mean": s["sum_image_max"] / n,
            "text_token_max_mean": s["sum_text_max"] / n,

            "image_token_count_avg": s["image_token_count_sum"] / n,
            "text_token_count_avg": s["text_token_count_sum"] / n,
        })

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

    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fresh-limit", type=int, default=-1)

    parser.add_argument("--out-csv", default="output/base_attention_mass_image_text.csv")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

    collector = AttentionMassCollector(num_layers=args.num_layers)

    num_seen = 0

    with patch_softmax_for_attention_mass(collector):
        with torch.no_grad():
            for sid, image in tqdm(iter_samples(loader), desc=f"collect mass {args.dataset}"):
                if args.fresh_limit > 0 and sid >= args.fresh_limit:
                    break
                if sid >= len(prompts):
                    break

                prompt = prompts[sid]

                collector.reset_sample()

                single_input = wrapper.processor(
                    text=prompt,
                    images=image,
                    padding="max_length",
                    return_tensors="pt",
                    max_length=args.max_length,
                ).to(args.device)

                image_start, image_end, text_indices = get_image_and_text_indices(
                    single_input,
                    wrapper.model,
                    num_image_tokens=args.num_image_tokens,
                )

                if image_start is None:
                    print("[WARN] no image token at sid", sid)
                    continue

                collector.set_indices(image_start, image_end, text_indices)

                if sid < 3:
                    print(
                        f"[DEBUG] sid={sid} image_start={image_start} image_end={image_end} "
                        f"num_text_tokens={len(text_indices)} prompt={str(prompt)[:80]}"
                    )

                wrapper.model.generate(
                    **single_input,
                    max_new_tokens=args.max_new_tokens,
                    output_scores=False,
                    return_dict_in_generate=True,
                )

                num_seen += 1

    write_csv(args.out_csv, collector, args.dataset, num_seen)

    print("[DONE]")
    print("num_seen:", num_seen)


if __name__ == "__main__":
    main()
