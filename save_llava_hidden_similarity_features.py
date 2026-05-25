#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save final-layer hidden-state features for LLaVA/AdaptVis prompt-only forward.

Purpose:
  For each sample and alpha value, save hidden states that allow analysis of:
    - text token <-> image patch feature similarity
    - obj1 token <-> image patches
    - obj2 token <-> image patches
    - alpha=0.5/1.5 representation changes relative to alpha=1.0

Default saves per sample per alpha:
  image_hidden:          [num_image_tokens, hidden_dim]  float16
  text_hidden:           [num_text_tokens, hidden_dim]   float16
  last_prompt_hidden:    [hidden_dim]                    float16
  obj1_hidden:           [hidden_dim]                    float16, if parsed
  obj2_hidden:           [hidden_dim]                    float16, if parsed
  sim_text_to_img:       [num_text_tokens, num_image_tokens] float16
  sim_obj1_to_img:       [num_image_tokens] float16, if parsed
  sim_obj2_to_img:       [num_image_tokens] float16, if parsed
  closed_set_logits:     left/right/on/under token logits, best-effort

Optional saves when --save-attn is enabled:
  attn_last_to_img_last_layer_heads: [num_heads, num_image_tokens] float32
  attn_last_to_img_last_layer_mean:  [num_image_tokens]            float32
  attn_last_to_text_last_layer_heads:[num_heads, num_text_tokens]  float32
  attn_last_to_text_last_layer_mean: [num_text_tokens]             float32
  attn_last_to_img_mass_per_head:    [num_heads]                   float32
  attn_last_to_text_mass_per_head:   [num_heads]                   float32

Important:
  These attention arrays are true post-softmax attention probabilities only if
  model_zoo/llama/modeling_llama_add_attn.py has been modified so that
  output_attentions returns the final attention used in the forward pass.

Run from AdaptVis repo root, e.g.:
  export TOKENIZERS_PARALLELISM=false
  python3 save_llava_hidden_similarity_features.py \
    --dataset Controlled_Images_A \
    --model-name llava1.5 \
    --method adapt_vis \
    --option four \
    --root-dir data \
    --device cuda \
    --alphas 1.0,0.5,1.5 \
    --output-dir output/hidden_similarity_features
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_zoo import get_model
from dataset_zoo import get_dataset
try:
    from misc import _default_collate
except Exception:
    _default_collate = None


REL_PROMPT_RE = re.compile(
    r"Where\s+(?:is|are)\s+(?:the\s+)?(?P<obj1>.+?)\s+in\s+relation\s+to\s+(?:the\s+)?(?P<obj2>.+?)\?",
    re.IGNORECASE,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Controlled_Images_A")
    p.add_argument("--model-name", default="llava1.5")
    p.add_argument("--method", default="adapt_vis", help="Use adapt_vis/scaling_vis to load Scal model.")
    p.add_argument("--option", default="four")
    p.add_argument("--root-dir", default="data", help="HF cache/model cache dir used by repo wrappers.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--download", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-length", type=int, default=77)
    p.add_argument("--alphas", default="1.0,0.5,1.5", help="Comma-separated alpha/weight values.")
    p.add_argument("--output-dir", default="output/hidden_similarity_features")
    p.add_argument("--limit", type=int, default=-1, help="Debug: only process first N samples.")
    p.add_argument("--sample-ids", default="", help="Optional comma-separated sample ids to process.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--compress", action="store_true", help="Use np.savez_compressed; saves disk but slower.")
    p.add_argument("--save-sim", action="store_true", default=True)
    p.add_argument("--no-save-sim", action="store_false", dest="save_sim")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument(
        "--save-attn",
        action="store_true",
        help=(
            "Save true post-softmax attention maps from outputs.attentions. "
            "Requires the modified modeling_llama_add_attn.py that returns final attention."
        ),
    )
    p.add_argument(
        "--save-all-layers-attn",
        action="store_true",
        help=(
            "When --save-attn is set, also save last-query attention to image/text tokens "
            "for all layers. This is more memory intensive because output_attentions=True "
            "returns full attention matrices."
        ),
    )
    return p.parse_args()


def safe_float_tag(x: float) -> str:
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


def load_prompts(dataset: str, option: str) -> Tuple[List[str], List[str]]:
    path = Path(f"prompts/{dataset}_with_answer_{option}_options.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Cannot find prompt file: {path}")
    prompts, answers = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompts.append(row["question"])
            ans = row.get("answer", "")
            if isinstance(ans, list):
                ans = ans[0] if ans else ""
            answers.append(str(ans))
    return prompts, answers


def parse_objects_with_spans(prompt: str) -> Tuple[str, str, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    m = REL_PROMPT_RE.search(prompt)
    if not m:
        return "", "", None, None
    obj1 = m.group("obj1").strip()
    obj2 = m.group("obj2").strip()
    s1, e1 = m.span("obj1")
    s2, e2 = m.span("obj2")
    # Strip leading/trailing spaces from spans if any.
    while s1 < e1 and prompt[s1].isspace():
        s1 += 1
    while e1 > s1 and prompt[e1 - 1].isspace():
        e1 -= 1
    while s2 < e2 and prompt[s2].isspace():
        s2 += 1
    while e2 > s2 and prompt[e2 - 1].isspace():
        e2 -= 1
    return obj1, obj2, (s1, e1), (s2, e2)


def get_image_token_id(wrapper) -> int:
    # The repo often uses 32001 in keys, while HF Llava commonly uses config.image_token_index.
    model = wrapper.model
    proc = wrapper.processor
    candidates = []
    for obj in [getattr(model, "config", None), getattr(getattr(model, "language_model", None), "config", None)]:
        if obj is not None and hasattr(obj, "image_token_index"):
            candidates.append(int(obj.image_token_index))
    try:
        tok_id = proc.tokenizer.convert_tokens_to_ids("<image>")
        if tok_id is not None and tok_id != proc.tokenizer.unk_token_id:
            candidates.append(int(tok_id))
    except Exception:
        pass
    # Common fallback used by this repo in keys.
    candidates += [32000, 32001]
    # Return first unique candidate; actual position search will verify.
    seen = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    return seen[0]


def find_image_token_pos(input_ids_1d: torch.Tensor, wrapper) -> Tuple[int, int]:
    ids = input_ids_1d.detach().cpu().tolist()
    candidates = []
    cfg = getattr(wrapper.model, "config", None)
    if cfg is not None and hasattr(cfg, "image_token_index"):
        candidates.append(int(cfg.image_token_index))
    try:
        tok_id = wrapper.processor.tokenizer.convert_tokens_to_ids("<image>")
        if tok_id is not None:
            candidates.append(int(tok_id))
    except Exception:
        pass
    candidates += [32000, 32001]

    for cand in candidates:
        if cand in ids:
            return ids.index(cand), cand

    # Last fallback: look for high special-token ids around image placeholder.
    for i, v in enumerate(ids):
        if int(v) >= 32000:
            return i, int(v)

    raise RuntimeError("Could not find <image> token in input_ids. Check tokenizer/model version.")


def build_inputs(wrapper, prompt: str, image, max_length: int, device: str):
    """Use repo-original processor path; fallback to tokenizer/image_processor split for newer HF."""
    try:
        inputs = wrapper.processor(
            text=prompt,
            images=image,
            padding="max_length",
            return_tensors="pt",
            max_length=max_length,
        ).to(device)
        return inputs
    except TypeError as e:
        if "patch_size" not in str(e) and "NoneType" not in str(e):
            raise
        # Newer LlavaProcessor tries to expand image tokens and fails if patch_size=None.
        text_inputs = wrapper.processor.tokenizer(
            text=prompt,
            padding="max_length",
            return_tensors="pt",
            max_length=max_length,
            truncation=False,
        )
        image_inputs = wrapper.processor.image_processor(images=image, return_tensors="pt")
        inputs = {}
        inputs.update(dict(text_inputs))
        inputs.update(dict(image_inputs))
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def tokenizer_offsets(wrapper, prompt: str, max_length: int):
    # Need Fast tokenizer offsets for object phrase -> token positions.
    tok = wrapper.tokenizer if hasattr(wrapper, "tokenizer") else wrapper.processor.tokenizer
    enc = tok(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"][0].tolist()
    offsets = enc["offset_mapping"][0].tolist()
    attention_mask = enc["attention_mask"][0].tolist()
    tokens = tok.convert_ids_to_tokens(input_ids)
    return input_ids, offsets, attention_mask, tokens


def token_positions_for_span(offsets, attention_mask, span: Optional[Tuple[int, int]], exclude_pos: Optional[int] = None):
    if span is None:
        return []
    s, e = span
    pos = []
    for i, ((a, b), m) in enumerate(zip(offsets, attention_mask)):
        if not m:
            continue
        if exclude_pos is not None and i == exclude_pos:
            continue
        if a == b == 0:
            continue
        # overlap with char span
        if max(a, s) < min(b, e):
            pos.append(i)
    return pos


def get_num_image_tokens_from_inputs(wrapper, inputs) -> int:
    """Infer the number of visual patch tokens used by LLaVA.

    For LLaVA-1.5 this is normally 24*24=576. We compute it from
    pixel_values and the vision patch size instead of inferring it from
    merged sequence length, because padding side / processor behavior can
    make that inference unstable.
    """
    pixel_values = inputs.get("pixel_values", None)
    if pixel_values is not None and hasattr(pixel_values, "shape"):
        h = int(pixel_values.shape[-2])
        w = int(pixel_values.shape[-1])
        patch_size = None
        cfg = getattr(wrapper.model, "config", None)
        vcfg = getattr(cfg, "vision_config", None)
        if vcfg is not None:
            patch_size = getattr(vcfg, "patch_size", None)
        if patch_size is None:
            # CLIP-L/336 in LLaVA-1.5 uses 14x14 patches.
            patch_size = 14
        n = (h // int(patch_size)) * (w // int(patch_size))
        # LLaVA-1.5 default selects patches without CLS, so no +1.
        return int(n)
    return 576


def nonpad_positions_from_mask(attention_mask) -> List[int]:
    return [i for i, m in enumerate(attention_mask) if int(m) == 1]


def image_start_after_merge(attention_mask, image_pos: int) -> int:
    """Position where image patch tokens start in merged hidden states.

    The custom LLaVA merge removes padding and replaces one <image> token by
    N visual tokens. Therefore the start index is not necessarily image_pos
    from the padded input_ids; it is the number of non-pad tokens before image_pos.
    """
    nonpad = nonpad_positions_from_mask(attention_mask)
    if image_pos not in nonpad:
        # Fallback: count nonpad positions before image_pos.
        return sum(1 for p in nonpad if p < image_pos)
    return nonpad.index(image_pos)


def map_premerge_to_merged_positions(token_positions: List[int], attention_mask, image_pos: int, num_img_tokens: int) -> List[int]:
    """Map tokenizer positions before image merge to merged hidden-state positions.

    Works for either left or right padding. Only non-pad token positions are kept.
    """
    nonpad = nonpad_positions_from_mask(attention_mask)
    rank = {p: i for i, p in enumerate(nonpad)}
    if image_pos not in rank:
        image_rank = sum(1 for p in nonpad if p < image_pos)
    else:
        image_rank = rank[image_pos]

    out = []
    for p in token_positions:
        if p == image_pos or p not in rank:
            continue
        r = rank[p]
        if r < image_rank:
            out.append(r)
        elif r > image_rank:
            out.append(r - 1 + num_img_tokens)
    return out


def nonpad_text_positions(attention_mask, image_pos: int, num_img_tokens: int) -> Tuple[List[int], List[int]]:
    pre = [i for i, m in enumerate(attention_mask) if int(m) == 1 and i != image_pos]
    merged = map_premerge_to_merged_positions(pre, attention_mask, image_pos, num_img_tokens)
    return pre, merged


def get_hidden_states_from_outputs(outputs):
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states
    if isinstance(outputs, dict) and outputs.get("hidden_states", None) is not None:
        return outputs["hidden_states"]
    # Some custom wrappers may store language model outputs.
    for name in ["language_model_outputs", "decoder_hidden_states"]:
        obj = getattr(outputs, name, None)
        if obj is not None:
            if hasattr(obj, "hidden_states") and obj.hidden_states is not None:
                return obj.hidden_states
            if isinstance(obj, dict) and obj.get("hidden_states", None) is not None:
                return obj["hidden_states"]
    raise RuntimeError("Forward output does not contain hidden_states. Make sure output_hidden_states=True is supported.")

def get_attentions_from_outputs(outputs):
    """Return outputs.attentions from HF/custom model outputs."""
    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        return outputs.attentions
    if isinstance(outputs, dict) and outputs.get("attentions", None) is not None:
        return outputs["attentions"]

    for name in ["language_model_outputs", "decoder_outputs"]:
        obj = getattr(outputs, name, None)
        if obj is not None:
            if hasattr(obj, "attentions") and obj.attentions is not None:
                return obj.attentions
            if isinstance(obj, dict) and obj.get("attentions", None) is not None:
                return obj["attentions"]

    raise RuntimeError(
        "Forward output does not contain attentions. "
        "Make sure output_attentions=True is passed and the model supports it."
    )


def _get_attr_from_candidates(candidates, attr: str):
    for obj in candidates:
        if obj is not None and hasattr(obj, attr):
            val = getattr(obj, attr)
            if val is not None:
                return val
    return None


def get_exact_image_text_indices_from_adaptvis(
    wrapper,
    merged_len: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Get the exact visual-token mask saved by the AdaptVis LLaVA forward.

    In the custom Scal LLaVA implementation, image_id/keys is the same mask used
    by the attention intervention. If available, this is more reliable than
    reconstructing image_start:image_end from tokenizer positions.
    """
    model = getattr(wrapper, "model", None)
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
    ]

    image_mask = _get_attr_from_candidates(candidates, "_adaptvis_last_image_id")
    valid_mask = _get_attr_from_candidates(candidates, "_adaptvis_last_attention_mask")

    if image_mask is None or not torch.is_tensor(image_mask):
        return None, None

    image_mask = image_mask.detach().cpu().bool()
    if image_mask.dim() == 2:
        image_mask = image_mask[0]
    elif image_mask.dim() > 2:
        image_mask = image_mask.reshape(-1)

    if int(image_mask.numel()) != int(merged_len):
        return None, None

    if valid_mask is not None and torch.is_tensor(valid_mask):
        valid_mask = valid_mask.detach().cpu().bool()
        if valid_mask.dim() == 2:
            valid_mask = valid_mask[0]
        elif valid_mask.dim() > 2:
            valid_mask = valid_mask.reshape(-1)
        if int(valid_mask.numel()) != int(merged_len):
            valid_mask = torch.ones_like(image_mask, dtype=torch.bool)
    else:
        valid_mask = torch.ones_like(image_mask, dtype=torch.bool)

    text_mask = valid_mask & (~image_mask)

    image_idx = torch.where(image_mask)[0].to(device=device, dtype=torch.long)
    text_idx = torch.where(text_mask)[0].to(device=device, dtype=torch.long)

    if image_idx.numel() == 0:
        return None, None

    return image_idx, text_idx


def attention_tensor_to_float32_numpy(x: torch.Tensor) -> np.ndarray:
    """Detach attention tensor without any normalization; values remain true probabilities."""
    return x.detach().float().cpu().numpy().astype(np.float32)


def add_attention_arrays(
    arrays: Dict[str, np.ndarray],
    outputs,
    wrapper,
    merged_len: int,
    last_merged_pos: int,
    image_start: int,
    image_end: int,
    text_merged_pos: List[int],
    save_all_layers_attn: bool,
):
    """Add post-softmax attention arrays to arrays.

    Expected attention shape per layer:
      [batch, num_heads, target_seq_len, source_seq_len]

    With the modified modeling_llama_add_attn.py, this is the final attention
    distribution actually used by the forward pass.
    """
    attentions = get_attentions_from_outputs(outputs)

    if attentions is None or len(attentions) == 0:
        raise RuntimeError("outputs.attentions is empty.")

    last_layer = attentions[-1]
    if last_layer is None:
        raise RuntimeError("The last layer attention is None.")

    # [batch, num_heads, tgt_len, src_len] -> [num_heads, tgt_len, src_len]
    last_layer_attn = last_layer[0].detach().float()
    src_len = int(last_layer_attn.shape[-1])
    tgt_len = int(last_layer_attn.shape[-2])

    if not (0 <= int(last_merged_pos) < tgt_len):
        raise RuntimeError(
            f"last_merged_pos={last_merged_pos} outside attention target length {tgt_len}."
        )

    exact_img_idx, exact_text_idx = get_exact_image_text_indices_from_adaptvis(
        wrapper=wrapper,
        merged_len=src_len,
        device=last_layer_attn.device,
    )

    if exact_img_idx is not None:
        img_idx = exact_img_idx
        text_idx = exact_text_idx if exact_text_idx is not None else torch.empty(0, dtype=torch.long, device=last_layer_attn.device)
        image_index_source = "adaptvis_mask"
    else:
        img_idx = torch.arange(int(image_start), int(image_end), device=last_layer_attn.device, dtype=torch.long)
        text_idx = torch.tensor(text_merged_pos, device=last_layer_attn.device, dtype=torch.long)
        image_index_source = "reconstructed_span"

    img_idx = img_idx[(img_idx >= 0) & (img_idx < src_len)]
    text_idx = text_idx[(text_idx >= 0) & (text_idx < src_len)]

    if img_idx.numel() == 0:
        raise RuntimeError("No valid image token indices for attention extraction.")

    attn_last_to_all = last_layer_attn[:, int(last_merged_pos), :]  # [num_heads, src_len]
    attn_last_to_img = attn_last_to_all[:, img_idx]                 # [num_heads, num_img_tokens]
    attn_last_to_text = attn_last_to_all[:, text_idx] if text_idx.numel() > 0 else attn_last_to_all[:, :0]

    arrays["attn_image_index_source"] = np.array(image_index_source)
    arrays["attn_image_indices"] = img_idx.detach().cpu().numpy().astype(np.int32)
    arrays["attn_text_indices"] = text_idx.detach().cpu().numpy().astype(np.int32)

    arrays["attn_last_to_img_last_layer_heads"] = attention_tensor_to_float32_numpy(attn_last_to_img)
    arrays["attn_last_to_img_last_layer_mean"] = attention_tensor_to_float32_numpy(attn_last_to_img.mean(dim=0))

    arrays["attn_last_to_text_last_layer_heads"] = attention_tensor_to_float32_numpy(attn_last_to_text)
    arrays["attn_last_to_text_last_layer_mean"] = attention_tensor_to_float32_numpy(attn_last_to_text.mean(dim=0))

    arrays["attn_last_to_img_mass_per_head"] = attention_tensor_to_float32_numpy(attn_last_to_img.sum(dim=-1))
    arrays["attn_last_to_text_mass_per_head"] = attention_tensor_to_float32_numpy(attn_last_to_text.sum(dim=-1))

    arrays["attn_last_to_img_mass_mean"] = np.array(
        float(attn_last_to_img.sum(dim=-1).mean().detach().cpu()),
        dtype=np.float32,
    )
    arrays["attn_last_to_text_mass_mean"] = np.array(
        float(attn_last_to_text.sum(dim=-1).mean().detach().cpu()) if text_idx.numel() > 0 else 0.0,
        dtype=np.float32,
    )

    # A simple sanity number: the total probability mass over all source tokens
    # should be near 1 for every head at the queried row.
    arrays["attn_last_to_all_mass_per_head"] = attention_tensor_to_float32_numpy(attn_last_to_all.sum(dim=-1))

    if save_all_layers_attn:
        all_img = []
        all_text = []
        all_img_mass = []
        all_text_mass = []

        for layer_attn in attentions:
            if layer_attn is None:
                continue
            layer_attn = layer_attn[0].detach().float()
            if int(last_merged_pos) >= int(layer_attn.shape[-2]):
                raise RuntimeError(
                    f"last_merged_pos={last_merged_pos} outside layer attention target length {layer_attn.shape[-2]}."
                )

            layer_last_to_all = layer_attn[:, int(last_merged_pos), :]
            layer_img = layer_last_to_all[:, img_idx]
            layer_text = layer_last_to_all[:, text_idx] if text_idx.numel() > 0 else layer_last_to_all[:, :0]

            all_img.append(attention_tensor_to_float32_numpy(layer_img))
            all_text.append(attention_tensor_to_float32_numpy(layer_text))
            all_img_mass.append(attention_tensor_to_float32_numpy(layer_img.sum(dim=-1)))
            all_text_mass.append(attention_tensor_to_float32_numpy(layer_text.sum(dim=-1)))

        arrays["attn_last_to_img_all_layers_heads"] = np.stack(all_img, axis=0)
        arrays["attn_last_to_text_all_layers_heads"] = np.stack(all_text, axis=0)
        arrays["attn_last_to_img_mass_all_layers_heads"] = np.stack(all_img_mass, axis=0)
        arrays["attn_last_to_text_mass_all_layers_heads"] = np.stack(all_text_mass, axis=0)



def closed_set_token_ids(wrapper) -> Dict[str, List[int]]:
    tok = wrapper.tokenizer if hasattr(wrapper, "tokenizer") else wrapper.processor.tokenizer
    out = {}
    for word in ["left", "right", "on", "under", "Left", "Right", "On", "Under"]:
        ids = []
        for text in [word, " " + word]:
            try:
                enc = tok.encode(text, add_special_tokens=False)
                if enc:
                    ids.append(int(enc[0]))
            except Exception:
                pass
        # unique
        uniq = []
        for x in ids:
            if x not in uniq:
                uniq.append(x)
        out[word] = uniq
    return out


def best_closed_set_logits(logits_1d: torch.Tensor, token_id_map: Dict[str, List[int]]):
    # For each word, use max over tokenization variants as a simple first-token proxy.
    result = {}
    for word, ids in token_id_map.items():
        if not ids:
            result[word] = float("nan")
        else:
            vals = logits_1d[torch.tensor(ids, device=logits_1d.device)].detach().float().cpu().numpy()
            result[word] = float(np.max(vals))
    return result


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1, eps=1e-12)


def to_numpy_dtype(x: torch.Tensor, dtype: str):
    arr = x.detach().cpu().numpy()
    if dtype == "float16":
        return arr.astype(np.float16)
    return arr.astype(np.float32)


def save_npz(path: Path, compress: bool, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **kwargs)
    else:
        np.savez(path, **kwargs)


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = args.device
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    sample_id_filter = None
    if args.sample_ids.strip():
        sample_id_filter = {int(x.strip()) for x in args.sample_ids.split(",") if x.strip()}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    print("[LOAD MODEL]", args.model_name, args.method, device)
    wrapper, image_preprocess = get_model(
        args.model_name,
        device,
        args.method,
        root_dir=args.root_dir,
    )
    wrapper.model.eval()

    print("[LOAD DATASET]", args.dataset)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)
    collate_fn = _default_collate if image_preprocess is None else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    prompts, answers = load_prompts(args.dataset, args.option)
    token_id_map = closed_set_token_ids(wrapper)
    print("[CLOSED SET TOKEN IDS]", token_id_map)
    print("[OUTPUT]", out_dir)
    print("[ALPHAS]", alphas)

    global_idx = 0
    written = 0

    # Open metadata in append mode so script can resume.
    meta_f = meta_path.open("a", encoding="utf-8")

    with torch.no_grad():
        pbar = tqdm(loader, desc="saving hidden states")
        for batch in pbar:
            for i_option in batch["image_options"]:
                for image in i_option:
                    sid = global_idx
                    global_idx += 1

                    if args.limit > 0 and sid >= args.limit:
                        meta_f.close()
                        print("[DONE limit] written", written)
                        return
                    if sample_id_filter is not None and sid not in sample_id_filter:
                        continue

                    prompt = prompts[sid]
                    gold = answers[sid]
                    obj1, obj2, obj1_span, obj2_span = parse_objects_with_spans(prompt)

                    # Build inputs once per sample.
                    inputs = build_inputs(wrapper, prompt, image, args.max_length, device)
                    input_ids_1d = inputs["input_ids"][0]
                    attention_mask_1d = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))[0]
                    image_pos, image_token_id = find_image_token_pos(input_ids_1d, wrapper)

                    tok_input_ids, offsets, tok_attention_mask, tokens = tokenizer_offsets(wrapper, prompt, args.max_length)
                    obj1_pre_pos = token_positions_for_span(offsets, tok_attention_mask, obj1_span, exclude_pos=image_pos)
                    obj2_pre_pos = token_positions_for_span(offsets, tok_attention_mask, obj2_span, exclude_pos=image_pos)

                    for alpha in alphas:
                        alpha_name = {1.0: "base", 0.5: "low", 1.5: "high"}.get(float(alpha), f"alpha{safe_float_tag(alpha)}")
                        save_path = out_dir / f"sid{sid:04d}_{alpha_name}_alpha{safe_float_tag(alpha)}.npz"
                        if save_path.exists() and not args.overwrite:
                            continue

                        # Forward prompt only. No generation answer tokens are appended.
                        # AdaptVis custom LLaMA attention code expects SAVE_ATTN_PATH to exist
                        # even when we only need hidden states. Set a per-sample/per-alpha temp path
                        # to avoid ValueError: SAVE_ATTN_PATH not set.
                        attn_tmp_dir = out_dir / "_tmp_attn_forward" / f"sid{sid:04d}_{alpha_name}"
                        attn_tmp_dir.mkdir(parents=True, exist_ok=True)
                        os.environ["SAVE_ATTN_PATH"] = str(attn_tmp_dir) + "/"

                        forward_kwargs = dict(inputs)
                        forward_kwargs.update({
                            "weight": float(alpha),
                            "output_hidden_states": True,
                            "output_attentions": bool(args.save_attn),
                            "return_dict": True,
                            "use_cache": False,
                        })

                        outputs = wrapper.model(**forward_kwargs)
                        hidden_states = get_hidden_states_from_outputs(outputs)
                        last_hidden = hidden_states[-1][0]  # [merged_seq_len, hidden_dim]
                        logits_last = outputs.logits[0, -1]

                        merged_len = int(last_hidden.shape[0])
                        # Use fixed visual patch count and padding-aware merge mapping.
                        # Do not infer num_img_tokens from merged_len, because left padding can
                        # make image_pos in padded input_ids differ from the merged position.
                        num_img_tokens = get_num_image_tokens_from_inputs(wrapper, inputs)
                        image_start = image_start_after_merge(tok_attention_mask, image_pos)
                        image_end = int(image_start + num_img_tokens)

                        if image_end > merged_len:
                            raise RuntimeError(
                                f"Image range exceeds merged hidden length: start={image_start}, "
                                f"end={image_end}, merged_len={merged_len}, num_img_tokens={num_img_tokens}, "
                                f"image_pos={image_pos}, nonpad={sum(int(m) for m in tok_attention_mask)}"
                            )

                        H_img = last_hidden[image_start:image_end]
                        if H_img.shape[0] != num_img_tokens:
                            raise RuntimeError(f"H_img length mismatch: {H_img.shape[0]} vs {num_img_tokens}")

                        text_pre_pos, text_merged_pos = nonpad_text_positions(tok_attention_mask, image_pos, num_img_tokens)
                        # Guard against tokenizer offsets and processor ids drifting.
                        text_pairs = [
                            (pre, merged)
                            for pre, merged in zip(text_pre_pos, text_merged_pos)
                            if 0 <= merged < merged_len
                        ]
                        text_pre_pos_valid = [pre for pre, _ in text_pairs]
                        text_merged_pos = [merged for _, merged in text_pairs]
                        H_text = last_hidden[torch.tensor(text_merged_pos, device=last_hidden.device)] if text_merged_pos else last_hidden[:0]

                        obj1_merged_pos = map_premerge_to_merged_positions(obj1_pre_pos, tok_attention_mask, image_pos, num_img_tokens)
                        obj2_merged_pos = map_premerge_to_merged_positions(obj2_pre_pos, tok_attention_mask, image_pos, num_img_tokens)
                        obj1_merged_pos = [p for p in obj1_merged_pos if 0 <= p < merged_len]
                        obj2_merged_pos = [p for p in obj2_merged_pos if 0 <= p < merged_len]

                        H_obj1 = last_hidden[torch.tensor(obj1_merged_pos, device=last_hidden.device)].mean(dim=0) if obj1_merged_pos else torch.full((last_hidden.shape[-1],), float("nan"), device=last_hidden.device)
                        H_obj2 = last_hidden[torch.tensor(obj2_merged_pos, device=last_hidden.device)].mean(dim=0) if obj2_merged_pos else torch.full((last_hidden.shape[-1],), float("nan"), device=last_hidden.device)

                        # Last non-pad text token position after merge.
                        last_pre_pos = max([i for i, m in enumerate(tok_attention_mask) if int(m) == 1 and i != image_pos])
                        last_merged_pos = map_premerge_to_merged_positions([last_pre_pos], tok_attention_mask, image_pos, num_img_tokens)[0]
                        H_last_prompt = last_hidden[last_merged_pos]

                        arrays = {
                            "sample_id": np.array(sid, dtype=np.int32),
                            "alpha": np.array(float(alpha), dtype=np.float32),
                            "alpha_name": np.array(alpha_name),
                            "gold": np.array(gold),
                            "prompt": np.array(prompt),
                            "obj1": np.array(obj1),
                            "obj2": np.array(obj2),
                            "image_token_id": np.array(image_token_id, dtype=np.int32),
                            "image_start": np.array(image_start, dtype=np.int32),
                            "image_end": np.array(image_end, dtype=np.int32),
                            "num_img_tokens": np.array(num_img_tokens, dtype=np.int32),
                            "merged_len": np.array(merged_len, dtype=np.int32),
                            "hidden_dim": np.array(last_hidden.shape[-1], dtype=np.int32),
                            "input_ids": np.array(tok_input_ids, dtype=np.int64),
                            "attention_mask": np.array(tok_attention_mask, dtype=np.int64),
                            "tokens": np.array(tokens, dtype=object),
                            "text_pre_positions": np.array(text_pre_pos_valid, dtype=np.int32),
                            "text_merged_positions": np.array(text_merged_pos, dtype=np.int32),
                            "obj1_pre_positions": np.array(obj1_pre_pos, dtype=np.int32),
                            "obj2_pre_positions": np.array(obj2_pre_pos, dtype=np.int32),
                            "obj1_merged_positions": np.array(obj1_merged_pos, dtype=np.int32),
                            "obj2_merged_positions": np.array(obj2_merged_pos, dtype=np.int32),
                            "last_pre_position": np.array(last_pre_pos, dtype=np.int32),
                            "last_merged_position": np.array(last_merged_pos, dtype=np.int32),
                            "image_hidden": to_numpy_dtype(H_img, args.dtype),
                            "text_hidden": to_numpy_dtype(H_text, args.dtype),
                            "obj1_hidden": to_numpy_dtype(H_obj1, args.dtype),
                            "obj2_hidden": to_numpy_dtype(H_obj2, args.dtype),
                            "last_prompt_hidden": to_numpy_dtype(H_last_prompt, args.dtype),
                        }

                        cls_logits = best_closed_set_logits(logits_last, token_id_map)
                        arrays["closed_set_words"] = np.array(list(cls_logits.keys()), dtype=object)
                        arrays["closed_set_logits"] = np.array([cls_logits[k] for k in cls_logits.keys()], dtype=np.float32)

                        if args.save_attn:
                            add_attention_arrays(
                                arrays=arrays,
                                outputs=outputs,
                                wrapper=wrapper,
                                merged_len=merged_len,
                                last_merged_pos=last_merged_pos,
                                image_start=image_start,
                                image_end=image_end,
                                text_merged_pos=text_merged_pos,
                                save_all_layers_attn=args.save_all_layers_attn,
                            )

                        if args.save_sim:
                            H_img_n = normalize_rows(H_img)
                            H_text_n = normalize_rows(H_text) if H_text.shape[0] > 0 else H_text.float()
                            if H_text.shape[0] > 0:
                                sim_text_img = H_text_n @ H_img_n.T
                                arrays["sim_text_to_img"] = to_numpy_dtype(sim_text_img, args.dtype)
                            if obj1_merged_pos:
                                sim_obj1 = normalize_rows(H_obj1[None, :]) @ H_img_n.T
                                arrays["sim_obj1_to_img"] = to_numpy_dtype(sim_obj1[0], args.dtype)
                                top10 = torch.topk(sim_obj1[0], k=min(10, sim_obj1.shape[-1])).indices.detach().cpu().numpy().astype(np.int32)
                                arrays["sim_obj1_top10_idx"] = top10
                            if obj2_merged_pos:
                                sim_obj2 = normalize_rows(H_obj2[None, :]) @ H_img_n.T
                                arrays["sim_obj2_to_img"] = to_numpy_dtype(sim_obj2[0], args.dtype)
                                top10 = torch.topk(sim_obj2[0], k=min(10, sim_obj2.shape[-1])).indices.detach().cpu().numpy().astype(np.int32)
                                arrays["sim_obj2_top10_idx"] = top10

                        save_npz(save_path, args.compress, **arrays)
                        written += 1

                        meta = {
                            "sample_id": sid,
                            "alpha": float(alpha),
                            "alpha_name": alpha_name,
                            "path": str(save_path),
                            "gold": gold,
                            "obj1": obj1,
                            "obj2": obj2,
                            "obj1_pre_positions": obj1_pre_pos,
                            "obj2_pre_positions": obj2_pre_pos,
                            "obj1_merged_positions": obj1_merged_pos,
                            "obj2_merged_positions": obj2_merged_pos,
                            "image_start": image_start,
                            "image_end": image_end,
                            "num_img_tokens": num_img_tokens,
                            "merged_len": merged_len,
                            "hidden_dim": int(last_hidden.shape[-1]),
                            "save_attn": bool(args.save_attn),
                        }
                        meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                        meta_f.flush()

                    if args.print_every > 0 and sid % args.print_every == 0:
                        pbar.set_postfix({"sid": sid, "written": written})
                        print(f"\n[SAMPLE {sid}] obj1={obj1!r} obj2={obj2!r} image_pos={image_pos} image_id={image_token_id}")

    meta_f.close()
    print("[DONE] written npz files:", written)
    print("[META]", meta_path)


if __name__ == "__main__":
    main()
