import os
import re
import json
import math
import random
import copy
import inspect
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from tqdm import tqdm
from PIL import Image
import requests
import pdb

import transformers
from transformers import (
    AutoProcessor,
    LlamaTokenizerFast,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
)

from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import (
    SampleOutput,
    SampleDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
)

from .llava import LlavaForConditionalGeneration, LlavaForConditionalGenerationScal


MODEL = "llava-hf/llava-1.5-7b-hf"


# ============================================================
# Probe / output helpers
# ============================================================

def load_probe_sample_id_set():
    """
    Optional filter for contribution probe.

    Env:
        PROBE_SAMPLE_IDS_FILE=output/relation_contribution_probe/ids_gold_on_under.txt

    The ids should be original dataset indices. Do NOT use Subset(dataset, ids),
    because AdaptVis prompts/answers are indexed by original index_of_total.
    """
    sample_id_file = os.getenv("PROBE_SAMPLE_IDS_FILE", "").strip()

    if not sample_id_file:
        return None

    if not os.path.exists(sample_id_file):
        raise FileNotFoundError(f"PROBE_SAMPLE_IDS_FILE not found: {sample_id_file}")

    sample_id_set = set()

    with open(sample_id_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sample_id_set.add(int(line))

    print(
        f"[PROBE FILTER] loaded {len(sample_id_set)} sample ids "
        f"from {sample_id_file}"
    )

    return sample_id_set


def make_tagged_output_path(dataset, method, weight, option, test_flag):
    """
    Original AdaptVis writes to a fixed filename. That is unsafe for multi-GPU
    probe jobs. If PROBE_RUN_TAG is set, append it to the output filename.
    """
    base = f"./output/results1.5_{dataset}_{method}_{weight}_{option}option_{test_flag}"
    tag = os.getenv("PROBE_RUN_TAG", "").strip()

    if tag:
        safe_tag = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", tag)
        return f"{base}_{safe_tag}.json"

    return f"{base}.json"


# ============================================================
# Object-mask CLIP helpers
# ============================================================

QUESTION_RE = re.compile(
    r"Where\s+is\s+(?:the\s+)?(.+?)\s+in\s+relation\s+to\s+(?:the\s+)?(.+?)\?\s*"
    r"Answer\s+with\s+left,\s*right,\s*on\s+or\s+under\.?",
    re.IGNORECASE,
)


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def _normalize_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.min()
    return x / (x.max() + eps)


def _clean_obj_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.IGNORECASE)
    return s.strip()


def is_generation_correct(golden: str, gen: str) -> bool:
    """
    Use the same correctness rule as the original AdaptVis evaluation,
    with a small normalization for in-front / in front / front.
    """
    golden = str(golden)
    gen = str(gen)

    golden_l = golden.strip().lower()
    gen_l = gen.strip().lower()

    golden_norm = golden_l.replace("in front", "in-front").replace("in_front", "in-front")
    gen_norm = gen_l.replace("in front", "in-front").replace("in_front", "in-front")

    ok = (golden in gen) or (golden_l in gen_l) or (golden_norm in gen_norm)

    if golden_norm in ["in-front", "front"]:
        ok = (
            ("front" in gen_norm)
            or ("in-front" in gen_norm)
            or ("in front" in gen_l)
        )

    if golden_norm == "on" and "front" in gen_l:
        ok = False

    return bool(ok)


RELATION_WORDS = ["left", "right", "on", "under"]


def get_probe_relation_set(dataset: Optional[str] = None) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Return canonical relation names and text/token aliases.

    You can override by env:
        PROBE_RELATION_SET=controlled_a
        PROBE_RELATION_SET=controlled_b
        PROBE_RELATION_SET=coco_two_obj
    """
    relation_set = os.getenv("PROBE_RELATION_SET", "").strip().lower()
    dataset_l = str(dataset or "").strip().lower()

    if not relation_set:
        if "controlled_images_b" in dataset_l:
            relation_set = "controlled_b"
        elif "coco_qa_two_obj" in dataset_l or "coco" in dataset_l:
            relation_set = "coco_two_obj"
        else:
            relation_set = "controlled_a"

    if relation_set in ["controlled_b", "b"]:
        canonical = ["left", "right", "in-front", "behind"]
        aliases = {
            "left": ["left"],
            "right": ["right"],
            "in-front": ["in-front", "in front", "front"],
            "behind": ["behind"],
        }
    elif relation_set in ["coco_two_obj", "coco", "coco_qa_two_obj"]:
        canonical = ["left", "right", "above", "below"]
        aliases = {
            "left": ["left"],
            "right": ["right"],
            "above": ["above"],
            "below": ["below"],
        }
    else:
        canonical = ["left", "right", "on", "under"]
        aliases = {
            "left": ["left"],
            "right": ["right"],
            "on": ["on"],
            "under": ["under"],
        }

    return canonical, aliases


def normalize_relation_token_text(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("▁", " ")
    s = s.strip()
    s = re.sub(r"^[\s\.,:;!\?\-\(\)\[\]\{\}'\"]+", "", s)
    s = re.sub(r"[\s\.,:;!\?\-\(\)\[\]\{\}'\"]+$", "", s)
    s = s.replace("in front", "in-front").replace("in_front", "in-front")
    return s


def canonicalize_relation_text(text: str, dataset: Optional[str] = None) -> str:
    s = str(text).strip().lower()
    s_norm = s.replace("in front", "in-front").replace("in_front", "in-front")

    canonical, aliases = get_probe_relation_set(dataset)

    if "behind" in canonical and re.search(r"\bbehind\b", s_norm):
        return "behind"

    if "in-front" in canonical:
        if (
            re.search(r"\bin\s*-\s*front\b", s_norm)
            or re.search(r"\bin\s+front\b", s)
            or re.search(r"\bfront\b", s_norm)
        ):
            return "in-front"

    for rel in canonical:
        if rel == "in-front":
            continue

        if re.search(rf"\b{re.escape(rel)}\b", s_norm):
            if rel == "on" and "front" in s_norm:
                continue
            return rel

    return "unknown"


def relation_exists_in_text(text: str, rel: str, dataset: Optional[str] = None) -> bool:
    s = str(text).strip().lower()
    s_norm = s.replace("in front", "in-front").replace("in_front", "in-front")

    if rel == "in-front":
        return bool(
            re.search(r"\bin\s*-\s*front\b", s_norm)
            or re.search(r"\bin\s+front\b", s)
            or re.search(r"\bfront\b", s_norm)
        )

    if rel == "on" and "front" in s_norm:
        return False

    return bool(re.search(rf"\b{re.escape(rel)}\b", s_norm))


def relations_in_text(text: str, dataset: Optional[str] = None) -> List[str]:
    canonical, _ = get_probe_relation_set(dataset)
    out = []
    for rel in canonical:
        if relation_exists_in_text(text, rel, dataset=dataset):
            out.append(rel)
    return out


def parse_relation_from_text(text: str, dataset: Optional[str] = None) -> str:
    return canonicalize_relation_text(text, dataset=dataset)


def _single_token_prob(tokenizer, probs: torch.Tensor, alias: str):
    candidates = [alias, " " + alias]
    best = None
    best_token_id = None
    best_decoded = None

    for cand in candidates:
        try:
            token_ids = tokenizer.encode(cand, add_special_tokens=False)
        except Exception:
            token_ids = []

        if len(token_ids) == 1:
            tid = int(token_ids[0])
            p = float(probs[tid].item())
            if best is None or p > best:
                best = p
                best_token_id = tid
                best_decoded = tokenizer.decode(
                    [tid],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

    return {
        "prob": best,
        "token_id": best_token_id,
        "decoded": best_decoded,
    }


def _generated_phrase_prob_if_matches(
    tokenizer,
    generated_ids,
    scores,
    end_step: int,
    phrase: str,
):
    variants = [phrase, " " + phrase]
    best = None
    best_ids = None
    best_decoded = None

    for v in variants:
        try:
            ids = tokenizer.encode(v, add_special_tokens=False)
        except Exception:
            ids = []

        if not ids:
            continue

        L = len(ids)
        start = end_step - L + 1

        if start < 0:
            continue

        actual = [int(x.item()) for x in generated_ids[start:end_step + 1]]

        if actual != [int(x) for x in ids]:
            continue

        prob = 1.0

        for off, tid in enumerate(ids):
            step = start + off
            logits = scores[step][0].detach().float()
            step_probs = torch.softmax(logits, dim=-1)
            prob *= float(step_probs[int(tid)].item())

        decoded = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if best is None or prob > best:
            best = prob
            best_ids = [int(x) for x in ids]
            best_decoded = decoded

    return {
        "prob": best,
        "token_ids": best_ids,
        "decoded": best_decoded,
    }


def build_relation_candidate_probs(
    tokenizer,
    scores,
    generated_ids,
    step: int,
    probs: torch.Tensor,
    dataset: Optional[str] = None,
):
    canonical, aliases = get_probe_relation_set(dataset)
    out = {}

    for rel in canonical:
        rel_info = {
            "aliases": aliases.get(rel, [rel]),
            "single_token": {},
            "generated_phrase": {},
        }

        for alias in aliases.get(rel, [rel]):
            alias_norm = alias.strip().lower()

            single = _single_token_prob(tokenizer, probs, alias_norm)
            rel_info["single_token"][alias_norm] = single

            phrase = _generated_phrase_prob_if_matches(
                tokenizer=tokenizer,
                generated_ids=generated_ids,
                scores=scores,
                end_step=step,
                phrase=alias_norm,
            )
            rel_info["generated_phrase"][alias_norm] = phrase

        if rel == "in-front":
            rel_info["in_front_parts"] = {}

            for part in ["in", "front", "in-front"]:
                rel_info["in_front_parts"][part] = _single_token_prob(
                    tokenizer,
                    probs,
                    part,
                )

            if step >= 1:
                prev_logits = scores[step - 1][0].detach().float()
                prev_probs = torch.softmax(prev_logits, dim=-1)
                rel_info["in_front_parts"]["prev_step_in"] = _single_token_prob(
                    tokenizer,
                    prev_probs,
                    "in",
                )
            else:
                rel_info["in_front_parts"]["prev_step_in"] = {
                    "prob": None,
                    "token_id": None,
                    "decoded": None,
                }

        out[rel] = rel_info

    return out


def extract_relation_token_topk_from_generate_output(
    output,
    input_len: int,
    tokenizer,
    topk: int = 10,
    dataset: Optional[str] = None,
):
    if output is None:
        return None

    if "sequences" not in output or "scores" not in output:
        return None

    canonical, _ = get_probe_relation_set(dataset)

    seq = output["sequences"][0]
    scores = output["scores"]

    if scores is None or len(scores) == 0:
        return None

    generated_ids = seq[input_len:]
    max_steps = min(len(generated_ids), len(scores))

    relation_hits = []
    prev_text = ""

    for step in range(max_steps):
        token_id = int(generated_ids[step].item())

        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        token_norm = normalize_relation_token_text(token_text)

        curr_text = tokenizer.decode(
            generated_ids[: step + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        relation = None

        for rel in canonical:
            if rel == "in-front":
                if token_norm in ["front", "in-front"]:
                    relation = "in-front"
                    break
            elif token_norm == rel:
                if not (rel == "on" and "front" in str(curr_text).lower()):
                    relation = rel
                    break

        if relation is None:
            before_relations = relations_in_text(prev_text, dataset=dataset)
            after_relations = relations_in_text(curr_text, dataset=dataset)
            newly_added = [rel for rel in after_relations if rel not in before_relations]

            if newly_added:
                curr_lower = str(curr_text).lower().replace("in front", "in-front")

                def last_pos(rel):
                    if rel == "in-front":
                        return max(
                            curr_lower.rfind("in-front"),
                            curr_lower.rfind("front"),
                        )
                    return curr_lower.rfind(rel)

                relation = max(newly_added, key=last_pos)

        if relation is not None:
            relation_hits.append(
                {
                    "step": int(step),
                    "relation": relation,
                    "token_id": int(token_id),
                    "token_text": token_text,
                    "token_norm": token_norm,
                    "text_until_relation": curr_text,
                }
            )

        prev_text = curr_text

    full_generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    if relation_hits:
        hit = relation_hits[-1]

        step = hit["step"]
        token_id = hit["token_id"]
        relation = hit["relation"]

        logits = scores[step][0].detach().float()
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=topk)

        top_tokens = []
        for rank, (tid, prob) in enumerate(
            zip(top_ids.tolist(), top_probs.tolist()),
            start=1,
        ):
            decoded = tokenizer.decode(
                [int(tid)],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            top_tokens.append(
                {
                    "rank": int(rank),
                    "token_id": int(tid),
                    "token": decoded,
                    "token_clean": normalize_relation_token_text(decoded),
                    "prob": float(prob),
                }
            )

        relation_candidate_probs = build_relation_candidate_probs(
            tokenizer=tokenizer,
            scores=scores,
            generated_ids=generated_ids,
            step=step,
            probs=probs,
            dataset=dataset,
        )

        return {
            "found": True,
            "probe_mode": "last_relation_token_dataset_aware",
            "relation_set": canonical,
            "relation": relation,
            "step": int(step),
            "generated_token_id": int(token_id),
            "generated_token": hit["token_text"],
            "generated_token_clean": hit["token_norm"],
            "generated_token_prob": float(probs[token_id].item()),
            "generated_text_until_relation": hit["text_until_relation"],
            "full_generated_text": full_generated_text,
            "num_relation_hits": int(len(relation_hits)),
            "relation_hits": relation_hits,
            "relation_candidate_probs": relation_candidate_probs,
            "topk": int(topk),
            "top_tokens": top_tokens,
        }

    first_logits = scores[0][0].detach().float()
    first_probs = torch.softmax(first_logits, dim=-1)
    top_probs, top_ids = torch.topk(first_probs, k=topk)

    top_tokens = []
    for rank, (tid, prob) in enumerate(
        zip(top_ids.tolist(), top_probs.tolist()),
        start=1,
    ):
        decoded = tokenizer.decode(
            [int(tid)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        top_tokens.append(
            {
                "rank": int(rank),
                "token_id": int(tid),
                "token": decoded,
                "token_clean": normalize_relation_token_text(decoded),
                "prob": float(prob),
            }
        )

    relation_candidate_probs = build_relation_candidate_probs(
        tokenizer=tokenizer,
        scores=scores,
        generated_ids=generated_ids,
        step=0,
        probs=first_probs,
        dataset=dataset,
    )

    return {
        "found": False,
        "probe_mode": "last_relation_token_dataset_aware",
        "relation_set": canonical,
        "relation": "unknown",
        "step": None,
        "generated_token_id": None,
        "generated_token": None,
        "generated_token_clean": None,
        "generated_token_prob": None,
        "generated_text_until_relation": full_generated_text,
        "full_generated_text": full_generated_text,
        "num_relation_hits": 0,
        "relation_hits": [],
        "relation_candidate_probs": relation_candidate_probs,
        "topk": int(topk),
        "top_tokens": top_tokens,
    }


def extract_objects_from_question(question: str) -> Optional[Tuple[str, str]]:
    if question is None:
        return None

    m = QUESTION_RE.search(question)

    if m is None:
        return None

    obj1 = _clean_obj_name(m.group(1))
    obj2 = _clean_obj_name(m.group(2))
    return obj1, obj2


def build_clip_text_prompts(obj_name: str) -> List[str]:
    obj_name = obj_name.strip()
    return [
        obj_name,
        f"a photo of a {obj_name}",
        f"an image of a {obj_name}",
        f"the {obj_name}",
    ]


def ensure_pil_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        x = image.detach().cpu()

        if x.dim() == 4:
            x = x[0]

        if x.dim() != 3:
            raise ValueError(f"Cannot convert tensor with shape {tuple(x.shape)} to PIL.")

        if x.shape[0] in [1, 3]:
            x = x.permute(1, 2, 0)

        x = x.float()

        if x.min() < 0 or x.max() > 1:
            x = x - x.min()
            x = x / (x.max() + 1e-8)

        arr = (x.numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            arr = arr - arr.min()
            arr = arr / (arr.max() + 1e-8)
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    raise TypeError(f"Unsupported image type for CLIP object mask: {type(image)}")


@torch.no_grad()
def get_clip_text_embed(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    obj_name: str,
    device: str,
) -> torch.Tensor:
    prompts = build_clip_text_prompts(obj_name)

    text_inputs = clip_processor.tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    text_outputs = clip_model.text_model(**text_inputs)
    pooled = text_outputs.pooler_output

    text_embeds = clip_model.text_projection(pooled)
    text_embeds = _l2_normalize(text_embeds, dim=-1)

    text_embed = text_embeds.mean(dim=0, keepdim=True)
    text_embed = _l2_normalize(text_embed, dim=-1)
    return text_embed


def dilate_binary_mask(mask_2d: torch.Tensor, dilate: int = 0) -> torch.Tensor:
    if dilate <= 0:
        return mask_2d

    x = mask_2d.float()[None, None, :, :]
    k = 2 * dilate + 1
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=dilate)
    return x[0, 0].bool()


@torch.no_grad()
def compute_clip_object_mask_binary(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    pil_image,
    question: str,
    device: str,
    clip_threshold: float = 0.85,
    invert: bool = True,
    dilate: int = 1,
) -> Optional[torch.Tensor]:
    objs = extract_objects_from_question(question)

    if objs is None:
        return None

    obj1, obj2 = objs
    pil_image = ensure_pil_image(pil_image)

    inputs = clip_processor(images=pil_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    vision_outputs = clip_model.vision_model(
        pixel_values=pixel_values,
        output_hidden_states=False,
        return_dict=True,
    )

    patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]

    if hasattr(clip_model.vision_model, "post_layernorm"):
        patch_tokens = clip_model.vision_model.post_layernorm(patch_tokens)

    patch_embeds = clip_model.visual_projection(patch_tokens)
    patch_embeds = _l2_normalize(patch_embeds, dim=-1)

    n_patches = patch_embeds.shape[1]
    grid_size = int(math.sqrt(n_patches))

    if grid_size * grid_size != n_patches:
        raise ValueError(f"CLIP patch number {n_patches} is not square.")

    text1 = get_clip_text_embed(clip_model, clip_processor, obj1, device)
    text2 = get_clip_text_embed(clip_model, clip_processor, obj2, device)

    sim1 = torch.matmul(patch_embeds, text1.T).squeeze(-1)[0]
    sim2 = torch.matmul(patch_embeds, text2.T).squeeze(-1)[0]

    if invert:
        sim1 = -sim1
        sim2 = -sim2

    score1 = _normalize_01(sim1).view(grid_size, grid_size)
    score2 = _normalize_01(sim2).view(grid_size, grid_size)

    object_score = torch.maximum(score1, score2)

    object_mask_2d = object_score >= clip_threshold
    object_mask_2d = dilate_binary_mask(object_mask_2d, dilate=dilate)

    object_patch_mask = object_mask_2d.reshape(-1)

    if object_patch_mask.sum() == 0:
        fallback_ratio = float(os.getenv("CLIP_OBJ_FALLBACK_TOP_RATIO", "0.05"))
        k = max(1, int(fallback_ratio * object_score.numel()))

        flat_score = object_score.reshape(-1)
        topk_idx = torch.topk(flat_score, k=k).indices

        object_patch_mask = torch.zeros_like(flat_score, dtype=torch.bool)
        object_patch_mask[topk_idx] = True

    debug = os.getenv("CLIP_OBJ_DEBUG", "False") == "True"
    if debug:
        print(
            f"[CLIP_OBJ_MASK] obj1={obj1}, obj2={obj2}, "
            f"threshold={clip_threshold}, dilate={dilate}, "
            f"selected={int(object_patch_mask.sum().item())}/{object_patch_mask.numel()}"
        )

    return object_patch_mask.detach()


def build_manual_patch_mask_from_env(device):
    mode = os.getenv("PATCH_MASK_MODE", "").strip()

    if mode == "":
        return None

    grid_size = int(os.getenv("PATCH_GRID_SIZE", "24"))
    block_grid = int(os.getenv("PATCH_BLOCK_GRID", "4"))
    block_id = int(os.getenv("PATCH_BLOCK_ID", "0"))
    block_ids_env = os.getenv("PATCH_BLOCK_IDS", "")

    num_patches = grid_size * grid_size
    mask_2d = torch.zeros((grid_size, grid_size), dtype=torch.bool)

    def parse_block_ids():
        if block_ids_env.strip() == "":
            return [block_id]

        ids = []
        for x in block_ids_env.split(","):
            x = x.strip()
            if x:
                ids.append(int(x))

        if len(ids) == 0:
            return [block_id]

        return ids

    def block_slice(bid):
        if grid_size % block_grid != 0:
            raise ValueError(
                f"PATCH_GRID_SIZE={grid_size} must be divisible by "
                f"PATCH_BLOCK_GRID={block_grid}"
            )

        total_blocks = block_grid * block_grid

        if not (0 <= bid < total_blocks):
            raise ValueError(
                f"Invalid block id={bid} for PATCH_BLOCK_GRID={block_grid}. "
                f"Valid range: 0..{total_blocks - 1}"
            )

        block_h = grid_size // block_grid
        block_w = grid_size // block_grid

        br = bid // block_grid
        bc = bid % block_grid

        r0 = br * block_h
        r1 = (br + 1) * block_h
        c0 = bc * block_w
        c1 = (bc + 1) * block_w

        return r0, r1, c0, c1

    if mode == "all":
        mask_2d[:, :] = True

    elif mode == "block":
        r0, r1, c0, c1 = block_slice(block_id)
        mask_2d[r0:r1, c0:c1] = True

    elif mode == "blocks":
        for bid in parse_block_ids():
            r0, r1, c0, c1 = block_slice(bid)
            mask_2d[r0:r1, c0:c1] = True

    elif mode == "except_block":
        mask_2d[:, :] = True
        r0, r1, c0, c1 = block_slice(block_id)
        mask_2d[r0:r1, c0:c1] = False

    elif mode == "except_blocks":
        mask_2d[:, :] = True
        for bid in parse_block_ids():
            r0, r1, c0, c1 = block_slice(bid)
            mask_2d[r0:r1, c0:c1] = False

    elif mode == "row":
        row_id = int(os.getenv("PATCH_ROW_ID", "0"))

        if not (0 <= row_id < grid_size):
            raise ValueError(f"Invalid PATCH_ROW_ID={row_id}")

        mask_2d[row_id, :] = True

    elif mode == "col":
        col_id = int(os.getenv("PATCH_COL_ID", "0"))

        if not (0 <= col_id < grid_size):
            raise ValueError(f"Invalid PATCH_COL_ID={col_id}")

        mask_2d[:, col_id] = True

    else:
        raise ValueError(f"Unknown PATCH_MASK_MODE={mode}")

    patch_mask = mask_2d.reshape(-1).to(device)

    if os.getenv("PATCH_MASK_DEBUG", "False") == "True":
        if mode in ["blocks", "except_blocks"]:
            selected_blocks = ",".join(str(x) for x in parse_block_ids())
        else:
            selected_blocks = str(block_id)

        print(
            f"[PATCH_MASK] mode={mode}, grid={grid_size}, "
            f"block_grid={block_grid}, block_ids={selected_blocks}, "
            f"selected={int(patch_mask.sum().item())}/{num_patches}"
        )

    return patch_mask


# ============================================================
# Custom greedy search
# ============================================================

def _add_weight_greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    output_logits: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    weight: Optional[float] = None,
    adjust_method: Optional[str] = None,
    pos: Optional[torch.Tensor] = None,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use "
            "`stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    raw_logits = () if (return_dict_in_generate and output_logits) else None
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    batch_size, cur_len = input_ids.shape
    if "inputs_embeds" in model_kwargs:
        cur_len = model_kwargs["inputs_embeds"].shape[1]

    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        for custom_key in [
            "keys",
            "object_patch_mask",
            "caption_length",
        ]:
            if custom_key not in model_inputs and custom_key in model_kwargs:
                model_inputs[custom_key] = model_kwargs.get(custom_key, None)

        if "Scal" not in str(type(self)):
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        else:
            outputs = self(
                **model_inputs,
                weight=weight,
                adjust_method=adjust_method,
                pos=pos,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :]

        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        for custom_key in [
            "keys",
            "object_patch_mask",
            "caption_length",
        ]:
            if custom_key in model_inputs and custom_key not in model_kwargs:
                model_kwargs[custom_key] = model_inputs[custom_key]

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )

    return input_ids


def change_greedy_to_add_weight():
    transformers.generation.utils.GenerationMixin._greedy_search = _add_weight_greedy_search


# ============================================================
# LLaVA Wrapper
# ============================================================

class LlavaWrapper:
    def __init__(self, root_dir, device, method):
        if method == "scaling_vis" or method == "adapt_vis":
            self.model = LlavaForConditionalGenerationScal.from_pretrained(
                MODEL,
                revision="a272c74",
                cache_dir=root_dir,
                ignore_mismatched_sizes=True,
            ).eval().to(device)
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                MODEL,
                revision="a272c74",
                cache_dir=root_dir,
                ignore_mismatched_sizes=True,
            ).eval().to(device)

        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            MODEL,
            revision="a272c74",
            cache_dir=root_dir,
        )
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            MODEL,
            revision="a272c74",
            cache_dir=root_dir,
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL,
            revision="a272c74",
            cache_dir=root_dir,
        )

        self.device = device

        self.use_clip_obj_mask = os.getenv("CLIP_OBJ_MASK", "False") == "True"

        if self.use_clip_obj_mask:
            clip_name = os.getenv("CLIP_OBJ_MODEL", "openai/clip-vit-large-patch14-336")
            print(f"[INFO] Loading external CLIP for object mask: {clip_name}")

            self.clip_obj_model = CLIPModel.from_pretrained(
                clip_name,
                cache_dir=root_dir,
            ).to(device).eval()

            self.clip_obj_processor = CLIPProcessor.from_pretrained(
                clip_name,
                cache_dir=root_dir,
            )
        else:
            self.clip_obj_model = None
            self.clip_obj_processor = None

    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=64, normalize=False):
        num_text = len(texts)
        text_embeds = []

        for i in tqdm(range(0, num_text, text_batch_size)):
            text = texts[i: min(num_text, i + text_batch_size)]
            text_input = self.tokenizer(
                text=text,
                return_tensors="pt",
                padding="max_length",
                max_length=77,
            ).to(self.device)

            text_feats = self.model.llava.get_text_features(**text_input).cpu().numpy()[:, 0, :]

            if normalize:
                text_feats = text_feats / np.linalg.norm(text_feats, axis=1, keepdims=True)

            text_embeds.append(text_feats)

        return np.concatenate(text_embeds, axis=0)

    @torch.no_grad()
    def get_image_embeddings(self, image_loader, normalize=False):
        image_embeds = []

        for batch in tqdm(image_loader):
            images = batch["image"]
            inputs = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
            image_feats = self.model.llava.get_image_features(**inputs).cpu().numpy()[:, 0, :]

            if normalize:
                image_feats = image_feats / np.linalg.norm(image_feats, axis=1, keepdims=True)

            image_embeds.append(image_feats)

        return np.concatenate(image_embeds, axis=0)

    def get_retrieval_scores_dataset(self, loader):
        texts = loader.dataset.text
        text_embeds = self.get_text_embeddings(texts, normalize=True)
        image_embeds = self.get_image_embeddings(loader, normalize=True)
        scores = image_embeds @ text_embeds.T
        return scores

    @torch.no_grad()
    def get_out_scores_wh_batched(
        self,
        dataset,
        joint_loader,
        method,
        weight,
        option,
        threshold,
        weight1,
        weight2,
    ):
        scores = []
        index_of_total = 0
        processed_count = 0
        skipped_count = 0
        acc = 0
        correct_id = []
        processed_sample_ids = []

        sample_id_set = load_probe_sample_id_set()

        qst_ans_file = f"prompts/{dataset}_with_answer_{option}_options.jsonl"

        with open(qst_ans_file, "r") as file:
            prompt_list = []
            answer_list = []

            for line in file:
                data = json.loads(line)
                prompt_list.append(data["question"])
                answer_list.append(data["answer"])

        SAMPLE = False
        TEST = os.getenv("TEST_MODE", "False") == "True"
        total_data_count = len(prompt_list)

        if SAMPLE:
            idx_file_path = f"./output/sampled_idx_{dataset}.npy"

            if os.path.exists(idx_file_path):
                sampled_indices = np.load(idx_file_path).tolist()
            else:
                sampled_indices = random.sample(range(total_data_count), int(0.2 * total_data_count))
                sampled_indices.sort()
                np.save(idx_file_path, np.array(sampled_indices))

            if TEST:
                all_indices = set(range(total_data_count))
                unsampled_indices = list(all_indices - set(sampled_indices))
                unsampled_indices.sort()
                sampled_indices = unsampled_indices

            prompt_list = [prompt_list[i] for i in sampled_indices]
            answer_list = [answer_list[i] for i in sampled_indices]

        attn_run_tag = os.getenv("ATTN_RUN_TAG", "")

        if attn_run_tag:
            save_attn_dir = f"./output/{attn_run_tag}"
        elif method == "scaling_vis":
            save_attn_dir = f"./output/{dataset}_scaling_w{weight:.2f}"
        elif method == "adapt_vis":
            save_attn_dir = (
                f"./output/{dataset}_adapt"
                f"_th{threshold:.2f}_w1{weight1:.2f}_w2{weight2:.2f}"
            )
        else:
            save_attn_dir = f"./output/{dataset}_{method}"

        os.makedirs(save_attn_dir, exist_ok=True)

        output_file_path = make_tagged_output_path(
            dataset=dataset,
            method=method,
            weight=weight,
            option=option,
            test_flag=TEST,
        )

        print(f"[OUTPUT] result path = {output_file_path}")

        results = []

        for batch in tqdm(joint_loader):
            # For Controlled_Images_* this loader is normally batch_size=1.
            # Keep index_of_total as the original dataset index.
            if sample_id_set is not None and index_of_total not in sample_id_set:
                skipped_count += 1
                index_of_total += 1
                continue

            batch_scores = []

            os.environ["SAVE_ATTN_PATH"] = f"{save_attn_dir}/{index_of_total}/"
            os.makedirs(os.environ["SAVE_ATTN_PATH"], exist_ok=True)

            for i_option in batch["image_options"]:
                im_scores = []

                for _ in i_option:
                    sample_id = int(index_of_total)
                    prompt = prompt_list[index_of_total]

                    single_input = self.processor(
                        text=prompt,
                        images=_,
                        padding="max_length",
                        return_tensors="pt",
                        max_length=77,
                    ).to(self.device)

                    keys = [
                        torch.where(input_id == 32001, 1, 0)
                        for input_id in single_input["input_ids"]
                    ]

                    adjust_method_env = os.getenv("ADJUST_METHOD", "last_query")
                    query_pos_env = os.getenv("QUERY_POS", "")

                    query_pos = None
                    if adjust_method_env == "text_offset":
                        if query_pos_env == "":
                            raise ValueError("ADJUST_METHOD=text_offset requires QUERY_POS.")
                        query_pos = torch.tensor(int(query_pos_env), device=self.device)

                    object_patch_mask = None

                    manual_patch_mask = build_manual_patch_mask_from_env(self.device)

                    if adjust_method_env == "object_mask" and manual_patch_mask is not None:
                        object_patch_mask = manual_patch_mask

                    elif self.use_clip_obj_mask and adjust_method_env == "object_mask":
                        clip_obj_threshold = float(os.getenv("CLIP_OBJ_THRESHOLD", "0.85"))
                        clip_obj_dilate = int(os.getenv("CLIP_OBJ_DILATE", "1"))
                        clip_obj_invert = os.getenv("CLIP_OBJ_INVERT", "True") == "True"

                        object_patch_mask = compute_clip_object_mask_binary(
                            clip_model=self.clip_obj_model,
                            clip_processor=self.clip_obj_processor,
                            pil_image=_,
                            question=prompt,
                            device=self.device,
                            clip_threshold=clip_obj_threshold,
                            invert=clip_obj_invert,
                            dilate=clip_obj_dilate,
                        )

                        if object_patch_mask is not None:
                            object_patch_mask = object_patch_mask.to(self.device)

                    selected_weight = None

                    if method == "scaling_vis":
                        change_greedy_to_add_weight()
                        selected_weight = weight

                        output = self.model.generate(
                            **single_input,
                            keys=keys,
                            weight=weight,
                            adjust_method=adjust_method_env,
                            pos=query_pos,
                            object_patch_mask=object_patch_mask,
                            max_new_tokens=100,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )

                        uncertainty = np.round(
                            float(max(torch.nn.functional.softmax(output["scores"][0], dim=-1)[0])),
                            2,
                        )

                        gen = self.processor.decode(
                            output["sequences"][0][len(single_input["input_ids"][-1]):],
                            skip_special_tokens=True,
                        )

                    elif method == "adapt_vis":
                        change_greedy_to_add_weight()

                        probe_single_pass = (
                            adjust_method_env in ["probe_bias", "probe_scale", "var_sink"]
                            or os.getenv("PROBE_RUN_TAG", "").strip() != ""
                        )

                        if probe_single_pass:
                            # In probe mode, intervention is controlled by env vars:
                            # PROBE_LAYER / PROBE_HEAD / PROBE_BLOCK_IDS / PROBE_SCALE.
                            # Do not run the original AdaptVis confidence branch twice.
                            selected_weight = 1.0

                            output = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=1.0,
                                adjust_method=adjust_method_env,
                                pos=query_pos,
                                object_patch_mask=object_patch_mask,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )

                            uncertainty = np.round(
                                float(max(torch.nn.functional.softmax(output["scores"][0], dim=-1)[0])),
                                2,
                            )

                            gen = self.processor.decode(
                                output["sequences"][0][len(single_input["input_ids"][-1]):],
                                skip_special_tokens=True,
                            )

                        else:
                            output = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=1.0,
                                adjust_method=adjust_method_env,
                                pos=query_pos,
                                object_patch_mask=object_patch_mask,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )

                            uncertainty = np.round(
                                float(max(torch.nn.functional.softmax(output["scores"][0], dim=-1)[0])),
                                2,
                            )

                            print(uncertainty, threshold)

                            if uncertainty < threshold:
                                selected_weight = weight1

                                output = self.model.generate(
                                    **single_input,
                                    keys=keys,
                                    weight=weight1,
                                    adjust_method=adjust_method_env,
                                    pos=query_pos,
                                    object_patch_mask=object_patch_mask,
                                    max_new_tokens=100,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                )
                            else:
                                selected_weight = weight2

                                output = self.model.generate(
                                    **single_input,
                                    keys=keys,
                                    weight=weight2,
                                    adjust_method=adjust_method_env,
                                    pos=query_pos,
                                    object_patch_mask=object_patch_mask,
                                    max_new_tokens=100,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                )

                            gen = self.processor.decode(
                                output["sequences"][0][len(single_input["input_ids"][-1]):],
                                skip_special_tokens=True,
                            )

                    else:
                        selected_weight = None

                        output = self.model.generate(
                            **single_input,
                            max_new_tokens=100,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )

                        gen = self.processor.decode(
                            output["sequences"][0][len(single_input["input_ids"][-1]):],
                            skip_special_tokens=True,
                        )

                        uncertainty = np.round(float(max(output["scores"][0][0])), 2)

                    relation_probe = None

                    if os.getenv("PROBE_RELATION_PROBS", "False") == "True":
                        probe_topk = int(os.getenv("PROBE_RELATION_TOPK", "10"))
                        input_len = len(single_input["input_ids"][-1])

                        relation_probe = extract_relation_token_topk_from_generate_output(
                            output=output,
                            input_len=input_len,
                            tokenizer=self.processor.tokenizer,
                            topk=probe_topk,
                            dataset=dataset,
                        )

                    golden = answer_list[index_of_total][0]
                    is_correct = is_generation_correct(golden, gen)

                    print(
                        f"Prompt: {prompt}\n"
                        f"Generation: {gen}\n"
                        f"Golden: {golden}\n"
                        f"Correct: {is_correct}"
                    )

                    c_option = batch["caption_options"]
                    num_options = len(list(c_option))

                    if num_options == 4:
                        if is_correct:
                            acc += 1
                            correct_id.append(sample_id)
                            answers = [1, 0, 0, 0]
                        else:
                            answers = [0, 0, 1, 0]

                    elif num_options == 2:
                        if is_correct:
                            acc += 1
                            correct_id.append(sample_id)
                            answers = [1, 0]
                        else:
                            answers = [0, 1]

                    else:
                        raise ValueError(f"Unexpected number of caption options: {num_options}")

                    patch_selected = None
                    if object_patch_mask is not None:
                        patch_selected = int(object_patch_mask.detach().bool().sum().item())

                    result = {
                        "sample_id": sample_id,
                        "Prompt": prompt,
                        "Generation": gen,
                        "Golden": golden,
                        "Correct": bool(is_correct),
                        "Uncertainty": float(uncertainty) if "uncertainty" in locals() else None,
                        "selected_weight": float(selected_weight) if selected_weight is not None else None,
                        "relation_probe": relation_probe,

                        "adjust_method": os.getenv("ADJUST_METHOD", "last_query"),
                        "probe_layer": os.getenv("PROBE_LAYER", ""),
                        "probe_head": os.getenv("PROBE_HEAD", ""),
                        "probe_block_ids": os.getenv("PROBE_BLOCK_IDS", ""),
                        "probe_beta": os.getenv("PROBE_BETA", ""),
                        "probe_scale": os.getenv("PROBE_SCALE", ""),
                        "probe_run_tag": os.getenv("PROBE_RUN_TAG", ""),

                        "patch_mask_mode": os.getenv("PATCH_MASK_MODE", ""),
                        "patch_grid_size": os.getenv("PATCH_GRID_SIZE", ""),
                        "patch_block_grid": os.getenv("PATCH_BLOCK_GRID", ""),
                        "patch_block_id": os.getenv("PATCH_BLOCK_ID", ""),
                        "patch_block_ids": os.getenv("PATCH_BLOCK_IDS", ""),
                        "patch_row_id": os.getenv("PATCH_ROW_ID", ""),
                        "patch_col_id": os.getenv("PATCH_COL_ID", ""),
                        "clip_obj_mask": os.getenv("CLIP_OBJ_MASK", "False"),
                        "clip_obj_threshold": os.getenv("CLIP_OBJ_THRESHOLD", ""),
                        "selected_patch_count": patch_selected,
                    }
                    results.append(result)

                    im_scores.append(np.expand_dims(np.array(answers), -1))

                    processed_count += 1
                    processed_sample_ids.append(sample_id)
                    index_of_total += 1

                if len(im_scores) > 0:
                    batch_scores.append(np.concatenate(im_scores, axis=-1))

            if len(batch_scores) > 0:
                scores.append(batch_scores)

            print("Saving results to", output_file_path)

            with open(output_file_path, "w", encoding="utf-8") as fout:
                json.dump(results, fout, ensure_ascii=False, indent=4)

            denom = processed_count if processed_count > 0 else 1
            print(
                f"[RUNNING] acc={acc}/{processed_count}={acc / denom:.6f}, "
                f"scanned={index_of_total}, skipped={skipped_count}"
            )

        denom = processed_count if processed_count > 0 else 1
        final_acc = acc / denom

        print(
            f"[FINAL] acc={acc}/{processed_count}={final_acc:.6f}, "
            f"scanned={index_of_total}, skipped={skipped_count}"
        )

        output_score_file = output_file_path.replace(".json", "scores.json")

        with open(output_score_file, "w", encoding="utf-8") as fout:
            json.dump(
                {
                    "acc": final_acc,
                    "correct_id": correct_id,
                    "processed_count": processed_count,
                    "skipped_count": skipped_count,
                    "processed_sample_ids": processed_sample_ids,
                    "sample_filter_file": os.getenv("PROBE_SAMPLE_IDS_FILE", ""),
                    "probe_run_tag": os.getenv("PROBE_RUN_TAG", ""),
                    "probe_scale": os.getenv("PROBE_SCALE", ""),
                },
                fout,
                ensure_ascii=False,
                indent=4,
            )

        if len(scores) > 0:
            all_scores = np.concatenate(scores, axis=0)
        else:
            if option == "four":
                all_scores = np.zeros((0, 4, 1))
            elif option == "two":
                all_scores = np.zeros((0, 2, 1))
            else:
                all_scores = np.zeros((0, 4, 1))

        if dataset in ["Controlled_Images_B", "Controlled_Images_A"]:
            return all_scores, []
        else:
            return final_acc, correct_id

    @torch.no_grad()
    def get_judge_scores_vsr_batched(
        self,
        dataset,
        joint_loader,
        method,
        weight,
        threshold,
        weight1,
        weight2,
    ):
        index = 0
        TP, TN, FP, FN = 0, 0, 0, 0

        save_attn_dir = f"/home/user/shiqi/mmlm_mech/whatsup_vlms/outputs/{dataset}_weight{weight:.2f}"

        if not os.path.exists(save_attn_dir):
            print("Creating directory for saving attention maps:", save_attn_dir)
            os.makedirs(save_attn_dir)

        index_of_total = 0
        results = []

        for batch in tqdm(joint_loader):
            batch_scores = []

            os.environ["SAVE_ATTN_PATH"] = f"{save_attn_dir}/{index_of_total}/"
            os.makedirs(os.environ["SAVE_ATTN_PATH"], exist_ok=True)

            for i_option in batch["image_options"]:
                im_scores = []

                for c_option in batch["caption_options"]:
                    prompt = (
                        "User: <image>\n Determine whether the description about the spatial relationship "
                        "is correct or not. Answer with yes or no: "
                    )

                    qst = [prompt] * len(list(c_option))
                    end_fix = [" Assistant:"] * len(list(c_option))

                    concatenated_list = [
                        s1 + s2 + s3
                        for s1, s2, s3 in zip(qst, c_option, end_fix)
                    ]

                    for idx, text in enumerate(concatenated_list):
                        single_input = self.processor(
                            text=text,
                            images=list(i_option)[idx],
                            padding="max_length",
                            return_tensors="pt",
                            max_length=77,
                        ).to(self.device)

                        keys = [
                            torch.where(input_id == 32001, 1, 0)
                            for input_id in single_input["input_ids"]
                        ]

                        if method == "scaling_vis":
                            change_greedy_to_add_weight()

                            output = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=weight,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )

                            uncertainty = np.round(
                                float(max(torch.nn.functional.softmax(output["scores"][0], dim=-1)[0])),
                                2,
                            )

                            gen = self.processor.decode(
                                output[0][0][len(single_input["input_ids"][-1]):],
                                skip_special_tokens=True,
                                output_attentions=True,
                            )

                        elif method == "adapt_vis":
                            change_greedy_to_add_weight()

                            output = self.model.generate(
                                **single_input,
                                weight=1.0,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )

                            gen = self.processor.decode(
                                output["sequences"][0][len(single_input["input_ids"][-1]):],
                                skip_special_tokens=True,
                                output_attentions=True,
                            )

                            uncertainty = np.round(float(max(output["scores"][0][0])), 2)

                            if uncertainty < threshold:
                                output = self.model.generate(
                                    **single_input,
                                    keys=keys,
                                    weight=weight1,
                                    max_new_tokens=100,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                )
                            else:
                                output = self.model.generate(
                                    **single_input,
                                    keys=keys,
                                    weight=weight2,
                                    max_new_tokens=100,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                )

                            gen = self.processor.decode(
                                output[0][0][len(single_input["input_ids"][-1]):],
                                skip_special_tokens=True,
                                output_attentions=True,
                            )

                        else:
                            output = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=weight,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )

                            uncertainty = np.round(
                                float(max(torch.nn.functional.softmax(output["scores"][0], dim=-1)[0])),
                                2,
                            )

                            gen = self.processor.decode(
                                output[0][0][len(single_input["input_ids"][-1]):],
                                skip_special_tokens=True,
                                output_attentions=True,
                            )

                        label = int(batch["labels"][0][idx])

                        if label == 1:
                            TP += 1 if "Yes" in gen else 0
                            FN += 1 if "Yes" not in gen else 0
                        else:
                            TN += 1 if "No" in gen else 0
                            FP += 1 if "No" not in gen else 0

                        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

                        gold = "Yes" if label == 1 else "No"

                        result = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Golden": gold,
                            "Uncertainty": uncertainty,
                        }

                        results.append(result)
                        index_of_total += 1

                index += 1

        precision = TP / (TP + FN)
        recall = TN / (TN + FP)
        f1_score = 2 * precision * recall / (precision + recall)

        print(
            f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n"
            f"Accuracy: {(TN + TP) / (TN + TP + FN + FP)}\n"
            f"Precision: {precision}\n"
            f"Recall: {recall}\n"
            f"F1 Score: {f1_score}"
        )

        all_scores = (TP, TN, FP, FN)

        output_file_path = f"./outputs/results_{dataset}_{method}_{weight}.json"

        with open(output_file_path, "w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=4)

        output_score_file = output_file_path.replace(".json", "_scores.json")

        with open(output_score_file, "w", encoding="utf-8") as fout:
            json.dump(
                {
                    "acc": (TN + TP) / (TN + TP + FN + FP),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1_score,
                },
                fout,
                ensure_ascii=False,
                indent=4,
            )

        return all_scores
