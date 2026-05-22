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

    Important:
        PROBE_RUN_TAG should only affect filename.
        It should NOT force single-pass generation.
        Use PROBE_SINGLE_PASS=True if you explicitly want single-pass probe mode.
    """
    base = f"./output/results1.5_{dataset}_{method}_{weight}_{option}option_{test_flag}"

    if use_closed_set_scoring_from_env():
        base = f"{base}_closedset"

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


# ============================================================
# Image-control helpers for bias / visual-grounding control
# ============================================================

def ensure_rgb_pil_for_control(image) -> Image.Image:
    """
    Convert input image to RGB PIL for IMAGE_CONTROL.
    This intentionally mirrors ensure_pil_image but is kept separate to make
    control logic explicit.
    """
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

    raise TypeError(f"Unsupported image type for IMAGE_CONTROL: {type(image)}")


def apply_image_control_from_env(image, sample_id: int = 0):
    """
    Env:
        IMAGE_CONTROL=none | blank_black | blank_gray | blank_white | blank_mean | shuffle_patches | random_noise
        IMAGE_CONTROL_SIZE=336
        IMAGE_CONTROL_GRID=24
        IMAGE_CONTROL_SEED=1
    """
    mode = os.getenv("IMAGE_CONTROL", "none").strip().lower()

    if mode in ["", "none", "original"]:
        return image

    pil = ensure_rgb_pil_for_control(image)

    seed = int(os.getenv("IMAGE_CONTROL_SEED", "1")) + int(sample_id)
    rng = np.random.default_rng(seed)

    if mode == "blank_black":
        return Image.new("RGB", pil.size, (0, 0, 0))

    if mode == "blank_gray":
        return Image.new("RGB", pil.size, (127, 127, 127))

    if mode == "blank_white":
        return Image.new("RGB", pil.size, (255, 255, 255))

    if mode == "blank_mean":
        arr = np.asarray(pil).astype(np.float32)
        mean_rgb = arr.reshape(-1, 3).mean(axis=0).clip(0, 255).astype(np.uint8)
        return Image.new("RGB", pil.size, tuple(int(x) for x in mean_rgb))

    if mode == "random_noise":
        size = int(os.getenv("IMAGE_CONTROL_SIZE", "336"))
        noise = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        return Image.fromarray(noise, mode="RGB")

    if mode == "shuffle_patches":
        size = int(os.getenv("IMAGE_CONTROL_SIZE", "336"))
        grid = int(os.getenv("IMAGE_CONTROL_GRID", "24"))

        if size % grid != 0:
            raise ValueError(
                f"IMAGE_CONTROL_SIZE={size} must be divisible by IMAGE_CONTROL_GRID={grid}"
            )

        patch = size // grid
        pil_resized = pil.resize((size, size), Image.BICUBIC)

        patches = []
        for r in range(grid):
            for c in range(grid):
                crop = pil_resized.crop(
                    (
                        c * patch,
                        r * patch,
                        (c + 1) * patch,
                        (r + 1) * patch,
                    )
                )
                patches.append(crop)

        perm = rng.permutation(len(patches))
        out = Image.new("RGB", (size, size))

        k = 0
        for r in range(grid):
            for c in range(grid):
                out.paste(patches[int(perm[k])], (c * patch, r * patch))
                k += 1

        return out

    raise ValueError(f"Unknown IMAGE_CONTROL={mode}")


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
# Closed-set relation scoring helpers
# ============================================================

def _env_flag(name: str, default: str = "False") -> bool:
    return str(os.getenv(name, default)).strip().lower() in [
        "1",
        "true",
        "yes",
        "y",
        "on",
    ]


def use_closed_set_scoring_from_env() -> bool:
    """
    Turn on closed-set decision without touching main_aro.py.

    Supported envs:
        CLOSED_SET_SCORING=True
        DECISION_MODE=closed_set
    """
    decision_mode = os.getenv("DECISION_MODE", "").strip().lower()
    return (
        _env_flag("CLOSED_SET_SCORING", "False")
        or decision_mode in ["closed_set", "closed-set", "closedset", "cs"]
    )


def extract_closed_set_candidates(
    prompt: str,
    dataset: Optional[str] = None,
    option: Optional[str] = None,
) -> List[str]:
    """
    Extract candidate answer strings from prompts like:
        "Answer with left, right, on or under."

    Falls back to dataset/option-aware relation sets when parsing fails.
    """
    prompt = str(prompt)

    m = re.search(
        r"Answer\s+with\s+(.+?)(?:\.|\n|$)",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if m is not None:
        raw = m.group(1).strip()
        raw = re.sub(r"\bonly\b", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(
            r"^(?:one\s+of\s+)?(?:the\s+)?(?:following\s*)?:",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip()
        raw = raw.replace(";", ",").replace("/", ",")
        raw = re.sub(r"\s+(?:or|and)\s+", ",", raw, flags=re.IGNORECASE)

        candidates = []
        for x in raw.split(","):
            x = x.strip()
            x = re.sub(r"^[\s\.:;\-\(\)\[\]\{\}'\"]+", "", x)
            x = re.sub(r"[\s\.:;\-\(\)\[\]\{\}'\"]+$", "", x)
            x = re.sub(r"\s+", " ", x).strip()

            if x and x.lower() not in ["answer", "answers"]:
                candidates.append(x)

        # Keep order while removing duplicates.
        deduped = []
        seen = set()
        for x in candidates:
            key = x.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(x)

        if len(deduped) >= 2:
            return deduped

    dataset_l = str(dataset or "").lower()
    option_l = str(option or "").lower()

    if "controlled_images_b" in dataset_l:
        return ["left", "right", "in front", "behind"]

    if "coco" in dataset_l:
        return ["left", "right", "above", "below"]

    if "vg" in dataset_l or option_l == "six":
        return ["left", "right", "above", "below", "in front", "behind"]

    return ["left", "right", "on", "under"]


def _candidate_answer_token_ids(tokenizer, candidate: str) -> List[int]:
    """
    Tokenize the appended answer exactly as used in full_text:
        prompt.rstrip() + " " + candidate
    """
    candidate = str(candidate).strip()
    token_ids = tokenizer.encode(" " + candidate, add_special_tokens=False)

    if len(token_ids) == 0:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)

    return [int(x) for x in token_ids]



def infer_llava_num_image_tokens(model, processor) -> int:
    """
    Infer the number of expanded visual tokens used by LLaVA attention.

    For llava-1.5 with 336x336 images and patch_size=14:
        24 * 24 = 576

    If vision_feature_select_strategy == "full", CLS may be kept, so return 577.
    You can override with:
        NUM_IMAGE_TOKENS=576
    """
    override = os.getenv("NUM_IMAGE_TOKENS", "").strip()
    if override:
        return int(override)

    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)

    image_size = getattr(vision_config, "image_size", None)
    patch_size = getattr(vision_config, "patch_size", None)

    if image_size is None:
        image_processor = getattr(processor, "image_processor", None)
        crop_size = getattr(image_processor, "crop_size", None)
        size = getattr(image_processor, "size", None)

        if isinstance(crop_size, dict):
            image_size = int(crop_size.get("height", crop_size.get("width", 336)))
        elif isinstance(crop_size, int):
            image_size = int(crop_size)
        elif isinstance(size, dict):
            if "height" in size:
                image_size = int(size["height"])
            elif "shortest_edge" in size:
                image_size = int(size["shortest_edge"])
            else:
                image_size = 336
        elif isinstance(size, int):
            image_size = int(size)
        else:
            image_size = 336

    if patch_size is None:
        patch_size = getattr(processor, "patch_size", 14)

    image_size = int(image_size)
    patch_size = int(patch_size)

    num_patches = (image_size // patch_size) ** 2

    strategy = getattr(
        config,
        "vision_feature_select_strategy",
        getattr(processor, "vision_feature_select_strategy", "default"),
    )

    if str(strategy).lower() == "full":
        return num_patches + 1

    return num_patches


def build_expanded_image_keys_from_input_ids(
    input_ids: torch.Tensor,
    image_token_id: int,
    num_image_tokens: int,
) -> torch.Tensor:
    """
    Convert input_ids-level <image> placeholder mask to attention-level image-token mask.

    input_ids normally contains one <image> placeholder, but LLaVA attention sees
    that placeholder expanded into many image patch tokens.

    Example:
        [text, <image>, text] -> [0..., 1 repeated num_image_tokens, 0...]

    If input_ids already contains multiple image tokens, fall back to the legacy mask
    instead of expanding each one again.
    """
    image_mask = input_ids == int(image_token_id)
    raw_count = int(image_mask.sum().item())

    if raw_count == 0:
        return torch.zeros_like(input_ids, dtype=torch.long)

    if raw_count > 1:
        # Already expanded or non-standard tokenizer behavior.
        return image_mask.long()

    out = []
    for tid in input_ids.detach().tolist():
        if int(tid) == int(image_token_id):
            out.extend([1] * int(num_image_tokens))
        else:
            out.append(0)

    return torch.tensor(out, dtype=torch.long, device=input_ids.device)


def build_adaptvis_keys_from_input_batch(
    input_ids_batch: torch.Tensor,
    model,
    processor,
    image_token_id: Optional[int] = None,
    debug_prefix: str = "",
) -> List[torch.Tensor]:
    """
    Build AdaptVis keys for a batch.

    Old behavior:
        keys mark only the single <image> placeholder in input_ids.

    New behavior:
        keys mark the expanded image-token range used inside LLaVA attention.

    Env:
        EXPAND_IMAGE_KEYS=True   # default, use expanded keys
        EXPAND_IMAGE_KEYS=False  # restore old placeholder-only behavior
        DEBUG_EXPANDED_KEYS=True # print key length / key sum
        NUM_IMAGE_TOKENS=576     # optional override
    """
    expand = _env_flag("EXPAND_IMAGE_KEYS", "True")

    if image_token_id is None:
        image_token_id = int(
            os.getenv(
                "IMAGE_TOKEN_ID",
                str(getattr(getattr(model, "config", None), "image_token_index", 32001)),
            )
        )

    if not expand:
        keys = [
            torch.where(input_id == image_token_id, 1, 0)
            for input_id in input_ids_batch
        ]
    else:
        num_image_tokens = infer_llava_num_image_tokens(model, processor)
        keys = [
            build_expanded_image_keys_from_input_ids(
                input_ids=input_id,
                image_token_id=image_token_id,
                num_image_tokens=num_image_tokens,
            )
            for input_id in input_ids_batch
        ]

    if _env_flag("DEBUG_EXPANDED_KEYS", "False"):
        num_image_tokens_dbg = infer_llava_num_image_tokens(model, processor)
        print(f"[DEBUG EXPANDED KEYS] {debug_prefix}")
        print("  input_ids shape:", tuple(input_ids_batch.shape))
        print("  image_token_id:", image_token_id)
        print("  inferred_num_image_tokens:", num_image_tokens_dbg)
        print("  expand:", expand)
        for i, input_id in enumerate(input_ids_batch):
            raw_count = int((input_id == image_token_id).sum().item())
            key_len = int(keys[i].numel())
            key_sum = int(keys[i].sum().item())
            print(
                f"  idx={i} raw_image_placeholders={raw_count} "
                f"key_len={key_len} key_sum={key_sum}"
            )

    return keys



def get_llava_pad_token_id(model, processor) -> int:
    """
    Match the pad_token_id used by LlavaForConditionalGenerationScal._merge_input_ids_with_image_features.
    """
    pad_token_id = getattr(model, "pad_token_id", None)

    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)

    if pad_token_id is None:
        tokenizer = getattr(processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)

    if pad_token_id is None:
        pad_token_id = -1

    return int(pad_token_id)


def build_original_to_expanded_position_map(
    input_ids: torch.Tensor,
    image_token_id: int,
    num_image_tokens: int,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Map original input_ids positions to expanded LLaVA sequence positions.

    Current AdaptVis/LLaVA-1.5 processor input_ids usually contains one <image>
    placeholder. The custom LlavaForConditionalGenerationScal expands this
    placeholder into many visual patch embeddings inside forward(), so
    outputs.logits may be indexed by the expanded sequence rather than the
    original input_ids sequence.

    This mirrors the position mapping in _merge_input_ids_with_image_features:
        new_token_positions = cumsum(mask * (num_image_tokens - 1) + 1) - 1

    Returns:
        LongTensor [batch, original_seq_len].
        Text-token positions map to their corresponding expanded positions.
        The <image> placeholder maps to the last image patch position; we do not
        use that placeholder position for answer scoring.
    """
    special_image_token_mask = input_ids == int(image_token_id)

    step = special_image_token_mask.long() * (int(num_image_tokens) - 1) + 1
    new_token_positions = torch.cumsum(step, dim=-1) - 1

    max_embed_dim = (
        int(special_image_token_mask.sum(dim=-1).max().item()) * (int(num_image_tokens) - 1)
    ) + input_ids.shape[1]

    # Same left-padding check as HF/LLaVA merge code:
    # if no sample has pad at the final position, assume left padding.
    last_is_pad = input_ids[:, -1] == int(pad_token_id)
    left_padding = not bool(last_is_pad.sum().item())

    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]

    if left_padding:
        new_token_positions = new_token_positions + nb_image_pad[:, None]

    return new_token_positions.long()

def _score_dict_to_prob_dict(
    score_dict: Dict[str, Dict[str, Any]],
    score_key: str = "score",
) -> Tuple[Dict[str, float], float, float]:
    candidates = list(score_dict.keys())

    if len(candidates) == 0:
        return {}, 0.0, 0.0

    score_tensor = torch.tensor(
        [float(score_dict[c][score_key]) for c in candidates],
        dtype=torch.float32,
    )

    probs = torch.softmax(score_tensor, dim=0)
    prob_dict = {
        c: float(p)
        for c, p in zip(candidates, probs.tolist())
    }

    sorted_scores = torch.sort(score_tensor, descending=True).values
    margin = (
        float((sorted_scores[0] - sorted_scores[1]).item())
        if len(candidates) >= 2
        else 0.0
    )

    confidence = float(probs.max().item())
    return prob_dict, confidence, margin


@torch.no_grad()
def score_candidates_closed_set(
    model,
    processor,
    image,
    prompt: str,
    candidates: List[str],
    device: str,
    method: Optional[str] = None,
    weight: float = 1.0,
    adjust_method: Optional[str] = "last_query",
    pos: Optional[torch.Tensor] = None,
    object_patch_mask: Optional[torch.Tensor] = None,
) -> Tuple[str, Dict[str, Dict[str, Any]], Dict[str, float], float, float]:
    """
    Closed-set relation scoring:
        S(candidate) = mean log p(candidate_tokens | prompt, image)

    For ScalingVis / AdaptVis, this function calls the Scal model forward with
    the same AdaptVis parameters used during generation.

    Important index detail:
        In this repository's custom LLaVA path, input_ids often contains only one
        <image> placeholder, while the forward pass expands that placeholder into
        many visual patch embeddings. In that case outputs.logits is indexed by
        the expanded sequence. Therefore candidate answer-token positions must be
        mapped from original input_ids positions to expanded positions before
        reading log-probabilities.
    """
    if len(candidates) == 0:
        raise ValueError("score_candidates_closed_set received empty candidates.")

    prompt = str(prompt).rstrip()
    full_texts = [prompt + " " + str(c).strip() for c in candidates]
    images = [image] * len(candidates)

    inputs = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Build keys from the actual closed-set batch, not from the shorter prompt-only
    # batch. This avoids length mismatch when the appended candidate adds tokens.
    image_token_id = int(
        os.getenv(
            "IMAGE_TOKEN_ID",
            str(getattr(getattr(model, "config", None), "image_token_index", 32001)),
        )
    )

    scoring_keys = build_adaptvis_keys_from_input_batch(
        input_ids_batch=inputs["input_ids"],
        model=model,
        processor=processor,
        image_token_id=image_token_id,
        debug_prefix="closed_set",
    )

    model_kwargs = {
        "return_dict": True,
    }

    # Only the Scal model accepts AdaptVis arguments.
    if "Scal" in str(type(model)):
        model_kwargs.update(
            {
                "keys": scoring_keys,
                "weight": float(weight),
                "adjust_method": adjust_method,
                "pos": pos,
                "object_patch_mask": object_patch_mask,
            }
        )

    outputs = model(**inputs, **model_kwargs)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tokenizer = processor.tokenizer

    # If outputs.logits has a different sequence length from input_ids, the model
    # returned expanded-sequence logits. Otherwise it returned input_ids-aligned logits.
    use_expanded_logits = outputs.logits.shape[1] != input_ids.shape[1]

    full_token_logprobs = F.log_softmax(outputs.logits.float(), dim=-1)

    if use_expanded_logits:
        num_image_tokens = infer_llava_num_image_tokens(model, processor)
        pad_token_id = get_llava_pad_token_id(model, processor)

        orig_to_expanded = build_original_to_expanded_position_map(
            input_ids=input_ids,
            image_token_id=image_token_id,
            num_image_tokens=num_image_tokens,
            pad_token_id=pad_token_id,
        )

        if os.getenv("DEBUG_CLOSED_SET_SHAPE", "False") == "True":
            print("[DEBUG CLOSED SET SHAPE]")
            print("  input_ids shape:", tuple(input_ids.shape))
            print("  logits shape:", tuple(outputs.logits.shape))
            print("  use_expanded_logits:", use_expanded_logits)
            print("  num_image_tokens:", num_image_tokens)
            print("  pad_token_id:", pad_token_id)
            print("  orig_to_expanded shape:", tuple(orig_to_expanded.shape))
    else:
        orig_to_expanded = None

        # Legacy/original-aligned case:
        # logits[:, t - 1] predicts input_ids[:, t].
        legacy_logits = outputs.logits[:, :-1, :]
        legacy_target_ids = input_ids[:, 1:]

        legacy_token_logprobs = F.log_softmax(legacy_logits.float(), dim=-1)
        legacy_token_logprobs = legacy_token_logprobs.gather(
            dim=-1,
            index=legacy_target_ids.unsqueeze(-1),
        ).squeeze(-1)

    score_dict: Dict[str, Dict[str, Any]] = {}

    for i, cand in enumerate(candidates):
        cand_ids = _candidate_answer_token_ids(tokenizer, cand)
        ans_len = len(cand_ids)

        if ans_len <= 0:
            raise ValueError(f"Candidate produced no tokens: {cand!r}")

        valid_positions = torch.where(attention_mask[i].bool())[0]

        if valid_positions.numel() == 0:
            raise ValueError("Empty input after tokenization in closed-set scoring.")

        last_input_pos = int(valid_positions[-1].item())
        answer_input_start = last_input_pos - ans_len + 1

        if answer_input_start < 0:
            raise ValueError(
                f"Invalid answer span for candidate={cand!r}: "
                f"answer_input_start={answer_input_start}, last_input_pos={last_input_pos}"
            )

        suffix_ids = [
            int(x)
            for x in input_ids[i, answer_input_start:last_input_pos + 1].detach().cpu().tolist()
        ]

        # The suffix should normally match cand_ids. If it does not, keep scoring
        # the final ans_len tokens but expose the mismatch for debugging.
        suffix_match = suffix_ids == cand_ids

        target_token_ids = input_ids[
            i,
            answer_input_start:last_input_pos + 1,
        ].long()

        if use_expanded_logits:
            answer_expanded_positions = orig_to_expanded[
                i,
                answer_input_start:last_input_pos + 1,
            ].long()

            # logits at expanded_pos - 1 predict the token at expanded_pos.
            predict_positions = answer_expanded_positions - 1

            if int(predict_positions.min().item()) < 0:
                raise ValueError(
                    f"Invalid expanded predict position for candidate={cand!r}: "
                    f"predict_positions={predict_positions.detach().cpu().tolist()}"
                )

            if int(predict_positions.max().item()) >= full_token_logprobs.shape[1]:
                raise ValueError(
                    f"Expanded answer span out of range for candidate={cand!r}: "
                    f"predict_positions={predict_positions.detach().cpu().tolist()}, "
                    f"logit_len={full_token_logprobs.shape[1]}"
                )

            cand_token_logprobs = full_token_logprobs[
                i,
                predict_positions,
                target_token_ids,
            ]

            answer_target_start = int(predict_positions[0].item())
            answer_target_end = int(predict_positions[-1].item()) + 1

            expanded_answer_positions_list = [
                int(x)
                for x in answer_expanded_positions.detach().cpu().tolist()
            ]
            predict_positions_list = [
                int(x)
                for x in predict_positions.detach().cpu().tolist()
            ]

        else:
            answer_target_start = answer_input_start - 1
            answer_target_end = last_input_pos

            if answer_target_start < 0 or answer_target_end > legacy_token_logprobs.shape[1]:
                raise ValueError(
                    f"Invalid answer span for candidate={cand!r}: "
                    f"answer_target_start={answer_target_start}, "
                    f"answer_target_end={answer_target_end}, "
                    f"logprob_len={legacy_token_logprobs.shape[1]}"
                )

            cand_token_logprobs = legacy_token_logprobs[
                i,
                answer_target_start:answer_target_end,
            ]

            expanded_answer_positions_list = None
            predict_positions_list = None

        cand_score_sum = float(cand_token_logprobs.sum().item())
        cand_score_mean = float(cand_token_logprobs.mean().item())

        score_dict[str(cand)] = {
            "score": cand_score_mean,
            "sum_logprob": cand_score_sum,
            "num_tokens": int(ans_len),
            "token_ids": cand_ids,
            "suffix_token_ids": suffix_ids,
            "suffix_match": bool(suffix_match),
            "use_expanded_logits": bool(use_expanded_logits),
            "answer_input_start_original": int(answer_input_start),
            "last_input_pos_original": int(last_input_pos),
            "answer_target_start": int(answer_target_start),
            "answer_target_end": int(answer_target_end),
            "expanded_answer_positions": expanded_answer_positions_list,
            "predict_positions": predict_positions_list,
            "token_logprobs": [
                float(x)
                for x in cand_token_logprobs.detach().cpu().tolist()
            ],
        }

        if os.getenv("DEBUG_CLOSED_SET_SPAN", "False") == "True":
            print(
                f"[DEBUG CLOSED SET SPAN] cand={cand!r} "
                f"use_expanded={use_expanded_logits} "
                f"orig_answer=({answer_input_start},{last_input_pos}) "
                f"expanded_answer={expanded_answer_positions_list} "
                f"predict_positions={predict_positions_list} "
                f"suffix_match={suffix_match} "
                f"score={cand_score_mean:.6f}"
            )

    prob_dict, confidence, margin = _score_dict_to_prob_dict(score_dict, score_key="score")
    pred = max(score_dict.keys(), key=lambda c: score_dict[c]["score"])

    return pred, score_dict, prob_dict, confidence, margin


@torch.no_grad()
def decide_closed_set_with_method(
    model,
    processor,
    image,
    prompt: str,
    candidates: List[str],
    device: str,
    dataset: Optional[str],
    method: str,
    weight: float,
    threshold: float,
    weight1: float,
    weight2: float,
    adjust_method: str,
    pos: Optional[torch.Tensor] = None,
    object_patch_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Run base / ScalingVis / AdaptVis using closed-set scoring instead of
    generation.

    For adapt_vis:
        pass 1: weight=1.0, compute closed-set confidence.
        pass 2: choose weight1 or weight2 by threshold, then rescore.
    """
    method = str(method)

    if method == "scaling_vis":
        pred, scores, probs, conf, margin = score_candidates_closed_set(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            candidates=candidates,
            device=device,
            method=method,
            weight=weight,
            adjust_method=adjust_method,
            pos=pos,
            object_patch_mask=object_patch_mask,
        )

        return {
            "prediction": pred,
            "scores": scores,
            "probs": probs,
            "confidence": conf,
            "confidence_raw": conf,
            "margin": margin,
            "selected_weight": float(weight),
            "base_prediction": None,
            "base_scores": None,
            "base_probs": None,
            "base_confidence": None,
            "base_margin": None,
            "probe_single_pass": False,
        }

    if method == "adapt_vis":
        probe_single_pass = (
            adjust_method in ["probe_bias", "probe_scale", "probe_add", "var_sink"]
            or os.getenv("PROBE_SINGLE_PASS", "False") == "True"
        )

        base_pred, base_scores, base_probs, base_conf, base_margin = score_candidates_closed_set(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            candidates=candidates,
            device=device,
            method=method,
            weight=1.0,
            adjust_method=adjust_method,
            pos=pos,
            object_patch_mask=object_patch_mask,
        )

        confidence_mode = os.getenv("CLOSED_SET_CONFIDENCE_MODE", "prob").strip().lower()
        confidence_value_raw = base_margin if confidence_mode == "margin" else base_conf
        confidence_value = float(np.round(confidence_value_raw, 2))

        if probe_single_pass:
            selected_weight = 1.0
        else:
            selected_weight = weight1 if confidence_value < threshold else weight2

        pred, scores, probs, conf, margin = score_candidates_closed_set(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            candidates=candidates,
            device=device,
            method=method,
            weight=selected_weight,
            adjust_method=adjust_method,
            pos=pos,
            object_patch_mask=object_patch_mask,
        )

        return {
            "prediction": pred,
            "scores": scores,
            "probs": probs,
            "confidence": confidence_value,
            "confidence_raw": float(confidence_value_raw),
            "margin": margin,
            "selected_weight": float(selected_weight),
            "base_prediction": base_pred,
            "base_scores": base_scores,
            "base_probs": base_probs,
            "base_confidence": float(base_conf),
            "base_margin": float(base_margin),
            "probe_single_pass": bool(probe_single_pass),
        }

    pred, scores, probs, conf, margin = score_candidates_closed_set(
        model=model,
        processor=processor,
        image=image,
        prompt=prompt,
        candidates=candidates,
        device=device,
        method=method,
        weight=1.0,
        adjust_method=adjust_method,
        pos=pos,
        object_patch_mask=object_patch_mask,
    )

    return {
        "prediction": pred,
        "scores": scores,
        "probs": probs,
        "confidence": conf,
        "confidence_raw": conf,
        "margin": margin,
        "selected_weight": None,
        "base_prediction": None,
        "base_scores": None,
        "base_probs": None,
        "base_confidence": None,
        "base_margin": None,
        "probe_single_pass": False,
    }



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

                    image_for_model = apply_image_control_from_env(_, sample_id=sample_id)

                    single_input = self.processor(
                        text=prompt,
                        images=image_for_model,
                        padding="max_length",
                        return_tensors="pt",
                        max_length=77,
                    ).to(self.device)

                    keys = build_adaptvis_keys_from_input_batch(
                        input_ids_batch=single_input["input_ids"],
                        model=self.model,
                        processor=self.processor,
                        image_token_id=None,
                        debug_prefix="generation_get_out_scores",
                    )

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
                            pil_image=image_for_model,
                            question=prompt,
                            device=self.device,
                            clip_threshold=clip_obj_threshold,
                            invert=clip_obj_invert,
                            dilate=clip_obj_dilate,
                        )

                        if object_patch_mask is not None:
                            object_patch_mask = object_patch_mask.to(self.device)

                    selected_weight = None
                    probe_single_pass = False
                    relation_probe = None
                    closed_set_info = None
                    output = None

                    if use_closed_set_scoring_from_env():
                        candidates = extract_closed_set_candidates(
                            prompt=prompt,
                            dataset=dataset,
                            option=option,
                        )

                        closed_set_info = decide_closed_set_with_method(
                            model=self.model,
                            processor=self.processor,
                            image=image_for_model,
                            prompt=prompt,
                            candidates=candidates,
                            device=self.device,
                            dataset=dataset,
                            method=method,
                            weight=weight,
                            threshold=threshold,
                            weight1=weight1,
                            weight2=weight2,
                            adjust_method=adjust_method_env,
                            pos=query_pos,
                            object_patch_mask=object_patch_mask,
                        )

                        gen = closed_set_info["prediction"]
                        selected_weight = closed_set_info["selected_weight"]
                        uncertainty = float(closed_set_info["confidence"])
                        probe_single_pass = bool(closed_set_info["probe_single_pass"])

                    else:
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

                            # Explicit probe modes should run single-pass.
                            # PROBE_RUN_TAG only controls output filename and must NOT disable AdaptVis.
                            # Use PROBE_SINGLE_PASS=True if you explicitly want single-pass with a tag.
                            probe_single_pass = (
                                adjust_method_env in ["probe_bias", "probe_scale", "probe_add", "var_sink"]
                                or os.getenv("PROBE_SINGLE_PASS", "False") == "True"
                            )

                            if probe_single_pass:
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

                        "decision_mode": "closed_set" if use_closed_set_scoring_from_env() else "generation",
                        "closed_set_candidates": (
                            list(closed_set_info["scores"].keys())
                            if closed_set_info is not None else None
                        ),
                        "closed_set_scores": (
                            closed_set_info["scores"]
                            if closed_set_info is not None else None
                        ),
                        "closed_set_probs": (
                            closed_set_info["probs"]
                            if closed_set_info is not None else None
                        ),
                        "closed_set_confidence": (
                            float(closed_set_info["confidence"])
                            if closed_set_info is not None else None
                        ),
                        "closed_set_confidence_raw": (
                            float(closed_set_info["confidence_raw"])
                            if closed_set_info is not None else None
                        ),
                        "closed_set_margin": (
                            float(closed_set_info["margin"])
                            if closed_set_info is not None else None
                        ),
                        "closed_set_base_prediction": (
                            closed_set_info["base_prediction"]
                            if closed_set_info is not None else None
                        ),
                        "closed_set_base_scores": (
                            closed_set_info["base_scores"]
                            if closed_set_info is not None else None
                        ),
                        "closed_set_base_probs": (
                            closed_set_info["base_probs"]
                            if closed_set_info is not None else None
                        ),
                        "closed_set_base_confidence": (
                            float(closed_set_info["base_confidence"])
                            if closed_set_info is not None and closed_set_info["base_confidence"] is not None else None
                        ),
                        "closed_set_base_margin": (
                            float(closed_set_info["base_margin"])
                            if closed_set_info is not None and closed_set_info["base_margin"] is not None else None
                        ),
                        "closed_set_confidence_mode": os.getenv("CLOSED_SET_CONFIDENCE_MODE", "prob"),

                        "adjust_method": os.getenv("ADJUST_METHOD", "last_query"),

                        # AdaptVis layer control metadata.
                        "adaptvis_exclude_layers": os.getenv("ADAPTVIS_EXCLUDE_LAYERS", ""),
                        "adaptvis_include_layers": os.getenv("ADAPTVIS_INCLUDE_LAYERS", ""),
                        "adaptvis_layer_debug": os.getenv("ADAPTVIS_LAYER_DEBUG", ""),

                        # Single-pass metadata.
                        "probe_single_pass": bool(probe_single_pass),
                        "probe_single_pass_env": os.getenv("PROBE_SINGLE_PASS", ""),

                        "probe_layer": os.getenv("PROBE_LAYER", ""),
                        "probe_head": os.getenv("PROBE_HEAD", ""),
                        "probe_block_ids": os.getenv("PROBE_BLOCK_IDS", ""),
                        "probe_beta": os.getenv("PROBE_BETA", ""),
                        "probe_scale": os.getenv("PROBE_SCALE", ""),

                        # Old probability-space probe_add fields.
                        # Kept for backward compatibility with earlier results.
                        "probe_add_mode": os.getenv("PROBE_ADD_MODE", ""),
                        "probe_add_mass": os.getenv("PROBE_ADD_MASS", ""),
                        "probe_add_value": os.getenv("PROBE_ADD_VALUE", ""),
                        "probe_add_renorm": os.getenv("PROBE_ADD_RENORM", ""),

                        # New logit-space probe_add fields.
                        "probe_add_beta": os.getenv("PROBE_ADD_BETA", ""),
                        "probe_add_alpha": os.getenv("PROBE_ADD_ALPHA", ""),
                        "probe_add_beta_mode": os.getenv("PROBE_ADD_BETA_MODE", ""),
                        "probe_add_beta_clamp": os.getenv("PROBE_ADD_BETA_CLAMP", ""),
                        "probe_add_std_eps": os.getenv("PROBE_ADD_STD_EPS", ""),

                        "probe_run_tag": os.getenv("PROBE_RUN_TAG", ""),

                        "image_control": os.getenv("IMAGE_CONTROL", "none"),
                        "image_control_size": os.getenv("IMAGE_CONTROL_SIZE", ""),
                        "image_control_grid": os.getenv("IMAGE_CONTROL_GRID", ""),
                        "image_control_seed": os.getenv("IMAGE_CONTROL_SEED", ""),

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
                    "probe_single_pass_env": os.getenv("PROBE_SINGLE_PASS", ""),
                    "decision_mode": "closed_set" if use_closed_set_scoring_from_env() else "generation",
                    "closed_set_scoring": bool(use_closed_set_scoring_from_env()),
                    "closed_set_confidence_mode": os.getenv("CLOSED_SET_CONFIDENCE_MODE", "prob"),

                    # AdaptVis layer control metadata.
                    "adaptvis_exclude_layers": os.getenv("ADAPTVIS_EXCLUDE_LAYERS", ""),
                    "adaptvis_include_layers": os.getenv("ADAPTVIS_INCLUDE_LAYERS", ""),
                    "adaptvis_layer_debug": os.getenv("ADAPTVIS_LAYER_DEBUG", ""),

                    "probe_scale": os.getenv("PROBE_SCALE", ""),

                    # Old probability-space probe_add fields.
                    "probe_add_mode": os.getenv("PROBE_ADD_MODE", ""),
                    "probe_add_mass": os.getenv("PROBE_ADD_MASS", ""),
                    "probe_add_value": os.getenv("PROBE_ADD_VALUE", ""),
                    "probe_add_renorm": os.getenv("PROBE_ADD_RENORM", ""),

                    # New logit-space probe_add fields.
                    "probe_add_beta": os.getenv("PROBE_ADD_BETA", ""),
                    "probe_add_alpha": os.getenv("PROBE_ADD_ALPHA", ""),
                    "probe_add_beta_mode": os.getenv("PROBE_ADD_BETA_MODE", ""),
                    "probe_add_beta_clamp": os.getenv("PROBE_ADD_BETA_CLAMP", ""),
                    "probe_add_std_eps": os.getenv("PROBE_ADD_STD_EPS", ""),

                    "image_control": os.getenv("IMAGE_CONTROL", "none"),
                    "image_control_size": os.getenv("IMAGE_CONTROL_SIZE", ""),
                    "image_control_grid": os.getenv("IMAGE_CONTROL_GRID", ""),
                    "image_control_seed": os.getenv("IMAGE_CONTROL_SEED", ""),
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
                        image_for_model = apply_image_control_from_env(
                            list(i_option)[idx],
                            sample_id=index_of_total,
                        )

                        single_input = self.processor(
                            text=text,
                            images=image_for_model,
                            padding="max_length",
                            return_tensors="pt",
                            max_length=77,
                        ).to(self.device)

                        keys = build_adaptvis_keys_from_input_batch(
                            input_ids_batch=single_input["input_ids"],
                            model=self.model,
                            processor=self.processor,
                            image_token_id=None,
                            debug_prefix="generation_get_scores",
                        )

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
                            "image_control": os.getenv("IMAGE_CONTROL", "none"),
                            "image_control_seed": os.getenv("IMAGE_CONTROL_SEED", ""),
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
                    "image_control": os.getenv("IMAGE_CONTROL", "none"),
                    "image_control_seed": os.getenv("IMAGE_CONTROL_SEED", ""),
                },
                fout,
                ensure_ascii=False,
                indent=4,
            )

        return all_scores
