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
    Use the same correctness rule as the original AdaptVis evaluation.

    This is saved into each result row so block-level ablations can later
    compare all-patch baseline vs except-block runs per sample.
    """
    golden = str(golden)
    gen = str(gen)

    ok = (golden in gen) or (golden.lower() in gen.lower())

    # Original special case: avoid counting "front" as "on".
    if golden.lower() == "on" and "front" in gen.strip().lower():
        ok = False

    return bool(ok)


def extract_objects_from_question(question: str) -> Optional[Tuple[str, str]]:
    """
    Parse:
        USER: Where is the obj1 in relation to the obj2?
        Answer with left, right, on or under.
    """
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
    """
    Controlled_Images_A normally returns PIL images because image_preprocess is None.
    This function keeps it robust.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        x = image.detach().cpu()

        if x.dim() == 4:
            x = x[0]

        if x.dim() != 3:
            raise ValueError(f"Cannot convert tensor with shape {tuple(x.shape)} to PIL.")

        # CHW -> HWC
        if x.shape[0] in [1, 3]:
            x = x.permute(1, 2, 0)

        x = x.float()

        # Best-effort normalization back to [0, 1]
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
    """
    mask_2d: [H, W], bool
    """
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
    """
    Binary object mask.

    Steps:
        1. parse obj1 / obj2 from question
        2. compute CLIP patch-text similarity for obj1 and obj2
        3. invert similarity because current observation shows -sim localizes objects better
        4. normalize each heatmap to [0, 1]
        5. object_score = max(score1, score2)
        6. object_mask = object_score >= clip_threshold
        7. optional dilation

    Return:
        object_patch_mask: [num_patches], bool tensor
    """
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

    # [1, 1 + N, hidden_dim] -> [1, N, hidden_dim]
    patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]

    # Keep consistent with the visualization script.
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

    # Use max, not average.
    object_score = torch.maximum(score1, score2)

    object_mask_2d = object_score >= clip_threshold
    object_mask_2d = dilate_binary_mask(object_mask_2d, dilate=dilate)

    object_patch_mask = object_mask_2d.reshape(-1)

    # Fallback: if threshold is too high and no patch is selected,
    # select top ratio according to object_score.
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
    """
    Manual patch/block mask for brute-force search.

    Assumes LLaVA-1.5 image tokens are 24x24 = 576.

    Environment variables:
        PATCH_MASK_MODE:
            ""             -> disabled
            "all"          -> select all patches
            "block"        -> select only one block
            "except_block" -> select all patches except one block
            "row"          -> select one row
            "col"          -> select one column

        PATCH_GRID_SIZE=24
        PATCH_BLOCK_GRID=4
        PATCH_BLOCK_ID=0
        PATCH_ROW_ID=0
        PATCH_COL_ID=0
    """
    mode = os.getenv("PATCH_MASK_MODE", "").strip()

    if mode == "":
        return None

    grid_size = int(os.getenv("PATCH_GRID_SIZE", "24"))
    block_grid = int(os.getenv("PATCH_BLOCK_GRID", "4"))
    block_id = int(os.getenv("PATCH_BLOCK_ID", "0"))

    num_patches = grid_size * grid_size
    mask_2d = torch.zeros((grid_size, grid_size), dtype=torch.bool)

    if mode == "all":
        mask_2d[:, :] = True

    elif mode in ["block", "except_block"]:
        if grid_size % block_grid != 0:
            raise ValueError(
                f"PATCH_GRID_SIZE={grid_size} must be divisible by "
                f"PATCH_BLOCK_GRID={block_grid}"
            )

        block_h = grid_size // block_grid
        block_w = grid_size // block_grid

        br = block_id // block_grid
        bc = block_id % block_grid

        if not (0 <= br < block_grid and 0 <= bc < block_grid):
            raise ValueError(
                f"Invalid PATCH_BLOCK_ID={block_id} for "
                f"PATCH_BLOCK_GRID={block_grid}"
            )

        r0 = br * block_h
        r1 = (br + 1) * block_h
        c0 = bc * block_w
        c1 = (bc + 1) * block_w

        if mode == "block":
            mask_2d[r0:r1, c0:c1] = True

        elif mode == "except_block":
            mask_2d[:, :] = True
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
        print(
            f"[PATCH_MASK] mode={mode}, grid={grid_size}, "
            f"block_grid={block_grid}, block_id={block_id}, "
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

        # Keep custom kwargs robustly across generation.
        # Some prepare_inputs_for_generation implementations may drop them.
        for custom_key in ["keys", "object_patch_mask"]:
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

        # Keep custom kwargs after HF updates model_kwargs.
        # This avoids losing object_patch_mask / keys after the first decode step.
        for custom_key in ["keys", "object_patch_mask"]:
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

        # External CLIP for object binary mask.
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
        acc = 0
        correct_id = []

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

        results = []

        for batch in tqdm(joint_loader):
            batch_scores = []

            os.environ["SAVE_ATTN_PATH"] = f"{save_attn_dir}/{index_of_total}/"
            os.makedirs(os.environ["SAVE_ATTN_PATH"], exist_ok=True)

            for i_option in batch["image_options"]:
                im_scores = []

                for _ in i_option:
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

                    # Manual brute-force mask has higher priority than CLIP mask.
                    # This lets you test which image patch/block is useful without using CLIP.
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

                    if method == "scaling_vis":
                        change_greedy_to_add_weight()

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

                        # First pass: weight=1.0 only estimates confidence.
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

                        # Second pass: original AdaptVis confidence rule.
                        if uncertainty < threshold:
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
                            correct_id.append(index_of_total)
                            answers = [1, 0, 0, 0]
                        else:
                            answers = [0, 0, 1, 0]

                    elif num_options == 2:
                        if is_correct:
                            acc += 1
                            correct_id.append(index_of_total)
                            answers = [1, 0]
                        else:
                            answers = [0, 1]

                    else:
                        raise ValueError(f"Unexpected number of caption options: {num_options}")

                    patch_selected = None
                    if object_patch_mask is not None:
                        patch_selected = int(object_patch_mask.detach().bool().sum().item())

                    result = {
                        "sample_id": int(index_of_total),
                        "Prompt": prompt,
                        "Generation": gen,
                        "Golden": golden,
                        "Correct": bool(is_correct),
                        "Uncertainty": float(uncertainty) if "uncertainty" in locals() else None,

                        # Mask / block metadata for per-image flip analysis.
                        "adjust_method": os.getenv("ADJUST_METHOD", "last_query"),
                        "patch_mask_mode": os.getenv("PATCH_MASK_MODE", ""),
                        "patch_grid_size": os.getenv("PATCH_GRID_SIZE", ""),
                        "patch_block_grid": os.getenv("PATCH_BLOCK_GRID", ""),
                        "patch_block_id": os.getenv("PATCH_BLOCK_ID", ""),
                        "patch_row_id": os.getenv("PATCH_ROW_ID", ""),
                        "patch_col_id": os.getenv("PATCH_COL_ID", ""),
                        "clip_obj_mask": os.getenv("CLIP_OBJ_MASK", "False"),
                        "clip_obj_threshold": os.getenv("CLIP_OBJ_THRESHOLD", ""),
                        "selected_patch_count": patch_selected,
                    }
                    results.append(result)

                    im_scores.append(np.expand_dims(np.array(answers), -1))
                    index_of_total += 1

                batch_scores.append(np.concatenate(im_scores, axis=-1))

            scores.append(batch_scores)

            output_file_path = f"./output/results1.5_{dataset}_{method}_{weight}_{option}option_{TEST}.json"
            print("Saving results to", output_file_path)

            with open(output_file_path, "w", encoding="utf-8") as fout:
                json.dump(results, fout, ensure_ascii=False, indent=4)

            print(acc, index_of_total, acc / index_of_total)

        print(acc / index_of_total)

        output_score_file = output_file_path.replace(".json", "scores.json")

        with open(output_score_file, "w", encoding="utf-8") as fout:
            json.dump(
                {"acc": acc / index_of_total, "correct_id": correct_id},
                fout,
                ensure_ascii=False,
                indent=4,
            )

        all_scores = np.concatenate(scores, axis=0)

        if dataset in ["Controlled_Images_B", "Controlled_Images_A"]:
            return all_scores, []
        else:
            return acc / index_of_total, correct_id

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
