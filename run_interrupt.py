import os
import csv
import json
import math
import inspect
import argparse
from typing import Dict, List, Optional, Set

import torch
from tqdm import tqdm

from misc import seed_all
from model_zoo import get_model
from dataset_zoo import get_dataset
from multiq_utils import build_object_pool, build_questions, parse_prediction


VALID_PERTURB_MODES = {"uniform", "random", "reverse"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--model-name", default="llava1.5", type=str)
    p.add_argument("--dataset", default="Controlled_Images_A", type=str)
    p.add_argument("--option", default="four", type=str)
    p.add_argument("--download", action="store_true")
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--sample-index", default=0, type=int)
    p.add_argument("--limit", default=-1, type=int)
    p.add_argument("--method", default="scaling_vis", type=str,
                   choices=["scaling_vis"],
                   help="Use the Scal LLaVA path so custom kwargs reach decoder attention.")
    p.add_argument("--weight", default=1.0, type=float,
                   help="Keep at 1.0 for structure-only perturbation experiments.")
    p.add_argument("--perturb-modes", default="uniform,random,reverse", type=str,
                   help="Comma-separated list from {uniform,random,reverse}.")
    p.add_argument("--target-layers", default="all", type=str,
                   help='"all" or comma-separated decoder layer indices, e.g. "16" or "8,16,24".')
    p.add_argument("--max-new-tokens", default=20, type=int)
    p.add_argument("--trace-topk", default=10, type=int)
    p.add_argument("--out-dir", default="./output_multiq_perturb", type=str)
    return p.parse_args()


def parse_mode_list(text: str) -> List[str]:
    modes = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not modes:
        raise ValueError("No perturb modes provided.")
    bad = [m for m in modes if m not in VALID_PERTURB_MODES]
    if bad:
        raise ValueError(f"Unsupported perturb modes: {bad}")
    return modes


def parse_target_layers(text: str) -> Optional[Set[int]]:
    if text.strip().lower() == "all":
        return None
    layers: Set[int] = set()
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        layers.add(int(x))
    if not layers:
        raise ValueError("target-layers must be 'all' or a non-empty comma-separated list of ints.")
    return layers


class _PerturbConfig:
    target_layers: Optional[Set[int]] = None


def install_attention_perturbation(target_layers: Optional[Set[int]]) -> None:
    import model_zoo.llama.modeling_llama_add_attn as mla

    _PerturbConfig.target_layers = target_layers
    mla.SAVE_ATTN = False
    mla.SAVE_LAYER_ONLY = True

    if getattr(mla.LLaMAAttention, "_perturb_patch_installed", False):
        return

    apply_rotary_pos_emb = mla.apply_rotary_pos_emb
    nn = mla.nn
    torch_mod = mla.torch
    math_mod = mla.math

    def _apply_image_block_perturbation(attn_probs: torch.Tensor, keys: Optional[torch.Tensor], mode: Optional[str]):
        if mode not in VALID_PERTURB_MODES or keys is None:
            return attn_probs

        if attn_probs.size(-2) != attn_probs.size(-1):
            # keep behavior simple and stable: perturb only prefill step where merged image block is explicit
            return attn_probs

        true_idx = torch_mod.where(keys)[1] if keys.ndim == 2 else torch_mod.where(keys)[0]
        if true_idx.numel() == 0:
            return attn_probs

        start_idx = int(true_idx[0].item())
        end_idx = int(true_idx[-1].item())
        img = attn_probs[..., start_idx:end_idx + 1]
        if img.numel() == 0:
            return attn_probs

        mass = img.sum(dim=-1, keepdim=True)
        M = img.shape[-1]
        eps = torch_mod.finfo(img.dtype).eps if img.is_floating_point() else 1e-12

        if mode == "uniform":
            new_img = torch_mod.full_like(img, 1.0 / float(M))
            new_img = new_img * mass

        elif mode == "random":
            rand = torch_mod.rand_like(img)
            rand = rand / rand.sum(dim=-1, keepdim=True).clamp_min(eps)
            new_img = rand * mass

        elif mode == "reverse":
            rev = img.max(dim=-1, keepdim=True).values - img
            rev_sum = rev.sum(dim=-1, keepdim=True)
            uniform = torch_mod.full_like(img, 1.0 / float(M))
            rev_norm = rev / rev_sum.clamp_min(eps)
            use_uniform = rev_sum <= eps
            rev_norm = torch_mod.where(use_uniform.expand_as(rev_norm), uniform, rev_norm)
            new_img = rev_norm * mass

        else:
            return attn_probs

        out = attn_probs.clone()
        out[..., start_idx:end_idx + 1] = new_img
        return out

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value=None,
        attention_mask=None,
        position_ids=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_head_hidden_states: bool = False,
        keys: Optional[torch.Tensor] = None,
        weight: Optional[float] = None,
        pos: Optional[torch.Tensor] = None,
        idx: Optional[int] = None,
        caption_length: Optional[list] = None,
        adjust_method: Optional[str] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, offset=offset)

        if past_key_value is not None:
            key_states = torch_mod.cat([past_key_value[0], key_states], dim=2)
            value_states = torch_mod.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states)

        attn_weights = torch_mod.matmul(query_states, key_states.transpose(2, 3)) / math_mod.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch_mod.max(
                attn_weights,
                torch_mod.tensor(torch_mod.finfo(attn_weights.dtype).min, device=attn_weights.device),
            )

        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch_mod.float32).to(query_states.dtype)

        should_perturb = (
            adjust_method in VALID_PERTURB_MODES
            and idx is not None
            and (_PerturbConfig.target_layers is None or idx in _PerturbConfig.target_layers)
        )
        if should_perturb:
            attn_probs = _apply_image_block_perturbation(attn_probs, keys=keys, mode=adjust_method)

        attn_probs = self.att_out(attn_probs)
        value_states = self.value_out(value_states)
        attn_output = torch_mod.matmul(attn_probs, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.head_out(attn_output)
        attn_output = self.o_proj(attn_output)

        return attn_output, (attn_probs if output_attentions else None), past_key_value

    mla.LLaMAAttention.forward = patched_forward
    mla.LLaMAAttention._perturb_patch_installed = True


def get_change_greedy_fn(model_name: str):
    if model_name == "llava1.5":
        from model_zoo.llava15 import change_greedy_to_add_weight
        return change_greedy_to_add_weight
    if model_name == "llava1.6":
        from model_zoo.llava16 import change_greedy_to_add_weight
        return change_greedy_to_add_weight
    raise ValueError(f"Unsupported model for this script: {model_name}")


@torch.no_grad()
def build_generation_trace(processor, generation_output, prompt_len: int, topk: int = 10):
    tokenizer = processor.tokenizer
    sequences = generation_output.sequences
    gen_ids = sequences[0][prompt_len:]

    step_scores = generation_output.scores or []
    step_logits = getattr(generation_output, "logits", None)

    trace = []
    for step_idx, token_id in enumerate(gen_ids.tolist()):
        step_score = step_scores[step_idx][0].detach().float().cpu()
        step_prob = torch.softmax(step_score, dim=-1)
        raw_logit = None
        if step_logits is not None:
            raw_logit = step_logits[step_idx][0].detach().float().cpu()

        chosen = {
            "step": step_idx,
            "token_id": int(token_id),
            "token_text": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "score": float(step_score[token_id].item()),
            "probability": float(step_prob[token_id].item()),
            "raw_logit": float(raw_logit[token_id].item()) if raw_logit is not None else None,
            "top10": [],
        }

        k = min(topk, step_prob.numel())
        topk_prob, topk_ids = torch.topk(step_prob, k=k)
        topk_score = step_score[topk_ids]
        topk_raw = raw_logit[topk_ids] if raw_logit is not None else None

        for rank in range(k):
            cand_id = int(topk_ids[rank].item())
            chosen["top10"].append({
                "rank": rank + 1,
                "token_id": cand_id,
                "token_text": tokenizer.decode([cand_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "score": float(topk_score[rank].item()),
                "probability": float(topk_prob[rank].item()),
                "raw_logit": float(topk_raw[rank].item()) if topk_raw is not None else None,
            })

        trace.append(chosen)
    return trace


@torch.no_grad()
def run_one_generation(wrapper, model_name: str, image, prompt: str, perturb_mode: str, max_new_tokens: int):
    processor = wrapper.processor
    model = wrapper.model

    single_input = processor(images=image, text=prompt, padding=True, return_tensors="pt")
    single_input = {
        k: (v.to(wrapper.device) if torch.is_tensor(v) else v)
        for k, v in single_input.items()
        if v is not None
    }

    image_id = (single_input["input_ids"] == model.config.image_token_index)
    prompt_len = int(single_input["input_ids"].shape[1])

    change_greedy = get_change_greedy_fn(model_name)
    change_greedy()

    gen_kwargs = dict(
        input_ids=single_input["input_ids"],
        pixel_values=single_input["pixel_values"],
        attention_mask=single_input.get("attention_mask", None),
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        use_cache=True,
        output_attentions=False,
        keys=image_id,
        weight=1.0,
        adjust_method=perturb_mode,
    )
    if "output_logits" in inspect.signature(model.generate).parameters:
        gen_kwargs["output_logits"] = True

    output = model.generate(**gen_kwargs)
    gen_ids = output.sequences[0][prompt_len:]
    gen_text = processor.decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return output, gen_text, prompt_len


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    seed_all(args.seed)

    perturb_modes = parse_mode_list(args.perturb_modes)
    target_layers = parse_target_layers(args.target_layers)
    install_attention_perturbation(target_layers)

    wrapper, image_preprocess = get_model(args.model_name, args.device, args.method)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)
    prompt_records, sampled_indices = wrapper.load_prompt_records_with_sampling(args.dataset, args.option)
    object_pool = build_object_pool(prompt_records)

    if sampled_indices is not None:
        sub_dataset = torch.utils.data.Subset(dataset, sampled_indices)
    else:
        sub_dataset = dataset

    start = args.sample_index
    end = len(prompt_records) if args.limit < 0 else min(len(prompt_records), start + args.limit)

    base_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(base_dir, exist_ok=True)
    summary_csv = os.path.join(base_dir, "summary.csv")
    summary_rows = []

    for local_idx in tqdm(range(start, end), desc="Samples"):
        rec = prompt_records[local_idx]
        item = sub_dataset[local_idx]
        image = item["image_options"][0]

        questions, meta = build_questions(
            base_prompt=rec["question"],
            base_answer=rec["answer"][0] if isinstance(rec["answer"], list) else rec["answer"],
            sample_idx=local_idx,
            object_pool=object_pool,
        )

        image_name = item.get("image_name", f"sample_{local_idx:04d}")
        image_path = item.get("image_path", "")
        image_stem = os.path.splitext(image_name)[0]
        sample_dir = os.path.join(base_dir, image_stem)
        os.makedirs(sample_dir, exist_ok=True)

        meta_path = os.path.join(sample_dir, "meta.json")
        if not os.path.exists(meta_path):
            write_json(meta_path, {
                "local_index": local_idx,
                "image_name": image_name,
                "image_path": image_path,
                **meta,
            })

        for perturb_mode in perturb_modes:
            for q in questions:
                output, gen_text, prompt_len = run_one_generation(
                    wrapper=wrapper,
                    model_name=args.model_name,
                    image=image,
                    prompt=q["prompt"],
                    perturb_mode=perturb_mode,
                    max_new_tokens=args.max_new_tokens,
                )

                trace = build_generation_trace(wrapper.processor, output, prompt_len, topk=args.trace_topk)
                pred = parse_prediction(gen_text, q["mode"])
                correct = (pred == q["gold"])

                trace_rel_path = os.path.join(image_stem, f"{q['qid']}_{perturb_mode}_trace.json")
                trace_path = os.path.join(base_dir, trace_rel_path)
                write_json(trace_path, {
                    "image_name": image_name,
                    "image_path": image_path,
                    "local_index": local_idx,
                    "qid": q["qid"],
                    "mode": q["mode"],
                    "gold": q["gold"],
                    "prompt": q["prompt"],
                    "perturb_mode": perturb_mode,
                    "target_layers": "all" if target_layers is None else sorted(target_layers),
                    "generated_text": gen_text,
                    "pred": pred,
                    "correct": correct,
                    "token_trace": trace,
                })

                first = trace[0] if trace else {}
                last = trace[-1] if trace else {}
                summary_rows.append({
                    "image_name": image_name,
                    "image_path": image_path,
                    "local_index": local_idx,
                    "qid": q["qid"],
                    "question_mode": q["mode"],
                    "gold": q["gold"],
                    "pred": pred,
                    "correct": correct,
                    "perturb_mode": perturb_mode,
                    "target_layers": "all" if target_layers is None else ",".join(str(x) for x in sorted(target_layers)),
                    "generated_text": gen_text,
                    "num_generated_tokens": len(trace),
                    "first_token": first.get("token_text", ""),
                    "first_raw_logit": first.get("raw_logit", None),
                    "first_score": first.get("score", None),
                    "first_probability": first.get("probability", None),
                    "final_token": last.get("token_text", ""),
                    "final_raw_logit": last.get("raw_logit", None),
                    "final_score": last.get("score", None),
                    "final_probability": last.get("probability", None),
                    "trace_json": trace_rel_path,
                })

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_name", "image_path", "local_index", "qid", "question_mode", "gold", "pred", "correct",
                    "perturb_mode", "target_layers", "generated_text", "num_generated_tokens",
                    "first_token", "first_raw_logit", "first_score", "first_probability",
                    "final_token", "final_raw_logit", "final_score", "final_probability", "trace_json",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
