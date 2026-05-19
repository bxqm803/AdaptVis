import argparse
import os
import json
import re
import pandas as pd
import pdb
from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores
import numpy as np
import random
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

IMAGE_TOKEN_ID = 32001


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num_workers", default=16, type=int)

    parser.add_argument(
        "--model-name",
        default="llava1.5",
        type=str,
        choices=["llava1.5", "llava1.6"],
    )

    parser.add_argument(
        "--dataset",
        default="Controlled_Images_A",
        type=str,
        choices=[
            "Controlled_Images_A",
            "Controlled_Images_B",
            "COCO_QA_one_obj",
            "COCO_QA_two_obj",
            "VG_QA_one_obj",
            "VG_QA_two_obj",
            "VSR",
        ],
    )

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--method", type=str)

    parser.add_argument("--dola-decoding", action="store_true")
    parser.add_argument("--info-layer", type=int)

    parser.add_argument(
        "--download",
        action="store_true",
        help="Whether to download the dataset if it doesn't exist. (Default: False)",
    )

    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="Whether to save the scores for the retrieval to analyze later.",
    )

    parser.add_argument("--output-dir", default="./outputs", type=str)

    parser.add_argument("--weight", default=1.0, type=float)
    parser.add_argument("--weight1", default=1.0, type=float)
    parser.add_argument("--weight2", default=1.0, type=float)
    parser.add_argument("--threshold", default=1.0, type=float)

    parser.add_argument(
        "--option",
        default="four",
        type=str,
        choices=["two", "four", "six"],
    )

    # ============================================================
    # Decision mode
    # ============================================================
    parser.add_argument(
        "--decision-mode",
        default=os.getenv("DECISION_MODE", "generation"),
        type=str,
        choices=["generation", "closed_set"],
        help="Use normal generation decoding or closed-set continuation scoring.",
    )

    parser.add_argument(
        "--closed-set-scoring",
        action="store_true",
        help="Shortcut for --decision-mode closed_set.",
    )

    # ============================================================
    # Relation Logit Trajectory
    # ============================================================
    parser.add_argument(
        "--relation-logit-trajectory",
        action="store_true",
        help=(
            "Run layer-wise relation logit trajectory analysis instead of "
            "normal evaluation."
        ),
    )

    parser.add_argument(
        "--trajectory-out-dir",
        default="./output/relation_logit_trajectory",
        type=str,
        help="Output directory for relation logit trajectory results.",
    )

    parser.add_argument(
        "--trajectory-relations",
        default="left,right,on,under",
        type=str,
        help="Comma-separated relation candidates for trajectory analysis.",
    )

    parser.add_argument(
        "--trajectory-max-samples",
        default=None,
        type=int,
        help="Optional maximum number of samples for trajectory analysis.",
    )

    parser.add_argument(
        "--trajectory-sample-ids-file",
        default="",
        type=str,
        help="Optional file containing one original sample id per line.",
    )

    parser.add_argument(
        "--trajectory-weight",
        default=None,
        type=float,
        help=(
            "Optional fixed weight for trajectory forward/generation. "
            "If unset, use --weight."
        ),
    )

    parser.add_argument(
        "--trajectory-adjust-method",
        default=None,
        type=str,
        help=(
            "Optional adjust method for trajectory forward. If unset, use "
            "ADJUST_METHOD env or last_query."
        ),
    )

    parser.add_argument(
        "--trajectory-query-pos",
        default=None,
        type=int,
        help="Query position used when adjust_method=text_offset.",
    )

    parser.add_argument(
        "--trajectory-save-jsonl",
        action="store_true",
        help="Save detailed per-sample per-layer trajectory jsonl.",
    )

    parser.add_argument(
        "--trajectory-no-final-norm",
        action="store_true",
        help=(
            "Disable applying final norm to intermediate hidden states before "
            "lm_head. By default final norm is applied logit-lens style."
        ),
    )

    parser.add_argument(
        "--trajectory-compare-generation",
        action="store_true",
        help=(
            "Also run normal generation for each sample and parse the generated "
            "relation word. This lets you compare closed-set logit-lens result "
            "with generation result."
        ),
    )

    parser.add_argument(
        "--trajectory-generation-max-new-tokens",
        default=100,
        type=int,
        help="max_new_tokens for trajectory generation comparison.",
    )

    parser.add_argument(
        "--trajectory-generation-relation-pick",
        default="last",
        choices=["first", "last"],
        type=str,
        help=(
            "When parsing generated text, choose the first or last occurrence "
            "of left/right/on/under. Default is last."
        ),
    )

    return parser.parse_args()


def setup_decision_mode_env(args):
    """
    Synchronize command-line decision mode with environment variables.

    The modified llava15.py checks:
        DECISION_MODE=closed_set
    or:
        CLOSED_SET_SCORING=True
    """
    if args.closed_set_scoring:
        args.decision_mode = "closed_set"

    args.decision_mode = str(args.decision_mode).strip().lower()

    if args.decision_mode not in ["generation", "closed_set"]:
        raise ValueError(
            f"Invalid decision mode: {args.decision_mode}. "
            f"Expected 'generation' or 'closed_set'."
        )

    os.environ["DECISION_MODE"] = args.decision_mode

    if args.decision_mode == "closed_set":
        os.environ["CLOSED_SET_SCORING"] = "True"
    else:
        os.environ["CLOSED_SET_SCORING"] = "False"

    print(
        f"[DECISION MODE] decision_mode={args.decision_mode}, "
        f"CLOSED_SET_SCORING={os.environ.get('CLOSED_SET_SCORING')}"
    )


# ============================================================
# Relation Logit Trajectory helpers
# ============================================================

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _normalize_relation_answer(x):
    if isinstance(x, list):
        x = x[0] if len(x) > 0 else ""

    x = str(x).strip().lower()
    x = x.replace("in front", "in-front").replace("in_front", "in-front")

    if x == "above":
        return "on"
    if x == "below":
        return "under"

    return x


def _load_prompt_answer_lists(dataset, option):
    qst_ans_file = f"prompts/{dataset}_with_answer_{option}_options.jsonl"

    if not os.path.exists(qst_ans_file):
        raise FileNotFoundError(f"Prompt file not found: {qst_ans_file}")

    prompt_list = []
    answer_list = []

    with open(qst_ans_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompt_list.append(data["question"])
            answer_list.append(data["answer"])

    return prompt_list, answer_list


def _load_sample_id_set(path):
    path = str(path or "").strip()

    if not path:
        return None

    if not os.path.exists(path):
        raise FileNotFoundError(f"trajectory sample ids file not found: {path}")

    ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(int(line))

    print(f"[TRAJECTORY FILTER] loaded {len(ids)} sample ids from {path}")
    return ids


def _nested_getattr(obj, path):
    cur = obj

    for name in path.split("."):
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)

    return cur


def _get_lm_head_and_final_norm(hf_model):
    """
    hf_model is wrapper.model, i.e. LlavaForConditionalGeneration or Scal.

    Expected repo structure:
        hf_model.language_model.lm_head
        hf_model.language_model.model.norm
    """
    lm_head_candidates = [
        "language_model.lm_head",
        "lm_head",
        "model.lm_head",
    ]

    norm_candidates = [
        "language_model.model.norm",
        "language_model.norm",
        "model.norm",
        "norm",
    ]

    lm_head = None
    for p in lm_head_candidates:
        lm_head = _nested_getattr(hf_model, p)
        if lm_head is not None:
            break

    final_norm = None
    for p in norm_candidates:
        final_norm = _nested_getattr(hf_model, p)
        if final_norm is not None:
            break

    if lm_head is None:
        raise RuntimeError(
            "Could not find lm_head. Please inspect wrapper.model structure."
        )

    if final_norm is None:
        print(
            "[WARN] Could not find final norm. "
            "Intermediate relation logits will use raw hidden states."
        )

    return lm_head, final_norm


def _build_relation_token_info(tokenizer, relations):
    """
    For left/right/on/under, LLaMA tokenizer should usually produce a single
    token for ' left', ' right', ' on', ' under'.

    If a relation is multi-token, this function uses the first token and warns.
    For Controlled_Images_A four-option setting, this should not be an issue.
    """
    out = {}

    for rel in relations:
        variants = [" " + rel, rel]
        chosen = None

        for text in variants:
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                ids = []

            if len(ids) == 1:
                tid = int(ids[0])
                chosen = {
                    "relation": rel,
                    "text": text,
                    "token_ids": [tid],
                    "token_id": tid,
                    "single_token": True,
                    "decoded": tokenizer.decode(
                        [tid],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                }
                break

        if chosen is None:
            ids = tokenizer.encode(" " + rel, add_special_tokens=False)

            if len(ids) == 0:
                raise RuntimeError(f"Could not tokenize relation: {rel}")

            tid = int(ids[0])
            chosen = {
                "relation": rel,
                "text": " " + rel,
                "token_ids": [int(x) for x in ids],
                "token_id": tid,
                "single_token": False,
                "decoded": tokenizer.decode(
                    [tid],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            }

            print(
                f"[WARN] relation={rel} is multi-token under tokenizer: "
                f"{ids}. Using first-token logit only."
            )

        out[rel] = chosen

    return out


def _get_last_prompt_position(inputs):
    """
    Last non-padding prompt token. Its hidden state predicts the first answer token.
    """
    attention_mask = inputs["attention_mask"][0]
    return int(attention_mask.sum().item()) - 1


def _build_image_keys(input_ids):
    return [
        torch.where(input_id == IMAGE_TOKEN_ID, 1, 0)
        for input_id in input_ids
    ]


def _selected_lm_head_logits(hidden_pos, lm_head, token_ids):
    """
    Compute logits only for selected relation token ids.

    hidden_pos: [hidden] or [1, hidden]
    token_ids: [num_relations]
    """
    if hidden_pos.dim() == 1:
        hidden_pos = hidden_pos.unsqueeze(0)

    token_ids = token_ids.to(lm_head.weight.device)

    weight = lm_head.weight.index_select(0, token_ids)
    weight = weight.to(dtype=hidden_pos.dtype)

    logits = hidden_pos @ weight.t()

    if getattr(lm_head, "bias", None) is not None:
        bias = lm_head.bias.index_select(0, token_ids)
        logits = logits + bias.to(dtype=logits.dtype)

    return logits[0]


def _relation_layer_record(layer_idx, layer_name, cand_logits, relations, gold):
    logits_np = cand_logits.detach().float().cpu().numpy()
    probs_np = torch.softmax(cand_logits.detach().float(), dim=-1).cpu().numpy()

    logits = {rel: float(logits_np[i]) for i, rel in enumerate(relations)}
    probs = {rel: float(probs_np[i]) for i, rel in enumerate(relations)}

    pred_idx = int(np.argmax(logits_np))
    pred = relations[pred_idx]

    if gold in relations:
        gold_idx = relations.index(gold)
        gold_logit = float(logits_np[gold_idx])

        non_gold = [i for i in range(len(relations)) if i != gold_idx]
        best_non_gold_idx = max(non_gold, key=lambda i: logits_np[i])

        best_non_gold = relations[int(best_non_gold_idx)]
        best_non_gold_logit = float(logits_np[best_non_gold_idx])

        gold_margin = gold_logit - best_non_gold_logit
        gold_prob = float(probs_np[gold_idx])
    else:
        gold_logit = None
        best_non_gold = None
        best_non_gold_logit = None
        gold_margin = None
        gold_prob = None

    return {
        "layer_idx": int(layer_idx),
        "layer_name": layer_name,
        "pred": pred,
        "correct": bool(pred == gold),
        "gold": gold,
        "gold_logit": gold_logit,
        "gold_prob_relation_softmax": gold_prob,
        "best_non_gold": best_non_gold,
        "best_non_gold_logit": best_non_gold_logit,
        "gold_margin": gold_margin,
        "logits": logits,
        "relation_softmax_probs": probs,
    }


def _parse_generated_relation(text, relations, pick="last"):
    """
    Parse generated answer text and extract first/last relation occurrence.

    Case-insensitive.
    Examples:
        "Left" -> left
        "The answer is UNDER." -> under
        "It is on the table." -> on
    """
    text_raw = str(text)
    text_l = text_raw.lower()
    text_norm = text_l.replace("in front", "in-front").replace("in_front", "in-front")

    hits = []

    for rel in relations:
        rel_norm = rel.lower().replace("in front", "in-front").replace("in_front", "in-front")

        if rel_norm == "in-front":
            patterns = [
                r"\bin\s*-\s*front\b",
                r"\bin\s+front\b",
                r"\bfront\b",
            ]
        else:
            patterns = [rf"\b{re.escape(rel_norm)}\b"]

        for pat in patterns:
            for m in re.finditer(pat, text_norm, flags=re.IGNORECASE):
                hits.append(
                    {
                        "relation": rel,
                        "start": int(m.start()),
                        "end": int(m.end()),
                        "matched_text": text_raw[m.start():m.end()],
                    }
                )

    if len(hits) == 0:
        return "unknown", []

    hits = sorted(hits, key=lambda x: x["start"])

    if pick == "first":
        return hits[0]["relation"], hits
    else:
        return hits[-1]["relation"], hits


def _maybe_import_llava15_helpers():
    """
    Import helpers from the current llava15.py.

    This keeps trajectory behavior consistent with your repo:
      - IMAGE_CONTROL
      - PATCH_MASK_MODE / PATCH_BLOCK_IDS
      - custom greedy search for scaling_vis/adapt_vis generation
    """
    try:
        from model_zoo.llava15 import (
            apply_image_control_from_env,
            build_manual_patch_mask_from_env,
            change_greedy_to_add_weight,
        )
    except Exception as e:
        raise ImportError(
            "Failed to import helpers from model_zoo.llava15. "
            "Please make sure llava15.py contains "
            "apply_image_control_from_env, build_manual_patch_mask_from_env, "
            "and change_greedy_to_add_weight."
        ) from e

    return (
        apply_image_control_from_env,
        build_manual_patch_mask_from_env,
        change_greedy_to_add_weight,
    )


@torch.no_grad()
def _build_forward_inputs_and_controls(
    wrapper,
    image,
    prompt,
    method,
    weight,
    adjust_method,
    query_pos,
):
    """
    Build processor inputs plus optional AdaptVis / ScalingVis controls.

    This is used by both:
      1. layer-wise closed-set trajectory forward
      2. normal generation comparison
    """
    processor = wrapper.processor
    device = wrapper.device

    inputs = processor(
        text=prompt,
        images=image,
        padding="max_length",
        return_tensors="pt",
        max_length=77,
    ).to(device)

    controls = {}

    if method in ["scaling_vis", "adapt_vis"]:
        keys = _build_image_keys(inputs["input_ids"])

        pos = None
        if adjust_method == "text_offset":
            if query_pos is None:
                raise ValueError(
                    "adjust_method=text_offset requires --trajectory-query-pos."
                )
            pos = torch.tensor(int(query_pos), device=device)

        object_patch_mask = None

        if adjust_method == "object_mask":
            _, build_manual_patch_mask_from_env, _ = _maybe_import_llava15_helpers()
            object_patch_mask = build_manual_patch_mask_from_env(device)

        controls = {
            "keys": keys,
            "weight": float(weight),
            "adjust_method": adjust_method,
            "pos": pos,
            "object_patch_mask": object_patch_mask,
        }

    return inputs, controls


@torch.no_grad()
def _compute_relation_logit_trajectory_one(
    wrapper,
    image,
    prompt,
    relations,
    relation_token_info,
    method,
    weight,
    adjust_method,
    query_pos,
    apply_final_norm_for_intermediate=True,
):
    """
    Closed-set relation logit trajectory.

    For each layer hidden state at the last prompt position:
        hidden[layer, answer_position] -> final_norm -> lm_head
    Then compare logits of relation tokens:
        left / right / on / under

    This is NOT actual generation from that layer.
    It is a logit-lens diagnostic using the original model's lm_head.
    """
    hf_model = wrapper.model
    device = wrapper.device

    inputs, controls = _build_forward_inputs_and_controls(
        wrapper=wrapper,
        image=image,
        prompt=prompt,
        method=method,
        weight=weight,
        adjust_method=adjust_method,
        query_pos=query_pos,
    )

    last_pos = _get_last_prompt_position(inputs)

    lm_head, final_norm = _get_lm_head_and_final_norm(hf_model)

    token_ids = torch.tensor(
        [relation_token_info[rel]["token_id"] for rel in relations],
        dtype=torch.long,
        device=device,
    )

    forward_kwargs = {
        "output_hidden_states": True,
        "return_dict": True,
    }

    if method in ["scaling_vis", "adapt_vis"]:
        outputs = hf_model(
            **inputs,
            **controls,
            **forward_kwargs,
        )
    else:
        outputs = hf_model(
            **inputs,
            **forward_kwargs,
        )

    if outputs.hidden_states is None:
        raise RuntimeError(
            "outputs.hidden_states is None. "
            "Check whether output_hidden_states=True is propagated correctly."
        )

    hidden_states = outputs.hidden_states
    num_states = len(hidden_states)

    layer_records = []

    for idx, h in enumerate(hidden_states):
        h_pos = h[0, last_pos, :]

        is_final_state = idx == num_states - 1

        if (
            apply_final_norm_for_intermediate
            and not is_final_state
            and final_norm is not None
        ):
            h_pos = final_norm(h_pos)

        if is_final_state:
            # Use actual forward logits for final layer.
            cand_logits = outputs.logits[0, last_pos, :].index_select(
                0,
                token_ids.to(outputs.logits.device),
            )
        else:
            cand_logits = _selected_lm_head_logits(
                hidden_pos=h_pos,
                lm_head=lm_head,
                token_ids=token_ids,
            )

        if idx == 0:
            layer_name = "emb"
        elif is_final_state:
            layer_name = "final"
        else:
            layer_name = f"layer_{idx}"

        layer_records.append(
            {
                "layer_idx": int(idx),
                "layer_name": layer_name,
                "candidate_logits": cand_logits.detach().float().cpu(),
            }
        )

    return layer_records, last_pos, inputs


@torch.no_grad()
def _run_generation_comparison_one(
    wrapper,
    image,
    prompt,
    relations,
    method,
    weight,
    adjust_method,
    query_pos,
    max_new_tokens,
    relation_pick,
):
    """
    Normal generation comparison.

    It runs model.generate() and parses first/last generated relation word.
    It is intended to compare with closed-set relation logits.
    """
    hf_model = wrapper.model
    processor = wrapper.processor

    inputs, controls = _build_forward_inputs_and_controls(
        wrapper=wrapper,
        image=image,
        prompt=prompt,
        method=method,
        weight=weight,
        adjust_method=adjust_method,
        query_pos=query_pos,
    )

    input_len = int(inputs["input_ids"].shape[-1])

    if method in ["scaling_vis", "adapt_vis"]:
        _, _, change_greedy_to_add_weight = _maybe_import_llava15_helpers()
        change_greedy_to_add_weight()

        output = hf_model.generate(
            **inputs,
            **controls,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )
    else:
        output = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_ids = output["sequences"][0][input_len:]

    generated_text = processor.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    generation_pred, relation_hits = _parse_generated_relation(
        generated_text,
        relations=relations,
        pick=relation_pick,
    )

    return {
        "generated_text": generated_text,
        "generation_pred": generation_pred,
        "generation_relation_hits": relation_hits,
    }


def run_relation_logit_trajectory(args, model, dataset, joint_loader):
    """
    Integrated trajectory runner.

    Saves:
      - relation_token_info.json
      - layer_rows.csv
      - final_layer_summary.csv
      - summary_by_gold_layer.csv
      - summary_all_layer.csv
      - trajectory.jsonl if --trajectory-save-jsonl is used

    layer_rows.csv records closed-set logit-lens prediction for every layer.
    final_layer_summary.csv also includes optional normal generation comparison.
    """
    apply_image_control_from_env, _, _ = _maybe_import_llava15_helpers()

    out_dir = args.trajectory_out_dir
    _ensure_dir(out_dir)
    _ensure_dir("./output")

    relations = [
        x.strip().lower()
        for x in str(args.trajectory_relations).split(",")
        if x.strip()
    ]

    if len(relations) < 2:
        raise ValueError("--trajectory-relations must contain at least two candidates.")

    prompt_list, answer_list = _load_prompt_answer_lists(args.dataset, args.option)
    sample_id_set = _load_sample_id_set(args.trajectory_sample_ids_file)

    tokenizer = model.processor.tokenizer
    relation_token_info = _build_relation_token_info(tokenizer, relations)

    token_info_path = os.path.join(out_dir, "relation_token_info.json")

    with open(token_info_path, "w", encoding="utf-8") as f:
        json.dump(relation_token_info, f, ensure_ascii=False, indent=2)

    print("[RELATION LOGIT TRAJECTORY]")
    print(f"  dataset={args.dataset}")
    print(f"  option={args.option}")
    print(f"  method={args.method}")
    print(f"  out_dir={out_dir}")
    print(f"  relations={relations}")
    print(f"  compare_generation={args.trajectory_compare_generation}")

    for rel in relations:
        info = relation_token_info[rel]
        print(
            f"  token {rel:>8s}: id={info['token_id']}, "
            f"text={repr(info['text'])}, "
            f"decoded={repr(info['decoded'])}, "
            f"single_token={info['single_token']}"
        )

    trajectory_weight = (
        float(args.trajectory_weight)
        if args.trajectory_weight is not None
        else float(args.weight)
    )

    trajectory_adjust_method = (
        str(args.trajectory_adjust_method)
        if args.trajectory_adjust_method is not None
        else os.getenv("ADJUST_METHOD", "last_query")
    )

    apply_final_norm = not bool(args.trajectory_no_final_norm)

    print(
        f"  trajectory_weight={trajectory_weight}, "
        f"trajectory_adjust_method={trajectory_adjust_method}, "
        f"apply_final_norm={apply_final_norm}"
    )

    jsonl_path = os.path.join(out_dir, "trajectory.jsonl")
    layer_rows_path = os.path.join(out_dir, "layer_rows.csv")
    final_summary_path = os.path.join(out_dir, "final_layer_summary.csv")
    summary_by_gold_path = os.path.join(out_dir, "summary_by_gold_layer.csv")
    summary_all_path = os.path.join(out_dir, "summary_all_layer.csv")

    fout = open(jsonl_path, "w", encoding="utf-8") if args.trajectory_save_jsonl else None

    all_layer_rows = []
    final_rows = []

    index_of_total = 0
    processed = 0
    skipped = 0

    for batch in tqdm(joint_loader):
        for i_option in batch["image_options"]:
            for img in i_option:
                sample_id = int(index_of_total)

                if sample_id_set is not None and sample_id not in sample_id_set:
                    skipped += 1
                    index_of_total += 1
                    continue

                if (
                    args.trajectory_max_samples is not None
                    and processed >= args.trajectory_max_samples
                ):
                    break

                prompt = prompt_list[sample_id]
                gold = _normalize_relation_answer(answer_list[sample_id])

                image_for_model = apply_image_control_from_env(
                    img,
                    sample_id=sample_id,
                )

                layer_raw, last_pos, _ = _compute_relation_logit_trajectory_one(
                    wrapper=model,
                    image=image_for_model,
                    prompt=prompt,
                    relations=relations,
                    relation_token_info=relation_token_info,
                    method=args.method,
                    weight=trajectory_weight,
                    adjust_method=trajectory_adjust_method,
                    query_pos=args.trajectory_query_pos,
                    apply_final_norm_for_intermediate=apply_final_norm,
                )

                per_layer = []

                for rec in layer_raw:
                    layer_rec = _relation_layer_record(
                        layer_idx=rec["layer_idx"],
                        layer_name=rec["layer_name"],
                        cand_logits=rec["candidate_logits"],
                        relations=relations,
                        gold=gold,
                    )

                    per_layer.append(layer_rec)

                    flat = {
                        "sample_id": sample_id,
                        "gold": gold,
                        "prompt": prompt,
                        "layer_idx": layer_rec["layer_idx"],
                        "layer_name": layer_rec["layer_name"],
                        "closed_set_pred": layer_rec["pred"],
                        "closed_set_correct": layer_rec["correct"],
                        "gold_logit": layer_rec["gold_logit"],
                        "gold_prob_relation_softmax": layer_rec[
                            "gold_prob_relation_softmax"
                        ],
                        "best_non_gold": layer_rec["best_non_gold"],
                        "best_non_gold_logit": layer_rec["best_non_gold_logit"],
                        "gold_margin": layer_rec["gold_margin"],
                    }

                    for rel in relations:
                        flat[f"logit_{rel}"] = layer_rec["logits"][rel]
                        flat[f"prob_{rel}"] = layer_rec["relation_softmax_probs"][rel]

                    all_layer_rows.append(flat)

                final_rec = per_layer[-1]

                generation_info = {
                    "generated_text": None,
                    "generation_pred": None,
                    "generation_correct": None,
                    "generation_relation_hits": [],
                }

                if args.trajectory_compare_generation:
                    generation_info = _run_generation_comparison_one(
                        wrapper=model,
                        image=image_for_model,
                        prompt=prompt,
                        relations=relations,
                        method=args.method,
                        weight=trajectory_weight,
                        adjust_method=trajectory_adjust_method,
                        query_pos=args.trajectory_query_pos,
                        max_new_tokens=args.trajectory_generation_max_new_tokens,
                        relation_pick=args.trajectory_generation_relation_pick,
                    )

                    generation_info["generation_correct"] = bool(
                        generation_info["generation_pred"] == gold
                    )

                final_rows.append(
                    {
                        "sample_id": sample_id,
                        "gold": gold,

                        # Closed-set final layer result:
                        "closed_set_pred_final": final_rec["pred"],
                        "closed_set_correct_final": final_rec["correct"],
                        "closed_set_gold_margin_final": final_rec["gold_margin"],
                        "closed_set_gold_prob_final": final_rec[
                            "gold_prob_relation_softmax"
                        ],

                        # Normal generation result:
                        "generation_pred": generation_info["generation_pred"],
                        "generation_correct": generation_info["generation_correct"],
                        "generated_text": generation_info["generated_text"],

                        "prompt": prompt,
                    }
                )

                if fout is not None:
                    obj = {
                        "sample_id": sample_id,
                        "prompt": prompt,
                        "gold": gold,
                        "last_prompt_position": int(last_pos),
                        "method": args.method,
                        "trajectory_weight": trajectory_weight,
                        "trajectory_adjust_method": trajectory_adjust_method,
                        "relations": relations,
                        "relation_token_info": relation_token_info,

                        # closed-set layer trajectory:
                        "per_layer": per_layer,

                        # normal generation comparison:
                        "generation": generation_info,
                    }

                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    fout.flush()

                processed += 1
                index_of_total += 1

                if processed % 20 == 0:
                    print(
                        f"[TRAJECTORY RUNNING] processed={processed}, "
                        f"skipped={skipped}, last_sample={sample_id}, "
                        f"gold={gold}, "
                        f"closed_set_final={final_rec['pred']}, "
                        f"closed_set_margin={final_rec['gold_margin']}, "
                        f"generation={generation_info['generation_pred']}"
                    )

            if (
                args.trajectory_max_samples is not None
                and processed >= args.trajectory_max_samples
            ):
                break

        if (
            args.trajectory_max_samples is not None
            and processed >= args.trajectory_max_samples
        ):
            break

    if fout is not None:
        fout.close()

    if len(all_layer_rows) == 0:
        print("[TRAJECTORY DONE] No samples processed.")
        return

    df = pd.DataFrame(all_layer_rows)
    df.to_csv(layer_rows_path, index=False)

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(final_summary_path, index=False)

    agg_dict = {
        "closed_set_correct": "mean",
        "gold_margin": ["mean", "median"],
        "gold_prob_relation_softmax": ["mean", "median"],
    }

    for rel in relations:
        agg_dict[f"logit_{rel}"] = "mean"
        agg_dict[f"prob_{rel}"] = "mean"

    summary_by_gold = (
        df.groupby(["gold", "layer_idx", "layer_name"])
        .agg(agg_dict)
        .reset_index()
    )

    summary_by_gold.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary_by_gold.columns
    ]

    summary_by_gold.to_csv(summary_by_gold_path, index=False)

    summary_all = (
        df.groupby(["layer_idx", "layer_name"])
        .agg(agg_dict)
        .reset_index()
    )

    summary_all.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary_all.columns
    ]

    summary_all.to_csv(summary_all_path, index=False)

    print("[TRAJECTORY DONE]")
    print(f"  processed={processed}, skipped={skipped}")
    print(f"  layer rows: {layer_rows_path}")
    print(f"  final summary: {final_summary_path}")
    print(f"  summary by gold/layer: {summary_by_gold_path}")
    print(f"  summary all/layer: {summary_all_path}")

    if args.trajectory_save_jsonl:
        print(f"  detailed jsonl: {jsonl_path}")


def is_probe_mode():
    """
    Probe / ablation runs may only process a filtered subset, e.g. gold=on/under ids.
    In that case, scores are no longer aligned with the full dataset and
    dataset.evaluate_scores() must be skipped.

    Important:
        CLOSED_SET_SCORING / DECISION_MODE should NOT by itself trigger probe mode.
        Closed-set scoring still processes the full dataset and returns aligned scores.
    """
    if os.getenv("FORCE_DATASET_EVAL", "False") == "True":
        return False

    adjust_method = os.getenv("ADJUST_METHOD", "").strip()

    if adjust_method in ["probe_bias", "probe_scale", "probe_add", "var_sink"]:
        return True

    probe_env_keys = [
        "PROBE_SAMPLE_IDS_FILE",
        "PROBE_RUN_TAG",
        "ATTN_RUN_TAG",

        "PROBE_LAYER",
        "PROBE_HEAD",
        "PROBE_BLOCK_IDS",

        "PROBE_BETA",

        "PROBE_SCALE",

        "PROBE_ADD_MODE",
        "PROBE_ADD_MASS",
        "PROBE_ADD_VALUE",
        "PROBE_ADD_RENORM",

        "PROBE_ADD_BETA",
        "PROBE_ADD_ALPHA",
        "PROBE_ADD_BETA_MODE",
        "PROBE_ADD_BETA_CLAMP",
        "PROBE_ADD_STD_EPS",

        "PROBE_RELATION_PROBS",
        "PROBE_RELATION_TOPK",
        "PROBE_RELATION_SET",

        "IMAGE_CONTROL",
        "IMAGE_CONTROL_SIZE",
        "IMAGE_CONTROL_GRID",
        "IMAGE_CONTROL_SEED",
    ]

    for key in probe_env_keys:
        val = os.getenv(key, "").strip()

        if key == "IMAGE_CONTROL" and val in ["", "none", "original"]:
            continue

        if val:
            return True

    return False


def main(args):
    seed_all(args.seed)

    setup_decision_mode_env(args)

    os.makedirs("./output", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model, image_preprocess = get_model(
        args.model_name,
        args.device,
        args.method,
    )

    dataset = get_dataset(
        args.dataset,
        image_preprocess=image_preprocess,
        download=args.download,
    )

    SAMPLE = False
    TEST = os.getenv("TEST_MODE", "False") == "True"
    sampled_indices = None

    collate_fn = _default_collate if image_preprocess is None else None

    if SAMPLE:
        total_data_count = len(dataset)
        idx_file_path = f"./output/sampled_idx_{args.dataset}.npy"

        if os.path.exists(idx_file_path):
            sampled_indices = np.load(idx_file_path).tolist()
        else:
            sampled_indices = random.sample(
                range(total_data_count),
                int(0.2 * total_data_count),
            )
            sampled_indices.sort()
            np.save(idx_file_path, np.array(sampled_indices))

        all_indices = set(range(total_data_count))

        if TEST:
            unsampled_indices = list(all_indices - set(sampled_indices))
            unsampled_indices.sort()
            sampled_indices = unsampled_indices

        sub_dataset = torch.utils.data.Subset(dataset, sampled_indices)

        joint_loader = DataLoader(
            sub_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    else:
        joint_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    print(args.dataset, args.model_name)

    if args.relation_logit_trajectory:
        run_relation_logit_trajectory(
            args=args,
            model=model,
            dataset=dataset,
            joint_loader=joint_loader,
        )
        return

    if args.dataset == "VSR":
        labels = dataset.get_labels()

        scores = model.get_judge_scores_vsr_batched(
            args.dataset,
            joint_loader,
            args.method,
            args.weight,
            args.threshold,
            args.weight1,
            args.weight2,
        )

        result_records = dataset.evaluate_scores(
            args.model_name,
            scores,
            labels,
            args.output_dir,
            args.dataset,
        )

    elif args.dataset in ["Controlled_Images_A", "Controlled_Images_B"]:
        scores, correct_id = model.get_out_scores_wh_batched(
            args.dataset,
            joint_loader,
            args.method,
            args.weight,
            args.option,
            args.threshold,
            args.weight1,
            args.weight2,
        )

        print("Got the following shape of scores", scores.shape)

        if is_probe_mode():
            print(
                "[PROBE MODE] Skip dataset.evaluate_scores because this run "
                "may contain a filtered subset and scores may not align with "
                "the full dataset."
            )
            print(
                "[PROBE MODE] Probe outputs have already been saved by "
                "model.get_out_scores_wh_batched()."
            )
            return

        scores = scores.transpose(0, 2, 1)

        dataset.evaluate_scores(
            scores,
            args.output_dir,
            args.dataset,
            args.model_name,
            args.method,
            args.weight,
            sampled_indices,
            args.option,
        )

    else:
        scores, correct_id = model.get_out_scores_wh_batched(
            args.dataset,
            joint_loader,
            args.method,
            args.weight,
            args.option,
            args.threshold,
            args.weight1,
            args.weight2,
        )

        os.makedirs(args.output_dir, exist_ok=True)

        dataset.save_scores(
            scores,
            correct_id,
            args.output_dir,
            args.dataset,
            args.method,
            args.weight,
            args.model_name,
            args.option,
        )


if __name__ == "__main__":
    args = config()
    main(args)
