import os
import atexit
from collections import defaultdict

import torch


VAR_STATS = defaultdict(int)


def _env_bool(name, default=False):
    v = os.getenv(name, str(default))
    return str(v).lower() in ["1", "true", "yes", "y"]


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _print_var_stats():
    if not _env_bool("VAR_DEBUG", False):
        return

    print("\n================ VAR-SINK STATS ================")
    for k in sorted(VAR_STATS.keys()):
        print(f"{k}: {VAR_STATS[k]}")
    print("================================================\n")


atexit.register(_print_var_stats)


def apply_var_sink_attention(
    attn_weights,
    image_token_start,
    image_token_end,
    layer_idx=None,
):
    """
    VAR-lite for AdaptVis.

    attn_weights:
        [bsz, num_heads, q_len, kv_len], after softmax.

    image_token_start / image_token_end:
        absolute KV positions of image tokens.

    This is not the full paper-faithful sink-dimension version.
    It redistributes attention mass from high-attention visual sink tokens
    to non-sink visual tokens, only for image-centric heads.

    It should be inserted AFTER softmax and BEFORE attn_output = attn_weights @ value_states.
    """

    if os.getenv("ADJUST_METHOD", "") != "var_sink":
        return attn_weights

    if image_token_start is None or image_token_end is None:
        VAR_STATS["skip_no_image_span"] += 1
        return attn_weights

    im = int(image_token_start)
    ie = int(image_token_end)

    if ie <= im:
        VAR_STATS["skip_bad_image_span"] += 1
        return attn_weights

    bsz, num_heads, q_len, kv_len = attn_weights.shape

    if im < 0 or ie > kv_len:
        VAR_STATS["skip_image_span_out_of_range"] += 1
        return attn_weights

    num_img = ie - im
    if num_img <= 1:
        VAR_STATS["skip_too_few_image_tokens"] += 1
        return attn_weights

    # Hyperparameters.
    sink_ratio = _env_float("VAR_SINK_RATIO", 0.05)
    sink_topk = _env_int("VAR_SINK_TOPK", 0)
    head_vis_thr = _env_float("VAR_HEAD_VIS_THR", 0.20)
    p = _env_float("VAR_P", 0.60)

    # p means sink token remaining ratio.
    # p=0.6 => move 40% sink attention mass to non-sink visual tokens.
    p = max(0.0, min(1.0, p))

    if sink_topk <= 0:
        sink_topk = max(1, int(round(num_img * sink_ratio)))

    sink_topk = max(1, min(sink_topk, num_img - 1))

    img_attn = attn_weights[:, :, :, im:ie]  # [B, H, Q, I]

    # Query mask:
    # - prefill stage: q_len is long, only modify text queries after image tokens
    # - decoding stage: q_len is usually 1, modify that query
    if q_len > 1:
        q_positions = torch.arange(q_len, device=attn_weights.device)
        query_mask = q_positions >= ie
        if not bool(query_mask.any()):
            VAR_STATS["skip_no_text_query"] += 1
            return attn_weights
    else:
        query_mask = torch.ones(q_len, dtype=torch.bool, device=attn_weights.device)

    img_attn_q = img_attn[:, :, query_mask, :]  # [B, H, Q', I]

    if img_attn_q.numel() == 0:
        VAR_STATS["skip_empty_query"] += 1
        return attn_weights

    # Image-centric heads.
    # Mean image attention mass over selected queries.
    head_img_mass = img_attn_q.sum(dim=-1).mean(dim=-1)  # [B, H]
    image_head_mask = head_img_mass >= head_vis_thr       # [B, H]

    if not bool(image_head_mask.any()):
        VAR_STATS["skip_no_image_centric_head"] += 1
        return attn_weights

    # Sink tokens: highest average image attention tokens per batch/head.
    avg_img_token_attn = img_attn_q.mean(dim=2)  # [B, H, I]
    sink_local_idx = torch.topk(
        avg_img_token_attn,
        k=sink_topk,
        dim=-1,
        largest=True,
    ).indices  # [B, H, K]

    out = attn_weights.clone()

    modified_heads = 0
    modified_queries = 0

    query_indices = torch.nonzero(query_mask, as_tuple=False).flatten()

    for b in range(bsz):
        for h in range(num_heads):
            if not bool(image_head_mask[b, h].item()):
                continue

            local_sinks = sink_local_idx[b, h]  # [K]
            abs_sinks = local_sinks + im

            # Full selected query attention.
            selected = out[b, h, query_indices, :]  # [Q', KV]
            if selected.numel() == 0:
                continue

            original_selected = selected.clone()

            # Sink budget from selected visual sink tokens.
            sink_attn = original_selected[:, abs_sinks]  # [Q', K]
            budget = sink_attn.sum(dim=-1) * (1.0 - p)  # [Q']

            if float(budget.abs().sum().item()) == 0.0:
                continue

            # Reduce sink tokens.
            selected[:, abs_sinks] = selected[:, abs_sinks] * p

            # Redistribute budget to non-sink visual tokens proportional to current attention.
            visual = original_selected[:, im:ie].clone()  # [Q', I]
            visual[:, local_sinks] = 0.0

            denom = visual.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            ratios = visual / denom

            selected[:, im:ie] = selected[:, im:ie] + budget.unsqueeze(-1) * ratios

            out[b, h, query_indices, :] = selected

            modified_heads += 1
            modified_queries += int(query_indices.numel())

    VAR_STATS["calls"] += 1
    VAR_STATS["modified_heads"] += modified_heads
    VAR_STATS["modified_queries"] += modified_queries

    if modified_heads == 0:
        VAR_STATS["skip_no_modified_head"] += 1

    return out
