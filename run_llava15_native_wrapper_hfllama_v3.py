#!/usr/bin/env python3
"""
Patch model_zoo/llava/modeling_llava_scal.py cache branch to simulate the
original zero-cache filtering in a safer "absolute full mask" way.

Original idea:
    first_layer_past_key_value = past_key_values[0][0][:, 0, :, 0]
    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value == 0)
    extended_attention_mask[batch_index, non_attended_tokens] = 0

Problem with HF LLaMA:
    non_attended_tokens are absolute cache positions, while extended_attention_mask
    only covers newly appended positions. This can go out of bounds.

This patch simulates the same filter by constructing a full attention mask of
length target_seqlen and zeroing absolute non_attended_tokens directly:
    full_attention_mask[batch_index, non_attended_tokens] = 0

Run from AdaptVis repo root:
    python patch_llava_scal_cache_zero_filter_fullmask.py
"""

from pathlib import Path
import shutil
import time

path = Path("model_zoo/llava/modeling_llava_scal.py")
if not path.exists():
    raise FileNotFoundError(path)

stamp = time.strftime("%Y%m%d_%H%M%S")
backup = path.with_suffix(path.suffix + f".bak_zero_filter_fullmask_{stamp}")
shutil.copy2(path, backup)
print("backup:", backup)

text = path.read_text()

old_original = """                # Zero-out the places where we don't need to attend
                extended_attention_mask[batch_index, non_attended_tokens] = 0
                attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
"""

new_fullmask = """                # Zero-out the places where we don't need to attend.
                #
                # Safe absolute-position simulation of the original zero-cache filter.
                # non_attended_tokens are absolute positions in the cache sequence.
                # Construct a full mask of length target_seqlen and write absolute indices.
                old_attention_mask = attention_mask
                full_attention_mask = torch.ones(
                    (attention_mask.shape[0], target_seqlen),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                copy_len = min(old_attention_mask.shape[1], target_seqlen)
                full_attention_mask[:, :copy_len] = old_attention_mask[:, :copy_len]

                valid = (non_attended_tokens >= 0) & (non_attended_tokens < target_seqlen)

                if valid.numel() > 0 and not bool(valid.all().detach().cpu()):
                    print(
                        "[zero_filter_fullmask] invalid abs positions ignored:",
                        "old_attention_mask_shape=", tuple(old_attention_mask.shape),
                        "target_seqlen=", int(target_seqlen),
                        "first_layer_past_shape=", tuple(first_layer_past_key_value.shape),
                        "num_zero=", int(non_attended_tokens.numel()),
                        "num_valid=", int(valid.sum().detach().cpu()),
                        "min_abs=", int(non_attended_tokens.min().detach().cpu()) if non_attended_tokens.numel() else None,
                        "max_abs=", int(non_attended_tokens.max().detach().cpu()) if non_attended_tokens.numel() else None,
                        flush=True,
                    )

                if bool(valid.any().detach().cpu()):
                    full_attention_mask[batch_index[valid], non_attended_tokens[valid]] = 0

                attention_mask = full_attention_mask
"""

old_debug = """                # Zero-out the places where we don't need to attend.
                #
                # Original code indexed extended_attention_mask with non_attended_tokens directly:
                #     extended_attention_mask[batch_index, non_attended_tokens] = 0
                # That assumes non_attended_tokens are local indices into extended_attention_mask.
                # With HF LLaMA cache, non_attended_tokens are absolute cache positions, so at
                # later decoding steps they can exceed extended_attention_mask.shape[1] and
                # trigger CUDA index out of bounds. Convert to relative indices first.
                ext_width = extended_attention_mask.shape[1]
                rel_non_attended_tokens = non_attended_tokens - attention_mask.shape[1]
                valid = (rel_non_attended_tokens >= 0) & (rel_non_attended_tokens < ext_width)

                if valid.numel() > 0 and not bool(valid.all().detach().cpu()):
                    print(
                        "[AdaptVis cache debug] invalid non_attended_tokens:",
                        "attention_mask_shape=", tuple(attention_mask.shape),
                        "target_seqlen=", int(target_seqlen),
                        "ext_width=", int(ext_width),
                        "first_layer_past_shape=", tuple(first_layer_past_key_value.shape),
                        "num_zero=", int(non_attended_tokens.numel()),
                        "num_valid=", int(valid.sum().detach().cpu()),
                        "min_abs=", int(non_attended_tokens.min().detach().cpu()) if non_attended_tokens.numel() else None,
                        "max_abs=", int(non_attended_tokens.max().detach().cpu()) if non_attended_tokens.numel() else None,
                        "min_rel=", int(rel_non_attended_tokens.min().detach().cpu()) if rel_non_attended_tokens.numel() else None,
                        "max_rel=", int(rel_non_attended_tokens.max().detach().cpu()) if rel_non_attended_tokens.numel() else None,
                        flush=True,
                    )

                if bool(valid.any().detach().cpu()):
                    extended_attention_mask[
                        batch_index[valid],
                        rel_non_attended_tokens[valid],
                    ] = 0

                attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
"""

if old_original in text:
    text = text.replace(old_original, new_fullmask)
    path.write_text(text)
    print("patched original block -> fullmask zero filter")
elif old_debug in text:
    text = text.replace(old_debug, new_fullmask)
    path.write_text(text)
    print("patched previous debug block -> fullmask zero filter")
else:
    print("target block not found. File may already be patched differently.")
    print("Search for 'extended_attention_mask[batch_index' or 'zero_filter_fullmask' manually.")

print("done:", path)
