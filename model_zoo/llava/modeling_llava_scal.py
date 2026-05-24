# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch Llava model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import pdb
import torch
import torch.utils.checkpoint
from torch import nn

from model_zoo import llama

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from transformers import AutoModel, AutoModelForCausalLM
from .configuration_llava import LlavaConfig
from model_zoo.llama.configuration_llama import LLaMAConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlavaConfig"

LLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "llava-hf/bakLlava-v1-hf",
]


@dataclass
class LlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
        )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


LLAVA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`].
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAVA_START_DOCSTRING,
)
class LlavaPreTrainedModel(PreTrainedModel):
    config_class = LlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAVA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Input token ids.

        pixel_values (`torch.FloatTensor`):
            Image pixel values.

        attention_mask (`torch.Tensor`, *optional*):
            Attention mask.

        position_ids (`torch.LongTensor`, *optional*):
            Position ids.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Cached key values.

        inputs_embeds (`torch.FloatTensor`, *optional*):
            Input embeddings.

        use_cache (`bool`, *optional*):
            Whether to use cache.

        output_attentions (`bool`, *optional*):
            Whether to return attentions.

        output_hidden_states (`bool`, *optional*):
            Whether to return hidden states.

        return_dict (`bool`, *optional*):
            Whether to return a ModelOutput.
"""


@add_start_docstrings(
    """The LLAVA model which consists of a vision backbone and a language model.""",
    LLAVA_START_DOCSTRING,
)
class LlavaForConditionalGenerationScal(LlavaPreTrainedModel):
    def __init__(self, config: LlavaConfig):
        super().__init__(config)

        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = LlavaMultiModalProjector(config)

        self.vocab_size = config.vocab_size

        self.language_model = llama.LLaMAForCausalLMScal._from_config(
            LLaMAConfig()
        )

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        # ---------------------------------------------------------------------
        # AdaptVis feature-saving support.
        #
        # These are filled during the first/full image-text forward, right after
        # <image> is expanded into visual tokens.
        #
        # _adaptvis_last_image_id is exactly the `image_id` mask passed to:
        #     language_model(..., keys=image_id)
        #
        # Therefore llava15.py can use this mask to slice hidden states at the
        # exact same visual-token positions used by AdaptVis attention scaling.
        #
        # Do NOT clear these in later cached decoding steps.
        # ---------------------------------------------------------------------
        self._adaptvis_last_image_id = None
        self._adaptvis_last_attention_mask = None

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings

        return model_embeds

    def _merge_input_ids_with_image_features(
        self,
        image_features,
        inputs_embeds,
        input_ids,
        attention_mask,
        position_ids,
    ):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape

        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))

        # 1. Create a mask to know where special image tokens are.
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)

        # Compute the maximum embed dimension.
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length

        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written.
        new_token_positions = torch.cumsum(
            (special_image_token_mask * (num_image_patches - 1) + 1),
            -1,
        ) - 1

        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]

        if left_padding:
            new_token_positions += nb_image_pad[:, None]

        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding.
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )

        # 4. Fill text embeddings.
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]

        # 5. Fill image embeddings.
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None]

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. "
                f"The number of image tokens is {torch.sum(special_image_token_mask)} while "
                f"the number of image given to the model is {num_images}. "
                f"This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim)
        final_attention_mask |= image_to_overwrite

        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
            (final_attention_mask == 0),
            1,
        )

        return final_embedding, final_attention_mask, position_ids, image_to_overwrite

    @add_start_docstrings_to_model_forward(LLAVA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=LlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        keys: Optional[torch.Tensor] = None,
        weight: Optional[float] = None,
        pos: Optional[torch.Tensor] = None,
        adjust_method: Optional[str] = None,
        caption_length: Optional[list] = None,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Forward pass.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Local visual-token mask for this forward.
        # It is set only on the first/full image-text forward.
        # Later cached decoding steps keep it as None.
        image_id = None

        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )

        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extract input embeddings.
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images.
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)

                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                image_features = self.multi_modal_projector(selected_image_feature)

                inputs_embeds, attention_mask, position_ids, image_id = self._merge_input_ids_with_image_features(
                    image_features,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                )

                # -----------------------------------------------------------------
                # Save exact merged-sequence masks.
                #
                # image_id is the same mask passed to:
                #     self.language_model(..., keys=image_id)
                #
                # Therefore it is exactly the visual-token span used by AdaptVis.
                # This is what llava15.py should use to slice hidden states.
                # -----------------------------------------------------------------
                self._adaptvis_last_image_id = image_id.detach().cpu()
                self._adaptvis_last_attention_mask = attention_mask.detach().cpu()

                if labels is None:
                    labels = torch.full_like(
                        attention_mask,
                        self.config.ignore_index,
                    ).to(torch.long)

            else:
                # Cached decoding step.
                # input_ids.shape[1] == 1 and pixel_values may still be present.
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    first_layer_past_key_value = past_key_values[0][0][:, 0, :, 0]
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value == 0)

                    target_seqlen = first_layer_past_key_value.shape[-1] + 1

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    extended_attention_mask[batch_index, non_attended_tokens] = 0

                    attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

                    # No new image merge in cached decoding.
                    # Keep self._adaptvis_last_image_id from the first/full forward.
                    image_id = None

        outputs = self.language_model(
            attention_mask=attention_mask,
            # position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            keys=image_id,
            weight=weight,
            pos=pos,
            adjust_method=adjust_method,
            caption_length=caption_length,
        )

        logits = outputs[0]

        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n.
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens.
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]

            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1:]

            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]):]

        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, only use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
        )

        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
