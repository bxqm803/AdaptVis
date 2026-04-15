# model_zoo/qwen25vl.py
import json
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

class QwenWrapper:
    def __init__(self, root_dir="data", device="cuda", method="base"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL,
            torch_dtype="auto",
            device_map="auto" if device.startswith("cuda") else None,
            attn_implementation="sdpa",
            cache_dir=root_dir,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL, cache_dir=root_dir)
        self.device = device

    def load_prompt_records_with_sampling(self, dataset, option):
        qst_ans_file = f'prompts/{dataset}_with_answer_{option}_options.jsonl'
        prompt_records = []
        with open(qst_ans_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                prompt_records.append({
                    "question": data["question"],
                    "answer": data["answer"],
                })
        return prompt_records, list(range(len(prompt_records)))

    def run_single_prompt(
        self,
        image,
        prompt,
        method,
        weight,
        threshold=1.0,
        weight1=1.0,
        weight2=1.0,
        return_trace=False,
        trace_topk=10,
        attn_target_text=None,
        attn_target_name=None,
        attn_layer=None,
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device
        inputs = {
            k: (v.to(model_device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=20,
        )

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        gen = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        if return_trace:
            return gen, []
        return gen
