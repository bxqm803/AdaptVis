#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract VG spatial relation hidden states.

Input:
data/vg/
    vg_spatial_6.json
    images/*.jpg

Each json item:
{
    "image_id": 123,
    "subject": "dog",
    "object": "car",
    "relation": "behind",
    "split": "train"
}

Output npz:
- subject_states: [N, L, D]
- reference_states: [N, L, D]
- relation_vectors: [N, L, D]
"""

import argparse
import gc
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo_id: str
    model_class: str
    dtype: torch.dtype


SPECS = {
    "llava-7b": ModelSpec(
        "llava-7b",
        "llava-hf/llava-1.5-7b-hf",
        "LlavaForConditionalGeneration",
        torch.float16,
    ),
    "llava-13b": ModelSpec(
        "llava-13b",
        "llava-hf/llava-1.5-13b-hf",
        "LlavaForConditionalGeneration",
        torch.float16,
    ),
    "qwen-3b": ModelSpec(
        "qwen-3b",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen2_5_VLForConditionalGeneration",
        torch.bfloat16,
    ),
    "qwen-7b": ModelSpec(
        "qwen-7b",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen2_5_VLForConditionalGeneration",
        torch.bfloat16,
    ),
}


def args():
    p = argparse.ArgumentParser()

    p.add_argument("--data-root", required=True)
    p.add_argument("--model", required=True, choices=SPECS.keys())
    p.add_argument("--split", choices=["train", "test", "all"], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    p.add_argument("--layer-fracs",
                   default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0")

    return p.parse_args()


def find_span_last(tokenizer, ids, word):

    for text in [" " + word, word]:

        token_ids = tokenizer(
            text,
            add_special_tokens=False
        ).input_ids

        n=len(token_ids)

        for i in range(len(ids)-n+1):
            if ids[i:i+n] == token_ids:
                return i+n-1

    raise RuntimeError(f"cannot find token {word}")


def load_data(root, split):

    path = Path(root)/"vg_spatial_6.json"

    data=json.loads(path.read_text())

    out=[]

    for i,x in enumerate(data):

        if split!="all" and x["split"]!=split:
            continue

        out.append(x)

    return out


def hidden_states(out):

    hs=getattr(out,"hidden_states",None)

    if hs is None:
        raise RuntimeError("hidden_states missing")

    return hs


def main():

    a=args()

    spec=SPECS[a.model]

    data=load_data(a.data_root,a.split)

    print("samples:",len(data))


    import transformers

    cls=getattr(
        transformers,
        spec.model_class
    )


    model=cls.from_pretrained(
        spec.repo_id,
        torch_dtype=spec.dtype,
        device_map={"":a.device}
    )

    model.eval()

    processor=AutoProcessor.from_pretrained(
        spec.repo_id
    )


    device=torch.device(a.device)


    subject_states=[]
    reference_states=[]
    relation_vectors=[]

    image_ids=[]
    subjects=[]
    references=[]
    relations=[]


    fractions=[
        float(x)
        for x in a.layer_fracs.split(",")
    ]


    selected=None


    for item in tqdm(data):

        image_path=(
            Path(a.data_root)
            /
            "images"
            /
            f"{item['image_id']}.jpg"
        )

        if not image_path.exists():
            continue


        image=Image.open(image_path).convert("RGB")


        prompt=(
            f"Where is the {item['subject']} "
            f"relative to the {item['object']}? "
            "Answer with one spatial relation."
        )


        messages=[
            {
                "role":"user",
                "content":[
                    {"type":"image"},
                    {"type":"text","text":prompt}
                ]
            }
        ]


        text=processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


        batch=processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        )


        batch={
            k:v.to(device)
            if torch.is_tensor(v)
            else v
            for k,v in batch.items()
        }


        ids=batch["input_ids"][0].cpu().tolist()

        sidx=find_span_last(
            processor.tokenizer,
            ids,
            item["subject"]
        )

        oidx=find_span_last(
            processor.tokenizer,
            ids,
            item["object"]
        )


        with torch.no_grad():

            out=model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )


        hs=hidden_states(out)

        nblock=len(hs)-1


        if selected is None:
            selected=[
                min(
                    nblock-1,
                    round(x*(nblock-1))
                )
                for x in fractions
            ]

            selected=sorted(set(selected))

            print("layers:",selected)


        sv=np.stack([
            hs[l+1][0,sidx].float().cpu().numpy()
            for l in selected
        ])

        ov=np.stack([
            hs[l+1][0,oidx].float().cpu().numpy()
            for l in selected
        ])


        subject_states.append(sv)
        reference_states.append(ov)
        relation_vectors.append(sv-ov)

        image_ids.append(str(item["image_id"]))
        subjects.append(item["subject"])
        references.append(item["object"])
        relations.append(item["relation"])


    output=Path(a.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output,
        image_id=np.array(image_ids,dtype=object),
        subject=np.array(subjects,dtype=object),
        reference=np.array(references,dtype=object),
        relation=np.array(relations,dtype=object),
        subject_states=np.stack(subject_states).astype(np.float16),
        reference_states=np.stack(reference_states).astype(np.float16),
        relation_vectors=np.stack(relation_vectors).astype(np.float16),
    )


    print("saved:",output)
    print("N:",len(image_ids))


    del model,processor
    gc.collect()
    torch.cuda.empty_cache()


if __name__=="__main__":
    main()
