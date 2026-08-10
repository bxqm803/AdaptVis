#!/usr/bin/env python3
"""
Centroid Top20 Head A/B Swap Generation Experiment

Goal:
    Find whether high-centroid heads contain
    causal A/B spatial-role information.

Procedure:
    1. Rank heads by centroid accuracy.
    2. For each head:
        Q_BA prefill
        swap this head's A/B token output
        continue generation
    3. Measure generation changes.
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm


# --------------------------------------------------
# head ranking
# --------------------------------------------------

def load_top_heads(summary_file, topk=20):

    data = json.load(open(summary_file))

    heads = data["top_attention_heads_by_accuracy"]

    result = []

    for x in heads[:topk]:

        result.append(
            (
                int(x["layer"]),
                int(x["head"]),
                float(x["accuracy"])
            )
        )

    return result


# --------------------------------------------------
# relation parser
# --------------------------------------------------

def normalize_answer(x):

    x = x.lower()

    if "left" in x:
        return "left"

    if "right" in x:
        return "right"

    if "above" in x:
        return "above"

    if "below" in x:
        return "below"

    return "unknown"



def opposite_relation(x):

    mp = {
        "left":"right",
        "right":"left",
        "above":"below",
        "below":"above",
    }

    return mp.get(x,"unknown")



# --------------------------------------------------
# A/B head swap hook
# --------------------------------------------------

class HeadSwapHook:


    def __init__(
        self,
        layer,
        head,
        head_dim,
        token_A,
        token_B
    ):

        self.layer = layer
        self.head = head
        self.head_dim = head_dim

        self.token_A = token_A
        self.token_B = token_B


    def hook(self,module,args):

        x=args[0]

        """
        x:
            [batch, seq, hidden]

        before o_proj

        hidden =
            n_heads * head_dim
        """

        h0=self.head*self.head_dim
        h1=(self.head+1)*self.head_dim


        xa=x[:,self.token_A,h0:h1].clone()

        xb=x[:,self.token_B,h0:h1].clone()


        x[:,self.token_A,h0:h1]=xb

        x[:,self.token_B,h0:h1]=xa


        return (x,)+args[1:]



# --------------------------------------------------
# generation
# --------------------------------------------------

def generate_with_swap(
    model,
    tokenizer,
    inputs,
    layer,
    head,
    token_A,
    token_B,
):


    block=model.model.layers[layer]


    hooker=HeadSwapHook(
        layer,
        head,
        model.config.hidden_size //
        model.config.num_attention_heads,
        token_A,
        token_B
    )


    handle=(
        block.self_attn.o_proj
        .register_forward_pre_hook(
            hooker.hook
        )
    )


    try:

        with torch.no_grad():

            out=model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )

    finally:

        handle.remove()


    return out



# --------------------------------------------------
# evaluation
# --------------------------------------------------

def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--centroid-summary",
        required=True
    )

    parser.add_argument(
        "--topk",
        default=20,
        type=int
    )

    parser.add_argument(
        "--output",
        default="centroid_head_swap_result.json"
    )


    args=parser.parse_args()


    top_heads=load_top_heads(
        args.centroid_summary,
        args.topk
    )


    print(
        "Testing heads:"
    )

    for h in top_heads:
        print(h)



    results=[]


    for layer,head,centroid_acc in top_heads:


        print(
            f"testing L{layer}H{head}"
        )


        # TODO:
        #
        # 这里接你的 coco loader
        # 和原 knockout script
        #
        # 逻辑：
        #
        # baseline = generate(Q_BA)
        #
        # patched =
        # generate_with_swap(
        #     Q_BA,
        #     layer,
        #     head
        # )
        #
        # compare


        results.append(
            {
                "layer":layer,
                "head":head,
                "centroid_acc":centroid_acc,
            }
        )


    json.dump(
        results,
        open(args.output,"w"),
        indent=2
    )



if __name__=="__main__":

    main()
