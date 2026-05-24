import argparse
import os
import json
import pandas as pd
import pdb
from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores
import numpy as np
import random
from torch.utils.data import DataLoader
import torch


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
    parser.add_argument("--option", default="four", type=str, choices=["two", "four", "six"])

    return parser.parse_args()


def _load_sample_ids_from_env():
    """
    Optional subset runner.

    Usage:
        export RUN_SAMPLE_IDS_FILE=lowprob_lt0p4_ids.json

    The json should be a list of original dataset indices, for example:
        [0, 1, 4, 11, ...]
    """
    sample_ids_file = os.environ.get("RUN_SAMPLE_IDS_FILE", "").strip()

    if not sample_ids_file:
        return None

    if not os.path.exists(sample_ids_file):
        raise FileNotFoundError(f"RUN_SAMPLE_IDS_FILE not found: {sample_ids_file}")

    with open(sample_ids_file, "r", encoding="utf-8") as f:
        sample_ids = json.load(f)

    sample_ids = [int(x) for x in sample_ids]

    print(f"[SUBSET] RUN_SAMPLE_IDS_FILE={sample_ids_file}")
    print(f"[SUBSET] num samples={len(sample_ids)}")
    print(f"[SUBSET] first 30 ids={sample_ids[:30]}")

    return sample_ids


def main(args):
    seed_all(args.seed)

    model, image_preprocess = get_model(args.model_name, args.device, args.method)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

    SAMPLE = False
    TEST = os.getenv("TEST_MODE", "False") == "True"

    sampled_indices = None
    collate_fn = _default_collate if image_preprocess is None else None

    # ------------------------------------------------------------
    # Highest priority: run explicit sample ids from env.
    # This is used for low-probability subset experiments.
    # ------------------------------------------------------------
    run_sample_ids = _load_sample_ids_from_env()

    if run_sample_ids is not None:
        sampled_indices = run_sample_ids
        sub_dataset = torch.utils.data.Subset(dataset, sampled_indices)

        joint_loader = DataLoader(
            sub_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    else:
        # ------------------------------------------------------------
        # Original SAMPLE split logic.
        # ------------------------------------------------------------
        if SAMPLE is True:
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
                os.makedirs("./output", exist_ok=True)
                np.save(idx_file_path, np.array(sampled_indices))

            all_indices = set(range(total_data_count))

            # use test set
            if TEST is True:
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

        # ------------------------------------------------------------
        # Original full-set logic.
        # ------------------------------------------------------------
        else:
            joint_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )

    print(args.dataset, args.model_name)

    if sampled_indices is not None:
        print(f"[RUNNING SUBSET] {len(sampled_indices)} samples")
    else:
        print(f"[RUNNING FULL DATASET] {len(dataset)} samples")

    if args.dataset == "VSR":
        labels = dataset.get_labels()

        if sampled_indices is not None:
            labels = [labels[i] for i in sampled_indices]

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

    elif args.dataset in ["Controlled_Images_B", "Controlled_Images_A"]:
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

        # change from (N, 4, 1) to (N, 1, 4)
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

        # dataset.save_scores(
        #     scores,
        #     correct_id,
        #     args.output_dir,
        #     args.dataset,
        #     args.method,
        #     args.weight,
        #     args.model_name,
        #     args.option,
        # )

    else:
        scores, correct_id = model.get_out_scores_wh_batched(
            args.dataset,
            joint_loader,
            args.method,
            args.weight,
            args.option,
        )

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
