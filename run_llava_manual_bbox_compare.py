import os
import re
import csv
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from dataset_zoo import get_dataset

MODEL_DEFAULT = 'llava-hf/llava-1.5-7b-hf'
REVISION_DEFAULT = 'a272c74'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manual-bbox-csv', required=True, type=str)
    p.add_argument('--dataset', default='Controlled_Images_A', type=str)
    p.add_argument('--option', default='four', type=str)
    p.add_argument('--download', action='store_true')
    p.add_argument('--model-id', default=MODEL_DEFAULT, type=str)
    p.add_argument('--revision', default=REVISION_DEFAULT, type=str)
    p.add_argument('--device', default='cuda', type=str)
    p.add_argument('--cache-dir', default=None, type=str)
    p.add_argument('--out-dir', default='output_llava_manual_bbox_compare', type=str)
    p.add_argument('--max-new-tokens', default=16, type=int)
    p.add_argument('--temperature', default=0.0, type=float)
    p.add_argument('--skip-existing', action='store_true')
    return p.parse_args()


def clean_text(x):
    x = '' if x is None else str(x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x


def strip_legacy_prompt(prompt: str) -> str:
    prompt = clean_text(prompt)
    prompt = prompt.replace('<image>', '').strip()
    prompt = re.sub(r'^\s*USER:\s*', '', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'\s*ASSISTANT:\s*$', '', prompt, flags=re.IGNORECASE)
    return clean_text(prompt)


def normalize_rel(answer):
    if isinstance(answer, (list, tuple)):
        if len(answer) == 0:
            return 'UNK'
        answer = answer[0]
    if answer is None:
        return 'UNK'
    rel = str(answer).strip().lower()
    mapping = {
        'left': 'left',
        'right': 'right',
        'on': 'on',
        'under': 'under',
        'below': 'under',
        'beneath': 'under',
        'top': 'on',
        'above': 'on',
        'to the left of': 'left',
        'to the right of': 'right',
        'on top of': 'on',
    }
    return mapping.get(rel, rel)


def parse_prediction(text: str):
    t = clean_text(text).lower()
    m = re.search(r'\b(left|right|on|under)\b', t)
    return m.group(1) if m else 'UNK'


def parse_bbox_string(x):
    x = clean_text(x)
    if not x:
        return None
    try:
        val = json.loads(x)
    except Exception:
        return None
    if not isinstance(val, list) or len(val) != 4:
        return None
    try:
        return [int(round(float(v))) for v in val]
    except Exception:
        return None


def normalize_object_name(name: str) -> str:
    name = clean_text(name).lower()
    name = name.replace('-', ' ').replace('_', ' ')
    name = re.sub(r'^(a|an|the)\s+', '', name)
    name = re.sub(r'[?.!,;:]+$', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def infer_objects_from_image_name(image_name: str):
    stem = os.path.splitext(os.path.basename(image_name))[0].lower()
    markers = ['_left_of_', '_right_of_', '_on_', '_under_']
    for marker in markers:
        if marker in stem:
            a, b = stem.split(marker, 1)
            return normalize_object_name(a), normalize_object_name(b)
    return 'object_1', 'object_2'


def load_manual_bbox_rows(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = clean_text(row.get('image_name', ''))
            if not image_name:
                continue

            status = clean_text(row.get('status', 'confirm')).lower()
            obj1_name = clean_text(row.get('object_1_name', ''))
            obj2_name = clean_text(row.get('object_2_name', ''))
            if not obj1_name or not obj2_name:
                obj1_name, obj2_name = infer_objects_from_image_name(image_name)

            obj1_bbox = parse_bbox_string(row.get('object_1_bbox', ''))
            obj2_bbox = parse_bbox_string(row.get('object_2_bbox', ''))

            rows.append({
                'image_name': image_name,
                'image_path': clean_text(row.get('image_path', '')),
                'object_1_name': obj1_name,
                'object_2_name': obj2_name,
                'object_1_bbox': obj1_bbox,
                'object_2_bbox': obj2_bbox,
                'status': status,
            })
    return rows


def load_prompt_records(dataset_name: str, option: str):
    prompt_path = Path('prompts') / f'{dataset_name}_with_answer_{option}_options.jsonl'
    if not prompt_path.exists():
        raise FileNotFoundError(f'Prompt file not found: {prompt_path}')
    records = []
    with open(prompt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_manual_bbox_prompt(base_question, image_w, image_h, obj1_name, obj1_bbox, obj2_name, obj2_bbox):
    return (
        'Use the image as the primary evidence. You are also given manually annotated bounding boxes '
        'for two objects in the same original image. '\
        f'Original image size: width={image_w}, height={image_h}. '\
        f'Object 1: {obj1_name}, bbox=[x1,y1,x2,y2]={json.dumps(obj1_bbox)}. '\
        f'Object 2: {obj2_name}, bbox=[x1,y1,x2,y2]={json.dumps(obj2_bbox)}. '\
        'These coordinates are original-image pixel coordinates, with origin at the top-left. '\
        'Use both the image and the bounding boxes to answer the question. '\
        f'Question: {base_question} '\
        'Answer with left, right, on or under only.'
    )


@torch.no_grad()
def run_llava_one(model, processor, image, question_text, max_new_tokens=16, temperature=0.0):
    prompt = f'USER: <image>\n{question_text}\nASSISTANT:'
    inputs = processor(images=image, text=prompt, return_tensors='pt')

    model_device = next(model.parameters()).device
    moved = {}
    model_dtype = next(model.parameters()).dtype
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if v.is_floating_point():
                moved[k] = v.to(model_device, dtype=model_dtype)
            else:
                moved[k] = v.to(model_device)
        else:
            moved[k] = v

    gen_kwargs = {
        'max_new_tokens': max_new_tokens,
        'pad_token_id': processor.tokenizer.eos_token_id,
        'do_sample': bool(temperature and temperature > 0),
    }
    if temperature and temperature > 0:
        gen_kwargs['temperature'] = temperature

    output = model.generate(**moved, **gen_kwargs)
    text = processor.batch_decode(output, skip_special_tokens=True)[0]
    if 'ASSISTANT:' in text:
        text = text.split('ASSISTANT:')[-1].strip()
    else:
        text = text.strip()
    return text


def load_model_and_processor(model_id, revision, cache_dir, device):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if device.startswith('cuda') and torch.cuda.is_available() else torch.float32,
        device_map='auto' if device.startswith('cuda') and torch.cuda.is_available() else None,
        ignore_mismatched_sizes=True,
    ).eval()
    if not (device.startswith('cuda') and torch.cuda.is_available()):
        model.to(device)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    return model, processor


def main():
    args = parse_args()

    cache_dir = args.cache_dir or f"/ddnB/work/{os.environ.get('USER', 'user')}/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault('HF_HOME', cache_dir)
    os.environ.setdefault('HF_HUB_CACHE', os.path.join(cache_dir, 'hub'))
    os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(cache_dir, 'transformers'))
    os.environ.setdefault('XDG_CACHE_HOME', cache_dir)

    manual_rows = load_manual_bbox_rows(args.manual_bbox_csv)
    # keep only confirmed rows with both bboxes
    manual_rows = [
        r for r in manual_rows
        if r['status'] != 'skip' and r['object_1_bbox'] is not None and r['object_2_bbox'] is not None
    ]
    if not manual_rows:
        raise ValueError('No usable manual bbox rows found.')

    manual_by_name = {r['image_name']: r for r in manual_rows}

    prompt_records = load_prompt_records(args.dataset, args.option)
    dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download)
    if len(prompt_records) != len(dataset):
        raise ValueError(f'Prompt count ({len(prompt_records)}) != dataset size ({len(dataset)}).')

    selected = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        image_name = item.get('image_name', '')
        if image_name in manual_by_name:
            rec = prompt_records[idx]
            selected.append({
                'local_index': idx,
                'image_name': image_name,
                'image_path': item.get('image_path', ''),
                'image': item['image_options'][0],
                'base_question': strip_legacy_prompt(rec['question']),
                'gold': normalize_rel(rec['answer']),
                'manual': manual_by_name[image_name],
            })

    if not selected:
        raise ValueError('No overlap found between manual bbox csv and dataset.')

    model, processor = load_model_and_processor(args.model_id, args.revision, cache_dir, args.device)

    out_root = os.path.join(args.out_dir, args.dataset, args.model_id.split('/')[-1])
    os.makedirs(out_root, exist_ok=True)
    out_csv = os.path.join(out_root, 'summary_manual_bbox_compare.csv')

    summary_rows = []
    for ex in tqdm(selected, desc='manual-bbox-compare'):
        image_name = ex['image_name']
        image_stem = os.path.splitext(image_name)[0]
        sample_dir = os.path.join(out_root, image_stem)
        os.makedirs(sample_dir, exist_ok=True)
        sample_json = os.path.join(sample_dir, 'compare.json')

        if args.skip_existing and os.path.exists(sample_json):
            continue

        img = ex['image']
        image_w, image_h = img.size
        manual = ex['manual']

        baseline_question = ex['base_question']
        bbox_question = build_manual_bbox_prompt(
            base_question=ex['base_question'],
            image_w=image_w,
            image_h=image_h,
            obj1_name=manual['object_1_name'],
            obj1_bbox=manual['object_1_bbox'],
            obj2_name=manual['object_2_name'],
            obj2_bbox=manual['object_2_bbox'],
        )

        baseline_text = run_llava_one(
            model=model,
            processor=processor,
            image=img,
            question_text=baseline_question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        bbox_text = run_llava_one(
            model=model,
            processor=processor,
            image=img,
            question_text=bbox_question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        baseline_pred = parse_prediction(baseline_text)
        bbox_pred = parse_prediction(bbox_text)
        baseline_correct = baseline_pred == ex['gold']
        bbox_correct = bbox_pred == ex['gold']

        payload = {
            'image_name': image_name,
            'image_path': ex['image_path'],
            'local_index': ex['local_index'],
            'gold': ex['gold'],
            'object_1_name': manual['object_1_name'],
            'object_2_name': manual['object_2_name'],
            'object_1_bbox': manual['object_1_bbox'],
            'object_2_bbox': manual['object_2_bbox'],
            'baseline_question': baseline_question,
            'bbox_question': bbox_question,
            'baseline_text': baseline_text,
            'bbox_text': bbox_text,
            'baseline_pred': baseline_pred,
            'bbox_pred': bbox_pred,
            'baseline_correct': baseline_correct,
            'bbox_correct': bbox_correct,
        }
        with open(sample_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        summary_rows.append({
            'image_name': image_name,
            'image_path': ex['image_path'],
            'local_index': ex['local_index'],
            'gold': ex['gold'],
            'object_1_name': manual['object_1_name'],
            'object_2_name': manual['object_2_name'],
            'object_1_bbox': json.dumps(manual['object_1_bbox'], ensure_ascii=False),
            'object_2_bbox': json.dumps(manual['object_2_bbox'], ensure_ascii=False),
            'baseline_pred': baseline_pred,
            'baseline_correct': baseline_correct,
            'bbox_pred': bbox_pred,
            'bbox_correct': bbox_correct,
            'baseline_text': baseline_text,
            'bbox_text': bbox_text,
            'delta': int(bbox_correct) - int(baseline_correct),
            'sample_json': os.path.relpath(sample_json, out_root),
        })

        fieldnames = [
            'image_name', 'image_path', 'local_index', 'gold',
            'object_1_name', 'object_2_name', 'object_1_bbox', 'object_2_bbox',
            'baseline_pred', 'baseline_correct',
            'bbox_pred', 'bbox_correct',
            'baseline_text', 'bbox_text',
            'delta', 'sample_json',
        ]
        with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    n = len(summary_rows)
    baseline_acc = sum(int(r['baseline_correct']) for r in summary_rows) / n if n else 0.0
    bbox_acc = sum(int(r['bbox_correct']) for r in summary_rows) / n if n else 0.0
    improved = sum(1 for r in summary_rows if int(r['delta']) > 0)
    worsened = sum(1 for r in summary_rows if int(r['delta']) < 0)

    report = {
        'num_images': n,
        'baseline_acc': baseline_acc,
        'bbox_acc': bbox_acc,
        'improved_count': improved,
        'worsened_count': worsened,
    }
    with open(os.path.join(out_root, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'Saved summary to: {out_csv}')
    print(f'Baseline acc: {baseline_acc:.4f}')
    print(f'BBox-prompt acc: {bbox_acc:.4f}')
    print(f'Improved: {improved} | Worsened: {worsened}')


if __name__ == '__main__':
    main()
