#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-find the best late last-token causal steering layers for Qwen3B / LLaVA7B.

Method:
  TRAIN is internally split into FIT/VAL.
  FIT learns, at every decoder layer l:
      d_i,l = h_last(real) - h_last(gray)
      s_r,l = mean(d_i,l | relation=r) - balanced_global_mean_l

  VAL uses oracle GT only to SEARCH the actuator location:
      1) scan every single decoder layer
      2) scan short contiguous windows around top single layers
      3) choose the best window by actual model.generate() accuracy

  Then templates are refit on FULL TRAIN using only the selected layers.
  A residual-Direction guide window is also selected on VAL from vectors.npz.
  Final TEST evaluates:
      oracle_all_add        # actuator upper bound only
      guide_all_add         # non-oracle
      guide_conflict_add    # non-oracle
      guide_all_contrast    # non-oracle
      guide_conflict_contrast

The script edits hidden[:, -1, :] on the FIRST decoder prefill call.
This is intentional and works for LLaVA's merged decoder sequence too.
"""

import argparse, contextlib, csv, gc, json, math, random, re, shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

RELS = ("left", "right", "above", "below")
RELSET = set(RELS)
EPS = 1e-12


# ---------------------------------------------------------------------
# basic
# ---------------------------------------------------------------------

def args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--model-id", required=True)
    p.add_argument("--direction-dir", required=True)
    p.add_argument("--direction-key", default="residual")

    p.add_argument("--prompt-jsonl",
                   default="prompts/COCO_QA_two_obj_with_answer_four_options.jsonl")
    p.add_argument("--annotation-json", default="data/coco_qa_two_obj.json")
    p.add_argument("--data-root", default="data")

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2", "none"])

    p.add_argument("--prompt-template", default=(
        "Determine the spatial relation of the {subject} to the {reference} "
        "in the image. Answer with left, right, above, or below."
    ))

    p.add_argument("--scan-layers", default="all",
                   help="all | late_half | explicit e.g. 10-35")
    p.add_argument("--template-filter", default="real_correct_gray_wrong",
                   choices=["real_correct_gray_wrong", "real_correct", "all"])

    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--min-val-per-relation", type=int, default=4)
    p.add_argument("--top-k-single", type=int, default=5)
    p.add_argument("--window-lengths", default="2,3,4")

    p.add_argument("--guide-window-lengths", default="1,2,3,4,5,6,7")
    p.add_argument("--guide-train-controls", default="correct",
                   choices=["correct", "all"])

    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--gray-value", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--max-fit", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    vals = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def rel(x):
    s = str(x).strip().lower()
    if re.search(r"\bleft\b", s): return "left"
    if re.search(r"\bright\b", s): return "right"
    if re.search(r"\babove\b", s) or re.search(r"\bover\b", s): return "above"
    if re.search(r"\bbelow\b", s) or re.search(r"\bunder", s) or re.search(r"\bbeneath\b", s):
        return "below"
    return s


def int_list(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def layer_spec(spec, n):
    spec = str(spec).strip().lower()
    if spec == "all":
        return list(range(n))
    if spec == "late_half":
        return list(range(n // 2, n))

    out = []
    for x in spec.split(","):
        x = x.strip()
        if not x:
            continue
        if "-" in x:
            a, b = map(int, x.split("-", 1))
            step = 1 if b >= a else -1
            out += list(range(a, b + step, step))
        else:
            out.append(int(x))

    out = sorted(set(out))
    bad = [x for x in out if x < 0 or x >= n]
    if bad:
        raise ValueError(f"bad layers {bad}; valid 0..{n-1}")
    return out


def choose_dtype(model_id, dtype):
    if dtype != "auto":
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
    return torch.float16 if "llava" in model_id.lower() else torch.bfloat16


# ---------------------------------------------------------------------
# saved Direction bundle + COCO records
# ---------------------------------------------------------------------

def load_bundle(root, key):
    root = Path(root)
    with np.load(root / "vectors.npz", allow_pickle=True) as z:
        sids = np.asarray(z["sample_index"], dtype=np.int64)
        labels = np.asarray([rel(x) for x in z["relation"]], dtype=object)
        arr = np.asarray(z[key], dtype=np.float32)

    n = len(sids)
    if arr.shape[0] == n:
        vecs = arr
    elif arr.shape[1] == n:
        vecs = np.transpose(arr, (1, 0, 2))
    else:
        raise RuntimeError(f"cannot align vectors {arr.shape} with N={n}")

    split, group = {}, {}
    for r in read_csv(root / "sample_split_and_generation.csv"):
        sid = int(r["sample_index"])
        split[sid] = str(r.get("split", "")).strip().lower()
        group[sid] = str(r.get("generation_group", "")).strip().lower()

    return dict(
        sids=sids, labels=labels, vecs=vecs, split=split, group=group,
        sid2i={int(s): i for i, s in enumerate(sids.tolist())},
    )


def parse_sr(question):
    q = str(question)
    for pat in [
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?\s*Answer",
        r"Where\s+is\s+(.+?)\s+in\s+relation\s+to\s+(.+?)\?",
    ]:
        m = re.search(pat, q, flags=re.I | re.S)
        if m:
            return (
                re.sub(r"\s+", " ", m.group(1)).strip(),
                re.sub(r"\s+", " ", m.group(2)).strip(),
            )
    return None, None


def load_records(prompt_jsonl, ann_json, data_root, bundle):
    with open(ann_json, "r", encoding="utf-8") as f:
        anns = json.load(f)

    prompts = []
    with open(prompt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    recs = {}

    for row_no, row in enumerate(prompts):
        sid = int(row.get("id", row_no))
        if sid not in bundle["sid2i"] or not (0 <= sid < len(anns)):
            continue

        sub, ref = parse_sr(row.get("question", ""))
        if sub is None:
            continue

        gt = rel(row.get("answer", ""))
        if gt not in RELSET:
            gt = str(bundle["labels"][bundle["sid2i"][sid]])
        if gt not in RELSET:
            continue

        image_id = int(anns[sid][0])
        image_path = Path(data_root) / "val2017" / f"{image_id:012d}.jpg"

        recs[sid] = dict(
            sid=sid, gt=gt, subject=sub, reference=ref,
            split=bundle["split"].get(sid, ""),
            image_path=str(image_path),
        )

    print(f"[records] N={len(recs)}")
    return recs


def stratified_split(records, seed, val_frac, min_val):
    rng = random.Random(seed)
    fit, val = [], []

    for r in RELS:
        xs = sorted(
            sid for sid, rec in records.items()
            if rec["split"] == "train" and rec["gt"] == r
        )
        rng.shuffle(xs)

        if len(xs) < 2:
            raise RuntimeError(f"too few TRAIN samples for {r}: {len(xs)}")

        nv = max(min_val, int(round(len(xs) * val_frac)))
        nv = min(nv, len(xs) - 1)

        val += xs[:nv]
        fit += xs[nv:]

    return sorted(fit), sorted(val)


# ---------------------------------------------------------------------
# model / processor
# ---------------------------------------------------------------------

def attr_path(obj, path):
    for x in path.split("."):
        obj = getattr(obj, x)
    return obj


def decoder_layers(model):
    for path in [
        "model.language_model.layers",          # Qwen2.5-VL
        "language_model.layers",
        "language_model.model.layers",          # HF LLaVA
        "model.language_model.model.layers",
        "model.model.layers",
        "model.layers",
    ]:
        try:
            layers = attr_path(model, path)
            if len(layers):
                return layers, path
        except Exception:
            pass
    raise RuntimeError("decoder layers not found")


def model_class(model_id):
    low = model_id.lower()

    if "qwen2.5" in low or "qwen2_5" in low:
        return transformers.Qwen2_5_VLForConditionalGeneration

    if "llava" in low:
        if ("1.6" in low or "next" in low) and hasattr(transformers, "LlavaNextForConditionalGeneration"):
            return transformers.LlavaNextForConditionalGeneration
        if hasattr(transformers, "LlavaForConditionalGeneration"):
            return transformers.LlavaForConditionalGeneration

    if hasattr(transformers, "AutoModelForImageTextToText"):
        return transformers.AutoModelForImageTextToText

    return transformers.AutoModelForVision2Seq


def load_model(args):
    cls = model_class(args.model_id)
    dtype = choose_dtype(args.model_id, args.dtype)

    kw = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    )
    if args.attn_impl != "none":
        kw["attn_implementation"] = args.attn_impl

    print(f"[model] {cls.__name__} | {args.model_id} | dtype={dtype}")

    try:
        model = cls.from_pretrained(args.model_id, dtype=dtype, **kw)
    except TypeError:
        model = cls.from_pretrained(args.model_id, torch_dtype=dtype, **kw)

    model.eval()

    try:
        proc = AutoProcessor.from_pretrained(
            args.model_id, trust_remote_code=True, use_fast=False
        )
    except TypeError:
        proc = AutoProcessor.from_pretrained(
            args.model_id, trust_remote_code=True
        )

    return model, proc


def prompt_for(proc, image, question, model_id):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]

    try:
        return proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        pass

    if "llava" in model_id.lower():
        return f"USER: <image>\n{question}\nASSISTANT:"

    try:
        return proc.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return question


def batch_for(proc, image, rec, args):
    q = args.prompt_template.format(
        subject=rec["subject"],
        reference=rec["reference"],
    )
    prompt = prompt_for(proc, image, q, args.model_id)

    last_error = None
    for fn in [
        lambda: proc(text=[prompt], images=[image], padding=True, return_tensors="pt"),
        lambda: proc(text=prompt, images=image, return_tensors="pt"),
    ]:
        try:
            batch = fn()
            break
        except Exception as e:
            last_error = e
    else:
        raise RuntimeError(f"processor failed: {last_error}")

    return {
        k: (v.to(args.device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def parse_pred(text):
    s = str(text).lower()
    hits = []
    for r, pat in [
        ("left", r"\bleft\b"),
        ("right", r"\bright\b"),
        ("above", r"\babove\b"),
        ("below", r"\bbelow\b"),
        ("below", r"\bunder(?:neath)?\b"),
        ("below", r"\bbeneath\b"),
    ]:
        m = re.search(pat, s)
        if m:
            hits.append((m.start(), r))
    return sorted(hits)[0][1] if hits else None


def generate(model, proc, batch, max_new):
    n_in = int(batch["input_ids"].shape[1])

    with torch.inference_mode():
        ids = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new,
            use_cache=True,
        )

    gen = ids[0, n_in:] if ids.shape[1] > n_in else ids[0]
    text = proc.tokenizer.decode(gen, skip_special_tokens=True).strip()
    pred = parse_pred(text)
    del ids
    return text, pred


# ---------------------------------------------------------------------
# block-output hook
# ---------------------------------------------------------------------

def get_hidden(out):
    if torch.is_tensor(out):
        return out, ("tensor", 0)

    if isinstance(out, tuple):
        for i, x in enumerate(out):
            if torch.is_tensor(x):
                return x, ("tuple", i)

    if isinstance(out, list):
        for i, x in enumerate(out):
            if torch.is_tensor(x):
                return x, ("list", i)

    raise RuntimeError(f"cannot get hidden from {type(out)}")


def put_hidden(out, desc, h):
    kind, i = desc
    if kind == "tensor":
        return h
    xs = list(out)
    xs[i] = h
    return tuple(xs) if kind == "tuple" else xs


class CaptureLast:
    def __init__(self, layers, selected):
        self.handles, self.states = [], {}
        self.done = {l: False for l in selected}
        for l in selected:
            self.handles.append(
                layers[l].register_forward_hook(self.hook(l))
            )

    def hook(self, l):
        def f(_m, _inp, out):
            if self.done[l]:
                return out
            h, _ = get_hidden(out)
            if h.ndim != 3:
                return out
            self.states[l] = h[0, -1, :].detach().float().cpu().numpy().astype(np.float32)
            self.done[l] = True
            return out
        return f

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


class SteerLast:
    def __init__(self, layers, templates, selected, target, scale,
                 mode="add", source=None):
        self.handles = []
        self.done = {l: False for l in selected}
        self.templates = templates
        self.target = target
        self.scale = scale
        self.mode = mode
        self.source = source

        for l in selected:
            self.handles.append(
                layers[l].register_forward_hook(self.hook(l))
            )

    def vec(self, l):
        t = np.asarray(self.templates[l]["shared"][self.target], dtype=np.float32)

        if self.mode == "add":
            return self.scale * t

        if self.source not in RELSET:
            return self.scale * t

        s = np.asarray(self.templates[l]["shared"][self.source], dtype=np.float32)
        return self.scale * (t - s)

    def hook(self, l):
        def f(_m, _inp, out):
            if self.done[l]:
                return out

            h, desc = get_hidden(out)
            if h.ndim != 3:
                return out

            v = torch.as_tensor(
                self.vec(l), device=h.device, dtype=h.dtype
            )

            y = h.clone()
            y[:, -1, :] += v[None, :]
            self.done[l] = True
            return put_hidden(out, desc, y)

        return f

    def close(self):
        for h in reversed(self.handles):
            with contextlib.suppress(Exception):
                h.remove()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


# ---------------------------------------------------------------------
# causal-template learning
# ---------------------------------------------------------------------

def gray_image(real, value):
    v = int(max(0, min(255, value)))
    return Image.new("RGB", real.size, (v, v, v))


def gen_capture(model, proc, layers, image, rec, selected, args):
    b = batch_for(proc, image, rec, args)

    with CaptureLast(layers, selected) as cap:
        text, pred = generate(model, proc, b, args.max_new_tokens)
        states = dict(cap.states)

    del b
    return pred, text, states


def allowed(rc, gc, mode):
    if mode == "all": return True
    if mode == "real_correct": return bool(rc)
    return bool(rc) and not bool(gc)


def collect(model, proc, layers, records, sids, selected, args, desc):
    bag = {
        l: {r: [] for r in RELS}
        for l in selected
    }
    rows = []

    for sid in tqdm(sids, desc=desc):
        rec = records[sid]
        real = gray = None

        try:
            real = Image.open(rec["image_path"]).convert("RGB")
            gray = gray_image(real, args.gray_value)

            rp, rt, rs = gen_capture(
                model, proc, layers, real, rec, selected, args
            )
            gp, gt, gs = gen_capture(
                model, proc, layers, gray, rec, selected, args
            )

            rc = int(rp == rec["gt"])
            gc = int(gp == rec["gt"])
            use = int(allowed(rc, gc, args.template_filter))

            rows.append(dict(
                sid=sid, gt=rec["gt"],
                real_pred=rp or "", gray_pred=gp or "",
                real_correct=rc, gray_correct=gc, used=use,
            ))

            if use:
                for l in selected:
                    if l in rs and l in gs:
                        bag[l][rec["gt"]].append(
                            (rs[l] - gs[l]).astype(np.float32)
                        )

        except Exception as e:
            rows.append(dict(
                sid=sid, gt=rec["gt"], used=0,
                error=f"{type(e).__name__}: {e}"
            ))
            tqdm.write(f"[collect sid={sid}] {type(e).__name__}: {e}")

        finally:
            if real is not None: real.close()
            if gray is not None: gray.close()
            gc_collect()

    return bag, rows


def fit_templates(bag, selected):
    templates = {}
    summary = []

    for l in selected:
        mus, counts = {}, {}

        for r in RELS:
            xs = bag[l][r]
            counts[r] = len(xs)
            if not xs:
                raise RuntimeError(
                    f"L{l}: no {r} vectors after filter={r}. "
                    f"Try --template-filter real_correct."
                )
            mus[r] = np.stack(xs).mean(0).astype(np.float32)

        global_mu = np.stack([mus[r] for r in RELS]).mean(0).astype(np.float32)

        shared = {
            r: (mus[r] - global_mu).astype(np.float32)
            for r in RELS
        }

        templates[l] = dict(
            relation_mean=mus,
            global_mean=global_mu,
            shared=shared,
        )

        summary.append(dict(
            layer=l,
            **{f"n_{r}": counts[r] for r in RELS},
            global_norm=float(np.linalg.norm(global_mu)),
            **{f"{r}_norm": float(np.linalg.norm(shared[r])) for r in RELS},
        ))

    return templates, summary


def gc_collect():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# baseline / steering eval
# ---------------------------------------------------------------------

def baseline(model, proc, records, sids, args, desc):
    rows = []

    for sid in tqdm(sids, desc=desc):
        rec = records[sid]
        im = None

        try:
            im = Image.open(rec["image_path"]).convert("RGB")
            b = batch_for(proc, im, rec, args)
            text, pred = generate(model, proc, b, args.max_new_tokens)

            rows.append(dict(
                sid=sid, gt=rec["gt"],
                pred=pred or "", correct=int(pred == rec["gt"]),
                text=text,
            ))
            del b

        except Exception as e:
            tqdm.write(f"[baseline sid={sid}] {type(e).__name__}: {e}")

        finally:
            if im is not None: im.close()
            gc_collect()

    return rows


def steer_eval(model, proc, layers, templates, records, base_rows,
               selected, targets, args, condition,
               mode="add", apply_mode="all"):
    out = []

    for base in tqdm(base_rows, desc=condition):
        sid = int(base["sid"])
        rec = records[sid]
        bp = rel(base["pred"])
        target = targets[sid]

        do = True
        if apply_mode == "conflict_only":
            do = bp in RELSET and target in RELSET and bp != target

        if not do:
            out.append(dict(
                condition=condition, sid=sid, gt=rec["gt"],
                target=target, baseline_pred=bp or "", edit_pred=bp or "",
                baseline_correct=int(base["correct"]),
                edit_correct=int(base["correct"]),
                applied=0, W2C=0, C2W=0, changed=0,
            ))
            continue

        im = None

        try:
            im = Image.open(rec["image_path"]).convert("RGB")
            b = batch_for(proc, im, rec, args)

            with SteerLast(
                layers, templates, selected, target,
                args.scale, mode=mode, source=bp
            ):
                text, pred = generate(
                    model, proc, b, args.max_new_tokens
                )

            bc = int(base["correct"])
            ec = int(pred == rec["gt"])

            out.append(dict(
                condition=condition, sid=sid, gt=rec["gt"],
                target=target, baseline_pred=bp or "", edit_pred=pred or "",
                baseline_correct=bc, edit_correct=ec, applied=1,
                W2C=int(not bc and ec),
                C2W=int(bc and not ec),
                changed=int((pred or "") != (bp or "")),
                text=text,
            ))
            del b

        except Exception as e:
            tqdm.write(f"[steer sid={sid} {condition}] {type(e).__name__}: {e}")

        finally:
            if im is not None: im.close()
            gc_collect()

    return out


def summarize(rows, condition, layers_text=""):
    n = len(rows)
    ba = mean(r["baseline_correct"] for r in rows)
    ea = mean(r["edit_correct"] for r in rows)
    nw = sum(1 - int(r["baseline_correct"]) for r in rows)
    nc = n - nw
    w2c = sum(int(r["W2C"]) for r in rows)
    c2w = sum(int(r["C2W"]) for r in rows)

    return dict(
        condition=condition, layers=layers_text, N=n,
        base_acc=ba, edit_acc=ea, gain=ea-ba,
        W2C=w2c, W2C_rate=w2c/nw if nw else float("nan"),
        C2W=c2w, C2W_rate=c2w/nc if nc else float("nan"),
        net=w2c-c2w,
        changed_rate=mean(r["changed"] for r in rows),
    )


def windows_around(scan_layers, anchors, lengths):
    allowed = set(scan_layers)
    out = set()

    for a in anchors:
        for n in lengths:
            for start in range(a-n+1, a+1):
                w = tuple(range(start, start+n))
                if a in w and all(x in allowed for x in w):
                    out.add(w)

    return sorted(out, key=lambda w: (len(w), w[0]))


# ---------------------------------------------------------------------
# old residual-Direction guide: auto-select guide window on FIT/VAL
# ---------------------------------------------------------------------

def fit_guide(bundle, train_sids, layers, controls):
    train_set = set(train_sids)
    idx = []

    for i, sid in enumerate(bundle["sids"].tolist()):
        sid = int(sid)
        if sid not in train_set:
            continue
        if controls == "correct" and bundle["group"].get(sid) != "correct":
            continue
        idx.append(i)

    cb = {}

    for l in layers:
        X = bundle["vecs"][idx, l].astype(np.float64)
        Y = bundle["labels"][idx]
        c = X.mean(0)
        p = {}

        for r in RELS:
            m = (X[Y == r] - c).mean(0)
            m /= max(np.linalg.norm(m), EPS)
            p[r] = m.astype(np.float32)

        cb[l] = dict(center=c.astype(np.float32), proto=p)

    return cb


def guide_pred(bundle, cb, sid, layers):
    if sid not in bundle["sid2i"]:
        return None

    i = bundle["sid2i"][sid]
    votes = {r: 0 for r in RELS}
    sums = {r: 0.0 for r in RELS}

    for l in layers:
        q = bundle["vecs"][i, l].astype(np.float64) - cb[l]["center"]
        q /= max(np.linalg.norm(q), EPS)

        sc = {
            r: float(np.dot(q, cb[l]["proto"][r]))
            for r in RELS
        }
        p = max(sc, key=sc.get)
        votes[p] += 1

        for r in RELS:
            sums[r] += sc[r]

    mv = max(votes.values())
    tied = [r for r in RELS if votes[r] == mv]
    return tied[0] if len(tied) == 1 else max(tied, key=lambda r: sums[r])


def choose_guide(bundle, fit_sids, val_sids, lengths, controls):
    n_layers = bundle["vecs"].shape[1]
    all_layers = list(range(n_layers))
    cb = fit_guide(bundle, fit_sids, all_layers, controls)

    rows = []

    for n in lengths:
        if n < 1 or n > n_layers:
            continue

        for start in range(n_layers-n+1):
            w = list(range(start, start+n))
            ok = tot = 0

            for sid in val_sids:
                if sid not in bundle["sid2i"]:
                    continue

                pred = guide_pred(bundle, cb, sid, w)
                gt = str(bundle["labels"][bundle["sid2i"][sid]])
                ok += int(pred == gt)
                tot += 1

            rows.append(dict(
                layers=",".join(map(str, w)),
                start=start, end=start+n-1, length=n,
                N=tot, acc=ok/tot if tot else float("nan"),
            ))

    best = max(
        rows,
        key=lambda r: (float(r["acc"]), -int(r["length"]))
    )
    chosen = [int(x) for x in best["layers"].split(",")]
    return chosen, rows, best


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    args = args_parser()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.output_dir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(args.direction_dir, args.direction_key)
    records = load_records(
        args.prompt_jsonl, args.annotation_json, args.data_root, bundle
    )

    fit_sids, val_sids = stratified_split(
        records, args.seed, args.val_frac, args.min_val_per_relation
    )
    test_sids = sorted(
        sid for sid, rec in records.items()
        if rec["split"] == "test"
    )

    if args.max_fit is not None:
        fit_sids = fit_sids[:args.max_fit]
    if args.max_val is not None:
        val_sids = val_sids[:args.max_val]
    if args.max_test is not None:
        test_sids = test_sids[:args.max_test]

    model, proc = load_model(args)
    layers, layer_path = decoder_layers(model)
    n_layers = len(layers)
    scan = layer_spec(args.scan_layers, n_layers)

    print("\n" + "="*120)
    print(f"MODEL={args.model_id}")
    print(f"decoder={layer_path}")
    print(f"decoder_blocks={n_layers}")
    print(f"scan={scan}")
    print(f"FIT={len(fit_sids)} VAL={len(val_sids)} TEST={len(test_sids)}")
    print("="*120)

    # FIT templates, all candidate layers
    fit_bag, fit_rows = collect(
        model, proc, layers, records, fit_sids, scan, args,
        "FIT Real-Gray deltas"
    )
    write_csv(outdir/"fit_real_gray_generation.csv", fit_rows)

    fit_temp, fit_sum = fit_templates(fit_bag, scan)
    write_csv(outdir/"fit_template_summary.csv", fit_sum)

    # VAL baseline
    val_base = baseline(
        model, proc, records, val_sids, args, "VAL baseline"
    )
    write_csv(outdir/"val_baseline.csv", val_base)

    val_targets = {
        int(r["sid"]): records[int(r["sid"])]["gt"]
        for r in val_base
    }

    # single scan
    single = []

    for l in scan:
        name = f"val_oracle_L{l:02d}"
        rows = steer_eval(
            model, proc, layers, fit_temp, records, val_base,
            [l], val_targets, args, name, "add", "all"
        )
        s = summarize(rows, name, str(l))
        s["window_length"] = 1
        single.append(s)
        write_csv(outdir/"val_single_layer_scan.csv", single)

    ranked = sorted(
        single,
        key=lambda r: (
            float(r["edit_acc"]), int(r["net"]), -int(r["C2W"])
        ),
        reverse=True,
    )

    top_layers = [
        int(r["layers"])
        for r in ranked[:args.top_k_single]
    ]

    # window scan
    candidates = windows_around(
        scan, top_layers, int_list(args.window_lengths)
    )

    win_rows = []

    for w in candidates:
        txt = ",".join(map(str, w))
        name = "val_oracle_" + "_".join(f"L{x:02d}" for x in w)

        rows = steer_eval(
            model, proc, layers, fit_temp, records, val_base,
            list(w), val_targets, args, name, "add", "all"
        )

        s = summarize(rows, name, txt)
        s["window_length"] = len(w)
        win_rows.append(s)
        write_csv(outdir/"val_window_scan.csv", win_rows)

    all_candidates = single + win_rows

    best = max(
        all_candidates,
        key=lambda r: (
            float(r["edit_acc"]),
            int(r["net"]),
            -int(r["window_length"]),
        ),
    )

    causal_layers = [
        int(x) for x in str(best["layers"]).split(",")
    ]

    print("\nTop single layers:")
    for r in ranked[:10]:
        print(
            f"  L{int(r['layers']):02d} "
            f"{r['base_acc']:.4f}->{r['edit_acc']:.4f} "
            f"{r['gain']:+.4f} W2C={r['W2C']} C2W={r['C2W']}"
        )

    print(
        f"\nSELECTED CAUSAL WINDOW = {causal_layers} | "
        f"VAL {best['base_acc']:.4f}->{best['edit_acc']:.4f} "
        f"{best['gain']:+.4f}"
    )

    # guide window select, also on FIT/VAL only
    guide_layers, guide_scan, guide_best = choose_guide(
        bundle, fit_sids, val_sids,
        int_list(args.guide_window_lengths),
        args.guide_train_controls,
    )
    write_csv(outdir/"val_guide_window_scan.csv", guide_scan)

    print(
        f"SELECTED GUIDE WINDOW = {guide_layers} | "
        f"VAL guide acc={guide_best['acc']:.4f}"
    )

    # refit causal templates on FULL TRAIN, selected layers only
    full_train = sorted(set(fit_sids + val_sids))

    full_bag, full_rows = collect(
        model, proc, layers, records, full_train,
        causal_layers, args, "FULL TRAIN refit"
    )
    write_csv(outdir/"full_train_real_gray_generation.csv", full_rows)

    final_temp, final_temp_sum = fit_templates(
        full_bag, causal_layers
    )
    write_csv(outdir/"final_template_summary.csv", final_temp_sum)

    # refit guide on FULL TRAIN
    final_guide = fit_guide(
        bundle, full_train, guide_layers,
        args.guide_train_controls,
    )

    # TEST baseline
    test_base = baseline(
        model, proc, records, test_sids, args, "TEST baseline"
    )
    write_csv(outdir/"test_baseline.csv", test_base)

    oracle_targets = {
        int(r["sid"]): records[int(r["sid"])]["gt"]
        for r in test_base
    }

    guide_targets = {}
    guide_rows = []

    for r in test_base:
        sid = int(r["sid"])
        p = guide_pred(
            bundle, final_guide, sid, guide_layers
        )

        guide_targets[sid] = p

        guide_rows.append(dict(
            sid=sid,
            gt=records[sid]["gt"],
            baseline_pred=r["pred"],
            baseline_correct=int(r["correct"]),
            guide_pred=p or "",
            guide_correct=int(p == records[sid]["gt"]),
            conflict=int(
                p in RELSET
                and rel(r["pred"]) in RELSET
                and p != rel(r["pred"])
            ),
        ))

    write_csv(outdir/"test_guide_predictions.csv", guide_rows)

    conditions = [
        ("oracle_all_add", oracle_targets, "add", "all"),
        ("guide_all_add", guide_targets, "add", "all"),
        ("guide_conflict_add", guide_targets, "add", "conflict_only"),
        ("guide_all_contrast", guide_targets, "contrast", "all"),
        ("guide_conflict_contrast", guide_targets, "contrast", "conflict_only"),
    ]

    summaries, details = [], []

    for name, targets, mode, apply_mode in conditions:
        rows = steer_eval(
            model, proc, layers, final_temp, records, test_base,
            causal_layers, targets, args, name, mode, apply_mode
        )

        details += rows
        summaries.append(
            summarize(
                rows, name,
                ",".join(map(str, causal_layers))
            )
        )

        write_csv(outdir/"test_steering_details.csv", details)
        write_csv(outdir/"test_summary.csv", summaries)

    test_base_acc = mean(r["correct"] for r in test_base)
    guide_acc = mean(r["guide_correct"] for r in guide_rows)

    print("\n" + "="*155)
    print("FINAL TEST — AUTO-SELECTED LAST-TOKEN CAUSAL WINDOW")
    print("="*155)
    print(f"model={args.model_id}")
    print(f"decoder_blocks={n_layers}")
    print(f"causal_layers={causal_layers}")
    print(f"guide_layers={guide_layers}")
    print(f"baseline={test_base_acc:.4f}")
    print(f"guide_acc={guide_acc:.4f}")
    print()
    print(
        "condition                 | acc base->edit gain | "
        "W2C/wrong | C2W/correct | net | changed"
    )

    for r in summaries:
        print(
            f"{r['condition']:25s} | "
            f"{r['base_acc']:.4f}->{r['edit_acc']:.4f} "
            f"{r['gain']:+.4f} | "
            f"{r['W2C']}/{r['W2C_rate']:.3f} | "
            f"{r['C2W']}/{r['C2W_rate']:.3f} | "
            f"{r['net']:+d} | "
            f"{r['changed_rate']:.3f}"
        )

    (outdir/"summary.json").write_text(
        json.dumps(dict(
            model_id=args.model_id,
            decoder_blocks=n_layers,
            scan_layers=scan,
            fit_n=len(fit_sids),
            val_n=len(val_sids),
            test_n=len(test_sids),
            selected_causal_layers=causal_layers,
            selected_val_acc=float(best["edit_acc"]),
            selected_val_gain=float(best["gain"]),
            selected_guide_layers=guide_layers,
            selected_guide_val_acc=float(guide_best["acc"]),
            test_baseline_acc=test_base_acc,
            test_guide_acc=guide_acc,
            note=(
                "Causal layer/window selection uses only held-out VAL from the "
                "original TRAIN split. TEST is untouched until final evaluation."
            ),
        ), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
