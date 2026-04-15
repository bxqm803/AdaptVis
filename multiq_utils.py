import re
import random

WH_RE = re.compile(
    r"Where (is|are) the (.+?) in relation to the (.+?)\?",
    flags=re.IGNORECASE | re.DOTALL,
)

def clean_prompt_to_wh_question(prompt: str):
    s = prompt.strip()
    s = s.replace("\n", " ")
    if "USER:" in s:
        s = s.split("USER:", 1)[1].strip()
    if "ASSISTANT:" in s:
        s = s.split("ASSISTANT:", 1)[0].strip()
    s = re.sub(
        r"Answer with left,\s*right,\s*on or under\.\s*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


# canonical labels used internally
RELS = ["left", "right", "on", "under"]

INV_REL = {
    "left": "right",
    "right": "left",
    "on": "under",
    "under": "on",
}

# natural phrases used in prompts
REL_TO_PHRASE = {
    "left": "to the left of",
    "right": "to the right of",
    "on": "on",
    "under": "under",
}

# paraphrases for q7
SYN_REL = {
    "left": "on the left of",
    "right": "on the right of",
    "on": "on top of",
    "under": "beneath",
}

def wrap_prompt_user(question_text: str):
    return f"<image>\nUSER: {question_text}\nASSISTANT:"

def clean_question_text(question: str):
    q = question.strip()
    q = q.replace("\n", " ")
    if "USER:" in q:
        q = q.split("USER:", 1)[1].strip()
    if "ASSISTANT:" in q:
        q = q.split("ASSISTANT:", 1)[0].strip()
    q = re.sub(
        r"Answer with left,\s*right,\s*on or under\.\s*$",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\s+", " ", q).strip()
    return q

def parse_wh_question(prompt: str):
    q = clean_prompt_to_wh_question(prompt)
    m = WH_RE.search(q)
    if not m:
        raise ValueError(f"Cannot parse question: {prompt}")
    be_verb = m.group(1).lower()
    obj1 = m.group(2).strip()
    obj2 = m.group(3).strip()
    return be_verb, obj1, obj2

def normalize_rel(answer):
    if isinstance(answer, (list, tuple)):
        if len(answer) == 0:
            raise ValueError("Empty answer list.")
        answer = answer[0]

    if answer is None:
        raise ValueError("Answer is None.")

    rel = str(answer).strip().lower()

    mapping = {
        "left": "left",
        "right": "right",
        "on": "on",
        "under": "under",
        "below": "under",
        "beneath": "under",
        "top": "on",
        "above": "on",
        "to the left of": "left",
        "to the right of": "right",
        "on top of": "on",
    }

    if rel not in mapping:
        raise ValueError(f"Unsupported relation answer: {answer}")

    return mapping[rel]

def build_object_pool(prompt_records):
    pool = set()
    for rec in prompt_records:
        _, obj1, obj2 = parse_wh_question(rec["question"])
        pool.add(obj1)
        pool.add(obj2)
    return sorted(pool)

def pick_obj3_obj4(object_pool, obj1, obj2, sample_idx):
    candidates = [x for x in object_pool if x not in {obj1, obj2}]
    if len(candidates) < 2:
        raise ValueError("Not enough distinct objects for Q5.")
    rng = random.Random(sample_idx)
    obj3, obj4 = rng.sample(candidates, 2)
    return obj3, obj4

def build_questions(base_prompt, base_answer, sample_idx, object_pool):
    _, obj1, obj2 = parse_wh_question(base_prompt)
    gold_rel = normalize_rel(base_answer)

    inv_rel = INV_REL[gold_rel]
    gold_phrase = REL_TO_PHRASE[gold_rel]
    inv_phrase = REL_TO_PHRASE[inv_rel]
    syn_phrase = SYN_REL[gold_rel]

    obj3, obj4 = pick_obj3_obj4(object_pool, obj1, obj2, sample_idx)

    items = []

    # q0: original WH question, no explicit relation phrase in the prompt body
    items.append({
        "qid": "q0",
        "mode": "orig",
        "prompt": wrap_prompt_user(
            f"Where is the {obj1} in relation to the {obj2}? "
            f"Answer with left, right, on or under."
        ),
        "gold": gold_rel,
        "target_texts": {
            "obj1": obj1,
            "obj2": obj2,
            "rel": None,
        },
    })

    question_specs = [
        (
            "q1",
            f"Is the {obj1} {gold_phrase} the {obj2}? Answer with True or False only.",
            "True",
            {"obj1": obj1, "obj2": obj2, "rel": gold_phrase},
        ),
        (
            "q2",
            f"Is the {obj2} {inv_phrase} the {obj1}? Answer with True or False only.",
            "True",
            {"obj1": obj2, "obj2": obj1, "rel": inv_phrase},
        ),
        (
            "q3",
            f"Is the {obj1} not {gold_phrase} the {obj2}? Answer with True or False only .",
            "False",
            {"obj1": obj1, "obj2": obj2, "rel": gold_phrase},
        ),
        (
            "q4",
            f"Given the {obj1} and the {obj2} in the image, is the {obj1} {gold_phrase} the {obj2}? "
            f"Answer with True or False only",
            "True",
            {"obj1": obj1, "obj2": obj2, "rel": gold_phrase},
        ),
        (
            "q5",
            f"Given the {obj3} and the {obj4} in the image, is the {obj1} {gold_phrase} the {obj2}? "
            f"Answer with True or False only.",
            "True",
            {"obj1": obj1, "obj2": obj2, "rel": gold_phrase},
        ),
        (
            "q6",
            f"Is the {obj2} {gold_phrase} the {obj1}? Answer with True or False only.",
            "False",
            {"obj1": obj2, "obj2": obj1, "rel": gold_phrase},
        ),
        (
            "q7",
            f"Is the {obj1} {syn_phrase} the {obj2}? Answer with True or False only.",
            "True",
            {"obj1": obj1, "obj2": obj2, "rel": syn_phrase},
        ),
        (
            "q8",
            f"In the image, is it true that the {obj1} is {gold_phrase} the {obj2}? "
            f"Answer with True or False only.",
            "True",
            {"obj1": obj1, "obj2": obj2, "rel": gold_phrase},
        ),
        (
            "q9",
            f"Would it be correct to say that the {obj1} is {gold_phrase} the {obj2}? "
            f"Answer with True or False only.",
            "True",
            {"obj1": obj1, "obj2": obj2, "rel": gold_phrase},
        ),
    ]

    for qid, qtext, gold, target_texts in question_specs:
        items.append({
            "qid": qid,
            "mode": "tf",
            "prompt": wrap_prompt_user(qtext),
            "gold": gold,
            "target_texts": target_texts,
        })

    meta = {
        "obj1": obj1,
        "obj2": obj2,
        "gold_rel": gold_rel,
        "gold_phrase": gold_phrase,
        "inv_rel": inv_rel,
        "inv_phrase": inv_phrase,
        "syn_phrase": syn_phrase,
        "obj3": obj3,
        "obj4": obj4,
        "base_prompt": base_prompt,
        "base_question_clean": clean_prompt_to_wh_question(base_prompt),
        "base_answer": gold_rel,
    }

    return items, meta

def parse_prediction(text: str, mode: str):
    t = text.strip().lower()

    if mode == "orig":
        m = re.search(r"\b(left|right|on|under)\b", t)
        return m.group(1) if m else "UNK"

    m = re.search(r"\b(t|true|f|false|yes|no)\b", t)
    if not m:
        return "UNK"

    tok = m.group(1)
    return "True" if tok in {"t", "true", "yes"} else "False"
