import re
import random

WH_RE = re.compile(
    r"Where (is|are) the (.+?) in relation to the (.+?)\?",
    flags=re.IGNORECASE | re.DOTALL
)

def clean_prompt_to_wh_question(prompt: str):
    s = prompt.strip()

    s = s.replace("<image>", " ")
    s = s.replace("\n", " ")

    if "USER:" in s:
        s = s.split("USER:", 1)[1].strip()

    if "ASSISTANT:" in s:
        s = s.split("ASSISTANT:", 1)[0].strip()

    s = re.sub(
        r"Answer with left,\s*right,\s*on or under\.\s*$",
        "",
        s,
        flags=re.IGNORECASE
    )

    s = re.sub(r"\s+", " ", s).strip()
    return s

INV_REL = {
    "left": "right",
    "right": "left",
    "on": "under",
    "under": "on",
}

# internal label -> natural surface form for prompts
REL_TO_PHRASE = {
    "left": "to the left of",
    "right": "to the right of",
    "on": "on",
    "under": "under",
}

# alternative natural paraphrase
SYN_REL = {
    "left": "on the left of",
    "right": "on the right of",
    "on": "on top of",
    "under": "beneath",
}

def wrap_prompt_user(question_text: str):
    return f"<image>\nUSER: {question_text}\nASSISTANT:"

def tf_be_verb(be_verb: str):
    return "Is" if be_verb == "is" else "Are"
    
def clean_question_text(question: str):
    q = question.strip()

    q = q.replace("<image>", " ")
    q = q.replace("\n", " ")

    if "USER:" in q:
        q = q.split("USER:", 1)[1].strip()

    if "ASSISTANT:" in q:
        q = q.split("ASSISTANT:", 1)[0].strip()

    q = re.sub(
        r"Answer with left,\s*right,\s*on or under\.\s*$",
        "",
        q,
        flags=re.IGNORECASE
    )

    q = re.sub(r"\s+", " ", q).strip()
    return q

def parse_wh_question(prompt: str):
    q = clean_prompt_to_wh_question(prompt)
    m = WH_RE.search(q)
    if not m:
        raise ValueError(f"Cannot parse question: {prompt}")
    be_verb = m.group(1).lower()   # is / are
    obj1 = m.group(2).strip()
    obj2 = m.group(3).strip()
    return be_verb, obj1, obj2

def normalize_rel(answer: str):
    rel = answer.strip().lower()
    if rel not in RELS:
        raise ValueError(f"Unexpected relation: {answer}")
    return rel

def build_object_pool(prompt_records):
    pool = set()
    for rec in prompt_records:
        q = rec["question"]
        _, obj1, obj2 = parse_wh_question(q)
        pool.add(obj1)
        pool.add(obj2)
    return sorted(pool)

def pick_obj3_obj4(object_pool, obj1, obj2, sample_idx):
    candidates = [x for x in object_pool if x not in {obj1, obj2}]
    if len(candidates) < 2:
        raise ValueError("Not enough distinct objects for Q5.")
    rng = random.Random(sample_idx)
    picks = rng.sample(candidates, 2)
    return picks[0], picks[1]
    
def build_questions(base_prompt, base_answer, sample_idx, object_pool):
    _, obj1, obj2 = parse_wh_question(base_prompt)
    gold_rel = normalize_rel(base_answer)

    inv_rel = INV_REL[gold_rel]
    gold_phrase = REL_TO_PHRASE[gold_rel]
    inv_phrase = REL_TO_PHRASE[inv_rel]
    syn_phrase = SYN_REL[gold_rel]

    obj3, obj4 = pick_obj3_obj4(object_pool, obj1, obj2, sample_idx)

    items = []

    items.append({
        "qid": "q0",
        "mode": "orig",
        "prompt": wrap_prompt_user(
            f"Where is the {obj1} in relation to the {obj2}? "
            f"Answer with left, right, on or under."
        ),
        "gold": gold_rel,
    })

    tf_questions = {
        "q1": (f"Is the {obj1} {gold_phrase} the {obj2}? Answer with True or False only.", "True"),
        "q2": (f"Is the {obj2} {inv_phrase} the {obj1}? Answer with True or False only.", "True"),
        "q3": (f"Is the {obj1} not {gold_phrase} the {obj2}? Answer with True or False only.", "False"),
        "q4": (f"Given the {obj1} and the {obj2} in the image, is the {obj1} {gold_phrase} the {obj2}? Answer with True or False only.", "True"),
        "q5": (f"Given the {obj3} and the {obj4} in the image, is the {obj1} {gold_phrase} the {obj2}? Answer with True or False only.", "True"),
        "q6": (f"Is the {obj2} {gold_phrase} the {obj1}? Answer with True or False only.", "False"),
        "q7": (f"Is the {obj1} {syn_phrase} the {obj2}? Answer with True or False only.", "True"),
        "q8": (f"In the image, is it true that the {obj1} is {gold_phrase} the {obj2}? Answer with True or False only.", "True"),
        "q9": (f"Would it be correct to say that the {obj1} is {gold_phrase} the {obj2}? Answer with True or False only.", "True"),
    }

    for qid, (qtext, gold) in tf_questions.items():
        items.append({
            "qid": qid,
            "mode": "tf",
            "prompt": wrap_prompt_user(qtext),
            "gold": gold,
        })

    meta = {
        "obj1": obj1,
        "obj2": obj2,
        "gold_rel": gold_rel,
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
