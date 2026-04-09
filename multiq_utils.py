import re
import random

WH_RE = re.compile(
    r"Where is the (.+?) in relation to the (.+?)\?",
    flags=re.IGNORECASE | re.DOTALL
)

RELS = ["left", "right", "on", "under"]

INV_REL = {
    "left": "right",
    "right": "left",
    "on": "under",
    "under": "on",
}

SYN_REL = {
    "left": "to the left of",
    "right": "to the right of",
    "on": "on top of",
    "under": "beneath",
}

def clean_question_text(question: str):
    q = question.strip()

    q = q.replace("<image>", " ")
    q = q.replace("\n", " ")

    if "USER:" in q:
        q = q.split("USER:", 1)[1].strip()

    if "ASSISTANT:" in q:
        q = q.split("ASSISTANT:", 1)[0].strip()

    # 去掉原始 prompt 里附带的答案说明
    q = re.sub(
        r"Answer with left,\s*right,\s*on or under\.\s*$",
        "",
        q,
        flags=re.IGNORECASE
    )

    q = re.sub(r"\s+", " ", q).strip()
    return q

def parse_wh_question(question: str):
    q = clean_question_text(question)
    m = WH_RE.search(q)
    if not m:
        raise ValueError(f"Cannot parse question: {question}")
    obj1, obj2 = m.group(1), m.group(2)
    return obj1.strip(), obj2.strip()

def normalize_rel(answer: str):
    rel = answer.strip().lower()
    if rel not in RELS:
        raise ValueError(f"Unexpected relation: {answer}")
    return rel

def build_object_pool(prompt_records):
    pool = set()
    for rec in prompt_records:
        q = rec["question"]
        obj1, obj2 = parse_wh_question(q)
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

def build_questions(base_question, base_answer, sample_idx, object_pool):
    obj1, obj2 = parse_wh_question(base_question)
    gold_rel = normalize_rel(base_answer)
    inv_rel = INV_REL[gold_rel]
    syn_rel = SYN_REL[gold_rel]
    obj3, obj4 = pick_obj3_obj4(object_pool, obj1, obj2, sample_idx)

    items = []

    items.append({
        "qid": "q0",
        "mode": "orig",
        "prompt": f"USER: <image>\nWhere is the {obj1} in relation to the {obj2}? "
                  f"Answer with left, right, on or under.\nASSISTANT:",
        "gold": gold_rel,
    })

    tf_questions = {
        "q1": (f"Is the {obj1} {gold_rel} the {obj2}? Answer with T or F only.", "T"),
        "q2": (f"Is the {obj2} {inv_rel} the {obj1}? Answer with T or F only.", "T"),
        "q3": (f"Is the {obj1} not {gold_rel} the {obj2}? Answer with T or F only.", "F"),
        "q4": (f"Given the {obj1} and the {obj2} in the image, is the {obj1} {gold_rel} the {obj2}? "
               f"Answer with T or F only.", "T"),
        "q5": (f"Given the {obj3} and the {obj4} in the image, is the {obj1} {gold_rel} the {obj2}? "
               f"Answer with T or F only.", "T"),
        "q6": (f"Is the {obj2} {gold_rel} the {obj1}? Answer with T or F only.", "F"),
        "q7": (f"Is the {obj1} {syn_rel} the {obj2}? Answer with T or F only.", "T"),
        "q8": (f"In the image, is it true that the {obj1} {gold_rel} the {obj2}? "
               f"Answer with T or F only.", "T"),
        "q9": (f"Would it be correct to say that the {obj1} {gold_rel} the {obj2}? "
               f"Answer with T or F only.", "T"),
    }

    for qid, (question_text, gold) in tf_questions.items():
        items.append({
            "qid": qid,
            "mode": "tf",
            "prompt": f"USER: <image>\n{question_text}\nASSISTANT:",
            "gold": gold,
        })

    meta = {
        "obj1": obj1,
        "obj2": obj2,
        "gold_rel": gold_rel,
        "obj3": obj3,
        "obj4": obj4,
        "base_question": clean_question_text(base_question),
        "base_answer": gold_rel,
    }
    return items, meta

def parse_prediction(text: str, mode: str):
    t = text.strip().lower()

    if mode == "orig":
        for rel in RELS:
            if rel in t:
                return rel
        return "UNK"

    if t.startswith("t"):
        return "T"
    if t.startswith("f"):
        return "F"
    return "UNK"
