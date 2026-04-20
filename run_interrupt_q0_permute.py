def extract_first_topk_candidates(trace, topn=10):
    """
    Extract top-k candidate tokens from the first generation step.

    Supports multiple possible trace schemas, including:
      - first["top10"]
      - first["topk"]
      - first["top_k"]
      - first["top_tokens"]
      - first["top_candidates"]
      - first["candidates"]

    Returns:
        A list of dicts, each with:
            {
                "token_text": str,
                "token_norm": str,
                "raw_logit": float | None,
                "probability": float | None,
            }
    """
    if not trace or not isinstance(trace, list):
        return []

    first = trace[0]
    if not isinstance(first, dict):
        return []

    candidate_keys = [
        "top10",
        "topk",
        "top_k",
        "top_tokens",
        "top_candidates",
        "candidates",
    ]

    candidates = None
    for key in candidate_keys:
        value = first.get(key, None)
        if isinstance(value, list):
            candidates = value
            break

    if not candidates:
        return []

    out = []
    for cand in candidates[:topn]:
        if not isinstance(cand, dict):
            continue

        token_text = (
            cand.get("token_text")
            if cand.get("token_text") is not None
            else cand.get("token")
            if cand.get("token") is not None
            else cand.get("text")
            if cand.get("text") is not None
            else cand.get("decoded")
            if cand.get("decoded") is not None
            else ""
        )

        raw_logit = (
            cand.get("raw_logit")
            if cand.get("raw_logit") is not None
            else cand.get("logit")
            if cand.get("logit") is not None
            else cand.get("raw_score")
            if cand.get("raw_score") is not None
            else None
        )

        probability = (
            cand.get("probability")
            if cand.get("probability") is not None
            else cand.get("prob")
            if cand.get("prob") is not None
            else None
        )

        out.append(
            {
                "token_text": token_text,
                "token_norm": normalize_token_text(token_text),
                "raw_logit": float(raw_logit) if raw_logit is not None else None,
                "probability": float(probability) if probability is not None else None,
            }
        )

    return out
