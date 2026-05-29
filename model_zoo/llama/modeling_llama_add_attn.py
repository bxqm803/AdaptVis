from pathlib import Path

path = Path("model_zoo/llama/modeling_llama_add_attn.py")
txt = path.read_text(encoding="utf-8")
orig = txt

# Insert effective_weight fallback once.
if "effective_weight = weight" not in txt:
    marker = """        start_idx, end_idx, square_size = -1, -1, -1
        image_key_mask = None
        query_indices = None
"""
    insert = """        start_idx, end_idx, square_size = -1, -1, -1
        image_key_mask = None
        query_indices = None

        # generate() may drop custom kwargs like weight before LLaMAAttention receives them.
        # Fallback to a per-sample environment variable set by the runner.
        effective_weight = weight
        if effective_weight is None:
            _env_w = os.environ.get("ADAPTVIS_WEIGHT", "").strip()
            if _env_w:
                effective_weight = float(_env_w)
"""
    if marker not in txt:
        raise RuntimeError("Could not find insertion marker for effective_weight.")
    txt = txt.replace(marker, insert, 1)

# Replace gating conditions and weight arguments.
txt = txt.replace(
    "if _adaptvis_layer_selected(idx) and keys is not None and weight is not None:",
    "if _adaptvis_layer_selected(idx) and keys is not None and effective_weight is not None:",
)
txt = txt.replace(
    "                            weight=weight,",
    "                            weight=effective_weight,",
)
txt = txt.replace(
    "weight=np.array(float(weight), dtype=np.float32),",
    "weight=np.array(float(effective_weight), dtype=np.float32),",
)
txt = txt.replace(
    "and weight is not None",
    "and effective_weight is not None",
)
txt = txt.replace(
    "                weight=weight,",
    "                weight=effective_weight,",
)

# Improve existing debug line if present.
txt = txt.replace(
    "weight={weight} square={attn_weights.size()[2] == attn_weights.size()[3]}",
    "weight={effective_weight} raw_weight={weight} square={attn_weights.size()[2] == attn_weights.size()[3]}",
)

if txt == orig:
    print("[NO CHANGE] modeling file already appears patched or markers not found.")
else:
    backup = path.with_suffix(".py.bak_effective_weight")
    backup.write_text(orig, encoding="utf-8")
    path.write_text(txt, encoding="utf-8")
    print("[BACKUP]", backup)
    print("[PATCHED]", path)
