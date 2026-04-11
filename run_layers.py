import subprocess
import sys

LAYERS = [0, 4, 8, 16, 24, 31]

BASE_CMD = [
    sys.executable,
    "new_run_multiq_attn.py",
    "--model-name", "llava1.5",
    "--dataset", "Controlled_Images_A",
    "--method", "scaling_vis",
    "--weight", "1.0",
    "--targets", "obj1,obj2,rel",
    "--limit", "3",
]

for layer in LAYERS:
    cmd = BASE_CMD + ["--attn-layer", str(layer)]
    print("\n" + "=" * 80)
    print("Running layer:", layer)
    print(" ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)
