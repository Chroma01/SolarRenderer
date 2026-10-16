"""Export a training run's LoRA (<checkpoint>/adapter.pt) as the safetensors scripts/render.py reads.

    python scripts/export_lora.py --checkpoint /path/to/runs/structure_guided_4step/checkpoint_model_000100 \
        --out weights/my_4step_lora.safetensors

Keys become <module>.lora_A.weight / <module>.lora_B.weight in bf16; the run's rank and alpha go in the metadata.
"""
import argparse

import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="a checkpoint directory holding adapter.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    saved = torch.load(f"{a.checkpoint.rstrip('/')}/adapter.pt", map_location="cpu", weights_only=True)
    meta = saved["metadata"]
    state = {k.removeprefix("base_model.model."): v.to(torch.bfloat16).contiguous() for k, v in saved["state"].items()}
    if not all(k.endswith((".lora_A.weight", ".lora_B.weight")) for k in state):
        raise SystemExit("adapter.pt holds more than LoRA pairs")
    save_file(state, a.out, metadata={"format": "pt", "rank": str(meta["rank"]), "alpha": str(meta["alpha"])})
    print(f"{len(state) // 2} LoRA pairs, rank {meta['rank']}, alpha {meta['alpha']} -> {a.out}")


if __name__ == "__main__":
    main()
