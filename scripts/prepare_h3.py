"""Download MiniMax-H3 and point its modular pipeline at the local copy.

    python scripts/prepare_h3.py --dir ./MiniMax-H3

SolarRenderer reads the released Ref2VA transformer (``transformer_ref``), the Qwen3-VL text encoder, its processor and
tokenizer, and the video VAE. Training reads them through <dir>/ref2va, links that name the Ref2VA transformer
``transformer``. The repository's ``modular_model_index.json`` names every component by its Hub id; it is
rewritten to name the local directory, so encoding runs offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = "MiniMaxAI/MiniMax-H3"
NEEDED = ["transformer_ref/*", "text_encoder/*", "tokenizer/*", "processor/*", "vae/*", "audio_vae/*", "scheduler/*",
          "audio_scheduler/*", "model_index.json", "modular_model_index.json", "LICENSE"]


def localize(directory: Path) -> None:
    index = directory / "modular_model_index.json"
    spec = json.loads(index.read_text())
    for value in spec.values():
        if isinstance(value, list) and len(value) == 3 and isinstance(value[2], dict):
            value[2]["pretrained_model_name_or_path"] = str(directory)
    index.write_text(json.dumps(spec, indent=2) + "\n")


def ref2va_layout(directory: Path) -> Path:
    """<dir>/ref2va: the Ref2VA transformer as ``transformer`` beside the VAEs and processors - the layout training
    reads (model.checkpoint_path in configs/), made of links into the download."""

    root = directory / "ref2va"
    root.mkdir(exist_ok=True)
    for name, target in (("transformer", "transformer_ref"), ("vae", "vae"), ("audio_vae", "audio_vae"),
                         ("processor", "processor"), ("tokenizer", "tokenizer"), ("scheduler", "scheduler")):
        link = root / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(Path("..") / target)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--skip-download", action="store_true", help="only rewrite the index of an existing copy")
    args = parser.parse_args()
    directory = args.dir.resolve()
    if not args.skip_download:
        from huggingface_hub import snapshot_download

        snapshot_download(REPO, local_dir=str(directory), allow_patterns=NEEDED)
    localize(directory)
    print(f"MiniMax-H3 ready at {directory}; training layout at {ref2va_layout(directory)}")


if __name__ == "__main__":
    main()
