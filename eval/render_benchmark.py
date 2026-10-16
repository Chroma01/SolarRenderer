"""Render the SolarRenderer benchmark with SolarRenderer, each model loaded once per GPU.

    python eval/render_benchmark.py --bench ./SolarRenderer-Data --out ./renders/solarrenderer \
        --h3 ./MiniMax-H3 --control weights/solarrenderer_control_lora.safetensors \
        --fewstep weights/solarrenderer_4step_lora.safetensors [--split hard|heldout|all] [--shard k --shards n]

Writes <out>/hard/<case_id>.mp4 and <out>/heldout/<clip_id>.mp4, the layout eval/metrics.py and eval/judge.py
read. Run one process per GPU with --shard / --shards (CUDA_VISIBLE_DEVICES=k). Every case is encoded first, the text
encoder is released, then every case is sampled; renders already on disk are skipped. Every case renders from its structure video, first
frame and prompt.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import bench  # noqa: E402
import render  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="local copy of the SolarRenderer-Data dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--h3", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--fewstep", required=True)
    ap.add_argument("--split", default="all", choices=("hard", "heldout", "all"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forwards", type=int, default=4)
    ap.add_argument("--causal", action="store_true", help="the causal LoRA pair, streamed block by block")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    cases = [c for c in bench.load(a.bench, a.split)][a.shard::a.shards]
    todo = [c for c in cases if not c.render(a.out).exists()]
    print(f"shard {a.shard}/{a.shards}: {len(cases)} cases, {len(todo)} to render", flush=True)
    if not todo:
        return
    with tempfile.TemporaryDirectory() as work:
        encoder = render.load_encoder(a.h3, a.device)
        members = {}
        for c in todo:
            member = f"{work}/{c.key.replace('/', '__')}.safetensors"
            began = time.perf_counter()
            try:
                render.encode_sample(encoder, render.sample_of(c.id, c.prompt, c.condition, c.first_frame,
                                                               target=c.condition), member)
            except Exception as error:  # one bad input must not stop the sweep
                print(f"SKIP encode {c.key}: {error}", flush=True)
                continue
            members[c.key] = (c, member)
            print(f"encoded {c.key} in {time.perf_counter() - began:.0f} s", flush=True)
        del encoder
        render.free_cuda()
        renderer = render.load_renderer(a.h3, a.control, a.fewstep, a.device, causal=a.causal)
        for key, (c, member) in members.items():
            try:
                took = render.render_member(renderer, member, str(c.render(a.out)), a.seed, a.forwards)
                print(f"rendered {key} in {took:.0f} s", flush=True)
            except Exception as error:
                print(f"SKIP render {key}: {error}", flush=True)


if __name__ == "__main__":
    main()
