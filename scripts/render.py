"""Render one clip with SolarRenderer: a structure video, an optional first frame and a six-part prompt in, a video out.

    python scripts/render.py --h3 ./MiniMax-H3 \
        --control solarrenderer_control_lora.safetensors --fewstep solarrenderer_4step_lora.safetensors \
        --structure structure.mp4 --first first.png --prompt prompt.txt --out render.mp4

    # streaming: the causal pair, rendered block by block (5 latents, about 20 frames) through a key/value cache
    python scripts/render.py --h3 ./MiniMax-H3 --causal \
        --control solarrenderer_causal_control_lora.safetensors --fewstep solarrenderer_causal_4step_lora.safetensors \
        --structure structure.mp4 --first first.png --prompt prompt.txt --out render.mp4

The inputs are encoded exactly as the training samples were (tools/omni_preencode.py): the structure video laid over
every target frame, the first frame over latent 0, the prompt through the released Qwen3-VL presentation. The
renderer is MiniMax-H3's Ref2VA transformer with the four-step LoRA merged into its weights and the control LoRA on
top, sampled in four evaluations with fresh noise between them. Output: 1344x768 (or 768x1344), 24 fps, 158 frames.
With --causal the causal pair renders block-causally: each block of at most 5 latents sees the prompt, the first frame,
the structure of the last 6 blocks and the clean frames of the 5 blocks before it - never a later block - and is
sampled in four evaluations before it is committed to the cache.

The text encoder and the transformer do not share the card: the clip is encoded first, the encoder is released, then
the transformer is loaded. One GPU runs it end to end. eval/render_benchmark.py uses the same functions to
render many clips with each model loaded once.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "tools"))

FRAMES = 158
RANK = ALPHA = 384
SILENCE = HERE.parent / "assets" / "h3_silence_153_158_170.safetensors"


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h3", required=True, help="local MiniMaxAI/MiniMax-H3 (scripts/prepare_h3.py)")
    p.add_argument("--control", required=True, help="solarrenderer_control_lora.safetensors")
    p.add_argument("--fewstep", required=True, help="solarrenderer_4step_lora.safetensors")
    p.add_argument("--structure", required=True, help="the structure video, frame-aligned with the target")
    p.add_argument("--first", default="", help="the first frame (optional)")
    p.add_argument("--prompt", required=True, help="a text file holding the six-part prompt (skills/solarrenderer-caption)")
    p.add_argument("--out", required=True, help="the output .mp4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--forwards", type=int, default=4, help="model evaluations; the four-step LoRA was trained for 4")
    p.add_argument("--causal", action="store_true", help="the causal LoRA pair: stream block by block")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


# ---------------------------------------------------------------------------------------------- encoding

def load_encoder(h3: str, device: str):
    """The released ref2va front end (text encoder, processor, reference VAE encoder), as training encoded with it."""

    os.environ["H3_REF_ROOT"] = str(Path(h3).resolve())
    import omni_preencode as pre

    pre.ROOT = os.environ["H3_REF_ROOT"]
    return pre, pre.build_pipelines(device)


def sample_of(name: str, prompt: str, structure: str | None, first: str | None, target: str | None = None) -> dict:
    """A clip's inputs as a training sample. The target only sets the canvas and the length - sampling never reads its
    latents - so it is the structure video itself, or a given clip when there is no structure video."""

    observations = []
    if structure:
        observations.append({"media": structure, "frames": "all"})
    if first:
        observations.append({"media": first, "frames": [0]})
    return {"id": name, "target": target or structure, "text": prompt, "observations": observations}


def encode_sample(encoder, sample: dict, member: str) -> None:
    from safetensors.torch import save_file

    pre, pipes = encoder
    tensors, meta = pre.encode(sample, pipes, FRAMES)
    save_file(tensors, member, metadata={"omni": json.dumps(meta)})


def free_cuda() -> None:
    """After the caller dropped its last reference to the encoder."""

    import torch

    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- rendering

def lora_pairs(path: str) -> tuple[dict, float]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(path, framework="pt") as handle:
        meta = handle.metadata() or {}
    return load_file(path), float(meta.get("alpha", ALPHA))


CAUSAL = {"mode": "teacher", "chunk": 5, "window": 6, "corrupt": 0.0}  # the causal pair's training plan


def load_renderer(h3: str, control: str, fewstep: str, device: str, causal: bool = False):
    """MiniMax-H3's Ref2VA transformer, the four-step LoRA merged in, the control LoRA on top; returns a renderer."""

    import torch

    from solarwm.backends.minimax_h3.artifacts import load_silence_latents
    from solarwm.backends.minimax_h3.lora import inject_h3_lora, merge_lora_delta
    from solarwm.backends.minimax_h3.observations import observation_profile_from_train_config
    from solarwm.backends.minimax_h3.optional import load_conditioners, load_transformer
    from solarwm.backends.minimax_h3.stage0p5 import H3Stage0p5Core

    torch.cuda.set_device(torch.device(device))
    model_cfg = {
        "checkpoint_path": str(Path(h3).resolve()),
        "transformer_subfolder": "transformer_ref",
        "torch_dtype": "bfloat16",
        "attention_backend": "flash",
        "adapter": {"rank": RANK, "alpha": ALPHA, "dropout": 0.0},
    }
    modules = load_transformer(model_cfg, device=torch.device(device))
    pairs, alpha = lora_pairs(fewstep)
    rank = int(next(v for k, v in pairs.items() if k.endswith("lora_A.weight")).shape[0])
    merged = merge_lora_delta(modules.transformer, pairs, alpha=alpha, rank=rank, scale=1.0)
    print(f"four-step LoRA merged into {merged} weights", flush=True)
    wrapped, lora = inject_h3_lora(modules.transformer, model_cfg["adapter"], base_identity={})
    pairs, _alpha = lora_pairs(control)
    lora.load_state_dict({f"base_model.model.{k}": v for k, v in pairs.items()}, broadcast=False)
    conditioners = load_conditioners(
        model_cfg, device=torch.device(device), qwen=False, video_vae=True, audio_vae=False, schedulers=False
    )
    silence, _ = load_silence_latents(str(SILENCE), pixel_frames=FRAMES)
    if causal:
        from solarwm.backends.minimax_h3.omni_causal import H3OmniCausalCore

        core = H3OmniCausalCore(wrapped, silence, torch.device(device), dict(CAUSAL), geometry=None,
                                observations=observation_profile_from_train_config({}))
    else:
        core = H3Stage0p5Core(wrapped, silence, torch.device(device), geometry=None,
                              observations=observation_profile_from_train_config({}), flow_map=False)
    return core, conditioners, torch.device(device)


def render_member(renderer, member: str, out: str, seed: int = 0, forwards: int = 4) -> float:
    """Sample one encoded clip and write its mp4; returns the sampling time in seconds."""

    import torch

    from solarwm.backends.minimax_h3.inference import _mp4_bytes, decode_h3_latents
    from solarwm.backends.minimax_h3.omni_data import H3OmniStream

    core, conditioners, device = renderer
    stream = object.__new__(H3OmniStream)
    stream.root, stream.current_plan_fingerprint, stream.aligned_control = "/", "render", False
    batch = stream.read({"id": Path(out).stem, "member": member})
    began = time.perf_counter()
    with torch.no_grad():
        if hasattr(core, "stream"):
            # block by block: each block's clean estimate re-noised between evaluations, as the student was trained;
            # the structure keys and values are recomputed per evaluation so one card holds the clip
            from solarwm.backends.minimax_h3.flow import make_shifted_schedule

            torch.manual_seed(seed)
            latents, _inputs, _times, _jitters = core.stream(
                batch, generator=torch.Generator(device=device).manual_seed(seed),
                schedule=make_shifted_schedule(forwards + 1, shift=12.0), exit_index=forwards - 1,
                cache_structure=False,
            )
        else:
            # generate() counts schedule points: N points run N - 1 evaluations
            latents = core.generate(batch, noise_seed=seed, num_inference_steps=forwards + 1, renoise=True)
    took = time.perf_counter() - began
    decoded = decode_h3_latents(conditioners.video_vae, latents, device=device)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out + ".tmp").write_bytes(_mp4_bytes(decoded))
    os.replace(out + ".tmp", out)
    return took


def main() -> None:
    args = parse()
    with tempfile.TemporaryDirectory() as work:
        began = time.perf_counter()
        encoder = load_encoder(args.h3, args.device)
        member = f"{work}/member.safetensors"
        prompt = Path(args.prompt).read_text()
        encode_sample(encoder, sample_of(Path(args.out).stem, prompt, args.structure, args.first or None), member)
        del encoder
        free_cuda()
        print(f"encoded in {time.perf_counter() - began:.0f} s", flush=True)
        renderer = load_renderer(args.h3, args.control, args.fewstep, args.device, causal=args.causal)
        took = render_member(renderer, member, args.out, args.seed, args.forwards)
    print(f"sampled {args.forwards} steps in {took:.0f} s; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
