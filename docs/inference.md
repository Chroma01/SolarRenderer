# Inference

## Install

```bash
git clone https://github.com/Enderfga/SolarRenderer && cd SolarRenderer
pip install -e . && pip install flash-attn==2.7.4.post1 --no-build-isolation    # Python 3.10+, CUDA; inference uses flash attention
```

flash-attn builds from source when no prebuilt wheel matches your torch and CUDA, which takes a while.

Tested with torch 2.8, diffusers 0.40.0, transformers 5.14.1, peft 0.20.0 and flash-attn 2.7.4, with
`ffmpeg` and `ffprobe` on the path. Inference runs on one GPU; the text encoder and the transformer are loaded one
after the other.

## Render a clip

**1. Download MiniMax-H3 and the weights** (about 135 GB and 8 GB). MiniMax-H3 is released under its own
[license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE); read it before use.

```bash
python scripts/prepare_h3.py --dir ./MiniMax-H3
hf download Enderfga/SolarRenderer config.json solarrenderer_control_lora.safetensors solarrenderer_4step_lora.safetensors \
    --local-dir ./weights
```

**2. Prepare the inputs.**

- `structure.mp4`: the structure video, 24 fps, at least 158 frames (6.6 s), landscape 1344×768 or portrait
  768×1344 (other sizes are resized and cropped to the canvas). Any representation works; the prompt names it.
- `first.png` (optional, recommended): the first frame, in the look you want. See [prompts and first frames](prompts.md).
- `prompt.txt`: the six-part prompt. See [prompts and first frames](prompts.md).

**3. Render.**

```bash
python scripts/render.py --h3 ./MiniMax-H3 \
    --control weights/solarrenderer_control_lora.safetensors \
    --fewstep weights/solarrenderer_4step_lora.safetensors \
    --structure structure.mp4 --first first.png --prompt prompt.txt --out render.mp4 --seed 0
```

Output: 158 frames at 24 fps. Encoding takes about 4 minutes and sampling about 3.5 minutes on one GPU.

## Stream a clip

The causal pair renders the same clip as a stream: the video is generated block by block, each block of at most 5
latents (about 20 frames) seeing the prompt, the first frame, the structure of its last 6 blocks and the frames already
generated for the 5 blocks before it, never a later block. Each block is sampled in four steps and then committed to a
key/value cache, so a block can be shown as soon as it is done. Download the causal LoRAs and add `--causal`:

```bash
hf download Enderfga/SolarRenderer config.json solarrenderer_causal_control_lora.safetensors \
    solarrenderer_causal_4step_lora.safetensors --local-dir ./weights
python scripts/render.py --h3 ./MiniMax-H3 --causal \
    --control weights/solarrenderer_causal_control_lora.safetensors \
    --fewstep weights/solarrenderer_causal_4step_lora.safetensors \
    --structure structure.mp4 --first first.png --prompt prompt.txt --out render.mp4 --seed 0
```

The inputs and the output are the same as above.

## Render a longer clip

`--seconds` sets the length, up to what the structure video holds (the default is the trained 6.6 s), with either pair:

```bash
python scripts/render.py --h3 ./MiniMax-H3 --causal \
    --control weights/solarrenderer_causal_control_lora.safetensors \
    --fewstep weights/solarrenderer_causal_4step_lora.safetensors \
    --structure structure.mp4 --first first.png --prompt prompt.txt --out render.mp4 --seconds 60
```

The clip is rendered whole either way. The bidirectional pair samples every frame at once, so its memory and time grow
with the length (attention over every frame). The causal pair streams it block by block, each block seeing the
structure of its last 6 blocks and the frames of the 5 blocks before it, so its memory stays that of a few blocks and
its time grows in step with the length. Write the prompt for the whole clip.

## Render the benchmark

`eval/render_benchmark.py` renders every benchmark case with each model loaded once per GPU; see [`eval/README.md`](../eval/README.md).
