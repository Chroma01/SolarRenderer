# Training


Training runs in three steps on one 8-GPU node with FSDP and sequence parallelism; the configurations in `configs/` are
set for it. The released weights were trained with the same recipe and the same number of samples at a larger global
batch. Training needs about 140 GB of memory per GPU. On one such node a control step takes about 84 s and a
post-training step 3 to 8 minutes, so the full schedules take weeks and days respectively. Replace the
`/path/to/...` entries before launching. Data and checkpoints can be local paths or remote object-storage
prefixes (`<scheme>://...`); remote ones are read and written through [fsspec](https://filesystem-spec.readthedocs.io),
so install the fsspec backend for your storage.

**1. Encode the training samples.** Each sample is one line of a JSONL file: the target clip, the prompt, and the
observations, each laid over the target frame by frame (`"all"`), on given latent frames (`[0]` for a first frame), or
not aligned (`null`):

```json
{"id": "clip_0001_full", "target": "clips/clip_0001.mp4", "text": "subject_definitions:\n<Video 1> is ...",
 "observations": [{"media": "structure/clip_0001.mp4", "frames": "all"},
                  {"media": "first/clip_0001.png", "frames": [0]}]}
```

To train on the released data, download the shards and let `tools/hf_samples.py` write the clips and the sample
list (each clip with the inputs and prompt training gave it; `--per-source N` keeps N clips per source for a quick test):

```bash
hf download Enderfga/SolarRenderer-Data --repo-type dataset --include "train/*" --local-dir ./SolarRenderer-Data
python tools/hf_samples.py --data ./SolarRenderer-Data --out ./samples     # writes ./samples/samples.jsonl
```

Each clip is one sample. Its inputs (structure and first frame, structure only, first frame only, or text only) are
fixed by a hash of its id, and the prompt carries the matching fixed sentences. Encoding runs one worker per GPU and writes `samples/` and one index per worker:

```bash
H3_REF_ROOT=./MiniMax-H3 torchrun --nproc-per-node 8 tools/omni_preencode.py --samples ./samples/samples.jsonl --out ./encoded
cat ./encoded/index.*.jsonl | sort -u > ./encoded/train.jsonl   # workers help each other, so rows can repeat
```

The [`data`](https://github.com/Enderfga/SolarRenderer/tree/data) branch holds the data processing, cleaning and
captioning pipeline that produced our training samples; the data itself is on
[Hugging Face](https://huggingface.co/datasets/Enderfga/SolarRenderer-Data).

**2. Train the control LoRA** (flow matching, 38,400 steps at global batch 4):

```bash
torchrun --standalone --nproc-per-node 8 -m solarwm train --config configs/control_lora.yaml
```

On Hopper GPUs, optional kernels make this step about 1.35x faster (80.8 s to 59.7 s per step on 8 H200s). They
are off by default and keep the original precision (bf16 data, fp32 norm statistics, softmax and accumulation):

```bash
export SOLAR_HOPPER_ATTENTION=1 SOLAR_HOPPER_FWD192=1 SOLAR_FUSED_ADALN=1 SOLAR_FUSED_QK_ROPE=1 SOLAR_ATTN_CPU_CACHE=24 \
       SOLAR_SP_PACK_B1=1
torchrun --standalone --nproc-per-node 8 -m solarwm train --config configs/control_lora.yaml
```

The attention kernels are FlashAttention's Hopper kernels, compiled on the first launch (about 3 minutes; the build is
kept under `TORCH_EXTENSIONS_DIR`, else the temporary directory) against the CUTLASS headers of the NGC PyTorch
container; elsewhere point `SOLAR_CUTLASS_INCLUDE` at a CUTLASS `include/` directory. `SOLAR_ATTN_CPU_CACHE=24` keeps
every block's attention output for the backward pass instead of recomputing it, 24 blocks of them in pinned host memory
(about 25 GB per GPU); raise it if GPU memory runs short. `SOLAR_SP_PACK_B1=1` packs the sequence-parallel exchange
directly by destination. These switches were measured and checked on this step only.

**3. Post-train the four-step LoRA with structure guidance** (800 steps at global batch 4). Convert the public DMAD LoRA first,
then point `train.dmd.init_lora` at it and `train.dmd.control` at the control run's checkpoint directory from step 2 (the
one holding `adapter.pt`):

```bash
hf download ZhengmingYu/DMAD minimax_h3/dmad_minimax_h3_4step_lora_critic.safetensors --local-dir ./dmad
python scripts/convert_dmad.py --h3 ./MiniMax-H3 \
    --src ./dmad/minimax_h3/dmad_minimax_h3_4step_lora_critic.safetensors --out ./weights/dmad_4step_lora.safetensors
torchrun --standalone --nproc-per-node 8 -m solarwm train --config configs/structure_guided_4step.yaml
```

The structure guidance is `train.dmd.structure_guidance` (1 = plain distribution matching); the released weights use
1.5; take the live adapter at the last step. To render with your own runs, export each run's LoRA with
`python scripts/export_lora.py --checkpoint <checkpoint dir> --out <file>.safetensors`.

## Causal version

The causal LoRAs follow the same two stages on top of the bidirectional ones. As above, the released causal LoRAs
saw the same number of samples as the configurations below, at a larger batch.

**4. Train the causal control LoRA** (flow matching with block-causal attention and teacher forcing, 19,200 steps at
global batch 2). It starts from the bidirectional control LoRA: point `checkpoint.initialization.student.path` at the
control run's checkpoint directory from step 2.

```bash
torchrun --standalone --nproc-per-node 8 -m solarwm train --config configs/causal_control_lora.yaml
```

**5. Post-train the causal four-step LoRA with structure guidance** (1,600 steps at global batch 1). The causal control
LoRA from step 4 (`train.dmd.student_init`) is frozen in the student's base; the student is a separate four-step LoRA
that starts as the bidirectional one (`train.dmd.accel_init`) and is distilled on its own block-by-block streams, with
the bidirectional control LoRA from step 2 (`train.dmd.control`) as the teacher:

```bash
torchrun --standalone --nproc-per-node 8 -m solarwm train --config configs/causal_structure_guided_4step.yaml
```

Export both runs with `scripts/export_lora.py` and render with `scripts/render.py --causal`.

