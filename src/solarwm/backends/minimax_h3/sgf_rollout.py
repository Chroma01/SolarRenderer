"""H3's two-pass SGF: detached raw-KV rollout, then parallel gradient replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .distributed import broadcast_sp_tensor as broadcast_sequence_parallel_tensor
from .layout import build_stage0p5_layout, build_stage1_layout, patchify_video, unpatchify_video
from .sgf import H3SGFWindow, h3_sgf_schedule, h3_sgf_windows
from .sgf_attention import H3RawKVCache, H3SGFAttention
from .torch_flow import predict_clean_sample, scale_noise
from .torch_geometry import native_video_position_grid
from .torch_layout import build_row_timesteps


@dataclass
class H3SGFInputs:
    prompt: torch.Tensor
    text_tags: torch.Tensor
    anchor_rows: torch.Tensor
    audio_rows: torch.Tensor
    audio_timestep: torch.Tensor
    camera_viewmats: torch.Tensor  # [B,51,4,4], anchor + 50 target cameras
    camera_K: torch.Tensor
    latent_height: int
    latent_width: int
    keyframe_timestep: float = 0.999
    sp_enabled: bool = False
    layouts: dict[str, Any] = field(default_factory=dict)
    # Geometry ControlNet profile and ``{name: [1,24,F,48,84]}`` renders covering
    # every rollout latent (tails repeat the last encoded latent).
    geometry: Any = None
    geometry_latents: dict[str, Any] | None = None

    def layout(self, mode):
        if mode not in self.layouts:
            builder = build_stage1_layout if mode == "replay" else build_stage0p5_layout
            self.layouts[mode] = builder(
                self.text_tags.cpu(),
                50 if mode == "replay" else 5,
                self.latent_height,
                self.latent_width,
                self.audio_rows.shape[1] // 2,
            ).to(self.prompt.device)
        return self.layouts[mode]


@dataclass(frozen=True)
class H3SGFRollout:
    cache_target: torch.Tensor
    noisy_at_exit: torch.Tensor
    exit_index: int
    native_exit_timestep: torch.Tensor


def _random_like(value, *, inputs, generator=None):
    noise = torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
    if inputs.sp_enabled:
        broadcast_sequence_parallel_tensor(noise)
    return noise


def _window_positions(layout, chunk_index: int):
    """Rotary grid of chunk ``chunk_index`` as the query of its own sliding window.

    The student's backbone rotates every chunk's queries at these local positions
    (``sgf_attention``); the chunk-local control trunk uses the same ones, so a
    chunk's control is identical in its rollout, its commit and the replay.
    """

    first = max(0, int(chunk_index) - 5)
    local_frames = (int(chunk_index) - first + 1) * 5
    grid = native_video_position_grid(
        local_frames,
        layout.latent_height,
        layout.latent_width,
        patch_size=layout.patch_size,
        origin=float(layout.text_indices.numel()),
    )
    return grid[-5 * int(layout.rows_per_video_frame) :]


def sgf_geometry_control(inputs: H3SGFInputs, layout, mode: str, chunk_index: int):
    """Chunk-local geometry control for a rollout/commit chunk or the full replay."""

    if inputs.geometry is None:
        return None
    if inputs.geometry_latents is None:
        raise ValueError("H3 SGF geometry run has no geometry latents")
    from .controlnet import geometry_control_for_layout

    positions = layout.position_ids.clone()
    rows = layout.video_indices[layout.num_condition_video_rows :]
    if mode == "rollout":
        frames, copies = slice(chunk_index * 5, (chunk_index + 1) * 5), 1
        local = _window_positions(layout, chunk_index)
    else:
        frames, copies = slice(0, 50), 2
        chunks = torch.cat([_window_positions(layout, index) for index in range(10)])
        local = torch.cat((chunks, chunks))
    positions[rows] = local.to(device=positions.device, dtype=positions.dtype)

    def patches(name: str):
        window = patchify_video(inputs.geometry_latents[name][:, :, frames])
        if int(window.shape[1]) * copies != int(rows.numel()):
            raise ValueError("H3 SGF geometry latents do not cover the rollout window")
        return torch.cat([window] * copies, dim=1)

    profile = inputs.geometry
    return geometry_control_for_layout(
        layout,
        {name: patches(name) for name in profile.modalities},
        profile=profile,
        wide_rows=(
            {name: patches(f"{name}_wide") for name in profile.wide_modalities}
            if profile.wide_enabled
            else None
        ),
        position_ids=positions,
        row_indices=rows,
        attention="segments",
    )


def h3_student_forward(
    student,
    inputs: H3SGFInputs,
    noisy,
    timestep,
    *,
    clean=None,
    chunk_index=0,
    cache=None,
    commit_cache=False,
):
    mode = "replay" if clean is not None else "rollout"
    layout = inputs.layout(mode)
    frames = 50 if mode == "replay" else 5
    video_parts = [inputs.anchor_rows]
    if clean is not None:
        video_parts.append(patchify_video(clean.detach()))
    video_parts.append(patchify_video(noisy))
    times, indices = build_row_timesteps(
        layout,
        timestep,
        inputs.audio_timestep,
        text_timestep=1.0,
        condition_video_timestep=inputs.keyframe_timestep,
        device=noisy.device,
    )
    control = H3SGFAttention(
        layout,
        inputs.camera_viewmats,
        inputs.camera_K,
        mode,
        chunk_index=chunk_index,
        cache=cache,
        commit_cache=commit_cache,
    )
    geometry = sgf_geometry_control(inputs, layout, mode, chunk_index)
    extra = {} if geometry is None else {"geometry_control": geometry}
    # SGF's camera controls are global immutable metadata. The dedicated
    # attention path selects local query/KV camera rows after Ulysses exchange.
    prediction, _ = student(
        hidden_states=torch.cat(video_parts, dim=1),
        audio_hidden_states=inputs.audio_rows,
        encoder_hidden_states=inputs.prompt,
        timestep=times,
        timestep_indices=indices,
        fused_prope=True,
        packed_sequence_parallel=inputs.sp_enabled,
        sgf_attention=control,
        return_dict=False,
        **layout.transformer_kwargs(),
        **extra,
    )
    velocity = unpatchify_video(
        prediction[:, layout.noisy_video_output_slice],
        frames,
        inputs.latent_height,
        inputs.latent_width,
    )
    return predict_clean_sample(noisy, velocity, timestep)


def h3_sgf_rollout(
    *,
    student,
    inputs: H3SGFInputs,
    noise,
    exit_index=None,
    per_rank_exit_step=True,
    last_step_only=False,
    generator=None,
    video_shift=12.0,
    progress=None,
    inference_full_length=False,
    output_device=None,
):
    if output_device is not None and not inference_full_length:
        raise ValueError("Output offload is inference-only")
    if inference_full_length:
        count = int(noise.shape[2])
        if count <= 0 or count % 5 or exit_index != 3:
            raise ValueError(
                "Full-length H3 inference requires complete five-latent chunks and exit_index=3"
            )
        if inputs.camera_viewmats.shape[1] != count + 1 or inputs.camera_K.shape[1] != count + 1:
            raise ValueError("Full-length H3 inference requires anchor plus every rollout camera")
        windows = tuple(
            H3SGFWindow(i, max(0, i - 5) * 5, i * 5, (i + 1) * 5, (i + 1) * 5)
            for i in range(count // 5)
        )
    elif noise.shape[2] != 50:
        raise ValueError("H3 SGF requires exactly 50 rollout latents")
    else:
        windows = h3_sgf_windows()
    schedule, _ = h3_sgf_schedule(video_shift=video_shift, device=noise.device)
    if exit_index is None:
        selected = (
            torch.full((), 3, device=noise.device, dtype=torch.long)
            if last_step_only
            else torch.randint(0, 4, (), device=noise.device, generator=generator)
        )
        if not per_rank_exit_step and torch.distributed.is_initialized():
            torch.distributed.broadcast(selected, src=0)
        if inputs.sp_enabled:
            broadcast_sequence_parallel_tensor(selected)
        exit_index = int(selected)
    if not 0 <= exit_index < 4:
        raise ValueError("H3 SGF exit index must lie in [0,4)")
    cache = H3RawKVCache()
    committed, noisy_exits = [], []
    try:
        with torch.no_grad():
            for window in windows:
                current = noise[:, :, window.current_start : window.stop]
                exit_x0 = None
                # Keep four forwards on every rank even when exit choices
                # differ across DP: FSDP collectives require identical counts.
                for step_index, timestep in enumerate(schedule.timesteps):
                    if step_index == exit_index:
                        noisy_exits.append(
                            current.detach().to(output_device).clone()
                            if output_device
                            else current.detach().clone()
                        )
                    x0 = h3_student_forward(
                        student,
                        inputs,
                        current,
                        timestep,
                        chunk_index=window.chunk_index,
                        cache=cache,
                    )
                    if not bool(torch.isfinite(x0).all()):
                        raise FloatingPointError(
                            f"non-finite H3 SGF chunk={window.chunk_index}, denoise={step_index}"
                        )
                    if step_index == exit_index:
                        exit_x0 = x0.detach().clone()
                    if step_index < 3:
                        current = scale_noise(
                            x0,
                            _random_like(x0, inputs=inputs, generator=generator),
                            schedule.timesteps[step_index + 1],
                        )
                committed.append(exit_x0.to(output_device) if output_device else exit_x0)
                h3_student_forward(
                    student,
                    inputs,
                    exit_x0,
                    noise.new_tensor(1.0),
                    chunk_index=window.chunk_index,
                    cache=cache,
                    commit_cache=True,
                )
                if progress is not None:
                    progress(window.chunk_index + 1)
        return H3SGFRollout(
            torch.cat(committed, dim=2),
            torch.cat(noisy_exits, dim=2),
            exit_index,
            schedule.timesteps[exit_index],
        )
    finally:
        cache.clear()


def h3_sgf_replay(*, student, inputs, rollout):
    return h3_student_forward(
        student,
        inputs,
        rollout.noisy_at_exit,
        rollout.native_exit_timestep,
        clean=rollout.cache_target,
    )
