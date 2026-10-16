"""AnyFlow flow-map distillation on omni samples, for either way of reading the condition.

The sample is prepared exactly as the ordinary omni recipe prepares it - the aligned observation
through the control trunk (a ControlNet run) or packed beside the target (a concatenation run),
free pictures through the prompt - and only the target rows move: every evaluation names where
they go (r), and everything else stays at its own time. The four AnyFlow evaluations share one
set of conditions; ``h3_anyflow_v15_loss`` holds the objective.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from solarwm.training.anyflow import apply_timestep_shift, sample_anyflow_time_pairs

from .anyflow_loss import h3_anyflow_v15_loss
from .distributed import get_dp_group, get_dp_rank, get_dp_world_size
from .layout import patchify_video
from .stage0p5 import H3Stage0p5Core


class H3OmniAnyFlowCore(H3Stage0p5Core):
    """Stage0.5 omni inputs with a flow-map objective and few-step sampling."""

    def __init__(
        self,
        model: Any,
        silence_latents: Any,
        device: Any,
        train: Any,
        *,
        geometry: Any = None,
        observations: Any = None,
    ) -> None:
        super().__init__(
            model,
            silence_latents,
            device,
            geometry=geometry,
            observations=observations,
            flow_map=True,
        )
        self.train = train

    def forward_loss(self, batch: Any, *, noise_seed: int | None) -> Any:
        import torch

        generator = (
            None
            if noise_seed is None
            else torch.Generator(device=self.device).manual_seed(int(noise_seed))
        )
        drawn: dict[str, Any] = {}

        def draw_time(draw: Any) -> Any:
            pairs = sample_anyflow_time_pairs(
                1,
                logical_dp_rank=get_dp_rank(),
                logical_dp_world_size=get_dp_world_size(),
                diffusion_ratio=float(self.train["diffusion_ratio"]),
                consistency_ratio=float(self.train["consistency_ratio"]),
                generator=draw,
                device=self.device,
            )
            self._broadcast_logical_sample(*pairs)
            drawn["sigma_t"] = apply_timestep_shift(pairs.t, 12.0)
            drawn["sigma_r"] = apply_timestep_shift(pairs.r, 12.0)
            drawn["is_diffusion"] = pairs.is_diffusion
            return (1.0 - drawn["sigma_t"]).reshape(()).float()

        # A flow map carries every target row from one time to another; per-row known degrees
        # from the data (edit masks) would hold rows still, which the objective has no place for.
        inputs = self.prepare(dataclasses.replace(batch, known_degrees=None), generator, draw_time)
        # The plan's row times are not read here: the flow map carries every target row from
        # sigma_t itself. Only its pattern matters (a time at the clean end frees the plan's last
        # frame alone, still under the shared pattern, and must train like any other).
        plan = inputs.plan
        if plan is not None and plan.pattern != "shared":
            raise ValueError("flow-map distillation needs the shared observation pattern alone")
        condition_rows = inputs.condition_rows

        def velocity(sample: Any, t: Any, r: Any) -> Any:
            return self._forward(
                video_rows=torch.cat((*condition_rows, sample), dim=1),
                audio_rows=inputs.audio_rows,
                prompt=inputs.prompt,
                layout=inputs.layout,
                video_t=t.reshape(()),
                audio_t=inputs.audio_t,
                viewmats=inputs.viewmats,
                K=inputs.K,
                frame_ids=inputs.frame_ids,
                geometry_control=inputs.geometry,
                observation_degrees=inputs.observation_degrees,
                video_r=r.reshape(()),
            )

        return h3_anyflow_v15_loss(
            velocity,
            patchify_video(inputs.clean),
            patchify_video(inputs.video_noise),
            drawn["sigma_t"],
            drawn["sigma_r"],
            drawn["is_diffusion"],
            epsilon=float(self.train["finite_difference_epsilon"]),
            shift=12.0,
            process_group=get_dp_group(),
        )


__all__ = ["H3OmniAnyFlowCore"]
