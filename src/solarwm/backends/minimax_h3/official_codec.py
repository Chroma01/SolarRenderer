"""Official Qwen/VisualVAE/AudioVAE codec shared by online and preencode paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from solarwm.errors import DataContractError
from solarwm.preencode import EncoderContract, validate_encoded_tensors

from .artifacts import h3_encoder_contract, h3_silence_profile
from .codec import H3Codec, H3RawSample, validate_raw_sample

H3_TEXT_ENCODER_LAYER = 50
H3_VIDEO_VAE_SEED = 42
H3_GEOMETRY_PIXEL_SHAPE = (158, 768, 1344, 3)
H3_AUDIO_HOP_LENGTH = 800
H3_AUDIO_LATENTS = 263
H3_SILENCE_METADATA = {"artifact": "official-audiovae-encoded-stereo-zero-waveform"}


def _pixels(value: Any, *, device: Any) -> Any:
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 4 and tensor.shape[-1] == 3:
        tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
    elif tensor.ndim == 5 and tensor.shape[1] == 3:
        pass
    else:
        raise DataContractError("H3 pixels must be [T,H,W,3] or [B,3,T,H,W]")
    if tuple(tensor.shape) != (1, 3, 158, 768, 1344):
        raise DataContractError(f"H3 pixels must be [1,3,158,768,1344], got {tuple(tensor.shape)}")
    return tensor.to(device=device, non_blocking=True).contiguous()


def encode_geometry_clips(video_vae: Any, read_frames: Any, num_frames: int, *, device: Any) -> Any:
    """Encode a render of any length to normalized ``[24,T,H/16,W/16]`` BF16 latents.

    ``read_frames(start, stop)`` returns ``[n,H,W,3]`` pixels in ``[0, 1]``. The
    H3 video VAE encodes every ``clip_length`` window independently (padding the
    tail by repeating the last frame) and drops ``token_drop`` trailing latents,
    so streaming its clips equals encoding the whole render while holding one
    clip in memory. Renders share the RGB path's ImageNet normalization and
    latent mean/std, but take the posterior mode: a conditioning signal is one
    deterministic encoding, not a sample of the encoder noise.
    """

    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    if num_frames < 2:
        raise DataContractError("H3 geometry renders need at least two frames")
    clip = int(video_vae.config.clip_length)
    drop = int(video_vae.config.token_drop)
    dtype = next(video_vae.encoder.parameters()).dtype
    pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, -1, 1, 1, 1)
    moments = []
    with torch.inference_mode():
        for start in range(0, num_frames, clip):
            frames = torch.as_tensor(
                np.asarray(read_frames(start, min(start + clip, num_frames))),
                dtype=torch.float32,
            )
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise DataContractError("H3 geometry render frames must be [n,H,W,3]")
            if frames.shape[0] < clip:
                tail = frames[-1:].expand(clip - frames.shape[0], -1, -1, -1)
                frames = torch.cat((frames, tail), dim=0)
            pixels = frames.permute(3, 0, 1, 2).unsqueeze(0).to(device, non_blocking=True)
            normalized = ((pixels - pixel_mean) / pixel_std).to(dtype)
            moments.append(video_vae._encode_clip(normalized).float().cpu())
    stacked = torch.cat(moments, dim=2)
    if drop:
        stacked = stacked[:, :, :-drop]
    latent = DiagonalGaussianDistribution(stacked).mode()
    mean = torch.tensor(video_vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(video_vae.config.latents_std).view(1, -1, 1, 1, 1)
    return ((latent - mean) / std).to(torch.bfloat16)[0].contiguous()


def encode_joint_prompt_condition(
    keyframe: Any, caption: str, *, processor: Any, tokenizer: Any, text_encoder: Any, device: Any
) -> tuple[Any, Any]:
    import torch
    from PIL import Image

    if not isinstance(keyframe, Image.Image):
        array = np.asarray(keyframe, dtype=np.uint8)
        keyframe = Image.fromarray(array, mode="RGB")
    vision = processor.image_processor(images=[keyframe], return_tensors="pt")
    grid = vision["image_grid_thw"]
    merge = processor.image_processor.merge_size**2
    image_tokens = int(grid[0].prod()) // merge
    label_ids = tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
    vision_ids = (
        [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
        + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * image_tokens
        + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
    )
    caption_ids = tokenizer(str(caption), add_special_tokens=False)["input_ids"]
    token_ids = label_ids + vision_ids + caption_ids
    tags = [1] * len(label_ids) + [0] * len(vision_ids) + [1] * len(caption_ids)
    if not token_ids:
        raise DataContractError("H3 Qwen presentation produced no tokens")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    token_types = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]),
        dtype=torch.long,
        device=device,
    )
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=token_types,
        use_cache=False,
        output_hidden_states=True,
        pixel_values=vision["pixel_values"].to(device, text_encoder.dtype),
        image_grid_thw=grid.to(device),
    )
    if len(outputs.hidden_states) <= H3_TEXT_ENCODER_LAYER:
        raise DataContractError("H3 Qwen has no hidden state 50")
    hidden = outputs.hidden_states[H3_TEXT_ENCODER_LAYER][0].to(torch.bfloat16).contiguous().cpu()
    tags_tensor = torch.tensor(tags, dtype=torch.long)
    if tuple(hidden.shape) != (len(token_ids), 5120):
        raise DataContractError("H3 Qwen hidden state has the wrong shape")
    return hidden, tags_tensor


class OfficialH3Codec(H3Codec):
    """Concrete official H3 encoder implementation."""

    def __init__(
        self,
        *,
        text_encoder: Any,
        tokenizer: Any,
        processor: Any,
        video_vae: Any,
        audio_vae: Any,
        device: Any,
        encoder_identity: str,
        geometry: Any = None,
    ) -> None:
        self.geometry = geometry
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.video_vae.set_attention_backend("native")
        self.audio_vae.set_attention_backend("native")
        self.device = device
        self.identity = str(encoder_identity).strip()
        if not self.identity:
            raise DataContractError("official H3 codec requires encoder identity")
        self._contract = h3_encoder_contract(encoder_identity=self.identity, geometry=geometry)
        self._silence: Any = None
        self._encoders_active = True

    @property
    def contract(self) -> EncoderContract:
        return self._contract

    def bind_silence_profile(self) -> None:
        """Freeze readable silence semantics into every row contract."""

        self._contract = h3_encoder_contract(
            encoder_identity=self.identity,
            silence_artifact_profile=h3_silence_profile(),
            geometry=self.geometry,
        )

    def silence_artifact_bytes(self) -> bytes:
        """Serialize the online/offline global silence artifact identically."""

        from safetensors.torch import save

        return save(
            {"silence_158f": self._silence_latents()},
            metadata=H3_SILENCE_METADATA,
        )

    def activate_encoders(self) -> None:
        """Move the per-sample video and prompt encoders onto their CUDA device."""

        if self._encoders_active:
            return
        self.text_encoder.to(self.device)
        self.video_vae.to(self.device)
        self._encoders_active = True

    def offload_encoders(self) -> None:
        """Release conditioner memory before the 33B training forward."""

        import torch

        self.text_encoder.to("cpu")
        self.video_vae.to("cpu")
        self.audio_vae.to("cpu")
        self._encoders_active = False
        if getattr(self.device, "type", str(self.device)) == "cuda":
            torch.cuda.empty_cache()

    def _video_latents(self, pixels: Any, *, seed: int) -> Any:
        import torch

        mean = torch.tensor(self.video_vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.video_vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=pixels.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor((0.229, 0.224, 0.225), device=pixels.device).view(1, -1, 1, 1, 1)
        normalized = (pixels.float().div(255.0) - pixel_mean) / pixel_std
        posterior = self.video_vae.encode(normalized, return_dict=False)[0]
        latent = posterior.sample(generator=torch.Generator().manual_seed(int(seed)))
        # This fp16 round before mean/std normalization is part of the required
        # cache identity, despite the final BF16 storage.
        latent = latent.to(torch.float16).float().cpu()
        return ((latent - mean) / std).to(torch.bfloat16).contiguous()

    def _geometry_latents(self, unit: Any) -> Any:
        """Encode one ``[158,768,1344,3]`` render in ``[0,1]`` to normalized latents."""

        shape = tuple(getattr(unit, "shape", ()))
        if shape != H3_GEOMETRY_PIXEL_SHAPE:
            raise DataContractError(
                f"H3 geometry render must be {list(H3_GEOMETRY_PIXEL_SHAPE)}, got {shape}"
            )
        return encode_geometry_clips(
            self.video_vae,
            lambda start, stop: unit[start:stop],
            int(shape[0]),
            device=self.device,
        )

    def _joint_prompt(self, keyframe: Any, caption: str) -> tuple[Any, Any]:
        return encode_joint_prompt_condition(
            keyframe,
            caption,
            processor=self.processor,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            device=self.device,
        )

    def _silence_latents(self) -> Any:
        import torch

        if self._silence is not None:
            return self._silence
        waveform = torch.zeros(
            (2, 1, H3_AUDIO_LATENTS * H3_AUDIO_HOP_LENGTH),
            dtype=torch.float32,
            device=self.device,
        )
        posterior = self.audio_vae.encode(waveform, return_dict=False)[0]
        latent = posterior.mode().float().cpu()
        mean = torch.tensor(self.audio_vae.config.latents_mean).view(1, -1, 1)
        std = torch.tensor(self.audio_vae.config.latents_std).view(1, -1, 1)
        self._silence = ((latent - mean) / std).to(torch.bfloat16).contiguous()
        if tuple(self._silence.shape) != (2, 32, H3_AUDIO_LATENTS):
            raise DataContractError("official AudioVAE silence is not [2,32,263]")
        return self._silence

    # H3Codec protocol -------------------------------------------------
    def encode_target_video(self, sample: H3RawSample) -> Any:
        validate_raw_sample(sample)
        return self._video_latents(
            _pixels(sample.frames, device=self.device), seed=H3_VIDEO_VAE_SEED
        )[0]

    def encode_visual_anchor(self, sample: H3RawSample) -> Any:
        validate_raw_sample(sample)
        pixels = _pixels(sample.frames, device=self.device)[:, :, :1]
        return self._video_latents(pixels, seed=H3_VIDEO_VAE_SEED)[0]

    def encode_joint_prompt(self, sample: H3RawSample) -> tuple[Any, Any]:
        validate_raw_sample(sample)
        return self._joint_prompt(np.asarray(sample.frames)[0], sample.caption)

    def encode_silence_audio(self, sample: H3RawSample) -> Any:
        validate_raw_sample(sample)
        return self._silence_latents()

    # offline preencode protocol --------------------------------------
    def encode(
        self,
        *,
        sample_id: str,
        pixels: Any,
        caption: str,
        camera: Any,
        seed: int,
        geometry: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Encode one sample; ``geometry`` maps artifact tensor keys to decoded renders."""

        import torch

        if not isinstance(camera, Mapping):
            raise DataContractError("H3 preencode camera must be a mapping")
        expected_geometry = set(self.geometry.tensor_keys()) if self.geometry is not None else set()
        if set(geometry or {}) != expected_geometry:
            raise DataContractError(
                f"H3 preencode geometry renders differ: got {sorted(geometry or {})}, "
                f"expected {sorted(expected_geometry)}"
            )
        c2w = torch.as_tensor(camera.get("camera_c2w"), dtype=torch.float32)
        K = torch.as_tensor(camera.get("camera_K"), dtype=torch.float32)
        source_indices = torch.as_tensor(camera.get("source_frame_indices"), dtype=torch.long)
        prepared = _pixels(pixels, device=self.device)
        with torch.inference_mode():
            target = self._video_latents(prepared, seed=int(seed))[0]
            anchor = self._video_latents(prepared[:, :, :1], seed=int(seed))[0]
            prompt, tags = self._joint_prompt(np.asarray(pixels)[0], str(caption))
            renders = {
                key: self._geometry_latents(value) for key, value in (geometry or {}).items()
            }
        values = {
            "target_latents": target,
            "anchor_latents": anchor,
            "prompt_embeds": prompt,
            "text_token_tags": tags,
            "source_frame_indices": source_indices.contiguous(),
            "camera_c2w": c2w.contiguous(),
            "camera_K": K.contiguous(),
            **renders,
        }
        validate_encoded_tensors(values, self.contract)
        return values


__all__ = [
    "H3_AUDIO_HOP_LENGTH",
    "H3_AUDIO_LATENTS",
    "H3_SILENCE_METADATA",
    "H3_TEXT_ENCODER_LAYER",
    "H3_VIDEO_VAE_SEED",
    "OfficialH3Codec",
    "encode_geometry_clips",
]
