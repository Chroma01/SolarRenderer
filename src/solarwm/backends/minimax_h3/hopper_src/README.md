FlashAttention Hopper (FlashAttention 3.0.0.b1, `hopper/`) sources as shipped in the NGC PyTorch 25.06 container
(`third_party/flash-attention`), BSD-3-Clause; see LICENSE and AUTHORS. The two `*_hdim128_bf16_sm90.cu` files are
the upstream `hopper/instantiations` files and `kernel_traits.h` comes from `csrc/flash_attn/src`. The one change is
in `tile_size.h`: an opt-in 192x128 forward tile for dense non-causal attention (`SOLAR_HOPPER_FWD192=1`).
Only the bf16, head dimension 128, dense forward/backward kernels are compiled. CUTLASS headers are not included here;
see `hopper_attention.py`.
