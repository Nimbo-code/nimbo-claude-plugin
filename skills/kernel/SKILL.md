---
name: kernel
description: "Optimize training speed with Triton kernels in Nimbo. Use when the user wants to speed up training, apply Triton optimizations, use custom kernels for RMSNorm, SwiGLU, or RoPE. Triggers on: Triton, kernel, optimize, speed up training, RMSNorm, SwiGLU, RoPE."
allowed-tools: Bash, Read, Write, Glob
---

# Triton Kernel Optimization with Nimbo

You are helping the user apply Triton kernel patches to accelerate LLM training.

## Prerequisites

- **CUDA GPU required** — Triton kernels only work on NVIDIA GPUs
- **triton package** — `pip install triton`

## Available Kernels

| Kernel | Speedup | Description |
|--------|---------|-------------|
| RMSNorm | **7-8x** | Fused Root Mean Square Layer Normalization |
| SwiGLU | **3-5x** | Fused SiLU activation with gated linear unit |
| RoPE | **1.9-2.3x** | Rotary Position Embeddings (including fused Q+K variant) |
| Attention | Experimental | Custom attention kernel (disabled by default) |

## Supported Models

**Full Triton support (dedicated patchers):**
- LLaMA 2 (7B, 13B, 70B)
- LLaMA 3 (8B, 70B)
- LLaMA 3.2 (1B, 3B)
- Mistral (7B) — via LlamaPatcher
- Phi (2, 3, 3.5) — via LlamaPatcher
- Qwen2 (0.5B–7B) — via LlamaPatcher
- EXAONE 3.5 (2.4B, 7.8B, 32B) — via ExaonePatcher
- EXAONE 4.0 (1.2B, 32B) — via ExaonePatcher

**Not supported for Triton:**
GPT-2, GPT-NeoX, Falcon, BLOOM, OPT, Qwen v1, Gemma, Gemma2

## Method 1: Automatic (via Nimbo Constructor)

The simplest way — kernels are applied automatically during training:

```python
from nimbo import Nimbo, KernelConfig

kernel_config = KernelConfig(
    use_triton_kernels=True,
    patch_rms_norm=True,       # 7-8x speedup
    patch_swiglu=True,         # 3-5x speedup
    patch_rope=True,           # 1.9-2.3x speedup
    patch_attention=False,     # Experimental, keep disabled
    use_flash_attention=True,  # Requires flash-attn package
)

trainer = Nimbo(
    base_model_name="meta-llama/Llama-3.2-1B",
    dataset="my_data",
    kernel_config=kernel_config,
    use_triton_kernels=True,
)
trainer.train()
```

Or more simply:
```python
trainer = Nimbo(
    base_model_name="meta-llama/Llama-3.2-1B",
    dataset="my_data",
    use_triton_kernels=True,      # Enables all Triton kernels
    use_flash_attention=True,     # Enables Flash Attention 2
)
```

## Method 2: Manual Patching

For advanced users who want to patch models outside of the Nimbo trainer:

```python
from nimbo.kernels import patch_model, unpatch_model, get_supported_models, is_triton_available
from transformers import AutoModelForCausalLM

# Check availability
print(f"Triton available: {is_triton_available()}")
print(f"Supported models: {get_supported_models()}")

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Apply patches
stats = patch_model(
    model,
    rms_norm=True,
    swiglu=True,
    rope=True,
    attention=False,
)
print(stats)
# PatchStats(rms_norm_count=16, swiglu_count=16, rope_count=16, attention_count=0)

# ... use model for training ...

# Remove patches when done
unpatch_model(model)
```

## Method 3: Individual Kernel Modules

For fine-grained control:

### RMSNorm (7-8x speedup)

```python
from nimbo.kernels import NimboRMSNorm

# Drop-in replacement for nn.RMSNorm
norm = NimboRMSNorm(hidden_size=2048, eps=1e-6)
output = norm(hidden_states)
```

### SwiGLU (3-5x speedup)

```python
from nimbo.kernels import NimboSwiGLU, swiglu

# As a module
activation = NimboSwiGLU()
output = activation(gate, up)

# As a function
output = swiglu(gate, up)
# Computes: silu(gate) * up, where silu(x) = x * sigmoid(x)
```

### RoPE (1.9-2.3x speedup)

```python
from nimbo.kernels import NimboRoPE, apply_rotary_pos_emb

# As a module
rope = NimboRoPE(dim=64, max_seq_len=2048)
q_rot, k_rot = rope(q, k, position_ids)

# HuggingFace-compatible function
q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

# Fused Q+K variant (faster)
from nimbo.kernels import triton_apply_rotary_pos_emb
q_rot, k_rot = triton_apply_rotary_pos_emb(q, k, cos, sin)
```

## Verification

After applying patches, verify they're active:

```bash
python3 -c "
from nimbo.kernels import is_triton_available(), get_supported_models
print(f'Triton available: {is_triton_available()}')
print(f'Supported models: {get_supported_models()}')
"
```

## Guidelines

- Always check `is_triton_available()` before attempting kernel patches
- Triton kernels are CUDA-only — they don't work on MPS or CPU
- Flash Attention requires the separate `flash-attn` package
- Keep `patch_attention=False` unless explicitly testing
- Kernel patches are most impactful for larger models (3B+)
- For small models (< 1B), the overhead may negate benefits
- Patches are automatically applied when using `use_triton_kernels=True` in the Nimbo constructor
