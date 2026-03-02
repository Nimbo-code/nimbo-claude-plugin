# Training Parameter Reference

## TrainingConfig — All Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | str | `"./nimbo_output"` | Directory for checkpoints and final model |
| `per_device_train_batch_size` | int | `1` | Training batch size per GPU |
| `per_device_eval_batch_size` | int | `1` | Evaluation batch size per GPU |
| `gradient_accumulation_steps` | int | `8` | Steps to accumulate before weight update |
| `num_train_epochs` | int | `3` | Number of training epochs |
| `max_steps` | int | `-1` | Max steps (-1 = use epochs) |
| `learning_rate` | float | `2e-4` | Peak learning rate |
| `lr_scheduler_type` | str | `"cosine"` | LR schedule: cosine, linear, constant, etc. |
| `warmup_ratio` | float | `0.1` | Fraction of steps for warmup |
| `warmup_steps` | int | `0` | Override warmup with step count (0 = use ratio) |
| `optim` | str | `"adamw_torch"` | Optimizer: adamw_torch, adamw_8bit, adafactor |
| `weight_decay` | float | `0.01` | L2 regularization |
| `max_grad_norm` | float | `1.0` | Gradient clipping threshold |
| `fp16` | bool | `False` | Mixed precision FP16 (pre-Ampere NVIDIA) |
| `bf16` | bool | `False` | Mixed precision BF16 (Ampere+, Apple Silicon) |
| `fp16_full_eval` | bool | `False` | FP16 during evaluation |
| `bf16_full_eval` | bool | `False` | BF16 during evaluation |
| `gradient_checkpointing` | bool | `False` | Trade compute for memory savings |
| `max_length` | int | `1024` | Max sequence length (tokens) |
| `logging_steps` | int | `10` | Log every N steps |
| `logging_first_step` | bool | `True` | Log the first training step |
| `eval_strategy` | str | `"steps"` | Evaluate every N steps or "epoch" |
| `eval_steps` | int | `100` | Evaluate every N steps |
| `save_strategy` | str | `"steps"` | Save checkpoint every N steps or "epoch" |
| `save_steps` | int | `100` | Save every N steps |
| `save_total_limit` | int | `3` | Max checkpoints to keep |
| `load_best_model_at_end` | bool | `True` | Load best checkpoint after training |
| `metric_for_best_model` | str | `"eval_loss"` | Metric for best model selection |
| `greater_is_better` | bool | `False` | Whether higher metric is better |
| `early_stopping_patience` | int | `3` | Epochs without improvement before stopping |
| `early_stopping_threshold` | float | `0.0` | Minimum improvement threshold |
| `seed` | int | `42` | Random seed |
| `dataloader_num_workers` | int | `0` | Data loading workers |
| `dataloader_pin_memory` | bool | `True` | Pin memory for faster GPU transfer |
| `train_on_responses_only` | bool | `False` | Only compute loss on response tokens |
| `extra_kwargs` | dict | `{}` | Pass-through to HuggingFace SFTConfig |

### Recommended Ranges

| Parameter | Small Model (<3B) | Medium (3-8B) | Large (8B+) |
|-----------|-------------------|---------------|-------------|
| `learning_rate` | 1e-4 to 5e-4 | 5e-5 to 2e-4 | 1e-5 to 1e-4 |
| `per_device_train_batch_size` | 2-4 | 1-2 | 1 |
| `gradient_accumulation_steps` | 2-4 | 4-8 | 8-16 |
| `num_train_epochs` | 3-5 | 2-3 | 1-3 |
| `max_length` | 512-2048 | 512-1024 | 512-1024 |

---

## LoRAConfig — All Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `r` | int | `8` | LoRA rank (4, 8, 16, 32, 64) |
| `lora_alpha` | int | `16` | Scaling factor (usually 2x rank) |
| `lora_dropout` | float | `0.1` | Dropout probability |
| `bias` | str | `"none"` | Bias training: "none", "all", "lora_only" |
| `task_type` | str | `"CAUSAL_LM"` | Task type for PEFT |
| `target_modules` | list/None | `None` | Modules to apply LoRA (None = auto-detect) |
| `modules_to_save` | list/None | `None` | Additional modules to train fully |
| `init_lora_weights` | bool/str | `True` | Init method: True, "olora", "pissa", "loftq", "gaussian" |
| `use_rslora` | bool | `False` | Rank-stabilized LoRA scaling |
| `use_dora` | bool | `False` | Weight-Decomposed LoRA |
| `extra_kwargs` | dict | `{}` | Pass-through to PEFT LoraConfig |

### Rank Selection Guide

| Rank (r) | Parameters | Quality | Memory | Use Case |
|----------|-----------|---------|--------|----------|
| 4 | Minimal | Basic adaptation | Lowest | Quick experiments |
| 8 | Low | Good | Low | Default, most tasks |
| 16 | Medium | Better | Medium | Complex tasks, instruction tuning |
| 32 | High | High | High | Domain-specific, quality-focused |
| 64 | Very High | Highest | Highest | Near full fine-tuning quality |

---

## QuantizationConfig — All Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `load_in_4bit` | bool | `False` | 4-bit quantization (QLoRA) |
| `load_in_8bit` | bool | `False` | 8-bit quantization |
| `bnb_4bit_compute_dtype` | str | `"float16"` | Compute dtype: "float16" or "bfloat16" |
| `bnb_4bit_quant_type` | str | `"nf4"` | Quantization type: "nf4" or "fp4" |
| `bnb_4bit_use_double_quant` | bool | `True` | Double quantization for extra savings |

**Notes:**
- Requires `bitsandbytes` package (CUDA only)
- `load_in_4bit` and `load_in_8bit` are mutually exclusive
- Use `bnb_4bit_compute_dtype="bfloat16"` on Ampere+ GPUs for best performance
- `nf4` (NormalFloat4) generally outperforms `fp4`

---

## KernelConfig — All Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_triton_kernels` | bool | `True` | Enable Triton kernel patching |
| `patch_rms_norm` | bool | `True` | RMSNorm kernel (7-8x speedup) |
| `patch_swiglu` | bool | `True` | SwiGLU kernel (3-5x speedup) |
| `patch_rope` | bool | `True` | RoPE kernel (1.9-2.3x speedup) |
| `patch_attention` | bool | `False` | Attention kernel (experimental) |
| `use_flash_attention` | bool | `True` | Flash Attention 2 |

**Notes:**
- Triton kernels require CUDA and the `triton` package
- Flash Attention requires `flash-attn` package
- Not all models support all kernel patches (see supported model list)
