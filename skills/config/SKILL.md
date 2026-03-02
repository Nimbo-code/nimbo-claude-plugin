---
name: config
description: "Create and manage Nimbo configuration files in YAML or JSON. Use when the user wants to create a training config, modify settings, generate a config file, or load existing configurations. Triggers on: config, settings, YAML, configuration file."
allowed-tools: Read, Write, Glob, Grep
---

# Nimbo Configuration Management

You are helping the user create, modify, or load Nimbo configuration files.

## Config File Formats

Nimbo supports both YAML and JSON configuration files. YAML is recommended for readability.

## Creating a Config File

### From Scratch

Generate a YAML config based on the user's requirements:

```yaml
device:
  device: null                    # null = auto-detect (cuda > mps > cpu)
  device_map: null                # "auto" for multi-GPU

lora:
  r: 16                           # LoRA rank
  lora_alpha: 32                  # Usually 2x rank
  lora_dropout: 0.05
  bias: none
  task_type: CAUSAL_LM
  target_modules: null            # null = auto-detect per model architecture
  modules_to_save: null
  init_lora_weights: true
  use_rslora: false
  use_dora: false

training:
  output_dir: ./nimbo_output
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  max_steps: -1
  learning_rate: 0.0001
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  warmup_steps: 0
  optim: adamw_torch
  weight_decay: 0.01
  max_grad_norm: 1.0
  fp16: false
  bf16: true
  gradient_checkpointing: true
  max_length: 1024
  logging_steps: 10
  logging_first_step: true
  eval_strategy: steps
  eval_steps: 100
  save_strategy: steps
  save_steps: 100
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  early_stopping_patience: 3
  early_stopping_threshold: 0.0
  seed: 42
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  train_on_responses_only: false

inference:
  max_new_tokens: 256
  do_sample: true
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1
  num_beams: 1
  num_return_sequences: 1
  stream: false
  use_cache: true

quantization:
  load_in_4bit: false
  load_in_8bit: false
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true

kernels:
  use_triton_kernels: true
  patch_rms_norm: true
  patch_swiglu: true
  patch_rope: true
  patch_attention: false
  use_flash_attention: true
```

### From Existing Trainer

```python
from nimbo import Nimbo
trainer = Nimbo(base_model_name="microsoft/phi-2", dataset="my_data")
trainer.save_config("nimbo_config.yaml")   # or .json
```

## Loading a Config File

### In Python

```python
from nimbo import Nimbo, NimboConfig

# Load config and create trainer
trainer = Nimbo.from_config(
    config_path="nimbo_config.yaml",
    base_model_name="microsoft/phi-2",
    dataset="my_dataset",
)

# Or load config object directly
config = NimboConfig.from_yaml("nimbo_config.yaml")
config = NimboConfig.from_json("nimbo_config.json")
```

### Saving Config Object

```python
config.to_yaml("output_config.yaml")
config.to_json("output_config.json")
```

## Config Templates

### Quick Experiment (Small Model, Low VRAM)

```yaml
lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.1
training:
  num_train_epochs: 1
  learning_rate: 0.0003
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  max_length: 512
  bf16: true
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
```

### Production (Quality-Focused)

```yaml
lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  use_rslora: true
training:
  num_train_epochs: 5
  learning_rate: 0.00005
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_length: 2048
  bf16: true
  gradient_checkpointing: true
  early_stopping_patience: 5
  save_total_limit: 5
```

### iOS Deployment Pipeline

```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
training:
  num_train_epochs: 3
  learning_rate: 0.0001
  bf16: true
  gradient_checkpointing: true
  max_length: 512
kernels:
  use_triton_kernels: true
```

## Guidelines

- Always write the config file for the user, don't just show it
- Adapt settings based on the user's hardware (ask if unknown)
- Use `bf16: true` for Ampere+ NVIDIA GPUs and Apple Silicon
- Use `fp16: true` for older NVIDIA GPUs
- Set `gradient_checkpointing: true` for models > 3B parameters
- Only include sections the user needs — omitted sections use defaults
