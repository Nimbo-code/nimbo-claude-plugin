---
name: train
description: "Fine-tune LLMs with Nimbo using LoRA or QLoRA. Use when the user wants to fine-tune, train, adapt, or customize a language model. Handles model selection, configuration, training execution, and results. Triggers on: fine-tune, train, LoRA, QLoRA, adapt model."
allowed-tools: Bash, Read, Write, Glob, Grep
---

# LLM Fine-Tuning with Nimbo

You are helping the user fine-tune a language model using Nimbo's LoRA/QLoRA framework.

Refer to `skills/train/reference.md` for complete parameter details when configuring training.

## Step 1: Model Selection

Help the user choose a base model based on their hardware:

| VRAM | Recommended Models | Method |
|------|-------------------|--------|
| < 8GB | Phi-2 (2.7B), LLaMA 3.2 1B | QLoRA (4-bit) |
| 8-16GB | LLaMA 3.2 3B, Mistral 7B | QLoRA (4-bit) |
| 16-24GB | LLaMA 3 8B, Phi-3 | QLoRA or LoRA |
| 24GB+ | Mistral 7B, LLaMA 3 8B | Full LoRA |
| Multi-GPU | Any size | `device_map="auto"` |
| Apple Silicon | Phi-2, LLaMA 3.2 1B | LoRA (MPS) |

For CoreML/iOS deployment, recommend **LLaMA architecture** models (LLaMA 3.2 1B/3B).

## Step 2: Configure Training

Build the training script based on user requirements. Ask about:
1. **Base model** — HuggingFace model ID
2. **Dataset** — Path to data or HF dataset name (use the dataset skill if needed)
3. **Task type** — Continual pre-training vs instruction tuning
4. **Hardware** — CUDA GPU / Apple Silicon / CPU

### Minimal Configuration

```python
from nimbo import Nimbo

trainer = Nimbo(
    base_model_name="microsoft/phi-2",
    dataset="my_dataset",
)
trainer.train()
trainer.save()
```

### Full Configuration

```python
from nimbo import (
    Nimbo, LoRAConfig, TrainingConfig, QuantizationConfig,
    KernelConfig, DeviceConfig, prepare_instruction_dataset,
    create_default_callbacks, WandbCallback,
)

# Dataset
dataset = prepare_instruction_dataset(
    "data.jsonl",
    template="### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n{output}",
)

# Configs
lora = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.05)
training = TrainingConfig(
    output_dir="./output",
    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    bf16=True,
    gradient_checkpointing=True,
    max_length=1024,
    logging_steps=10,
    eval_steps=100,
    save_steps=100,
)
quant = QuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")

# Callbacks
callbacks = create_default_callbacks(enable_progress=True, early_stopping_patience=5)

# Train
trainer = Nimbo(
    base_model_name="meta-llama/Llama-3.2-1B",
    dataset=dataset,
    lora_config=lora,
    training_config=training,
    quantization_config=quant,
    use_triton_kernels=True,
    use_flash_attention=True,
    callbacks=callbacks,
)
metrics = trainer.train()
output_path = trainer.save(merge=True)
```

### From Config File

```python
from nimbo import Nimbo
trainer = Nimbo.from_config("nimbo_config.yaml", base_model_name="microsoft/phi-2", dataset="data")
trainer.train()
trainer.save()
```

## Step 3: Generate & Run Training Script

1. Write the training script to a file (e.g., `train.py`)
2. Run it:
```bash
python3 train.py
```

## Step 4: Monitor & Report

After training completes, report:
- Final train loss and eval loss
- Number of training steps completed
- Output directory with saved model
- Training duration
- Any warnings or errors

Check loss history:
```python
history = trainer.get_loss_history()
# {"steps": [...], "train_losses": [...], "eval_losses": [...]}
```

## Guidelines

- Always use `gradient_checkpointing=True` for models > 3B parameters
- Use QLoRA (`load_in_4bit=True`) when VRAM is limited
- Set `bf16=True` on Ampere+ NVIDIA GPUs or Apple Silicon
- Set `fp16=True` on older NVIDIA GPUs (pre-Ampere)
- Use `use_triton_kernels=True` on CUDA for speed boost
- Default `lora_alpha` should be 2x the `r` value
- For instruction tuning, always use `prepare_instruction_dataset` with a template
- Save with `merge=True` if the model will be exported to CoreML
