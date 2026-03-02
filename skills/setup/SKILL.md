---
name: setup
description: "Install and set up Nimbo for LLM fine-tuning. Use when the user wants to install Nimbo, create a virtual environment, set up dependencies, detect GPU, or initialize a Nimbo project. Triggers on: install, setup, environment, venv, GPU check."
allowed-tools: Bash, Read, Write, Glob
---

# Nimbo Environment Setup

You are setting up the Nimbo LLM fine-tuning framework. Follow these steps in order.

## Step 1: Python & Virtual Environment

Check Python version (>= 3.9 required):
```bash
python3 --version
```

Create and activate a virtual environment if one doesn't exist:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 2: Install Nimbo

Install base package:
```bash
pip install nimbo
```

Ask the user which optional extras they need, then install accordingly:

| Extra | Command | When to use |
|-------|---------|-------------|
| QLoRA | `pip install nimbo[qlora]` | 4/8-bit quantized training (needs CUDA) |
| Flash Attention | `pip install nimbo[flash]` | Faster attention on CUDA GPUs |
| W&B Logging | `pip install nimbo[wandb]` | Weights & Biases experiment tracking |
| CoreML | `pip install nimbo[coreml]` | CoreML export (macOS only) |
| All | `pip install nimbo[all]` | Everything |

If the user is unsure, recommend:
- **CUDA GPU**: `pip install nimbo[qlora,flash]`
- **Apple Silicon**: `pip install nimbo[coreml]`
- **CPU only**: `pip install nimbo` (base only)

## Step 3: GPU Detection

Detect available hardware:
```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'MPS available: {torch.backends.mps.is_available()}')
"
```

## Step 4: HuggingFace Login (Optional)

If the user needs gated models (LLaMA, Mistral, etc.):
```bash
pip install huggingface_hub
huggingface-cli login
```

## Step 5: Verify Installation

```bash
python3 -c "
import nimbo
print(f'Nimbo {nimbo.__version__} installed successfully')
from nimbo import Nimbo, NimboInference, LoRAConfig, TrainingConfig
print('All core modules imported OK')
"
```

## Report

After completing setup, summarize:
- Python version
- Nimbo version
- Device detected (CUDA/MPS/CPU)
- VRAM available (if GPU)
- Installed extras
- Any warnings or issues
