# Nimbo Claude Code Plugin

A [Claude Code](https://claude.ai/code) plugin for the [Nimbo](https://github.com/Nimbo-code/Nimbo) LLM fine-tuning framework. Fine-tune language models, convert to CoreML, and deploy to iOS — all through natural language.

```
You: "Fine-tune LLaMA 3.2 1B on my cooking dataset and deploy it to my iPhone"

Claude: Sets up environment → Prepares data → Trains with LoRA → Tests → Converts to CoreML → Deploys to iOS
```

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Skills Reference](#skills-reference)
- [Usage Examples](#usage-examples)
- [Supported Models](#supported-models)
- [Requirements](#requirements)
- [Plugin Architecture](#plugin-architecture)
- [Development](#development)
- [License](#license)

---

## Installation

### Step 1: Add Marketplace (once)

```bash
claude plugin marketplace add https://github.com/Nimbo-code/nimbo-claude-plugin
```

### Step 2: Install Plugin

```bash
claude plugin install nimbo
```

### Verify Installation

```bash
claude plugin list
```

If `nimbo` appears in the list, installation is complete. The plugin loads automatically when you start Claude Code.

### Alternative: Local Development

```bash
git clone https://github.com/Nimbo-code/nimbo-claude-plugin.git
claude --plugin-dir ./nimbo-claude-plugin
```

---

## Usage

After installation, the Nimbo plugin loads automatically when you start Claude Code. No additional configuration is needed.

### Natural Language (Automatic Skill Matching)

No commands to memorize. Just describe what you want and Claude automatically selects the right skill:

```
"Install Nimbo and set up my environment"           → setup
"Prepare my JSONL data for instruction tuning"      → dataset
"Fine-tune LLaMA 3.2 1B on my dataset"              → train
"Test the fine-tuned model with sample prompts"     → inference
"Convert to CoreML for my iPhone"                    → export
"Create a training config file"                      → config
"Speed up training with Triton kernels"              → kernel
```

### Direct Skill Invocation

You can also invoke skills directly:

```
/nimbo:setup       # Environment setup & initialization
/nimbo:dataset     # Dataset preparation
/nimbo:train       # LoRA/QLoRA fine-tuning
/nimbo:inference   # Inference & model testing
/nimbo:export      # CoreML conversion
/nimbo:deploy      # iOS deployment (manual-invoke only)
/nimbo:config      # Config file management
/nimbo:kernel      # Triton kernel optimization
```

### Your First Fine-Tuning in 3 Steps

**Step 1** — Setup:
```
You: "Set up Nimbo with QLoRA support"
```
Claude checks Python, creates a virtual environment, installs Nimbo, and detects your GPU.

**Step 2** — Train:
```
You: "Fine-tune microsoft/phi-2 on my_data.jsonl with LoRA rank 16"
```
Claude generates a training script with optimal configuration for your hardware and runs it.

**Step 3** — Test:
```
You: "Test the model — ask it what LoRA is"
```
Claude loads the fine-tuned model and runs inference with your prompt.

### One-Liner E2E Pipeline

You can run the entire pipeline with a single sentence:

```
You: "Fine-tune LLaMA 3.2 1B on my cooking data and deploy it to my iPhone"
```

Claude automatically executes the full pipeline:

```
Setup → Data prep → LoRA training → Model test → CoreML conversion → iOS deployment guide
```

---

## Skills Reference

The plugin provides 8 skills, triggered by natural language or invoked directly with `/nimbo:<skill>`.

### `/nimbo:setup` — Environment Setup

Install Nimbo, configure virtual environment, detect GPU, and set up dependencies.

**Natural language triggers:**
```
"Install Nimbo"
"Set up the environment"
"Check my GPU"
"Configure Nimbo with QLoRA support"
```

**What it does:**
1. Checks Python version (>= 3.9)
2. Creates/activates virtual environment
3. Installs Nimbo with appropriate extras (qlora, flash, coreml, wandb)
4. Detects CUDA GPU / Apple Silicon MPS / CPU
5. Optionally runs HuggingFace login for gated models
6. Verifies the installation

---

### `/nimbo:dataset` — Dataset Preparation

Load, convert, chunk, filter, and validate training data in various formats.

**Natural language triggers:**
```
"Prepare my JSONL data for training"
"Convert my CSV to instruction format"
"Load and chunk my text files"
"Create an instruction tuning dataset"
```

**Supported formats:** JSONL, CSV, Parquet, plain text, HuggingFace datasets

**Key APIs:**
```python
# Instruction tuning (most common)
from nimbo import prepare_instruction_dataset
dataset = prepare_instruction_dataset(
    "instructions.jsonl",
    template="### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n{output}"
)

# Plain text / continual pre-training
from nimbo import prepare_dataset
dataset = prepare_dataset("data.jsonl", chunk_size=256, deduplicate=True)

# Multi-turn chat
from nimbo import prepare_chat_dataset
dataset = prepare_chat_dataset("chat_data.jsonl", tokenizer=tokenizer)
```

**Instruction JSONL format:**
```json
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
```

---

### `/nimbo:train` — Fine-Tuning

LoRA/QLoRA fine-tuning with automatic model selection, configuration, and execution.

**Natural language triggers:**
```
"Fine-tune Phi-2 on my dataset"
"Train LLaMA 3.2 1B with QLoRA"
"Run instruction tuning on Mistral 7B"
"Continue training from checkpoint"
```

**Minimal example:**
```python
from nimbo import Nimbo
trainer = Nimbo(base_model_name="microsoft/phi-2", dataset="my_data")
trainer.train()
trainer.save()
```

**Full example:**
```python
from nimbo import Nimbo, LoRAConfig, TrainingConfig, QuantizationConfig

trainer = Nimbo(
    base_model_name="meta-llama/Llama-3.2-1B",
    dataset=dataset,
    lora_config=LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.05),
    training_config=TrainingConfig(
        learning_rate=2e-4, bf16=True, gradient_checkpointing=True,
        num_train_epochs=3, max_length=1024,
    ),
    quantization_config=QuantizationConfig(load_in_4bit=True),
    use_triton_kernels=True,
)
trainer.train()
trainer.save(merge=True)
```

**Hardware recommendations:**

| VRAM | Recommended Models | Method |
|------|-------------------|--------|
| < 8GB | Phi-2 (2.7B), LLaMA 3.2 1B | QLoRA |
| 8-16GB | LLaMA 3.2 3B, Mistral 7B | QLoRA |
| 16-24GB | LLaMA 3 8B | QLoRA or LoRA |
| 24GB+ | Mistral 7B, LLaMA 3 8B | Full LoRA |
| Apple Silicon | Phi-2, LLaMA 3.2 1B | LoRA (MPS) |

---

### `/nimbo:inference` — Inference & Testing

Generate text, test fine-tuned models, run batch inference, streaming, and chat.

**Natural language triggers:**
```
"Test my fine-tuned model"
"Generate text with the trained model"
"Chat with the model"
"Run batch inference on my test set"
```

**Usage modes:**

```python
from nimbo import load_for_inference

model = load_for_inference("./nimbo_output/final_merged", quantize="4bit")

# Single generation
result = model.generate("What is LoRA?")

# Batch
results = model.generate(["prompt1", "prompt2", "prompt3"])

# Streaming
for token in model.stream("Once upon a time"):
    print(token, end="", flush=True)

# Chat
response = model.chat([
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Explain fine-tuning."},
])
```

---

### `/nimbo:export` — CoreML Conversion

Convert fine-tuned models to CoreML `.mlpackage` format for Apple devices.

**Natural language triggers:**
```
"Convert to CoreML"
"Export for iOS"
"Create an mlpackage"
"Quantize model with LUT for iPhone"
```

**Requirements:** macOS only, LLaMA architecture models only

```python
from nimbo.export.coreml.hf_converter import convert_hf_to_coreml

result = convert_hf_to_coreml(
    model_id="./nimbo_output/final_merged",
    output_dir="./coreml_output",
    lut_bits=6,                   # 4 (smallest), 6 (balanced), 8 (best quality)
    lut_embeddings_bits=-1,       # -1 = keep float16
    split_model=True,
    num_chunks=1,                 # 2 for models > 1B params
    context_length=512,
)
```

**LUT quantization comparison:**

| Bits | Size Reduction | Quality | Use Case |
|------|---------------|---------|----------|
| 4-bit | ~75% | Good | Prototyping |
| 6-bit | ~62% | Balanced | Production (recommended) |
| 8-bit | ~50% | Best | Quality-critical |

---

### `/nimbo:deploy` — iOS Deployment

Deploy CoreML models to iOS/macOS using NimboChat.

This skill is **manual-invoke only** — Claude won't trigger it automatically to prevent accidental builds.

**Invoke manually:**
```
/nimbo:deploy
```

**Pipeline:**
1. Compile CoreML model: `xcrun coremlcompiler compile model.mlpackage output/`
2. Rename to NimboChat convention: `model_chunk_01of01.mlmodelc`
3. Create `meta.yaml` with model parameters
4. Copy into NimboChat Xcode project
5. Build and run on device

---

### `/nimbo:config` — Configuration Management

Create, modify, and load YAML/JSON configuration files.

**Natural language triggers:**
```
"Create a training config"
"Make a YAML config for QLoRA"
"Show me all training parameters"
```

**Config file example:**
```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
training:
  learning_rate: 0.0001
  bf16: true
  gradient_checkpointing: true
  num_train_epochs: 3
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
```

**Load from config:**
```python
from nimbo import Nimbo
trainer = Nimbo.from_config("config.yaml", base_model_name="microsoft/phi-2", dataset="data")
```

---

### `/nimbo:kernel` — Triton Kernel Optimization

Accelerate training with custom Triton kernels. CUDA GPU required.

**Natural language triggers:**
```
"Speed up training with Triton"
"Apply kernel optimizations"
"Use Triton kernels for RMSNorm"
```

**Available kernels:**

| Kernel | Speedup |
|--------|---------|
| RMSNorm | 7-8x |
| SwiGLU | 3-5x |
| RoPE | 1.9-2.3x |

**Usage:**
```python
# Automatic (via constructor)
trainer = Nimbo(..., use_triton_kernels=True, use_flash_attention=True)

# Manual patching
from nimbo.kernels import patch_model
stats = patch_model(model, rms_norm=True, swiglu=True, rope=True)
```

---

## Usage Examples

### Example 1: Instruction Tuning

```
You: "Fine-tune LLaMA 3.2 1B on my cooking dataset"

Claude:
  1. Installs Nimbo and detects GPU
  2. Converts cooking.jsonl to instruction dataset
  3. Configures LoRA (r=16) with bf16 precision
  4. Trains for 3 epochs with gradient checkpointing
  5. Tests with sample cooking questions
```

### Example 2: QLoRA on Limited VRAM

```
You: "I only have 8GB VRAM. Fine-tune Mistral 7B on my Q&A dataset."

Claude:
  - Automatically selects QLoRA (4-bit NF4 quantization)
  - Batch size 1, gradient accumulation 8
  - Gradient checkpointing enabled
  - Triton kernels for extra speed
```

### Example 3: Full Pipeline to iOS

```
You: "Train a small model on my data and deploy it to my iPhone"

Claude:
  1. Recommends LLaMA 3.2 1B (best for mobile)
  2. Prepares dataset, trains with LoRA
  3. Merges adapter: trainer.save(merge=True)
  4. Converts to CoreML with 6-bit LUT
  5. Guides through NimboChat Xcode setup
```

### Example 4: Config File Workflow

```
You: "Create a reusable training config for my team"

Claude:
  - Generates nimbo_config.yaml
  - Sets optimal defaults for your hardware
  - Includes comments explaining each parameter
```

### Example 5: Training Optimization

```
You: "Training is too slow. Optimize it with Triton kernels."

Claude:
  1. Verifies Triton installation
  2. Checks model architecture compatibility
  3. Applies RMSNorm (7-8x), SwiGLU (3-5x), RoPE (1.9-2.3x) kernels
  4. Compares speed before and after optimization
```

### Example 6: Post-Training Inference

```
You: "Training just finished. Test the model performance with streaming output."

Claude:
  1. Locates the saved model path
  2. Loads model with load_for_inference()
  3. Streams tokens in real-time
  4. Evaluates quality across diverse prompts
```

---

## Supported Models

### Full Feature Matrix

| Model | LoRA | Triton Kernels | CoreML Export |
|-------|------|----------------|---------------|
| LLaMA 2 (7B, 13B, 70B) | Yes | Yes | Yes |
| LLaMA 3 (8B, 70B) | Yes | Yes | Yes |
| LLaMA 3.2 (1B, 3B) | Yes | Yes | Yes |
| Mistral (7B) | Yes | Yes | No |
| Phi (2, 3, 3.5) | Yes | Yes | No |
| Qwen2 (0.5B-7B) | Yes | Yes | No |
| EXAONE 3.5 (2.4B, 7.8B, 32B) | Yes | Yes | No |
| EXAONE 4.0 (1.2B, 32B) | Yes | Yes | No |
| GPT-2 | Yes | No | No |
| GPT-NeoX | Yes | No | No |
| Falcon | Yes | No | No |
| BLOOM | Yes | No | No |
| OPT | Yes | No | No |
| Gemma / Gemma2 | Yes | No | No |
| Mixtral 8x7B | Yes | No | No |

---

## Requirements

### Minimum

- **Claude Code** >= 2.1.0
- **Python** >= 3.9

### For Training

- PyTorch >= 2.0.0
- NVIDIA GPU (CUDA) or Apple Silicon (MPS)
- `pip install nimbo-code`

> **Note:** The PyPI package name is `nimbo-code` (not `nimbo`). The `nimbo` package on PyPI is a completely different project.

### For CoreML Export

- macOS
- coremltools
- `ruamel.yaml` (required for export config parsing):
  ```bash
  pip install nimbo-code[coreml] ruamel.yaml
  ```

### For Triton Kernels

- NVIDIA GPU with CUDA
- `pip install triton`

### For iOS Deployment

- macOS with Xcode
- Apple Developer account (for device deployment)

---

## Plugin Architecture

```
nimbo-claude-plugin/
├── .claude-plugin/
│   ├── plugin.json              # Plugin manifest
│   └── marketplace.json         # Marketplace registration
├── CLAUDE.md                    # Nimbo API knowledge base (872 lines)
├── skills/
│   ├── setup/SKILL.md           # Environment setup
│   ├── dataset/SKILL.md         # Data preparation
│   ├── train/
│   │   ├── SKILL.md             # Fine-tuning
│   │   └── reference.md         # Parameter reference
│   ├── inference/SKILL.md       # Model testing
│   ├── export/
│   │   ├── SKILL.md             # CoreML conversion
│   │   └── reference.md         # Export options reference
│   ├── deploy/SKILL.md          # iOS deployment (manual only)
│   ├── config/SKILL.md          # Config file management
│   └── kernel/SKILL.md          # Triton optimization
├── agents/
│   └── nimbo-expert.md          # Multi-step workflow agent
├── hooks/
│   ├── hooks.json               # Session start environment detection
│   └── scripts/
│       └── check-environment.sh # Auto-detects Python, GPU, Nimbo
├── settings.json                # Default agent configuration
└── tests/
    └── validation-log.md        # Test results
```

### How It Works

1. **CLAUDE.md** provides the complete Nimbo API knowledge (all class signatures, parameters, defaults)
2. **Skills** match natural language input via description keywords and execute specific tasks
3. **Hooks** auto-detect your environment (Python version, GPU, Nimbo installation) on session start
4. **nimbo-expert agent** orchestrates complex multi-step workflows

---

## Development

### Local Testing

```bash
git clone https://github.com/Nimbo-code/nimbo-claude-plugin.git
claude --plugin-dir ./nimbo-claude-plugin
```

### Validate Plugin Structure

```bash
claude plugin validate ./nimbo-claude-plugin
```

### Debug Mode

```bash
claude --debug --plugin-dir ./nimbo-claude-plugin
```

### Test Specific Skills

In a Claude Code session with the plugin loaded:
```
/nimbo:setup      # Environment setup
/nimbo:dataset    # Data preparation
/nimbo:train      # Fine-tuning
/nimbo:inference  # Model testing
/nimbo:export     # CoreML conversion
/nimbo:deploy     # iOS deployment
/nimbo:config     # Config management
/nimbo:kernel     # Triton optimization
```

---

## License

Apache 2.0 — See [LICENSE](LICENSE)
