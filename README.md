# Nimbo Claude Code Plugin

A [Claude Code](https://claude.ai/code) plugin for the [Nimbo](https://github.com/Nimbo-code/Nimbo) LLM fine-tuning framework. Fine-tune language models, convert to CoreML, and deploy to iOS — all through natural language.

```
You: "LLaMA 3.2 1B를 한국어 요리 데이터로 파인튜닝하고 iPhone에 배포해줘"

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

`nimbo` 플러그인이 목록에 표시되면 설치 완료입니다. 이후 Claude Code를 실행하면 자동으로 로드됩니다.

### Alternative: Local Development

```bash
git clone https://github.com/Nimbo-code/nimbo-claude-plugin.git
claude --plugin-dir ./nimbo-claude-plugin
```

---

## Usage

설치 후 Claude Code를 실행하면 Nimbo 플러그인이 자동으로 로드됩니다. 별도 설정 없이 바로 사용할 수 있습니다.

### Natural Language (Automatic Skill Matching)

명령어를 외울 필요 없이 자연어로 말하면 Claude가 적절한 스킬을 자동으로 선택합니다:

```
"Nimbo 설치해줘"                              → setup 스킬 실행
"내 JSONL 데이터 학습용으로 준비해줘"              → dataset 스킬 실행
"LLaMA 3.2 1B를 내 데이터로 파인튜닝해줘"         → train 스킬 실행
"학습된 모델로 테스트해봐"                        → inference 스킬 실행
"CoreML로 변환해줘"                             → export 스킬 실행
"학습 설정 파일 만들어줘"                         → config 스킬 실행
"Triton으로 학습 속도 올려줘"                     → kernel 스킬 실행
```

영어도 동일하게 동작합니다:

```
"Install Nimbo and set up my environment"
"Prepare my JSONL data for instruction tuning"
"Fine-tune Phi-2 on my dataset"
"Test the model with some sample prompts"
"Convert to CoreML for my iPhone"
```

### Direct Skill Invocation

스킬을 직접 호출할 수도 있습니다:

```
/nimbo:setup       # 환경 설치 & 초기화
/nimbo:dataset     # 데이터셋 준비
/nimbo:train       # LoRA/QLoRA 파인튜닝
/nimbo:inference   # 추론 & 모델 테스트
/nimbo:export      # CoreML 변환
/nimbo:deploy      # iOS 배포 (수동 전용)
/nimbo:config      # 설정 파일 관리
/nimbo:kernel      # Triton 커널 최적화
```

### Your First Fine-Tuning in 3 Steps

**Step 1** — Setup:
```
You: "Nimbo 설치하고 환경 세팅해줘"
```
Claude가 Python 확인, 가상환경 생성, Nimbo 설치, GPU 감지까지 자동으로 수행합니다.

**Step 2** — Train:
```
You: "microsoft/phi-2를 my_data.jsonl로 LoRA rank 16으로 파인튜닝해줘"
```
Claude가 하드웨어에 맞는 최적 설정으로 학습 스크립트를 생성하고 실행합니다.

**Step 3** — Test:
```
You: "학습된 모델에 '한국의 수도는?' 이라고 물어봐"
```
Claude가 파인튜닝된 모델을 로드하고 추론을 실행합니다.

### One-Liner E2E Pipeline

한 문장으로 전체 파이프라인을 실행할 수도 있습니다:

```
You: "LLaMA 3.2 1B를 한국어 요리 데이터로 파인튜닝하고 iPhone에 배포해줘"
```

Claude가 자동으로 아래 전체 파이프라인을 수행합니다:

```
환경설정 → 데이터준비 → LoRA 학습 → 모델 테스트 → CoreML 변환 → iOS 배포 안내
```

---

## Skills Reference

8개의 스킬이 제공되며, 자연어 또는 `/nimbo:<skill>` 명령으로 실행할 수 있습니다.

### `/nimbo:setup` — Environment Setup

Nimbo 설치, 가상환경 구성, GPU 감지, 의존성 설치를 수행합니다.

**Natural language triggers:**
```
"Install Nimbo"                    "Nimbo 설치해줘"
"Set up the environment"           "환경 세팅해줘"
"Check my GPU"                     "내 GPU 확인해줘"
"Configure Nimbo with QLoRA"       "QLoRA 쓸 수 있게 세팅해줘"
```

**What it does:**
1. Python 버전 확인 (>= 3.9)
2. 가상환경 생성/활성화
3. Nimbo 및 선택적 의존성 설치 (qlora, flash, coreml, wandb)
4. CUDA GPU / Apple Silicon MPS / CPU 감지
5. HuggingFace 로그인 (gated 모델 사용 시)
6. 설치 검증

---

### `/nimbo:dataset` — Dataset Preparation

다양한 형식의 학습 데이터를 로드, 변환, 청킹, 필터링합니다.

**Natural language triggers:**
```
"Prepare my JSONL data for training"     "내 JSONL 데이터 학습용으로 준비해줘"
"Convert my CSV to instruction format"   "CSV를 인스트럭션 형식으로 변환해줘"
"Load and chunk my text files"           "텍스트 파일 불러와서 청킹해줘"
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

LoRA/QLoRA 파인튜닝을 수행합니다. 모델 선택, 설정 구성, 실행, 결과 보고까지 자동화합니다.

**Natural language triggers:**
```
"Fine-tune Phi-2 on my dataset"            "Phi-2를 내 데이터로 파인튜닝해줘"
"Train LLaMA 3.2 1B with QLoRA"            "LLaMA 3.2 1B QLoRA로 학습해줘"
"Run instruction tuning on Mistral 7B"     "Mistral 7B 인스트럭션 튜닝 실행해줘"
"Continue training from checkpoint"         "체크포인트에서 이어서 학습해줘"
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

텍스트 생성, 모델 테스트, 배치 추론, 스트리밍, 챗 모드를 지원합니다.

**Natural language triggers:**
```
"Test my fine-tuned model"                 "학습된 모델 테스트해봐"
"Generate text with the trained model"     "모델로 텍스트 생성해줘"
"Chat with the model"                      "모델이랑 대화해봐"
"Run batch inference on my test set"       "테스트셋으로 배치 추론 돌려줘"
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

파인튜닝된 모델을 CoreML `.mlpackage`로 변환합니다. Apple 디바이스 배포용입니다.

**Natural language triggers:**
```
"Convert to CoreML"                        "CoreML로 변환해줘"
"Export for iOS"                            "iOS용으로 내보내줘"
"Quantize model with LUT for iPhone"       "iPhone용 LUT 양자화해줘"
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

CoreML 모델을 NimboChat 앱으로 iOS/macOS에 배포합니다.

이 스킬은 **수동 호출 전용**입니다 — 실수로 빌드가 실행되는 것을 방지하기 위해 Claude가 자동으로 트리거하지 않습니다.

**Invoke manually:**
```
/nimbo:deploy
```

**Pipeline:**
1. CoreML 모델 컴파일: `xcrun coremlcompiler compile model.mlpackage output/`
2. NimboChat 파일명 규칙으로 변환: `model_chunk_01of01.mlmodelc`
3. `meta.yaml` 생성 (모델 파라미터)
4. NimboChat Xcode 프로젝트에 복사
5. 디바이스에 빌드 & 실행

---

### `/nimbo:config` — Configuration Management

YAML/JSON 설정 파일을 생성, 수정, 로드합니다.

**Natural language triggers:**
```
"Create a training config"                 "학습 설정 파일 만들어줘"
"Make a YAML config for QLoRA"             "QLoRA용 YAML 설정 만들어줘"
"Show me all training parameters"          "학습 파라미터 전부 보여줘"
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

커스텀 Triton 커널로 학습 속도를 가속합니다. CUDA GPU 필요.

**Natural language triggers:**
```
"Speed up training with Triton"            "Triton으로 학습 속도 올려줘"
"Apply kernel optimizations"               "커널 최적화 적용해줘"
"Use Triton kernels for RMSNorm"           "RMSNorm에 Triton 커널 적용해줘"
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

### Example 1: Korean Cooking Instruction Tuning

```
You: "한국어 요리 데이터로 LLaMA 3.2 1B 파인튜닝해줘"

Claude:
  1. Nimbo 환경 설치 및 GPU 감지
  2. korean_cooking.jsonl을 인스트럭션 데이터셋으로 변환
  3. LoRA (r=16) + bf16 정밀도로 설정
  4. 3 에포크 학습 (gradient checkpointing 적용)
  5. 샘플 요리 질문으로 모델 테스트
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
You: "팀에서 재사용할 학습 설정 파일 만들어줘"

Claude:
  - nimbo_config.yaml 생성
  - 하드웨어에 맞는 최적 기본값 설정
  - 각 파라미터 설명 포함
```

### Example 5: Model Optimization

```
You: "학습 속도가 너무 느려. Triton 커널로 최적화해줘"

Claude:
  1. Triton 설치 확인
  2. 모델 아키텍처 호환성 확인
  3. RMSNorm (7-8x), SwiGLU (3-5x), RoPE (1.9-2.3x) 커널 적용
  4. 최적화 전후 속도 비교
```

### Example 6: Post-Training Inference

```
You: "방금 학습 끝났는데, 모델 성능 테스트해봐. 스트리밍으로 보여줘"

Claude:
  1. 저장된 모델 경로 확인
  2. load_for_inference()로 모델 로드
  3. 스트리밍 모드로 토큰 단위 실시간 출력
  4. 다양한 프롬프트로 품질 평가
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
