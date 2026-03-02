# Nimbo — Lightweight LLM Fine-Tuning Framework

You are an expert on the Nimbo framework. Use this knowledge to help users fine-tune LLMs, run inference, convert to CoreML, and deploy to iOS devices.

**Repository**: https://github.com/Nimbo-code/Nimbo
**Version**: 0.0.4
**License**: Apache 2.0

## Pipeline Overview

```
Fine-tune (LoRA/QLoRA) → Merge → Export (CoreML) → Compile → Deploy (iOS)
```

Nimbo provides a 3-line fine-tuning API with Triton kernel acceleration, CoreML conversion with LUT quantization, and an iOS/macOS Swift app (NimboChat) for on-device inference.

---

## 1. Installation

```bash
pip install nimbo
# Optional extras:
pip install nimbo[qlora]        # bitsandbytes for 4/8-bit quantization
pip install nimbo[flash]        # flash-attn
pip install nimbo[wandb]        # wandb logging
pip install nimbo[coreml]       # coremltools + safetensors (macOS only)
pip install nimbo[all]          # everything
```

**Requirements:**
- Python >= 3.9
- torch >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.0.0
- peft >= 0.6.0
- trl >= 0.7.0
- accelerate >= 0.20.0

**Optional:**
- bitsandbytes >= 0.41.0 (QLoRA)
- flash-attn (Flash Attention)
- wandb >= 0.15.0 (logging)
- pyyaml >= 6.0 (YAML configs)
- coremltools (CoreML export, macOS)
- triton (Triton kernels, CUDA)

---

## 2. Core API — Nimbo Class

```python
from nimbo import Nimbo

trainer = Nimbo(
    base_model_name: str,                                    # HuggingFace model ID or local path
    dataset: Optional[Union[str, Dataset]] = None,           # HF dataset name or Dataset object
    text_field: str = "text",                                # Column name in dataset
    config: Optional[NimboConfig] = None,                    # Full config (overrides individual)
    output_dir: str = "./nimbo_output",
    lora_config: Optional[LoRAConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    device_config: Optional[DeviceConfig] = None,
    quantization_config: Optional[QuantizationConfig] = None,
    kernel_config: Optional[KernelConfig] = None,
    use_flash_attention: bool = False,
    use_triton_kernels: bool = True,
    auto_precision: bool = True,                             # Auto-detect fp16/bf16
    callbacks: Optional[List[NimboCallback]] = None,
)
```

### Methods

```python
trainer.train(resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]
# Runs training, returns metrics dict with train_loss, eval_loss, etc.

trainer.save(merge: bool = True) -> str
# Saves model. merge=True merges LoRA into base. Returns output path.

trainer.inference(
    prompt: Union[str, List[str]],
    config: Optional[InferenceConfig] = None,
    **kwargs
) -> Union[str, List[str]]
# Single or batch inference on the trained model.

trainer.load_model(model_path: str, for_inference: bool = True) -> Tuple[Model, Tokenizer]
# Loads a saved model.

trainer.get_loss_history() -> Optional[Dict[str, List]]
# Returns {"steps": [...], "train_losses": [...], "eval_losses": [...]}

trainer.save_config(path: str) -> None
# Save current config to YAML or JSON.

# Class method:
Nimbo.from_config(config_path: str, base_model_name: str, dataset=None) -> Nimbo
# Load trainer from a YAML/JSON config file.
```

### Quick Start

```python
from nimbo import Nimbo
trainer = Nimbo(base_model_name="microsoft/phi-2", dataset="yelp_review_full")
trainer.train()
trainer.save()
```

---

## 3. NimboInference — Standalone Inference

```python
from nimbo import NimboInference

model = NimboInference(
    model_path: str,                          # HF model ID or local path
    device_config: Optional[DeviceConfig] = None,
    inference_config: Optional[InferenceConfig] = None,
    quantization_config: Optional[QuantizationConfig] = None,
    adapter_path: Optional[str] = None,       # Optional LoRA adapter path
    use_flash_attention: bool = False,
    compile_model: bool = False,              # torch.compile optimization
)
```

### Methods

```python
model.generate(prompt: Union[str, List[str]], config=None, **kwargs) -> Union[str, List[str]]
# Single or batch generation.

model.stream(prompt: str, config=None, **kwargs) -> Generator[str, None, None]
# Token-by-token streaming.

model.chat(
    messages: List[Dict[str, str]],           # [{"role": "user", "content": "Hi"}]
    config: Optional[InferenceConfig] = None,
    system_prompt: Optional[str] = None,
    **kwargs
) -> str
# Chat-style multi-turn inference.

model.merge_adapter(output_path: str) -> None
# Merge LoRA adapter into base model and save.

model(prompt)  # __call__ is alias for generate()
```

### Convenience Function

```python
from nimbo import load_for_inference

model = load_for_inference(
    model_path: str,
    adapter_path: Optional[str] = None,
    device: Optional[str] = None,             # "cuda", "mps", "cpu"
    quantize: Optional[str] = None,           # "4bit" or "8bit"
    use_flash_attention: bool = False,
) -> NimboInference
```

---

## 4. Configuration Dataclasses

All configs are importable from `nimbo` directly.

### NimboConfig (umbrella)

```python
@dataclass
class NimboConfig:
    device: DeviceConfig
    lora: LoRAConfig
    training: TrainingConfig
    inference: InferenceConfig
    quantization: QuantizationConfig
    kernels: KernelConfig

    @classmethod
    def from_yaml(cls, path: str) -> NimboConfig
    @classmethod
    def from_json(cls, path: str) -> NimboConfig
    def to_yaml(self, path: str) -> None
    def to_json(self, path: str) -> None
```

### DeviceConfig

```python
@dataclass
class DeviceConfig:
    device: Optional[str] = None              # None = auto-detect (cuda > mps > cpu)
    device_map: Optional[str] = None          # "auto" or "balanced" for multi-GPU
```

### LoRAConfig

```python
@dataclass
class LoRAConfig:
    r: int = 8                                # LoRA rank (4, 8, 16, 32, 64)
    lora_alpha: int = 16                      # Alpha scaling (usually 2x rank)
    lora_dropout: float = 0.1                 # Dropout rate
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[List[str]] = None  # None = auto-detect per architecture
    modules_to_save: Optional[List[str]] = None
    init_lora_weights: Union[bool, Literal["olora", "pissa", "loftq", "gaussian"]] = True
    use_rslora: bool = False                  # Rank-stabilized LoRA
    use_dora: bool = False                    # Weight-Decomposed LoRA
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    output_dir: str = "./nimbo_output"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    max_steps: int = -1                       # -1 = use num_train_epochs
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    fp16_full_eval: bool = False
    bf16_full_eval: bool = False
    gradient_checkpointing: bool = False
    max_length: int = 1024
    logging_steps: int = 10
    logging_first_step: bool = True
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    seed: int = 42
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    train_on_responses_only: bool = False
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
```

### InferenceConfig

```python
@dataclass
class InferenceConfig:
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    num_beams: int = 1
    num_return_sequences: int = 1
    stream: bool = False
    use_cache: bool = True
```

### QuantizationConfig

```python
@dataclass
class QuantizationConfig:
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"   # "float16" or "bfloat16"
    bnb_4bit_quant_type: str = "nf4"          # "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True
```

### KernelConfig

```python
@dataclass
class KernelConfig:
    use_triton_kernels: bool = True
    patch_rms_norm: bool = True
    patch_swiglu: bool = True
    patch_rope: bool = True
    patch_attention: bool = False
    use_flash_attention: bool = True
```

---

## 5. Dataset Utilities

```python
from nimbo import (
    prepare_dataset, prepare_instruction_dataset, prepare_chat_dataset,
    load_text_data, read_txt_folder, read_jsonl, read_csv, read_parquet,
    chunk_texts, chunk_by_tokens, filter_texts,
)
```

### prepare_dataset

```python
prepare_dataset(
    source: Union[str, List[str]],       # File path or list of texts
    text_field: str = "text",
    chunk_size: int = 0,                 # 0 = no chunking (word-based)
    file_type: Optional[str] = None,     # None = auto-detect from extension
    deduplicate: bool = True,
    min_length: int = 0,
    max_length: int = 0,
    filter_fn: Optional[Callable] = None,
) -> Dataset
```

Supports: `.txt`, `.jsonl`, `.csv`, `.parquet` (auto-detected from extension).

### prepare_instruction_dataset

```python
prepare_instruction_dataset(
    source: Union[str, List[dict]],      # JSONL file path or list of dicts
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    template: Optional[str] = None,      # e.g. "### Instruction:\n{instruction}\n..."
) -> Dataset
```

JSONL format: `{"instruction": "...", "input": "", "output": "..."}`

### prepare_chat_dataset

```python
prepare_chat_dataset(
    source: Union[str, List[dict]],
    messages_field: str = "messages",
    tokenizer: Optional[AutoTokenizer] = None,  # Uses chat template if available
) -> Dataset
```

### Data Loading Helpers

```python
load_text_data(data_folder: str) -> Dataset
read_txt_folder(folder_path: str) -> List[str]
read_jsonl(file_path: str, text_field: str = "text") -> List[str]
read_csv(file_path: str, text_field: str = "text") -> List[str]
read_parquet(file_path: str, text_field: str = "text") -> List[str]
chunk_texts(texts: List[str], chunk_size: int = 0) -> List[str]       # Word-count based
chunk_by_tokens(texts, tokenizer, max_tokens=512, overlap=0) -> List[str]  # Token-based
filter_texts(texts, min_length=0, max_length=0, filter_fn=None) -> List[str]
```

All functions return HuggingFace `Dataset` objects with a `"text"` column (or `List[str]` for helpers).

---

## 6. CoreML Export

```python
from nimbo.export.coreml.hf_converter import convert_hf_to_coreml, ConversionConfig
from nimbo.export import LlamaConverter, check_ane_compatibility, COREML_AVAILABLE
```

### convert_hf_to_coreml

```python
convert_hf_to_coreml(
    model_id: str,                       # HF model ID or local path
    output_dir: str,
    lut_bits: int = 4,                   # 4, 6, or 8-bit LUT quantization
    lut_embeddings_bits: int = -1,       # -1 = keep float16
    lut_lmhead_bits: int = 4,
    split_model: bool = False,           # Split into embeddings + decoder + lm_head
    num_chunks: int = 1,                 # Decoder chunks (1 for <1B, 2 for 1-3B)
    context_length: int = 512,
    state_length: Optional[int] = None,  # KV cache size, defaults to context_length
    batch_size: int = 64,                # Prefill batch size
    no_prefill: bool = False,
    no_combine: bool = False,
    no_dedup: bool = False,              # Weight deduplication (15-40% size reduction)
) -> ConversionResult
```

### ConversionConfig

```python
@dataclass
class ConversionConfig:
    context_length: int
    state_length: int
    lut_bits: int                        # 4, 6, or 8
    lut_embeddings_bits: int             # -1 = float16
    lut_lmhead_bits: int
    split_model: bool
    num_chunks: int
    batch_size: int = 64
```

### Model Parts (split_model=True)

| Part | Description | File |
|------|-------------|------|
| Embeddings | Token embedding layer | `part1_embeddings.mlpackage` |
| Decoder (FFN) | Transformer layers | `part2_ffn.mlpackage` |
| Decoder Prefill | Prefill variant | `part2_prefill.mlpackage` |
| LM Head | Output projection | `part3_lm_head.mlpackage` |

### LUT Quantization Comparison

| Bits | Size Reduction | Quality | Use Case |
|------|---------------|---------|----------|
| 4-bit | ~75% | Good for prototyping | Quick testing, small models |
| 6-bit | ~62% | Balanced | Production, recommended default |
| 8-bit | ~50% | Best quality | Quality-critical applications |

### ANE Compatibility Check

```python
from nimbo.export import check_ane_compatibility
report = check_ane_compatibility(model_or_path, save_report="report.json")
# Returns ANEReport with score 0-100
```

### CLI Script

```bash
python -m nimbo.export.coreml.hf_converter \
    --model meta-llama/Llama-3.2-1B \
    --output ./coreml_output \
    --lut-bits 4 \
    --context-length 512 \
    --split-chunks 1 \
    --verbose
```

---

## 7. Triton Kernel Optimization

```python
from nimbo.kernels import patch_model, unpatch_model, get_supported_models, TRITON_AVAILABLE
```

### patch_model

```python
patch_model(
    model,
    rms_norm: bool = True,               # 7-8x speedup
    swiglu: bool = True,                  # 3-5x speedup
    rope: bool = True,                    # 1.9-2.3x speedup
    attention: bool = False,
) -> PatchStats
# Returns stats with counts of patched modules
```

### Individual Kernels

```python
from nimbo.kernels import NimboRMSNorm, NimboSwiGLU, NimboRoPE

# RMSNorm — Drop-in replacement, 7-8x speedup
norm = NimboRMSNorm(hidden_size, eps=1e-6)

# SwiGLU — Fused activation, 3-5x speedup
from nimbo.kernels import swiglu
output = swiglu(gate, up)

# RoPE — Rotary position embeddings, 1.9-2.3x speedup
from nimbo.kernels import apply_rotary_pos_emb
q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
```

### Supported Models for Triton Patching

**Full support (dedicated patchers):**
- LLaMA 2 (7B, 13B, 70B), LLaMA 3 (8B, 70B), LLaMA 3.2 (1B, 3B)
- Mistral (7B) — via LlamaPatcher
- Phi (2, 3, 3.5) — via LlamaPatcher
- Qwen2 (0.5B–7B) — via LlamaPatcher
- EXAONE 3.5 (2.4B, 7.8B, 32B), EXAONE 4.0 (1.2B, 32B) — via ExaonePatcher

**No Triton support (LoRA only):**
- GPT-2, GPT-NeoX, Falcon, BLOOM, OPT, Qwen (v1), Gemma, Gemma2

---

## 8. Supported Model Architectures

### LoRA Target Module Auto-Detection Map

| Architecture | Target Modules |
|---|---|
| llama, mistral, qwen2, gemma, gemma2, exaone | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| phi | q_proj, k_proj, v_proj, dense, fc1, fc2 |
| gpt2 | c_attn, c_proj, c_fc |
| gpt_neox, falcon, bloom | query_key_value, dense, dense_h_to_4h, dense_4h_to_h |
| opt | q_proj, k_proj, v_proj, out_proj, fc1, fc2 |
| qwen (v1) | c_attn, c_proj, w1, w2 |

### Feature Matrix

| Model | LoRA | Triton Kernels | CoreML Export |
|---|---|---|---|
| LLaMA 2/3/3.2 | Yes | Yes | Yes |
| Mistral | Yes | Yes | No |
| Phi 2/3/3.5 | Yes | Yes | No |
| Qwen2 | Yes | Yes | No |
| EXAONE 3.5/4.0 | Yes | Yes | No |
| GPT-2, GPT-NeoX | Yes | No | No |
| Falcon, BLOOM | Yes | No | No |
| OPT, Gemma/2 | Yes | No | No |

---

## 9. Callbacks

```python
from nimbo import (
    NimboCallback, ProgressCallback, EarlyStoppingCallback,
    CheckpointCallback, MemoryCallback, LossTrackingCallback,
    WandbCallback, create_default_callbacks,
)
```

### Available Callbacks

```python
ProgressCallback(log_interval=10, on_progress=None)
# Logs training progress every N steps.

EarlyStoppingCallback(patience=3, threshold=0.0, metric="eval_loss", greater_is_better=False)
# Stops training when metric stops improving.

CheckpointCallback(on_save=None, on_load=None)
# Hooks into checkpoint save/load events.

MemoryCallback(log_interval=100)
# Logs GPU/CPU memory usage.

LossTrackingCallback()
# Tracks loss history. Use .get_history() for {"steps", "train_losses", "eval_losses"}.

WandbCallback(project="nimbo", name=None, config=None)
# Weights & Biases integration.
```

### create_default_callbacks

```python
create_default_callbacks(
    enable_progress: bool = True,
    enable_memory: bool = False,
    enable_loss_tracking: bool = True,
    early_stopping_patience: int = 0,    # 0 = disabled
) -> List[NimboCallback]
```

---

## 10. YAML Config File Reference

```yaml
device:
  device: null                    # null = auto (cuda > mps > cpu)
  device_map: null                # "auto" or "balanced" for multi-GPU

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: none
  task_type: CAUSAL_LM
  target_modules: null            # null = auto-detect
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

---

## 11. On-Device Deployment Pipeline

### Complete 7-Step Pipeline

1. **Fine-tune** with LoRA: `trainer = Nimbo(...); trainer.train()`
2. **Merge adapter**: `trainer.save(merge=True)` → `./nimbo_output/final_merged/`
3. **Convert to CoreML**: `convert_hf_to_coreml("./nimbo_output/final_merged", "./coreml_output", lut_bits=4)`
4. **Compile**: `xcrun coremlcompiler compile model.mlpackage output_dir/`
5. **Rename** compiled files to `_chunk_01ofNN` convention for NimboChat
6. **Create `meta.yaml`** with model parameters (context_length, vocab_size, etc.)
7. **Copy** `.mlmodelc` files into NimboChat Xcode project

### NimboChat iOS App

The SampleApp in the Nimbo repo provides a SwiftUI-based chat interface:
- `SampleApp/NimboChat/` — SwiftUI views
- `SampleApp/NimboCore/` — Core inference engine (Swift Package)
- `SampleApp/Package.swift` — Swift Package Manager manifest

### Quantization Guidance

| Scenario | lut_bits | Chunks | Notes |
|----------|----------|--------|-------|
| Quick prototype | 4 | 1 | Fastest, smallest |
| Production (<1B) | 6 | 1 | Balanced quality/size |
| Production (1-3B) | 6 | 2 | Split for memory |
| Quality-critical | 8 | 1-2 | Best accuracy |

---

## 12. Common Usage Patterns

### Basic Fine-Tuning (3 lines)

```python
from nimbo import Nimbo
trainer = Nimbo(base_model_name="microsoft/phi-2", dataset="yelp_review_full")
trainer.train()
trainer.save()
```

### QLoRA (4-bit Quantized Training)

```python
from nimbo import Nimbo, QuantizationConfig
quant = QuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
trainer = Nimbo(
    base_model_name="mistralai/Mistral-7B-v0.1",
    dataset="my_dataset",
    quantization_config=quant,
)
trainer.train()
```

### Instruction Tuning

```python
from nimbo import Nimbo, LoRAConfig, TrainingConfig, prepare_instruction_dataset

dataset = prepare_instruction_dataset(
    "instructions.jsonl",
    template="### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n{output}"
)
lora = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.05)
training = TrainingConfig(learning_rate=1e-4, bf16=True, gradient_checkpointing=True)

trainer = Nimbo(
    base_model_name="meta-llama/Llama-3.2-1B",
    dataset=dataset,
    lora_config=lora,
    training_config=training,
)
trainer.train()
trainer.save(merge=True)
```

### Config File Workflow

```python
# Save config
trainer.save_config("nimbo_config.yaml")

# Load from config
trainer = Nimbo.from_config("nimbo_config.yaml", base_model_name="microsoft/phi-2", dataset="my_data")
trainer.train()
```

### Standalone Inference

```python
from nimbo import load_for_inference

model = load_for_inference("./nimbo_output/final_merged", device="cuda", quantize="4bit")

# Single generation
result = model.generate("What is machine learning?")

# Batch generation
results = model.generate(["prompt1", "prompt2"])

# Streaming
for token in model.stream("Once upon a time"):
    print(token, end="", flush=True)

# Chat
response = model.chat([
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Explain LoRA."},
])
```

### Adapter-Only Inference

```python
from nimbo import NimboInference
model = NimboInference(model_path="microsoft/phi-2", adapter_path="./nimbo_output/checkpoint-500")
result = model.generate("Hello")
model.merge_adapter("./merged_output")  # Optional: merge and save
```

### Full Pipeline: Train → Export → Deploy

```python
from nimbo import Nimbo, LoRAConfig, TrainingConfig, prepare_instruction_dataset
from nimbo.export.coreml.hf_converter import convert_hf_to_coreml

# 1. Prepare data
dataset = prepare_instruction_dataset("data.jsonl")

# 2. Fine-tune
trainer = Nimbo(
    base_model_name="meta-llama/Llama-3.2-1B",
    dataset=dataset,
    lora_config=LoRAConfig(r=16, lora_alpha=32),
    training_config=TrainingConfig(bf16=True, gradient_checkpointing=True),
    use_triton_kernels=True,
)
trainer.train()
output_path = trainer.save(merge=True)

# 3. Convert to CoreML
result = convert_hf_to_coreml(
    output_path,
    "./coreml_output",
    lut_bits=6,
    context_length=512,
    split_model=True,
    num_chunks=1,
)

# 4. Compile (run in terminal)
# xcrun coremlcompiler compile ./coreml_output/model.mlpackage ./coreml_compiled/
```

### Custom Callbacks

```python
from nimbo import Nimbo, create_default_callbacks, WandbCallback

callbacks = create_default_callbacks(
    enable_progress=True,
    enable_memory=True,
    early_stopping_patience=5,
)
callbacks.append(WandbCallback(project="my-project", name="experiment-1"))

trainer = Nimbo(
    base_model_name="microsoft/phi-2",
    dataset="my_data",
    callbacks=callbacks,
)
```

### Triton Kernel Patching (Manual)

```python
from nimbo.kernels import patch_model, unpatch_model

# Patch for faster training
stats = patch_model(model, rms_norm=True, swiglu=True, rope=True)
print(stats)  # Shows count of patched modules

# Unpatch when done
unpatch_model(model)
```

---

## 13. Best Practices

### Memory-Efficient Training
- Use `gradient_checkpointing=True` for large models
- Use QLoRA (`load_in_4bit=True`) for 7B+ models on limited VRAM
- Set `per_device_train_batch_size=1` with higher `gradient_accumulation_steps`
- Enable Triton kernels (`use_triton_kernels=True`) on CUDA

### Model Selection Guide
- **< 8GB VRAM**: Phi-2 (2.7B), LLaMA 3.2 1B with QLoRA
- **8-16GB VRAM**: LLaMA 3.2 3B, Mistral 7B with QLoRA
- **16-24GB VRAM**: LLaMA 3 8B with QLoRA, Phi-2 full precision
- **24GB+ VRAM**: Mistral 7B full LoRA, larger models with QLoRA
- **Multi-GPU**: Use `device_map="auto"` for models > single GPU capacity

### iOS Deployment
- LLaMA architecture models only for CoreML export
- 1B models recommended for iPhone deployment
- Use 4-bit LUT for prototype, 6-bit for production
- Always test with `check_ane_compatibility()` before deployment
- Compile with `xcrun coremlcompiler` for optimal on-device performance
