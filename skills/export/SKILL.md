---
name: export
description: "Convert fine-tuned models to CoreML for Apple devices. Use when the user wants to export to CoreML, convert for iOS/macOS, create .mlpackage files, run LUT quantization, or prepare models for Apple Neural Engine. Triggers on: CoreML, export, convert, mlpackage, Apple, ANE."
allowed-tools: Bash, Read, Write, Glob
---

# CoreML Export with Nimbo

You are helping the user convert a fine-tuned HuggingFace model to CoreML format for on-device deployment on Apple platforms.

Refer to `skills/export/reference.md` for complete conversion options.

## Prerequisites

- **macOS only** — CoreML conversion requires macOS
- **coremltools** — `pip install nimbo[coreml]` or `pip install coremltools safetensors`
- **Merged model** — The model must be saved with `trainer.save(merge=True)`
- **LLaMA architecture** — Currently only LLaMA-based models are supported for CoreML export

## Step 1: Verify Environment

```bash
python3 -c "
import platform
assert platform.system() == 'Darwin', 'CoreML export requires macOS'
import coremltools
print(f'coremltools: {coremltools.__version__}')
from nimbo.export import COREML_AVAILABLE
print(f'Nimbo CoreML support: {COREML_AVAILABLE}')
"
```

## Step 2: Choose Conversion Settings

Ask the user about their deployment target:

| Setting | Prototype | Production | Quality |
|---------|-----------|------------|---------|
| `lut_bits` | 4 | 6 | 8 |
| `context_length` | 256-512 | 512 | 512-1024 |
| `split_model` | False | True | True |
| `num_chunks` | 1 | 1 (< 1B) / 2 (1-3B) | 1-2 |

## Step 3: Convert

### Basic Conversion

```python
from nimbo.export.coreml.hf_converter import convert_hf_to_coreml

result = convert_hf_to_coreml(
    model_id="./nimbo_output/final_merged",
    output_dir="./coreml_output",
    lut_bits=4,
    context_length=512,
)
```

### Production Conversion (Split Model)

```python
from nimbo.export.coreml.hf_converter import convert_hf_to_coreml

result = convert_hf_to_coreml(
    model_id="./nimbo_output/final_merged",
    output_dir="./coreml_output",
    lut_bits=6,
    lut_embeddings_bits=-1,       # Keep embeddings in float16
    lut_lmhead_bits=6,
    split_model=True,
    num_chunks=1,                 # 2 for models > 1B params
    context_length=512,
    state_length=512,             # KV cache size
    batch_size=64,                # Prefill batch size
)
```

### Full Options

```python
result = convert_hf_to_coreml(
    model_id="./nimbo_output/final_merged",   # HF model ID or local path
    output_dir="./coreml_output",
    lut_bits=6,                   # 4, 6, or 8
    lut_embeddings_bits=-1,       # -1 = float16
    lut_lmhead_bits=6,
    split_model=True,
    num_chunks=1,
    context_length=512,
    state_length=None,            # None = same as context_length
    batch_size=64,
    no_prefill=False,             # Skip prefill model
    no_combine=False,             # Skip combining parts
    no_dedup=False,               # Skip weight deduplication
)
```

## Step 4: Check ANE Compatibility

```python
from nimbo.export import check_ane_compatibility

report = check_ane_compatibility("./coreml_output/model.mlpackage")
print(f"ANE compatibility score: {report.score}/100")
```

## Step 5: Compile (Optional)

Compile for optimal on-device performance:
```bash
xcrun coremlcompiler compile ./coreml_output/model.mlpackage ./coreml_compiled/
```

## Step 6: Verify Output

```bash
ls -la ./coreml_output/
python3 -c "
import coremltools as ct
model = ct.models.MLModel('./coreml_output/model.mlpackage')
print(model.get_spec().description)
"
```

## Guidelines

- Always verify macOS and coremltools availability first
- Only LLaMA architecture models support CoreML export
- Use `lut_embeddings_bits=-1` (float16) for embedding layers to preserve quality
- Weight deduplication saves 15-40% model size — keep it enabled
- For iPhone deployment, target 4-bit or 6-bit quantization
- Split models for better memory management on mobile devices
- Compile with `xcrun coremlcompiler` for deployment
