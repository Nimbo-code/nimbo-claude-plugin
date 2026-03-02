# CoreML Export Reference

## convert_hf_to_coreml — All Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | str | required | HuggingFace model ID or local path |
| `output_dir` | str | required | Output directory for .mlpackage files |
| `lut_bits` | int | `4` | LUT quantization bits: 4, 6, or 8 |
| `lut_embeddings_bits` | int | `-1` | Embedding quantization (-1 = float16) |
| `lut_lmhead_bits` | int | `4` | LM head quantization bits |
| `split_model` | bool | `False` | Split into embeddings + decoder + lm_head |
| `num_chunks` | int | `1` | Number of decoder chunks |
| `context_length` | int | `512` | Maximum sequence length |
| `state_length` | int | `None` | KV cache size (None = context_length) |
| `batch_size` | int | `64` | Prefill batch size |
| `no_prefill` | bool | `False` | Skip prefill model generation |
| `no_combine` | bool | `False` | Skip combining model parts |
| `no_dedup` | bool | `False` | Skip weight deduplication |

## ConversionConfig Dataclass

```python
@dataclass
class ConversionConfig:
    context_length: int
    state_length: int
    lut_bits: int                    # 4, 6, or 8
    lut_embeddings_bits: int         # -1 = float16
    lut_lmhead_bits: int
    split_model: bool
    num_chunks: int
    batch_size: int = 64
```

## Split Model Parts

When `split_model=True`, the model is split into separate components:

| Part | Enum | File Pattern | Description |
|------|------|-------------|-------------|
| Embeddings | `ModelPart.EMBEDDINGS` ("1") | `part1_*.mlpackage` | Token embedding layer |
| Decoder (FFN) | `ModelPart.FFN` ("2") | `part2_*.mlpackage` | Transformer decoder layers |
| Decoder Prefill | `ModelPart.PREFILL` ("2_prefill") | `part2_prefill_*.mlpackage` | Prefill variant for batch processing |
| LM Head | `ModelPart.LM_HEAD` ("3") | `part3_*.mlpackage` | Output projection + sampling |
| Full (monolithic) | `ModelPart.FULL` ("123") | `model.mlpackage` | Single combined model |
| Monolithic | `ModelPart.MONOLITHIC` | — | Non-split single model |
| Monolithic Prefill | `ModelPart.MONOLITHIC_PREFILL` | — | Non-split prefill variant |

## LUT Quantization Comparison

| Bits | Size vs FP16 | Perplexity Impact | Speed | Use Case |
|------|-------------|-------------------|-------|----------|
| 4-bit | ~25% original | Noticeable (+0.5-2.0) | Fastest | Prototyping, small models |
| 6-bit | ~38% original | Minimal (+0.1-0.5) | Fast | **Production (recommended)** |
| 8-bit | ~50% original | Negligible | Good | Quality-critical tasks |
| -1 (float16) | 100% | None | Baseline | Embedding layers only |

## Model Size Estimates (LLaMA 3.2 1B)

| Configuration | Approximate Size |
|--------------|-----------------|
| FP16 (no quant) | ~2.0 GB |
| 8-bit LUT | ~1.0 GB |
| 6-bit LUT | ~0.75 GB |
| 4-bit LUT | ~0.5 GB |
| 4-bit + dedup | ~0.35-0.43 GB |

## Chunking Guide

| Model Size | Recommended `num_chunks` | Notes |
|-----------|------------------------|-------|
| < 500M params | 1 | Single chunk sufficient |
| 500M - 1B | 1 | Single chunk usually works |
| 1B - 3B | 2 | Split for mobile memory |
| 3B+ | 2-4 | May not fit on mobile |

## Weight Deduplication

Enabled by default (`no_dedup=False`). Deduplicates shared weights across model parts:
- Typical size reduction: **15-40%**
- Uses cosine similarity threshold (0.9999) to detect identical weights
- No quality impact — only exact/near-exact duplicates are merged
- Most effective with split models that share embedding/output weights

## ANE Compatibility

The Apple Neural Engine (ANE) provides the fastest inference on Apple Silicon:
- Nimbo's LLaMA implementation uses Conv2d instead of Linear for ANE optimization
- Use `check_ane_compatibility()` to verify compatibility (score 0-100, >= 50 is good)
- Float16 precision is used throughout for ANE compatibility

## CLI Usage

```bash
python -m nimbo.export.coreml.hf_converter \
    --model meta-llama/Llama-3.2-1B \
    --output ./coreml_output \
    --lut-bits 6 \
    --context-length 512 \
    --state-length 512 \
    --batch-size 64 \
    --split-chunks 1 \
    --verbose

# ANE compatibility check
python -m nimbo.export.coreml.ane_checker \
    --model ./coreml_output/model.mlpackage \
    --format json
```

## Compilation for Deployment

After conversion, compile for optimal on-device performance:

```bash
# Compile single model
xcrun coremlcompiler compile ./coreml_output/model.mlpackage ./compiled/

# Compile split parts
for pkg in ./coreml_output/*.mlpackage; do
    xcrun coremlcompiler compile "$pkg" ./compiled/
done
```

Compiled `.mlmodelc` directories are smaller and load faster on device.
