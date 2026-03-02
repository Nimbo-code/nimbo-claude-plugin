---
name: inference
description: "Run inference and test models with Nimbo. Generate text, test fine-tuned models, run batch inference, streaming, and chat. Use when the user wants to test a model, generate text, run predictions, chat with a model, or evaluate outputs."
allowed-tools: Bash, Read, Write, Glob
---

# Inference & Model Testing with Nimbo

You are helping the user run inference on a base or fine-tuned model using Nimbo.

## Step 1: Determine Model Source

Ask the user:
1. **Fine-tuned model** — Local path from `trainer.save()` (e.g., `./nimbo_output/final_merged/`)
2. **Adapter checkpoint** — LoRA adapter to load on base model
3. **HuggingFace model** — Model ID from the Hub (e.g., `"microsoft/phi-2"`)

## Step 2: Choose Inference Mode

### Quick Loading (Recommended)

```python
from nimbo import load_for_inference

model = load_for_inference(
    model_path="./nimbo_output/final_merged",   # or HF model ID
    adapter_path=None,                          # Optional LoRA adapter
    device=None,                                # None = auto-detect
    quantize=None,                              # "4bit" or "8bit" for memory savings
    use_flash_attention=False,
)
```

### Full Control

```python
from nimbo import NimboInference, InferenceConfig, DeviceConfig, QuantizationConfig

config = InferenceConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    do_sample=True,
)

model = NimboInference(
    model_path="./nimbo_output/final_merged",
    inference_config=config,
    device_config=DeviceConfig(device="cuda"),
    quantization_config=QuantizationConfig(load_in_4bit=True),
    adapter_path=None,
    use_flash_attention=False,
    compile_model=False,              # torch.compile for speed
)
```

## Step 3: Generate

### Single Generation

```python
result = model.generate("What is machine learning?")
print(result)
```

### Batch Generation

```python
prompts = [
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
    "What is the capital of France?",
]
results = model.generate(prompts)
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}\nA: {result}\n")
```

### Streaming

```python
for token in model.stream("Once upon a time in a galaxy far away"):
    print(token, end="", flush=True)
print()
```

### Chat (Multi-Turn)

```python
response = model.chat(
    messages=[
        {"role": "user", "content": "Hello! What can you do?"},
        {"role": "assistant", "content": "I can help with many things!"},
        {"role": "user", "content": "Tell me about LoRA fine-tuning."},
    ],
    system_prompt="You are a helpful AI assistant.",
)
print(response)
```

### Using from Trainer (Post-Training)

```python
# IMPORTANT: After trainer.save(), the model is unloaded from memory.
# You must use load_for_inference() for post-save inference:
output_path = trainer.save(merge=True)
model = load_for_inference(output_path, device="cuda")
result = model.generate("Test prompt here")

# Alternatively, call inference BEFORE save:
result = trainer.inference("Test prompt here")
trainer.save(merge=True)
```

## Step 4: Adapter Operations

### Load Adapter on Base Model

```python
model = NimboInference(
    model_path="microsoft/phi-2",
    adapter_path="./nimbo_output/checkpoint-500",
)
result = model.generate("Test prompt")
```

### Merge Adapter and Save

```python
model.merge_adapter("./merged_model")
# Creates a standalone merged model at ./merged_model/
```

## InferenceConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 256 | Maximum tokens to generate |
| `temperature` | 0.7 | Sampling temperature (0 = greedy, higher = more random) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 50 | Top-K sampling |
| `repetition_penalty` | 1.1 | Penalize repeated tokens |
| `do_sample` | True | Enable sampling (False = greedy decoding) |
| `num_beams` | 1 | Beam search width (1 = no beam search) |
| `num_return_sequences` | 1 | Number of sequences to return |
| `stream` | False | Enable streaming mode |
| `use_cache` | True | Use KV cache for faster generation |

## Guidelines

- Use `quantize="4bit"` in `load_for_inference` to save memory on large models
- For deterministic outputs, set `temperature=0` and `do_sample=False`
- For creative text, increase `temperature` (0.8-1.2) and `top_p` (0.95)
- Use streaming for interactive / real-time applications
- Use `compile_model=True` for repeated inference (has startup cost)
- Always test with representative prompts from your training data domain
