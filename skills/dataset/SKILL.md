---
name: dataset
description: "Prepare datasets for LLM fine-tuning with Nimbo. Load, convert, chunk, filter, and validate JSONL, CSV, Parquet, or text data. Use when the user wants to prepare training data, create instruction datasets, format chat data, or inspect dataset contents."
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Dataset Preparation for Nimbo

You are helping the user prepare a dataset for fine-tuning with Nimbo. Determine the data format and use the appropriate preparation function.

## Step 1: Identify Data Format

Ask or detect the user's data format:
- **JSONL** (`.jsonl`): Most common for instruction tuning
- **CSV** (`.csv`): Tabular data with a text column
- **Parquet** (`.parquet`): Compressed columnar format
- **Text files** (`.txt`): Plain text, one file or folder of files
- **HuggingFace dataset**: Dataset name on the Hub (e.g., `"yelp_review_full"`)

Inspect the file to understand its structure before processing.

## Step 2: Choose Preparation Method

### Plain Text / Continual Pre-training

```python
from nimbo import prepare_dataset

dataset = prepare_dataset(
    source="./data.jsonl",          # or "./data/" folder, or list of strings
    text_field="text",              # column name containing text
    chunk_size=0,                   # 0 = no chunking, or word count per chunk
    file_type=None,                 # None = auto-detect from extension
    deduplicate=True,               # Remove exact duplicates
    min_length=0,                   # Minimum text length (chars)
    max_length=0,                   # Maximum text length (0 = no limit)
    filter_fn=None,                 # Optional custom filter function
)
```

### Instruction Tuning (Recommended for most tasks)

JSONL format required:
```json
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
{"instruction": "Summarize", "input": "Long article...", "output": "Summary..."}
```

The `input` field can be empty string `""` if not applicable.

```python
from nimbo import prepare_instruction_dataset

dataset = prepare_instruction_dataset(
    source="./instructions.jsonl",
    instruction_field="instruction",
    input_field="input",
    output_field="output",
    template="### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
)
```

### Chat / Multi-Turn

JSONL format:
```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

```python
from nimbo import prepare_chat_dataset

dataset = prepare_chat_dataset(
    source="./chat_data.jsonl",
    messages_field="messages",
    tokenizer=tokenizer,            # Optional: applies chat template
)
```

## Step 3: Advanced Processing

### Token-Based Chunking

For precise control over sequence lengths:
```python
from nimbo import chunk_by_tokens
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
chunks = chunk_by_tokens(texts, tokenizer, max_tokens=512, overlap=64)
```

### Filtering

```python
from nimbo import filter_texts

filtered = filter_texts(
    texts,
    min_length=50,
    max_length=2048,
    filter_fn=lambda t: "spam" not in t.lower(),
)
```

## Step 4: Validate

After preparation, always validate:
```python
print(f"Dataset size: {len(dataset)} examples")
print(f"Columns: {dataset.column_names}")
print(f"Sample:\n{dataset[0]['text'][:500]}")
```

## Step 5: Generate Script

Create a Python script that the user can run to prepare their dataset. Write it to a file like `prepare_data.py`.

## Guidelines

- Always inspect data before processing to detect the format
- For instruction tuning, verify the JSONL has instruction/input/output fields
- Recommend `deduplicate=True` to avoid training on duplicate examples
- Suggest appropriate `chunk_size` or `max_tokens` based on the model's context length
- If the data is in a non-standard format, help the user convert it to JSONL first
- Warn if the dataset is very small (< 100 examples) or very large (> 100K examples)
