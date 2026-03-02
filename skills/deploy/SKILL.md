---
name: deploy
description: "Deploy fine-tuned models to iOS/macOS with NimboChat. Use when the user wants to deploy to iPhone, iPad, Mac, create an iOS app, or set up on-device inference with Xcode. Triggers on: deploy, iOS, iPhone, iPad, Xcode, NimboChat, on-device."
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Glob
---

# iOS Deployment with NimboChat

You are helping the user deploy a CoreML model to iOS/macOS using the NimboChat sample app from the Nimbo repository.

**Important**: This skill is manual-invoke only (`/nimbo:deploy`) because it involves Xcode builds and device deployment that require user interaction.

## Prerequisites

- macOS with Xcode installed
- CoreML model (`.mlpackage` or compiled `.mlmodelc`) from the export step
- Apple Developer account (for device deployment)

## Step 1: Get NimboChat

Clone the Nimbo repository and locate the sample app:

```bash
git clone https://github.com/Nimbo-code/Nimbo.git
cd Nimbo/SampleApp
```

The SampleApp structure:
```
SampleApp/
├── Package.swift           # Swift Package Manager manifest
├── NimboChat/              # SwiftUI chat interface
└── NimboCore/              # Core inference engine
```

## Step 2: Prepare Model Files

### If using compiled models (`.mlmodelc`)

1. Compile the CoreML model if not already done:
```bash
xcrun coremlcompiler compile ./coreml_output/model.mlpackage ./compiled/
```

2. Rename compiled files following NimboChat convention:
```
model_chunk_01of01.mlmodelc    # Single chunk
# or for split models:
model_chunk_01of02.mlmodelc    # Chunk 1 of 2
model_chunk_02of02.mlmodelc    # Chunk 2 of 2
```

### Create meta.yaml

Create a `meta.yaml` file alongside the model:

```yaml
model_name: "My Fine-Tuned Model"
context_length: 512
vocab_size: 32000               # Check your model's tokenizer
lut_bits: 6
num_chunks: 1
batch_size: 64
```

## Step 3: Add Model to Xcode Project

1. Open `SampleApp/Package.swift` in Xcode
2. Copy compiled `.mlmodelc` files into the app's Resources directory
3. Copy `meta.yaml` into the same directory
4. Verify the files appear in Xcode's file navigator

## Step 4: Build & Run

1. Select your target device (iPhone, iPad, or Mac)
2. Build the project (Cmd+B)
3. Run on device (Cmd+R)

### Simulator Note

CoreML models can run on the iOS Simulator but will use CPU only (no ANE acceleration). For realistic performance testing, deploy to a physical device.

## Step 5: Test

In the NimboChat app:
1. The model should load automatically on app launch
2. Type a prompt in the chat interface
3. Verify the model generates appropriate responses
4. Check for reasonable generation speed

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model fails to load | Check file naming convention and meta.yaml |
| Out of memory | Use smaller model or lower quantization (4-bit) |
| Slow generation | Verify ANE compatibility with `check_ane_compatibility()` |
| Xcode build fails | Ensure macOS and Xcode are up to date |
| Code signing error | Configure Apple Developer team in Xcode project settings |

## Guidelines

- Test on Simulator first for quick iteration, then deploy to device
- 1B parameter models are recommended for iPhone (4-6 GB RAM limit)
- 3B models may work on iPad Pro and Mac
- Always compile models with `xcrun coremlcompiler` for production
- Monitor memory usage — iOS will kill apps that exceed memory limits
