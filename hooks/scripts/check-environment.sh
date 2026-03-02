#!/usr/bin/env bash
# Nimbo environment detection script
# Runs on session start to detect Python, Nimbo, GPU, and CoreML availability

set -euo pipefail

report=""

# Python
if command -v python3 &>/dev/null; then
    py_version=$(python3 --version 2>&1 | awk '{print $2}')
    report="Python: ${py_version}"
else
    report="Python: not found"
fi

# Nimbo
nimbo_version=$(python3 -c "import nimbo; print(nimbo.__version__)" 2>/dev/null || echo "")
if [ -n "$nimbo_version" ]; then
    report="${report} | Nimbo: ${nimbo_version}"
else
    report="${report} | Nimbo: not installed"
fi

# GPU / Device
device_info=$(python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'CUDA: {name} ({mem:.0f}GB)')
elif torch.backends.mps.is_available():
    print('MPS (Apple Silicon)')
else:
    print('CPU only')
" 2>/dev/null || echo "unknown")
report="${report} | Device: ${device_info}"

# CoreML
coreml_info=$(python3 -c "
import coremltools
print(f'coremltools {coremltools.__version__}')
" 2>/dev/null || echo "not available")
report="${report} | CoreML: ${coreml_info}"

# Triton
triton_info=$(python3 -c "
import triton
print(f'triton {triton.__version__}')
" 2>/dev/null || echo "not available")
report="${report} | Triton: ${triton_info}"

# Output as JSON for Claude Code to parse
echo "{\"result\": \"${report}\"}"
