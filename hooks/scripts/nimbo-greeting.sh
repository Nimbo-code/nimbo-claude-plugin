#!/usr/bin/env bash
# Nimbo greeting — runs on Claude Code session start

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
O='\033[38;5;208m'  # Orange (Nimbo accent)
W='\033[38;5;255m'  # White
G='\033[38;5;245m'  # Gray
GR='\033[38;5;34m'  # Green
D='\033[2m'         # Dim
B='\033[1m'         # Bold
R='\033[0m'         # Reset

# Mascot display — use chafa for real image, fallback to ASCII art
MASCOT_IMG="${SCRIPT_DIR}/nimbo-mascot.png"

show_mascot() {
  if command -v chafa &>/dev/null && [ -f "$MASCOT_IMG" ]; then
    # Real image via chafa (symbol mode for universal terminal support)
    printf '\033[?25l'
    printf "\n"
    chafa --format symbols --size 25 --colors 256 --symbols block "$MASCOT_IMG" 2>/dev/null
    printf '\033[?25h'
  else
    # Fallback: ASCII art with blink animation
    printf '\033[?25l'
    printf "\n"
    printf "${G}         .  *  .        ${R}\n"
    printf "${G}      .    ${O}___${G}    .     ${R}\n"
    printf "${G}    *   ${O}_(   )_${G}   *   ${R}\n"
    printf "${G}      ${O}(_  ${W}${B}N${O}  _)${G}        ${R}\n"
    printf "${G}   .  ${O}  (_ _)${G}  .      ${R}\n"
    printf "${G}       ${O} /   \\${G}         ${R}\n"
    printf "${G}      ${O}/  ${W}o o${O}  \\${G}        ${R}\n"
    printf "${G}     ${O}|   ${W} v ${O}  |${G}       ${R}\n"
    printf "${G}      ${O}\\  ${W}___${O} /${G}        ${R}\n"
    printf "${G}       ${O}\\_____/${G}        ${R}\n"
    sleep 0.3

    # Blink
    printf '\033[9A'
    printf "${G}         .  *  .        ${R}\n"
    printf "${G}      .    ${O}___${G}    .     ${R}\n"
    printf "${G}    *   ${O}_(   )_${G}   *   ${R}\n"
    printf "${G}      ${O}(_  ${W}${B}N${O}  _)${G}        ${R}\n"
    printf "${G}   .  ${O}  (_ _)${G}  .      ${R}\n"
    printf "${G}       ${O} /   \\${G}         ${R}\n"
    printf "${G}      ${O}/  ${W}- -${O}  \\${G}        ${R}\n"
    printf "${G}     ${O}|   ${W} v ${O}  |${G}       ${R}\n"
    printf "${G}      ${O}\\  ${W}___${O} /${G}        ${R}\n"
    printf "${G}       ${O}\\_____/${G}        ${R}\n"
    sleep 0.15

    # Eyes open + smile
    printf '\033[9A'
    printf "${G}         .  *  .        ${R}\n"
    printf "${G}      .    ${O}___${G}    .     ${R}\n"
    printf "${G}    *   ${O}_(   )_${G}   *   ${R}\n"
    printf "${G}      ${O}(_  ${W}${B}N${O}  _)${G}        ${R}\n"
    printf "${G}   .  ${O}  (_ _)${G}  .      ${R}\n"
    printf "${G}       ${O} /   \\${G}         ${R}\n"
    printf "${G}      ${O}/  ${W}o o${O}  \\${G}        ${R}\n"
    printf "${G}     ${O}|   ${W} v ${O}  |${G}       ${R}\n"
    printf "${G}      ${O}\\  ${W}^_^${O} /${G}        ${R}\n"
    printf "${G}       ${O}\\_____/${G}        ${R}\n"
    printf '\033[?25h'
  fi
}

show_mascot

# Title
printf "\n"
printf "  ${O}${B}Nimbo${R}${G} — LLM Fine-Tuning Framework${R}\n"
printf "  ${D}────────────────────────────────${R}\n"

# Environment info (quick detection)
py_ver=$(python3 --version 2>&1 | awk '{print $2}' 2>/dev/null || echo "not found")
nimbo_ver=$(python3 -c "import nimbo; print(nimbo.__version__)" 2>/dev/null || echo "not installed")

if command -v python3 &>/dev/null && python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
  gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
  device="${GR}CUDA${R}${G}: ${gpu_name}${R}"
elif command -v python3 &>/dev/null && python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q True; then
  device="${GR}MPS${R}${G} (Apple Silicon)${R}"
else
  device="${G}CPU${R}"
fi

printf "  ${G}Python: ${W}${py_ver}${R}  ${G}Nimbo: ${W}${nimbo_ver}${R}\n"
printf "  ${G}Device: ${device}${R}\n"
printf "  ${D}────────────────────────────────${R}\n"
printf "  ${D}Say ${W}\"Fine-tune LLaMA on my data\"${D} to start${R}\n"
printf "\n"
