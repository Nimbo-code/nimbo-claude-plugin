# Nimbo Claude Plugin — Validation Log

**Date**: 2026-03-02
**Claude Code Version**: 2.1.34
**Plugin Version**: 1.0.0
**Platform**: macOS Darwin 24.6.0

---

## Test 1: CLI Version & Plugin Flag Support

```
$ which claude && claude --version
/Users/optai/.local/bin/claude
2.1.34 (Claude Code)
```

**Result**: PASS — `--plugin-dir` flag confirmed in `claude --help` output.

---

## Test 2: Plugin Manifest Validation

```
$ claude plugin validate /Users/optai/Documents/nimbo-claude-plugin

Validating plugin manifest: /Users/optai/Documents/nimbo-claude-plugin/.claude-plugin/plugin.json

✔ Validation passed
```

**Result**: PASS

---

## Test 3: Skill Discovery

```
$ claude --plugin-dir /Users/optai/Documents/nimbo-claude-plugin \
    -p "List all available nimbo skills using /nimbo: prefix. Just list the skill names, nothing else."

- /nimbo:config
- /nimbo:dataset
- /nimbo:setup
- /nimbo:train
- /nimbo:inference
- /nimbo:export
- /nimbo:kernel
```

**Result**: PASS — 7 auto-invocable skills detected. `deploy` correctly excluded (disable-model-invocation: true).

---

## Test 4: Manual Skill Invocation (/nimbo:deploy)

```
$ claude --plugin-dir /Users/optai/Documents/nimbo-claude-plugin \
    -p "/nimbo:deploy 실행해줘. 실제로 실행하지 말고 이 스킬이 인식되는지만 확인해줘." \
    --allowedTools ""

`/deploy` 스킬이 정상적으로 인식되었습니다.

이 스킬은 **iOS Deployment with NimboChat** 스킬로, CoreML 모델을 NimboChat 샘플 앱을 통해
iOS/macOS에 배포하는 과정을 안내합니다.

요청대로 실제 실행은 하지 않았습니다. 스킬 인식 확인 완료!
```

**Result**: PASS — deploy skill recognized via manual `/nimbo:deploy` invocation.

---

## Test 5: Natural Language Trigger — Setup

```
$ claude --plugin-dir /Users/optai/Documents/nimbo-claude-plugin \
    -p "Nimbo 설치해줘. 실제로 실행하지 말고, 어떤 단계들을 수행할지 계획만 알려줘." \
    --allowedTools ""
```

**Output** (summarized):
- Step 1: Python 3.9+ 확인
- Step 2: 가상환경 생성 (`python3 -m venv .venv`)
- Step 3: `pip install -e ".[all]"` 패키지 설치
- Step 4: GPU 감지 (CUDA/MPS/CPU)
- Step 5: 설정 파일 생성
- Step 6: `from nimbo import Nimbo` 검증

**Result**: PASS — "Nimbo 설치해줘" triggered setup skill correctly.

---

## Test 6: Natural Language Trigger — Train

```
$ claude --plugin-dir /Users/optai/Documents/nimbo-claude-plugin \
    -p "Phi-2 모델을 파인튜닝하고 싶어. 실행하지 말고 어떤 Nimbo API를 사용할지만 알려줘." \
    --allowedTools ""
```

**Output** (summarized):
1. `nimbo:setup` — 환경 설정
2. `nimbo:config` — 학습 설정 (모델명 `microsoft/phi-2`, LoRA/QLoRA)
3. `nimbo:dataset` — 데이터셋 준비
4. `nimbo:train` — 파인튜닝 실행
5. `nimbo:inference` — 모델 테스트
6. `nimbo:export` (선택) — CoreML 변환
7. `nimbo:kernel` (선택) — Triton 최적화

워크플로우 순서: `setup → config → dataset → train → inference → (export)`

**Result**: PASS — Natural language correctly triggered multi-skill pipeline recommendation.

---

## Test 7: Natural Language Trigger — Export (CoreML)

```
$ claude --plugin-dir /Users/optai/Documents/nimbo-claude-plugin \
    -p "CoreML로 변환하려면 어떻게 해? 실행하지 말고 API 사용법만 알려줘." \
    --allowedTools ""
```

**Output** (summarized):
- LLaMA 아키텍처만 CoreML 변환 지원 확인
- `convert_hf_to_coreml()` API 시그니처 정확히 제공
- LUT 양자화 옵션 테이블 (4/6/8-bit)
- `check_ane_compatibility()` ANE 호환성 체크 안내
- CLI 사용법: `python -m nimbo.export.coreml.hf_converter --model ... --output ...`
- 전체 파이프라인: Fine-tune → merge → convert → compile → Xcode

**Result**: PASS — Export skill triggered with accurate Nimbo API details.

---

## Test 8: E2E Pipeline Knowledge

```
$ claude --plugin-dir /Users/optai/Documents/nimbo-claude-plugin \
    -p "Nimbo에서 LLaMA 3.2 1B를 한국어 요리 데이터로 파인튜닝하고 iOS에 배포하는 \
        전체 파이프라인을 설명해줘. 실행하지 말고 각 단계별로 어떤 Nimbo 코드를 쓸지만 보여줘." \
    --allowedTools ""
```

**Output** (summarized):
- 7-step pipeline: 환경설정 → 데이터준비 → 설정 → 학습 → 테스트 → CoreML변환 → iOS배포
- 각 단계별 Python 코드 및 YAML 설정 예시 제공
- LLaMA 3.2 1B + QLoRA + LUT 양자화 조합 권장

**Result**: PASS — CLAUDE.md knowledge base correctly informed full pipeline explanation.

---

## Summary

| # | Test | Result |
|---|------|--------|
| 1 | CLI version & --plugin-dir flag | PASS |
| 2 | `claude plugin validate` | PASS |
| 3 | Skill discovery (7 auto + 1 manual) | PASS |
| 4 | Manual `/nimbo:deploy` invocation | PASS |
| 5 | Natural language → setup | PASS |
| 6 | Natural language → train (multi-skill) | PASS |
| 7 | Natural language → export (CoreML) | PASS |
| 8 | E2E pipeline knowledge | PASS |

**All 8 tests passed. Plugin is functional on Claude Code 2.1.34.**
