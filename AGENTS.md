# AGENTS.md — Seamless Communication

## Overview

Seamless Communication is Meta's multilingual multimodal machine translation library. It provides models for speech-to-speech (S2ST), speech-to-text (S2TT), text-to-speech (T2ST), text-to-text (T2TT), and automatic speech recognition (ASR) across ~100 languages. The project includes SeamlessM4T (foundation model), SeamlessExpressive (prosody-preserving), SeamlessStreaming (real-time), and MuTox (toxicity detection).

**Primary language**: Python (PyTorch-based)
**Core dependency**: [fairseq2](https://github.com/facebookresearch/fairseq2) — provides model building blocks, asset loading, tokenizers, and generation utilities.

## Commands

### Installation

```bash
pip install .                    # Install the package (from repo root)
pip install -r dev_requirements.txt  # Install dev tools (black, flake8, isort, mypy, pytest)
```

> **Important**: fairseq2 has pre-built packages only for **Linux x86-64** and **Apple-silicon Mac**. It also requires `libsndfile` to be installed on the system. The `ffmpeg` CLI tool is required for Whisper-based transcription.

### Testing

```bash
pytest                           # Run all tests (unit + integration)
pytest tests/unit/               # Unit tests only (fast, no model downloads)
pytest tests/integration/        # Integration tests (downloads models, requires GPU for speed)
pytest --device cpu              # Explicit device selection (default: cpu)
pytest --device cuda:0           # Run on GPU
```

- Test config is in `pyproject.toml` under `[tool.pytest.ini_options]`.
- Test root: `tests/`
- Integration tests download large model checkpoints and audio files from the internet.
- Unit tests are self-contained and fast.

### Linting & Formatting

```bash
black .                          # Format code
isort .                          # Sort imports (profile: black)
flake8                           # Lint (extends ignore E, Y for Black compat)
mypy src/                        # Type check (strict mode, Python 3.8 target)
pre-commit run --all-files       # Run all pre-commit hooks
```

- Pre-commit hooks: `trailing-whitespace`, `check-ast`, `check-merge-conflict`, `check-added-large-files` (max 2000KB), `end-of-file-fixer`, `black`.
- mypy is configured with `strict = true` but `ignore_missing_imports = true`.

### CLI Entry Points

Installed as console scripts via `setup.py`:

```bash
m4t_predict <input> --task <task> --tgt_lang <lang>         # Run M4T inference
m4t_evaluate <args>                                          # Evaluate M4T models
m4t_finetune --train_dataset <path> --eval_dataset <path> --save_model_to <path>  # Finetune M4T
m4t_prepare_dataset <args>                                   # Prepare finetuning datasets
m4t_audio_to_units <args>                                    # Convert audio to discrete units
expressivity_predict <audio> --tgt_lang <lang> --model_name seamless_expressivity  # Expressive inference
expressivity_evaluate <args>                                 # Evaluate expressive models
streaming_evaluate <args>                                    # Evaluate streaming models
```

### GGML (C++ inference)

```bash
cd ggml
make build                       # Build unity.cpp with OpenBLAS + Tracy
make tests                       # Run GGML Python integration tests
./build/bin/unity --model seamlessM4T_medium.ggml input.wav  # Run inference
```

### Demos (Gradio)

```bash
cd demo
pip install -r requirements.txt
python app.py                    # Run local Gradio demo
```

## Code Organization

```
src/seamless_communication/
├── __init__.py              # Package init; registers asset cards with fairseq2
├── store.py                 # Gated asset registration for SeamlessExpressive
├── cards/                   # Model asset cards (YAML) — define model names, checkpoints, configs
├── models/                  # Model architectures
│   ├── unity/               # Core UnitY/UnitY2 encoder-decoder models (the main model family)
│   ├── vocoder/             # CodeHiFiGAN / HiFiGAN vocoders for unit-to-waveform
│   ├── pretssel/            # PretsseL expressive vocoder (ECAPA-TDNN based)
│   ├── generator/           # Streamable vocoder and ECAPA-TDNN builders
│   ├── monotonic_decoder/   # Monotonic attention decoder for streaming
│   ├── conformer_shaw/      # W2v-BERT 2.0 Conformer speech encoder
│   ├── aligner/             # UnitY2 forced alignment extractor
│   └── unit_extractor/      # Wav2Vec2 unit extraction + k-means
├── inference/               # High-level inference APIs
│   ├── translator.py        # Translator class — main user-facing inference API
│   ├── generator.py         # UnitYGenerator for sequence generation
│   └── transcriber.py       # Audio transcription
├── streaming/               # Streaming translation pipeline
│   ├── agents/              # SimulEval agents composing streaming pipelines
│   └── dataloaders/         # Streaming data loading
├── toxicity/                # Toxicity detection
│   ├── etox_bad_word_checker.py  # ETOX word-list based checker
│   ├── mintox.py            # MinTox pipeline
│   └── mutox/               # MuTox neural audio toxicity classifier
├── datasets/                # Dataset types and HuggingFace loading
├── denoise/                 # Audio denoising (Demucs wrapper)
├── segment/                 # Audio segmentation (Silero VAD)
└── cli/                     # CLI entry points
    ├── m4t/                 # M4T predict, evaluate, finetune, audio_to_units
    ├── expressivity/        # Expressive predict, evaluate, data
    ├── streaming/           # Streaming evaluate
    ├── toxicity/            # ETOX, MuTox, MuTox group annotations
    └── eval_utils/          # Shared evaluation metric utilities

tests/
├── conftest.py              # Pytest fixtures, --device option, audio download helper
├── common.py                # Shared test utilities (assert_close, assert_equal, device, etc.)
├── unit/                    # Fast unit tests (no model downloads)
│   ├── models/unity/        # UnitTokenizer tests
│   ├── denoise/             # Demucs tests
│   └── segment/             # Silero VAD tests
└── integration/             # Slow integration tests (require model downloads)
    ├── inference/           # Translator and MiNTox end-to-end tests
    └── models/              # Conformer-Shaw, Unity2 aligner tests

ggml/                        # C/C++ GGML inference engine for on-device translation
demo/                        # Gradio demo apps (m4tv1, m4tv2, expressive)
docs/                        # Documentation (m4t, streaming, expressive)
```

## Architecture & Key Patterns

### fairseq2 Asset System

Models are registered via YAML **asset cards** in `src/seamless_communication/cards/`. The package `__init__.py` registers these cards with `fairseq2.assets.asset_store` at import time. This is how model names like `"seamlessM4T_v2_large"` resolve to configs and checkpoint URLs.

```python
# Loading a model by name:
from seamless_communication.models.unity import load_unity_model
model = load_unity_model("seamlessM4T_v2_large", device=device, dtype=dtype)
```

Each model family follows the pattern:
- `builder.py` — `@dataclass` config + `ArchitectureRegistry` + builder class
- `loader.py` — `ConfigLoader` + `ModelLoader` instances + checkpoint conversion
- `model.py` — `nn.Module` subclass

### Model Builder Pattern (fairseq2 arch registry)

Models use fairseq2's `ArchitectureRegistry` to register architecture variants:

```python
unity_archs = ArchitectureRegistry[UnitYConfig]("unity")

@unity_arch("base")
def _base() -> UnitYConfig:
    return UnitYConfig(...)

@unity_arch("base_v2")
def _base_v2() -> UnitYConfig:
    return UnitYConfig(...)
```

### Translator — Main Inference API

`seamless_communication.inference.Translator` is the primary user-facing class:

```python
translator = Translator("seamlessM4T_v2_large", "vocoder_v2", device, dtype=torch.float16)
text_output, speech_output = translator.predict(input, "t2tt", "deu", src_lang="eng")
```

Tasks are defined as an enum: `S2ST`, `S2TT`, `T2ST`, `T2TT`, `ASR`.

### Streaming Pipeline Pattern

Streaming uses **SimulEval agents** composed into pipelines. Each agent handles one stage:

```python
class SeamlessStreamingS2STAgent(UnitYAgentPipeline):
    pipeline = [
        OnlineFeatureExtractorAgent,
        OfflineWav2VecBertEncoderAgent,
        UnitYMMATextDecoderAgent,
        NARUnitYUnitDecoderAgent,
        VocoderAgent,
    ]
```

Agents are in `src/seamless_communication/streaming/agents/`.

### Checkpoint Conversion

`loader.py` files contain `convert_*_checkpoint()` functions that handle converting fairseq1-style checkpoints to fairseq2 format via key mapping.

## Naming Conventions & Style

### Python Style
- **Formatter**: Black (v22.3.0)
- **Import sorting**: isort with `profile = "black"`
- **Type hints**: Used throughout. `from typing import ...` style (Python 3.8 compatible, no `X | Y` union syntax).
- **Docstrings**: Triple-quoted, placed under class/dataclass fields as inline documentation.
- **Logging**: `logging.basicConfig(...)` at module level in CLI scripts; `logger = logging.getLogger(__name__)` pattern.

### File Header
Every file starts with the Meta copyright header:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
```

Some files use slight variations (period after "affiliates", "BSD-style license" wording).

### Naming Patterns
- Model classes: `UnitYModel`, `UnitYT2UModel`, `UnitYNART2UModel`
- Config dataclasses: `UnitYConfig`, `UnitYT2UConfig`, `MutoxConfig`
- Builder classes: `UnitYBuilder`, `UnitYNART2UBuilder`
- Loader functions: `load_unity_model()`, `load_unity_text_tokenizer()`, `load_mutox_model`
- Architecture registries: `unity_archs`, `unity_t2u_archs`, `mutox_archs`
- CLI functions: `main()` in each CLI module
- Test classes: `TestUnitTokenizer`, `TestUnitEncoder` (class-based with pytest)
- Language codes: ISO 639-3 codes (`eng`, `deu`, `spa`, `fra`, `cmn`, etc.)

### Module `__init__.py` Style
Uses explicit re-exports with `as` aliasing for public API clarity:

```python
from seamless_communication.models.unity.builder import UnitYConfig as UnitYConfig
```

## Testing Patterns

### Test Structure
- **Unit tests** (`tests/unit/`): Test individual components in isolation. Fast, no network access needed. Use `pytest` classes.
- **Integration tests** (`tests/integration/`): End-to-end tests that load real models and run inference. Require downloading model checkpoints.

### Test Utilities (`tests/common.py`)
- `device` — global `Device` variable, set by `--device` CLI option in conftest.py
- `assert_close(a, b, rtol, atol)` — tensor comparison with tolerance
- `assert_equal(a, b)` — exact tensor comparison
- `assert_unit_close(a, b, num_unit_tol, percent_unit_tol)` — unit sequence comparison
- `has_no_inf(a)`, `has_no_nan(a)` — tensor validation
- `tmp_rng_seed(device, seed)` — context manager for reproducible RNG
- `get_default_dtype()` — returns `float32` on CPU, `float16` on CUDA
- `convert_to_collated_fbank(audio_dict, dtype)` — audio to filterbank feature conversion

### Test Fixtures (`tests/conftest.py`)
- `example_rate16k_audio` — module-scoped fixture that downloads a sample WAV from Meta's CDN

### Test Patterns
- Integration tests use `Final` constants for expected translations to validate model output
- Tests use `tests.common.device` (not `torch.device` directly) for device portability
- Unit tests mock external dependencies (e.g., `@patch` for subprocess calls in demucs tests)

## Important Gotchas

1. **fairseq2 version pinned**: The project requires `fairseq2==0.2.*`. Breaking changes between fairseq2 versions can affect model loading, tokenization, and generation APIs. Always check compatibility.

2. **Model downloads on first use**: Models are lazily downloaded from HuggingFace when first referenced by name. Integration tests will download multi-GB checkpoints. Unit tests avoid this.

3. **Device handling**: The `Translator` and models require explicit `device` and `dtype` parameters. CPU uses `float32`, CUDA uses `float16`. The test framework manages this via `tests.common.get_default_dtype()`.

4. **Asset cards are YAML, not Python**: Model configurations (checkpoint URLs, architecture names, supported languages) live in `src/seamless_communication/cards/*.yaml`, not in Python code. Adding a new model variant requires a new YAML card.

5. **`src/` layout**: Source code is under `src/seamless_communication/`, not at the top level. The `setup.py` uses `find_packages(where="src")` with `package_dir={"": "src"}`.

6. **SeamlessExpressive is gated**: The expressive model requires manual download via a request form. The `store.py` module handles gated asset registration for these models.

7. **Python 3.8 target**: Despite modern PyTorch usage, the project targets Python 3.8 (see `pyproject.toml` mypy config and `setup.py`). Use `typing.Optional`, `typing.List`, etc. instead of `X | Y` or `list[X]`.

8. **No CI pipeline**: There are no GitHub Actions, CircleCI, or other CI config files. Quality enforcement is via pre-commit hooks only.

9. **Git submodule**: The `ggml/tracy` directory is a git submodule. Run `git submodule update --init` if the `ggml/tracy` directory is empty.

10. **`# fmt: off` / `# fmt: on`**: Used in tests and CLI code to disable Black formatting for aligned constant definitions (e.g., test reference sentences).

## Dependencies

### Core Runtime
- `fairseq2==0.2.*` — Model framework (encoders, decoders, tokenizers, generation)
- `sonar-space==0.2.*` — Sentence embeddings
- `torch` / `torchaudio` — Tensor computation and audio I/O
- `simuleval~=1.1.3` — Streaming evaluation framework
- `datasets==2.18.0` — HuggingFace datasets
- `openai-whisper` — ASR for evaluation metrics
- `librosa`, `soundfile`, `scipy` — Audio processing
- `fire` — CLI argument parsing (some scripts)
- `tqdm` — Progress bars

### Dev
- `black`, `isort`, `flake8`, `mypy` — Formatting, linting, type checking
- `pytest` (>=7.1) — Testing
- `pre-commit` — Git hooks
- `audiocraft` — Audio codec (dev dependency)

## License

Three license tiers:
- **MIT**: Code, W2v-BERT 2.0, mExpresso text data, aligner, ETOX, MuTox
- **CC-BY-NC 4.0**: SeamlessM4T models (v1/v2), SeamlessStreaming models
- **Seamless License**: Seamless and SeamlessExpressive models (requires request form)
