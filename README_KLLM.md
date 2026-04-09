# K-LLM README

This repository is maintained as a fork of the upstream Jacobi Forcing project
for K-LLM-related integration, environment management, and downstream
customization work.

## Installation

K-LLM-specific environment guidance is maintained here instead of the upstream
`README.md`.

1. Install `uv` and sync the default environment.
   This path is enough for the Hugging Face and Jacobi Forcing flows and does
   not install `vllm`.

```bash
uv sync
```

2. When you need `vllm`, install it as an extra instead of mixing it into the
   default environment by accident.

```bash
uv sync --extra vllm
```

3. If you want to keep separate virtual environments in the same repo, use two
   venv directories and sync the active one:

```bash
uv venv .venv
source .venv/bin/activate
uv sync --active
```

```bash
uv venv .venv-vllm
source .venv-vllm/bin/activate
uv sync --active --extra vllm
```

4. Run commands inside the managed environment:

```bash
uv run streamlit run applications/jacobi_model_chat.py
```

5. The current K-LLM environment target is:

- Python 3.12
- `transformers==5.5.0`
- `torch==2.11.0+cu130`
- CUDA 13 wheel index via PyTorch

## Notes

- This fork pins `flash-attn==2.8.3` to a prebuilt CUDA 13 / Torch 2.11 /
  Python 3.12 wheel to avoid long local source builds during `uv sync`.
- `deepspeed` remains configured without PEP 517 isolation so it builds
  against the project `torch` install.
- `JacobiForcing/offline_benchmark.py` supports `hf`, `jacobi`, `vllm`,
  `nano_vllm_ar`, and `nano_vllm_jacobi`.
- The recommended split in this repo is:
  default env for `hf`, `jacobi`, and `nano_vllm_*`, separate `vllm` env only
  when comparing against the real `vllm` backend.

## Tracking Policy

- Record K-LLM-specific changes in this file.
- Keep upstream project documentation in `README.md`.
- Use dated entries when behavior, dependencies, scripts, or model support change.


## Benchmark

`JacobiForcing/offline_benchmark.py` is the common offline benchmark entrypoint.
It records per-prompt timing/stat rows to CSV and can also save generated text
via `--output-generations-jsonl`.

The latest measured comparison report lives at
`docs/full_benchmark_report.md`.

Key options:

- `--backend`: `hf`, `jacobi`, `vllm`, `nano_vllm_ar`, `nano_vllm_jacobi`
- `--output-csv`: per-prompt metrics CSV
- `--output-generations-jsonl`: generated text dump
- `--warmup`: number of warmup prompts to skip from measurement
- `--limit`: number of measured prompts

Environment split:

- `./.venv/bin/python` for `hf`, `jacobi`, `nano_vllm_ar`, `nano_vllm_jacobi`
- `./.venv-vllm/bin/python` for `vllm`

### HF baseline

```bash
./.venv/bin/python JacobiForcing/offline_benchmark.py \
  --backend hf \
  --dataset /workspace/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet \
  --model-name /workspace/JacobiForcing_Coder_7B_v1 \
  --tokenizer-name /workspace/Qwen2.5-Coder-7B-Instruct \
  --output-csv results/hf.full.csv \
  --output-generations-jsonl results/hf.full.generations.jsonl \
  --warmup 3 \
  --limit 100 \
  --max-new-tokens 1024 \
  --attention-impl flash_attention_2
```

### HF Jacobi

```bash
./.venv/bin/python JacobiForcing/offline_benchmark.py \
  --backend jacobi \
  --dataset /workspace/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet \
  --model-name /workspace/JacobiForcing_Coder_7B_v1 \
  --tokenizer-name /workspace/Qwen2.5-Coder-7B-Instruct \
  --output-csv results/jacobi.full.csv \
  --output-generations-jsonl results/jacobi.full.generations.jsonl \
  --warmup 3 \
  --limit 100 \
  --max-new-tokens 1024 \
  --n-token-seq-len 64 \
  --jacobi-K 2 \
  --jacobi-r 0.85 \
  --jacobi-n-gram-pool-size 4
```

### nano-vLLM baseline

Use a dedicated port if another run recently used the default distributed port.

```bash
env TORCH_DISTRIBUTED_PORT=2350 \
  ./.venv/bin/python JacobiForcing/offline_benchmark.py \
  --backend nano_vllm_ar \
  --dataset /workspace/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet \
  --model-name /workspace/JacobiForcing_Coder_7B_v1 \
  --tokenizer-name /workspace/Qwen2.5-Coder-7B-Instruct \
  --output-csv results/nano_vllm_ar.full.csv \
  --output-generations-jsonl results/nano_vllm_ar.full.generations.jsonl \
  --warmup 3 \
  --limit 100 \
  --max-new-tokens 1024 \
  --max-model-len 2048
```

### nano-vLLM Jacobi

This path runs with `--nano-vllm-enforce-eager=False` by default. For the
current `Qwen2.5-Coder-7B` setup, Jacobi CUDA graph capture fails during init
and the engine falls back to eager Jacobi decode automatically.

```bash
env TORCH_DISTRIBUTED_PORT=2351 \
  ./.venv/bin/python JacobiForcing/offline_benchmark.py \
  --backend nano_vllm_jacobi \
  --dataset /workspace/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet \
  --model-name /workspace/JacobiForcing_Coder_7B_v1 \
  --tokenizer-name /workspace/Qwen2.5-Coder-7B-Instruct \
  --output-csv results/nano_vllm_jacobi.full.csv \
  --output-generations-jsonl results/nano_vllm_jacobi.full.generations.jsonl \
  --warmup 3 \
  --limit 100 \
  --max-new-tokens 1024 \
  --max-model-len 2048 \
  --n-token-seq-len 64 \
  --jacobi-K 2 \
  --jacobi-r 0.85 \
  --jacobi-n-gram-pool-size 4
```

### vLLM baseline

Use the `vllm` extra or the dedicated `.venv-vllm` environment first.

```bash
env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  ./.venv-vllm/bin/python JacobiForcing/offline_benchmark.py \
  --backend vllm \
  --dataset /workspace/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet \
  --model-name /workspace/Qwen2.5-Coder-7B-Instruct \
  --tokenizer-name /workspace/Qwen2.5-Coder-7B-Instruct \
  --output-csv results/vllm.csv \
  --output-generations-jsonl results/vllm.generations.jsonl \
  --warmup 3 \
  --limit 100 \
  --max-new-tokens 1024
```

## Change Log

### 2026-04-09

- Declared this repository as a K-LLM fork in `README.md`.
- Renamed the fork-specific document to `README_KLLM.md`.
- Added K-LLM-specific installation guidance based on `uv`.
- Added `uv`-based environment management files targeting Python 3.12,
  `transformers==5.5.0`, and `torch==2.11.0+cu130`.
- Configured `uv` to build `flash-attn` and `deepspeed` without build
  isolation so `uv sync` succeeds.
- Replaced the `flash-attn-4[cu13]` experiment with a pinned prebuilt
  `flash-attn==2.8.3` CUDA 13 wheel to keep the existing repo API working
  without long source builds.
- Split `vllm` into an optional `uv` extra so the default environment stays
  focused on the Hugging Face and Jacobi Forcing paths.
- Added repo-local guidance for keeping a separate `.venv-vllm` alongside the
  default environment when `vllm` comparisons are needed.
