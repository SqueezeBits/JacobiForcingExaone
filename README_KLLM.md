# K-LLM README

This repository is maintained as a fork of the upstream Jacobi Forcing project
for K-LLM-related integration, environment management, and downstream
customization work.

## Installation

K-LLM-specific environment guidance is maintained here instead of the upstream
`README.md`.

1. Install `uv` and sync the managed environment:

```bash
uv sync
```

2. Run commands inside the managed environment:

```bash
uv run streamlit run applications/jacobi_model_chat.py
```

3. The current K-LLM environment target is:

- Python 3.12
- `transformers==5.5.0`
- `torch==2.11.0+cu130`
- CUDA 13 wheel index via PyTorch

## Notes

- `flash-attn` imports `torch` during its build step, so this fork configures
  `uv` to build `flash-attn` and `deepspeed` without PEP 517 isolation.
- If `uv sync` previously failed with `ModuleNotFoundError: No module named 'torch'`
  while building `flash-attn`, pull the latest `pyproject.toml` and rerun
  `uv sync`.

## Tracking Policy

- Record K-LLM-specific changes in this file.
- Keep upstream project documentation in `README.md`.
- Use dated entries when behavior, dependencies, scripts, or model support change.

## Change Log

### 2026-04-09

- Declared this repository as a K-LLM fork in `README.md`.
- Renamed the fork-specific document to `README_KLLM.md`.
- Added K-LLM-specific installation guidance based on `uv`.
- Added `uv`-based environment management files targeting Python 3.12,
  `transformers==5.5.0`, and `torch==2.11.0+cu130`.
- Configured `uv` to build `flash-attn` and `deepspeed` without build
  isolation so `uv sync` succeeds.
