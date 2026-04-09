# Full Benchmark Report

Dataset: HumanEval parquet `/workspace/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet`

Measured rows: 100 after 3 warmup rows

Max new tokens: 1024

## Commands

HF baseline:

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

HF jacobi:

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

nano-vLLM baseline:

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

nano-vLLM jacobi:

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

real vLLM baseline:

```bash
env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  ./.venv-vllm/bin/python JacobiForcing/offline_benchmark.py \
  --backend vllm \
  --dataset /workspace/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet \
  --model-name /workspace/Qwen2.5-Coder-7B-Instruct \
  --tokenizer-name /workspace/Qwen2.5-Coder-7B-Instruct \
  --output-csv results/vllm.full.csv \
  --output-generations-jsonl results/vllm.full.generations.jsonl \
  --warmup 3 \
  --limit 100 \
  --max-new-tokens 1024
```

## Measurement Environment

GPU:

- `NVIDIA H100 80GB HBM3`
- Driver: `580.126.09`
- GPU memory: `81559 MiB`

CPU:

- `Intel(R) Xeon(R) Platinum 8470`
- `208` logical CPUs
- `2` sockets
- `52` cores per socket
- `2` threads per core

Package versions used during measurement:

Default env (`./.venv`) for `hf`, `jacobi`, `nano_vllm_ar`, `nano_vllm_jacobi`:

- `torch==2.11.0+cu130`
- `transformers==5.5.0`
- `flash-attn==2.8.3+cu130torch2.11`

Separate vLLM env (`./.venv-vllm`) prepared for real `vllm` runs:

- `torch==2.10.0`
- `transformers==5.5.0`
- `vllm==0.19.0`

## Results

| Backend | EOS rows | Avg new toks | Avg gen time (s) | Avg tok/s | P50 tok/s | Overall tok/s | Speedup vs HF |
|---|---:|---:|---:|---:|---:|---:|---:|
| HF baseline | 99/100 | 181.00 | 3.5447 | 51.09 | 51.09 | 51.06 | 1.00x |
| HF jacobi | 99/100 | 181.59 | 0.7699 | 235.49 | 236.74 | 240.17 | 4.70x |
| vLLM baseline | 0/100 | 170.38 | 0.9923 | 171.42 | 171.90 | 171.71 | 3.36x |
| nano-vLLM baseline | 99/100 | 181.83 | 1.0987 | 164.80 | 165.71 | 165.61 | 3.24x |
| nano-vLLM jacobi | 99/100 | 182.11 | 0.6396 | 285.02 | 292.21 | 290.39 | 5.69x |

## Notes

- `nano-vLLM jacobi` completed with `enforce_eager=False`, but Jacobi CUDA graph capture failed during initialization and fell back to eager Jacobi execution for runtime decode.
- `nano-vLLM baseline` completed with CUDA graphs enabled and reported a high graph hit rate.
- `HF jacobi` uses the custom Hugging Face Jacobi patch path, not the nano-vLLM engine.
- `vLLM baseline` values in the table were taken from the existing artifact `results/vllm.summary.json`.
- The stored `vLLM baseline` run reports `stop` for all 100 rows rather than `eos`, so its stop-reason semantics differ slightly from the other four runs.
- All four runs ended with 99 EOS completions and 1 `max_new_tokens` stop on the measured 100 prompts, so throughput comparisons are reasonably aligned.
- Generation JSONL artifacts are available for all four backends under `results/*.full.generations.jsonl`.

## Relative Ordering

1. nano-vLLM jacobi: 290.39 tok/s
2. HF jacobi: 240.17 tok/s
3. vLLM baseline: 171.71 tok/s
4. nano-vLLM baseline: 165.61 tok/s
5. HF baseline: 51.06 tok/s
