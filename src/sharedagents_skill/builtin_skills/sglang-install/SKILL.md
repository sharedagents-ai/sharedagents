---
name: sglang-install
description: "Install and serve a model with SGLang on a local GPU box. Covers: (1) install (with the right Torch/CUDA combo), (2) launch commands for qwen3.6-27b / gemma-4-31b with RadixAttention prefix sharing, (3) the BF16 KV cache gotcha for Qwen3.5/3.6 Gated DeltaNet families, (4) wiring the endpoint into sharedagents's `models[]`. Use whenever the user asks 'install sglang', 'serve with sglang', 'set up sglang locally', or wants prefix-sharing across many short prompts that share a system prompt."
metadata:
  sharedagents:
    always: false
---

# SGLang — install + serve a local model from sharedagents

Sibling skill to `vllm-install`. Pick SGLang when:

- The workload is **many short prompts that share a system prompt**
  (SGLang's RadixAttention shares prefixes across DIFFERENT prompts,
  not just replays of the same one).
- You want the Llama 3 / Mixtral / DeepSeek-specific kernels SGLang
  has tuned that vLLM doesn't.
- You're on Hopper / Blackwell and want chunked prefill +
  speculative decoding together.

Otherwise vLLM is the safe default (`vllm-install`).

## 1. Detect the hardware

Same check as vLLM (`nvidia-smi` → map compute capability → pick
dtype). See `vllm-install` step 1.

## 2. Install

```bash
# uv-driven; falls back to pip if uv isn't installed.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Pin to the latest stable; SGLang nightlies churn fast.
uv pip install --system "sglang[all]>=0.4"

# CUDA 12.4+ with FlashInfer is the supported path.
python -c "import flashinfer; print(flashinfer.__version__)"
```

For Blackwell (5090):

```bash
uv pip install --system --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install --system 'sglang[all]'
```

## 3. Pull the weights

Same model files as `vllm-install` (Qwen3.6-27B FP8 for Hopper /
Blackwell, AWQ-BF16-INT4 mixed for Ampere/Ada, Gemma-4-31B-it). See
that skill's step 3.

## 4. Serve

**Blackwell / Hopper (FP8 weights, BF16 KV cache, single GPU):**

```bash
python -m sglang.launch_server \
  --model-path ~/models/Qwen3.6-27B-FP8 \
  --served-model-name qwen3.6-27b-fp8 \
  --port 8001 \
  --mem-fraction-static 0.92 \
  --context-length 254000 \
  --kv-cache-dtype auto \
  --max-running-requests 4 \
  --enable-mixed-chunk
```

⚠️ **Do NOT pass `--kv-cache-dtype fp8_e4m3` or `--kv-cache-dtype
fp8_e5m2` for Qwen3.5 / Qwen3.6 / any Gated-DeltaNet family.** It
produces `!!!`/repetition silently (filed upstream as sglang#19603).
Stick to `auto` (BF16) for those models.

**Ampere/Ada dual-GPU (AWQ-BF16-INT4 mixed):**

```bash
python -m sglang.launch_server \
  --model-path ~/models/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b \
  --port 8000 \
  --tp 2 \
  --mem-fraction-static 0.88 \
  --context-length 98304 \
  --max-running-requests 2
```

**Common knobs:**

- `--mem-fraction-static` — like vLLM's `gpu-memory-utilization`.
- `--max-running-requests` — concurrency cap. Same preempt-free math:
  total KV budget ≥ `max-running-requests × context-length`.
- `--enable-mixed-chunk` — chunked prefill; usually a win.
- `--torch-compile-max-bs N` — JIT-compiles the model graph; first
  request after start is slow, all subsequent are faster. Leave off
  on dev boxes (long startup), turn on for production.

## 5. Register it as a sharedagents model

`~/.sharedagents/config.json` → `models[]`:

```json
{
  "alias": "sgl",
  "baseUrl": "http://127.0.0.1:8001/v1",
  "id": "qwen3.6-27b-fp8",
  "backend": "sglang",
  "contextWindow": 254000,
  "maxTokens": 0,
  "capabilities": []
}
```

`/model sgl` to switch a live session.

## 6. Health check

```bash
curl -s http://127.0.0.1:8001/get_model_info | jq .
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.6-27b-fp8","messages":[{"role":"user","content":"ping"}],"max_tokens":4}' | jq .
```

If startup hangs on FlashInfer compilation, set `FLASHINFER_LOGGING=1`
to confirm progress.
