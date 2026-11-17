---
name: vllm-install
description: "Install and serve a model with vLLM on a local GPU box. Covers: (1) one-line install (uv + nightly), (2) picking the right quant/dtype for the operator's GPU (Ampere 3090/4090/5090 vs Hopper H100/H200), (3) launch commands for the sharedagents-default models qwen3.6-27b / gemma-4-31b at 32k/96k/254k context, (4) wiring the running endpoint into `models[]` in ~/.sharedagents/config.json so /model can switch to it. Use whenever the user asks 'install vllm', 'serve a model', 'run qwen/gemma locally', 'set up local inference', or wants to add a new local endpoint to sharedagents."
metadata:
  sharedagents:
    always: false
---

# vLLM — install + serve a local model from sharedagents

This is the local-inference skill. Goal: get a vLLM server running on
the operator's GPU and registered as a sharedagents model alias they can
pick with `/model`.

## 1. Detect the hardware first

ALWAYS run `nvidia-smi` before recommending anything — quant choice is
hardware-bound. Map the result:

| GPU | VRAM | Compute capability | Best dtype path |
|---|---|---|---|
| RTX 3090 / 4090 | 24 GB | 8.6 / 8.9 (Ampere/Ada) | `AWQ-BF16-INT4` mixed (NOT FP8 — Marlin OOMs on Ampere) |
| RTX 5090 / PRO 6000 | 32 / 96 GB | 12.x (Blackwell) | FP8 weights, BF16 KV cache (Hopper/Blackwell path) |
| L40 / A100 / H100 / H200 | 48 / 80 / 141 GB | 8.0–9.0+ | FP8 if available, else BF16 |

For dual-3090 (48 GB total) AVOID `fp8-dynamic` — Marlin dequant
workspace OOMs and you'll see the worker crash with the misleading
`ShmRingBuffer / SemLock FileNotFoundError` (the real error is CUDA
OOM upstream). Use `cpatonn/...-AWQ-BF16-INT4` mixed quants instead.

## 2. Install

```bash
# uv is the fastest path; falls back to pip if uv isn't installed.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Pin to a recent release (nightly only if the user needs a feature
# that isn't in the latest tag yet).
uv pip install --system 'vllm>=0.7' 'huggingface_hub[cli]'
```

For Blackwell (5090) you currently need the nightly Torch + nightly
vLLM:

```bash
uv pip install --system --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install --system --pre vllm \
  --extra-index-url https://wheels.vllm.ai/nightly
```

## 3. Pull the weights

```bash
# Qwen3.6-27B FP8 (Hopper/Blackwell) — 27 GB, 254k context with BF16 KV.
huggingface-cli download Qwen/Qwen3.6-27B-Instruct-FP8 \
  --local-dir ~/models/Qwen3.6-27B-FP8

# Qwen3.6-27B AWQ-BF16-INT4 mixed (Ampere/Ada safe) — 17 GB.
huggingface-cli download cpatonn/Qwen3.6-27B-Instruct-AWQ-BF16-INT4 \
  --local-dir ~/models/Qwen3.6-27B-AWQ

# Gemma-4-31B (Google) — 18 GB AWQ-INT4 / 31 GB BF16.
huggingface-cli download google/gemma-4-31b-it \
  --local-dir ~/models/gemma-4-31b
```

## 4. Serve

The launch command must include **prefix caching** (the single
biggest perf win when sharedagents replays a turn) and a **sane round
cap**.

**Blackwell / Hopper (FP8 weights + BF16 KV):**

```bash
vllm serve ~/models/Qwen3.6-27B-FP8 \
  --served-model-name qwen3.6-27b-fp8 \
  --port 8001 \
  --enable-prefix-caching \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype auto \
  --max-num-seqs 4
```

**Ampere / Ada dual-GPU (24 GB × 2 = 48 GB) AWQ-BF16-INT4 mixed:**

```bash
vllm serve ~/models/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b \
  --port 8000 \
  --enable-prefix-caching \
  --max-model-len 98304 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 2
```

**Common knobs:**

- `--max-num-seqs` = concurrency. **Preempt-free math:** total KV
  budget ≥ `max_num_seqs × max_model_len`. Going wider than that
  thrashes: read the KV cache size vLLM logs at startup, divide it by
  the context length you need, and use that as the ceiling.
- `--enable-chunked-prefill` — long-context throughput; on by default
  in 0.7+.
- `--kv-cache-dtype fp8_e4m3` — DO NOT enable on Qwen3.5/3.6 with
  Gated DeltaNet — it silently corrupts (produces `!!!`/repetition).
  Stick to BF16 KV for those families.

## 5. Register it as a sharedagents model

Edit `~/.sharedagents/config.json` and add (or get the user to) an entry
in `models[]`:

```json
{
  "alias": "local",
  "baseUrl": "http://127.0.0.1:8001/v1",
  "id": "qwen3.6-27b-fp8",
  "backend": "vllm",
  "contextWindow": 262144,
  "maxTokens": 0,
  "capabilities": []
}
```

Then in sharedagents: `/model local` switches the live session.

## 6. Quick health check

```bash
curl -s http://127.0.0.1:8001/v1/models | jq .
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.6-27b-fp8","messages":[{"role":"user","content":"ping"}],"max_tokens":4}' | jq .
```

If the worker reports OOM after starting, drop one of:
`--gpu-memory-utilization` (try 0.90), `--max-model-len`, or
`--max-num-seqs`. Re-check the preempt-free math above.

## 7. Run it as a service (optional)

For an always-on server, write a systemd unit so it survives reboots:

```bash
sudo tee /etc/systemd/system/vllm.service >/dev/null <<EOF
[Unit]
Description=vLLM server
After=network.target

[Service]
User=$USER
ExecStart=/usr/bin/env bash -lc 'vllm serve ~/models/Qwen3.6-27B-FP8 \
  --served-model-name qwen3.6-27b-fp8 --port 8001 \
  --enable-prefix-caching --max-model-len 262144 \
  --gpu-memory-utilization 0.97 --max-num-seqs 4'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now vllm
```

## When to recommend SGLang instead

vLLM is the safe default. Recommend `sglang-install` instead when the
user wants RadixAttention prefix-cache prefix sharing across DIFFERENT
prompts (best for many short prompts that share a system prompt) or
when they need the Llama 3 / Mixtral / DeepSeek-specific kernels
SGLang has tuned. Both run the same Qwen/Gemma checkpoints.
