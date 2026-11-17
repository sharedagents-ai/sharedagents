---
name: vast-ai-rent
description: "Rent GPUs on vast.ai from the CLI and connect sharedagents to the rented endpoint. Covers: (1) install + auth (`vastai`), (2) search offers filtered by GPU type, VRAM, $/hr, region, and uptime, (3) launching with a vLLM/SGLang Docker image and the right port-forward, (4) SSH-tunneling the inference port back to the local box so sharedagents can reach it as 127.0.0.1, (5) registering the rented endpoint in `models[]`. Use when the user says 'rent a gpu', 'vast.ai', 'I need an H100/H200 for an hour', 'spin up a remote vllm', or wants compute they don't own."
metadata:
  sharedagents:
    always: false
---

# vast.ai — rent a GPU, point sharedagents at it

Use this when the local box can't run the model the user wants (the
27/31B FP8 paths need ~32 GB just for weights; the dense 70B / DeepSeek
variants need 80–141 GB). Vast.ai is the cheapest spot-style market;
expect $0.30–$2/hr per H100-80 depending on region + interruptible
status.

## 1. Install the CLI + auth

```bash
# vastai is a single Python script.
uv pip install --system vastai   # or: pip install --user vastai

# Get an API key from https://vast.ai/account/  (avoid pasting in
# shell history; vastai stores it locally for reuse):
vastai set api-key <key-from-the-web-ui>
vastai show user                  # sanity check
```

## 2. Find an offer

The query language is `field op value` joined by spaces. Useful
filters:

```bash
# H100 80GB, On-demand (not interruptible), < $2/hr, > 99% uptime,
# > 8 TB/s GPU bandwidth, in Datacenter (no residential), 1 GPU.
vastai search offers \
  'gpu_name=H100_SXM num_gpus=1 verified=True rentable=True \
   gpu_ram>=80 dph<2.0 inet_down>=500 reliability>0.99 \
   datacenter=True' --order 'dph asc' --limit 8
```

Tweak for your need:

| Want | Filter |
|---|---|
| Hopper / 80 GB / cheapest | `gpu_name=H100_SXM gpu_ram>=80 dph<1.5` |
| Hopper / 141 GB | `gpu_name=H200 gpu_ram>=141` |
| Ampere / 80 GB | `gpu_name=A100_SXM4 gpu_ram>=80` |
| Ada / 48 GB | `gpu_name=L40S gpu_ram>=48` |
| Spot (cheaper, can be evicted) | `rentable=True` plus `dph<...` and bid via `--bid_price` later |
| EU only | `country_code in [DE,FR,NL,SE,FI,IS]` |

Each result row has an `ID`, host machine info, and a `dph_total`
(dollar-per-hour). Pick the cheapest + most reliable.

## 3. Launch the rental

Use a prebuilt vLLM image. Vast.ai will expose the port number it
maps `8000` to in `Direct mapped ports`.

```bash
OFFER=12345678  # the chosen ID from the search
[ -s ~/.hf_token ] || { echo "error: ~/.hf_token missing or empty" >&2; exit 1; }
export HF_TOKEN=$(cat ~/.hf_token)   # from https://huggingface.co/settings/tokens
vastai create instance $OFFER \
  --image vllm/vllm-openai:latest \
  --disk 80 \
  --env "-p 8000:8000 -e HF_TOKEN=$HF_TOKEN" \
  --onstart-cmd 'vllm serve Qwen/Qwen3.6-27B-Instruct-FP8 \
    --served-model-name qwen3.6-27b-fp8 \
    --port 8000 \
    --enable-prefix-caching \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.97 \
    --max-num-seqs 4'
```

For SGLang use `lmsysorg/sglang:latest` and replace `--onstart-cmd`
with the SGLang launch (see the `sglang-install` skill).

Watch the boot log live:

```bash
INSTANCE=$(vastai show instances --raw | jq -r '.[0].id')
vastai logs $INSTANCE --tail 80
```

## 4. Get the connection info

```bash
vastai show instance $INSTANCE
```

Look for `ssh_host`, `ssh_port`, and the public port mapping for
`8000` (under `public_ipaddr` / `ports`). Use the public mapping if
the vast.ai host exposes it directly; otherwise SSH-tunnel.

**SSH tunnel (treat the remote endpoint as `127.0.0.1` locally):**

```bash
SSH_HOST=$(vastai show instance $INSTANCE --raw | jq -r '.ssh_host')
SSH_PORT=$(vastai show instance $INSTANCE --raw | jq -r '.ssh_port')

# Background tunnel: local :8001 -> remote :8000
ssh -fN -L 8001:127.0.0.1:8000 \
    -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    -p $SSH_PORT root@$SSH_HOST
```

(Vast.ai injects your `~/.ssh/id_rsa.pub` if you uploaded it before
launching; otherwise `vastai change-ssh-key` first.)

## 5. Register in sharedagents

`~/.sharedagents/config.json` → `models[]`:

```json
{
  "alias": "vast",
  "baseUrl": "http://127.0.0.1:8001/v1",
  "id": "qwen3.6-27b-fp8",
  "backend": "vllm",
  "contextWindow": 262144,
  "maxTokens": 0,
  "capabilities": []
}
```

Switch the live session with `/model vast`. The first request to a
fresh boot can take 30–90s while weights download to the instance's
local disk; subsequent requests are normal latency.

## 6. Cost discipline

```bash
# Show current $/hr + accumulated cost.
vastai show instance $INSTANCE --raw | jq '{dph,cur_billed,uptime_mins}'

# Stop billing when you're done (KEEPS the instance, no compute).
vastai stop instance $INSTANCE

# Or destroy it (and the disk).
vastai destroy instance $INSTANCE
```

Set a session-end checklist: sharedagents's `/restart`-style cleanup
shouldn't leave a rented H200 spinning at $2/hr overnight.

## 7. When to use this vs your own box

Run locally when:
- The workload fits in your VRAM (`vllm-install` step 1).
- The session is long-running (the SSH tunnel + boot delay isn't
  worth it for 10 minutes of work).

Rent vast.ai when:
- You need a quant/dtype your card can't host (FP8 on a 3090).
- The model size needs >24 GB (Qwen3.6-27B FP8 needs ≥32 GB; the
  dense 70B+ needs ≥80 GB).
- You're benchmarking or doing one-off heavy inference and don't want
  to keep the GPU on.

## 8. VALIDATED 2026-08-09 — DeepSeek-V4-Flash on 2× RTX PRO 6000 Blackwell (SM120)

Full validated recipe for serving `deepseek-ai/DeepSeek-V4-Flash-0731`
(167 GB FP8 weights, 256K ctx) on a dual RTX PRO 6000 WS box with vLLM
`v0.26.0`. Validate the flags against the selected host and keep all rental
identifiers and measurements in the operator's sharedagents home.

### 8.1 GPU naming gotcha

vast.ai lists these cards as **`RTX PRO 6000 WS`** and **`RTX PRO 6000 S`** —
they are NOT named "Blackwell" in the marketplace. Searching
`gpu_name in ["RTX PRO 6000 Blackwell"]` returns **ZERO results**.

```bash
vastai search offers \
  'gpu_name in ["RTX PRO 6000 WS","RTX PRO 6000 S"] num_gpus >= 2 disk_space >= 350' \
  --order 'dph asc'
```

Then prefer Gen5 PCIe (`pcie_gen >= 5`, `pcie_bw >= 50` GB/s) and
`reliability >= 0.99`; pick the cheapest `dph`.

### 8.2 Create the instance

```bash
vastai create instance <OFFER_ID> \
  --image vllm/vllm-openai:v0.26.0 \
  --disk 320 \
  --ssh --direct \
  --env '-p 8000:8000'
```

- `--disk` is the **TOTAL partition in GB and is PERMANENT** — it cannot
  be resized after creation. Offer must have `disk_space >= 350`
  (requesting more disk than the offer has free → instance won't start
  or comes up unusable).
- **Storage is billed per GB/hr from creation time** (even while the
  instance is stopped) — see section 10 for the disk-rate trap.
- GB vs GiB: 320 GB ≈ 298 GiB; HuggingFace counts GiB.
- The vast.ai-baked `vastai/vllm:v0.26.0-cuda-13.0` image also works
  (both validated). Do NOT use the stale `vllm/vllm-openai:deepseekv4-cu130`
  image — see row g in the troubleshooting table.

Set the self-destruct alarm immediately after create (section 12).

### 8.3 Full working launch command (use verbatim)

```bash
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000 \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.98 \
  --max-model-len 262144 \
  --max-num-seqs 128 \
  --download-dir /workspace/models
```

Rationale for every flag:

| Setting | Why |
|---|---|
| `NCCL_P2P_DISABLE=1` | Virtualized host: NCCL communicator busy-waits in `pynccl.__init__` (`cudaStreamSynchronize` spin) → workers hang at 100% CPU before any download starts. Disables P2P transport. |
| `NCCL_SHM_DISABLE=1` | Same root-cause family; forces non-SHM transport. Both together are required on this host class. |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600` | Default (600 s) is too short: SM120 kernel warmup + 167 GB download + weight load blow past it → "Timed out waiting for engine core processes". |
| `--data-parallel-size 2` | One rank per GPU; the FP8-quantized model fits in 96 GB, so DP beats TP here. |
| `--enable-expert-parallel` | MoE expert parallelism across the 2 ranks (matches this model's EP-friendly layout). |
| `--disable-custom-all-reduce` | Custom all-reduce is another hang/crash vector on the virtualized host; disable it. |
| `--kv-cache-dtype fp8` | Needed to fit a full 262144-token context into KV cache. |
| `--gpu-memory-utilization 0.98` | CUDA graphs eat ~3.07 GiB; at 0.95 you get "need 7.55 GiB KV, only 5.59 GiB available" → OOM at engine init. |
| `--max-model-len 262144` | Full 256K context. |
| `--max-num-seqs 128` | Sampler warmup OOMs at lower values; 128 fixes warmup. |
| `--download-dir /workspace/models` | Persist the 167 GB download on the 320 GB disk; avoids re-download on relaunch. |

### 8.4 Troubleshooting (all rows VALIDATED 2026-08-09)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| a | Workers spin at 100% CPU, no model download; nvidia-smi shows GPUs "100%" with 0 compute apps | NCCL communicator busy-wait in `pynccl.__init__` (`cudaStreamSynchronize`) on virtualized hosts | `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`. **NOTE: nvidia-smi util/power metrics can be PHANTOM on virtualized hosts — judge progress by CPU time and files, not nvidia-smi.** |
| b | "Timed out waiting for engine core processes... Waited 600s" | default `VLLM_ENGINE_READY_TIMEOUT_S` (600 s) too short for the SM120 kernel warmup that runs BEFORE weight download | set `VLLM_ENGINE_READY_TIMEOUT_S=3600` |
| c | "need 7.55 GiB KV, only 5.59 GiB available" | `--gpu-memory-utilization 0.95` leaves too little after weights + 3.07 GiB CUDA graphs | use `--gpu-memory-utilization 0.98` |
| d | Sampler warmup OOM | warmup over-reserves at low seq counts | `--max-num-seqs 128` |
| e | Instance won't start / tiny disk | offer's free disk < requested `--disk` | filter `disk_space >= 350` in search; `--disk` is TOTAL partition GB, PERMANENT (cannot resize after creation); storage billed per GB/hr from creation time (even while stopped); GB vs GiB (320 GB ≈ 298 GiB — HF counts GiB) |
| f | `huggingface_hub` ImportError on pre-baked images | image ships a stale huggingface-hub | `pip install --no-cache-dir "huggingface-hub>=0.34.0,<1.0"` |
| g | STALE IMAGE WARNING: crashes on SM120 with `AssertionError: auto_functionalized was not removed` | `vllm/vllm-openai:deepseekv4-cu130` (pushed 2026-04-24) | use `vllm/vllm-openai:v0.25.0` or `v0.26.0` (v0.26.0 pins FlashInfer 0.6.14 + SM120-capable DeepGEMM nv_dev) |

### 8.5 First-boot readiness

Watch kernel warmup, model download, weight loading, and graph capture in
sequence. Judge progress by logs, CPU time, and downloaded files rather than
GPU utilization alone. Wait for the serving process to report readiness,
then validate inference. Measure elapsed times on the selected host and keep
them only in the operator's sharedagents home.

## 9. Benchmark the selected host

Measure first-token latency, steady-state throughput, concurrent throughput,
and cold-start time with the intended workload. Include GPU, disk, warmup,
and idle charges. Keep offer identifiers, spend records, and measurements
in the operator's sharedagents home; do not copy them into the repository.

## 10. Official vast.ai DeepSeek-V4-Flash templates (VALIDATED 2026-08-09)

Team-created templates (all download-at-boot; the model is **ungated**):

| Template | Hash (template_id) |
|---|---|
| vLLM 0.26 + 0731 (1M ctx) | `74abba01dd9faaaaf06784de07269033` → `cloud.vast.ai/?template_id=...` |
| SGLang | `e49e4c90e59be305d09b31b5bec064b1` |
| vLLM 0.25 preview | `2bdf3a2ef690d2df83dc7d86b1089972` |
| DSpark SGLang (H100-compatible) | `b2ae6477dd89cc2c06a59b86c58afc4d` |

- **vLLM ≥ 0.25.0 is MANDATORY** — the older `vastai/vllm` template
  ships vLLM 0.15.0, which cannot serve V4.
- Flagship-class filter (B200-class only): `num_gpus >= 4`,
  `compute_cap >= 1000`, `gpu_ram >= 700` (i.e. ≥ 700 GB).

## 11. Compare rental and API costs

Use current provider pricing and the selected host's measured useful-token
throughput. Include idle time, warmup, and storage before choosing a rental.
Renting can also support validation and workflow testing. Keep the detailed
cost calculation in the operator's sharedagents home.

## 12. Self-destruct alarm pattern (MANDATORY for unattended runs)

Set this **immediately after `create instance`** — billing keeps
running until destroy:

```bash
nohup bash -c 'sleep 14400; ~/.local/bin/vastai destroy instance <ID> >/tmp/vast_selfdestruct.log 2>&1' >/dev/null 2>&1 &
```

- `14400` s = 4 h backstop; adjust to your budget.
- **Always destroy + verify** before finishing, in every outcome:

```bash
vastai destroy instance <ID>
vastai show instances --raw   # must print [] — empty list
```
