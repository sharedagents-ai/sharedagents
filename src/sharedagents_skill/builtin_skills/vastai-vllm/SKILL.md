---
name: vastai-vllm
description: >
  Provision a GPU on vast.ai and serve any HuggingFace model via vLLM — the full cold-start playbook.
  Use when the user asks to "spin a vast.ai GPU and serve a model via vLLM", "rent a GPU on vast.ai
  and run inference", "provision a B200/A100/H100 for one-shot inference", or "set up a remote vLLM
  endpoint on rented hardware". Covers provisioning, SSH, vLLM launch, port discovery, curl verification,
  and teardown — with every gotcha that has bitten us in real runs.
---

# vastai-vllm — GPU Provisioning + vLLM Serving Playbook

Provision a GPU instance on [vast.ai](https://vast.ai), launch vLLM to serve a HuggingFace model,
and verify with a `curl` probe. This skill encodes the full cold-start workflow with every gotcha
discovered in live runs.

**Top level is model-agnostic** (vLLM + any HF model). DeepSeek-V4-Flash is the worked example;
model-specific flags are in the [Model-specific flags](#model-specific-flags) table.

---

## ✅ VALIDATED 2026-08-09 — DeepSeek-V4-Flash on 2× RTX PRO 6000 Blackwell (SM120)

This is a hardware-class recipe for vLLM v0.26.0. Validate it against the
operator's runtime, model, and available memory before use.

### Working launch command (use verbatim)

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

| Setting | Why |
|---|---|
| `NCCL_P2P_DISABLE=1` | Virtualized host: NCCL communicator busy-waits in `pynccl.__init__` (`cudaStreamSynchronize` spin) → workers hang at 100% CPU before any download starts. |
| `NCCL_SHM_DISABLE=1` | Same root-cause family; both required together on this host class. |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600` | Default 600 s is too short (SM120 warmup + download + load) → "Timed out waiting for engine core processes... Waited 600s". |
| `--data-parallel-size 2` | One rank per GPU; the FP8 model fits in 96 GB, so DP beats TP here. |
| `--enable-expert-parallel` | MoE expert parallelism across the 2 ranks. |
| `--disable-custom-all-reduce` | Custom all-reduce is another hang/crash vector on virtualized hosts. |
| `--kv-cache-dtype fp8` | Needed to fit the full 262144-token context into KV cache. |
| `--gpu-memory-utilization 0.98` | CUDA graphs eat ~3.07 GiB; at 0.95 → "need 7.55 GiB KV, only 5.59 GiB available" OOM. |
| `--max-model-len 262144` | Full 256K context. |
| `--max-num-seqs 128` | Sampler warmup OOMs at lower values. |
| `--download-dir /workspace/models` | Persist the 167 GB download on the disk; avoids re-download on relaunch. |

### Cold-start stages

Kernel compilation, model download, weight loading, and graph capture can
all delay readiness. Watch logs and downloaded files rather than assuming
GPU utilization alone proves inference is ready. Measure startup time on the
operator's host and keep those measurements outside the repository.

### Validated troubleshooting (2026-08-09)

| Symptom | Root cause | Fix |
|---|---|---|
| Workers spin 100% CPU, no download; nvidia-smi "100%" with 0 apps | NCCL communicator busy-wait in `pynccl.__init__` (`cudaStreamSynchronize`) on virtualized host | `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` |
| "Timed out waiting for engine core processes... Waited 600s" | default `VLLM_ENGINE_READY_TIMEOUT_S` (600 s) too short for SM120 warmup BEFORE weight download | `VLLM_ENGINE_READY_TIMEOUT_S=3600` |
| "need 7.55 GiB KV, only 5.59 GiB available" | `--gpu-memory-utilization 0.95` too low; CUDA graphs eat 3.07 GiB | `--gpu-memory-utilization 0.98` |
| Sampler warmup OOM | warmup over-reserves at low seq counts | `--max-num-seqs 128` |
| Instance won't start / tiny disk | offer free disk < requested `--disk` (TOTAL partition GB, PERMANENT, billed per GB/hr from creation) | filter `disk_space >= 350`; GB vs GiB: 320 GB ≈ 298 GiB (HF counts GiB) |
| `huggingface_hub` ImportError on pre-baked images | stale hub in bake | `pip install --no-cache-dir "huggingface-hub>=0.34.0,<1.0"` |
| STALE IMAGE WARNING: `AssertionError: auto_functionalized was not removed` on SM120 | `vllm/vllm-openai:deepseekv4-cu130` (pushed 2026-04-24) | use `vllm/vllm-openai:v0.25.0` or `v0.26.0` (v0.26.0 pins FlashInfer 0.6.14 + SM120-capable DeepGEMM nv_dev) |

### Measure before renting

Measure throughput, first-token latency, and cold-start time on the selected
host. Keep offer identifiers, spend records, and results in the operator's
sharedagents home. Include storage, idle time, and warmup in cost comparisons.

### Status corrections (vs the historical sections below)

- **DeepGEMM / SM120:** now SUPPORTED — vLLM **v0.26.0 pins an SM120-capable DeepGEMM nv_dev
  + FlashInfer 0.6.14**. (Earlier v0.25.0-preview / Apr-2024 bakes crashed on SM120 with
  `auto_functionalized` errors.)
- **Speculative decoding (MTP / DSpark) is STILL fork-only on SM120:** `jasl/vllm` #41834
  preview tags, `voipmonitor/lucifer` b12x tags, `fraserprice/dspark-vllm`. Stock v0.26.0 =
  plain serving only.
- **`--kv-cache-dtype fp8`:** validated as NEEDED on this stack to fit the full 256K context
  (see the FP8 KV caveat section for when it's optional).
- **Cost comparison:** compare the current provider price with the measured
  cost per useful token on the selected rental, including idle time and storage.

---

## ⛔ KNOWN-FAILED COMBOS — DO NOT REPEAT WITHOUT NEW EVIDENCE

### DeepSeek-V4-Flash on a single B200 with tool-call workflows (historical compatibility caveats)

**Validation:** a plain `hello world` round trip is insufficient.
**Tool-call workflow end-to-end through sharedagents FAILED** — tools fire correctly, but synthesis
after the tool results degenerates into token-level repetition garbage
(`"table acrossing table across table acrossing-table..."` loops).

What was tried (all failed the same way):

| Stack | Flags | Outcome |
|---|---|---|
| vLLM 0.14 / `vllm/vllm-openai:deepseekv4-cu130` | `--tool-call-parser deepseek_v3` | Tool calls not decoded — DSML leaks raw |
| vLLM 0.14 | `--tool-call-parser deepseek_v4 --enable-auto-tool-choice --kv-cache-dtype fp8` | Tools fire, synthesis = repetition garbage |
| vLLM 0.14 | Same as above without FP8 KV (BF16 KV) | Same — repetition garbage |
| vLLM 0.14 | Same, `enable_thinking: false` | Same — repetition garbage |
| SGLang / `lmsysorg/sglang:deepseek-v4-blackwell` | `--tp 1 --moe-runner-backend flashinfer_mxfp4` | Same path — also fails (see note below) |

**SGLang attempt:** pivot driven by "DeepSeek themselves use SGLang as day-0 stack." On a single B200
with TP=1, `lmsysorg/sglang:deepseek-v4-blackwell` was launched with the recommended flags
(`--mem-fraction-static 0.9 --moe-runner-backend flashinfer_mxfp4 --skip-server-warmup --disable-flashinfer-autotune --chunked-prefill-size 4096 --context-length 32768`).
Trial was abandoned before final verification — but probes during the 30-tick window never showed
clean tool-call+synthesis, and the sharedagents-side synthesis-garbage symptom is reproducible
across runtimes, which points the cause **away from the inference engine**.

**Suspected root causes (in order of likelihood):**
1. **sharedagents-side parser:** raw `<｜DSML｜tool_calls>…` fragments leak into the assistant message
   history that gets sent back to the model on the next turn. The model then conditions on
   contaminated context and degenerates. The bug is upstream of the runtime — switching vLLM ↔ SGLang
   wouldn't fix it.
2. **Long tool-trajectory weakness in DeepSeek-V4-Flash itself.** Plain chat is fine; complex
   multi-tool synthesis is where it falls apart. Known issue community-side too.
3. **FP8 KV cache numerics** (less likely — symptom reproduced with BF16 KV).

**When synthesis fails across runtimes, inspect the conversation builder:**
inspect the actual `messages[]` payload sent on turn N+1 after a tool call, confirm whether
DSML tags survive into the next request, and patch the conversation builder if so. If that's
clean and the bug persists, the model is the problem — switch to DeepSeek-V4-Pro (API or rented)
or a Qwen3.6 / Llama 4 alternative that doesn't use DSML.

**Bug references suspected/observed during the trial:**
- vLLM #41132 thinking-mode broken
- vLLM #40800 DSML streaming leak
- vLLM #41240 typed-arg mishandling
- vLLM #41262 speculative decoding token loops

---

## 1. Prerequisites

Verify these exist before doing anything else:

```bash
# vast.ai CLI
test -x ~/.local/bin/vastai || echo "MISSING: pipx install vastai"

# API key
test -f ~/.config/vastai/vast_api_key || echo "MISSING: ~/.config/vastai/vast_api_key"

# SSH identity (MUST pass -i). Override if your key is not the default.
VAST_SSH_IDENTITY="${VAST_SSH_IDENTITY:-$HOME/.ssh/id_ed25519}"
test -f "$VAST_SSH_IDENTITY" || echo "MISSING: $VAST_SSH_IDENTITY"
```

All SSH commands MUST include `-i "$VAST_SSH_IDENTITY"`. A non-default
identity filename is not auto-discovered — omitting `-i` causes silent
auth failures.

---

## 2. Step-by-Step Playbook

### Step 1 — Search for an available GPU

```bash
# Example: B200, interruptible, sorted by score
~/.local/bin/vastai search offers 'gpu_name == "B200" num_gpus >= 1' \
  -i --limit 5 --order 'score-'

# Generic: any GPU with enough VRAM
~/.local/bin/vastai search offers 'gpu_ram >= 80000 num_gpus >= 1' \
  -i --limit 10 --order 'score-'
```

Key flags:
- `-i` / `--interruptible` → bid/spot pricing (3–5× cheaper than on-demand)
- `--order 'score-'` → highest-score (best value) first

Pick an **offer ID** from the output. Each offer ID provisions exactly one instance.

> **2026-08-09 note (VALIDATED):** RTX PRO 6000 Blackwell cards are listed as `RTX PRO 6000 WS`
> / `RTX PRO 6000 S` — searching "Blackwell" returns zero results. Filter
> `gpu_name in ["RTX PRO 6000 WS","RTX PRO 6000 S"] num_gpus >= 2 disk_space >= 350`,
> prefer `pcie_gen >= 5` + `pcie_bw >= 50` GB/s and `reliability >= 0.99`, sort `dph asc`.

### Step 2 — Create the instance

```bash
~/.local/bin/vastai create instance <OFFER_ID> \
  --image vllm/vllm-openai:v0.26.0 \
  --disk 320 \
  --ssh \
  --direct \
  --env '-p 8000:8000'
```

Flags explained:
| Flag | Purpose |
|---|---|
| `--image <image>` | Docker image. Use a recent vLLM image (see the STALE IMAGE warning below). |
| `--disk 320` | Local disk in GB — **TOTAL partition size, PERMANENT (cannot resize after creation), billed per GB/hr from creation time even while stopped.** Offer must have `disk_space >= 350`. GB vs GiB: 320 GB ≈ 298 GiB (HF counts GiB). |
| `--ssh` | Launch as SSH-accessible instance. |
| `--direct` | Use direct connections (faster, avoids proxy hop). |
| `--env '-p 8000:8000'` | Expose container port 8000. vast.ai NAT-maps this to an external port. |

> ⛔ **STALE IMAGE WARNING (VALIDATED 2026-08-09):** `vllm/vllm-openai:deepseekv4-cu130`
> (pushed 2026-04-24) crashes on SM120 with `AssertionError: auto_functionalized was not removed`.
> Use `vllm/vllm-openai:v0.25.0` or `v0.26.0` (v0.26.0 pins FlashInfer 0.6.14 + SM120-capable
> DeepGEMM nv_dev). The vast.ai-baked `vastai/vllm:v0.26.0-cuda-13.0` also works.
> vLLM ≥ 0.25.0 is MANDATORY for DeepSeek-V4 — the older `vastai/vllm` template ships 0.15.0.

The command returns an **instance ID**. Save it — every subsequent step needs it.

> ⛔ **Set a self-destruct alarm IMMEDIATELY after create (VALIDATED 2026-08-09, mandatory for
> unattended runs):** `nohup bash -c 'sleep 14400; ~/.local/bin/vastai destroy instance <ID> >/tmp/vast_selfdestruct.log 2>&1' >/dev/null 2>&1 &`
> (14400 s = 4 h backstop). Billing keeps running until destroy.

### Step 3 — Wait for provisioning (30–120 s)

```bash
# Poll until status is "running"
~/.local/bin/vastai show instance <INSTANCE_ID> --raw 2>/dev/null | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('actual_status','?'))"
```

The instance goes through: `created` → `scheduled` → `building` → `running`.
Wait until `running` before proceeding.

### Step 4 — Get the direct SSH target

```bash
~/.local/bin/vastai ssh-url <INSTANCE_ID>
# Output: ssh -p 17160 root@203.0.113.10 -o StrictHostKeyChecking=no
```

This returns the **direct** `IP:port` — NOT the proxy (`<ssh-proxy>`).
Use this exact `IP:port` pair for all SSH commands.

> ⛔ **Never use `<ssh-proxy>` directly.** It's a proxy, not the instance host.
> Always run `vastai ssh-url <id>` to get the real target.

### Step 5 — SSH in and fix the hf-hub version (pre-baked images only)

The `vllm/vllm-openai:deepseekv4-cu130` image ships with a conflicting
`huggingface-hub` version that breaks model loading. Fix it first:

```bash
ssh -i "$VAST_SSH_IDENTITY" -o StrictHostKeyChecking=no \
  -p <PORT> root@<IP> \
  'pip install --no-cache-dir "huggingface-hub>=0.34.0,<1.0"'
```

Symptom: vLLM crashes at startup with `ImportError` or `AttributeError` from
`huggingface_hub`. Root cause: pre-baked image has a stale hf-hub that conflicts
with vLLM's pinned dependency. Fix: pin to `>=0.34.0,<1.0`.
(Not hit on the v0.26.0 images — documented for older bakes.)

### Step 6 — Launch vLLM in a background Monitor

Startup includes model loading and kernel warmup. Use a background
`Bash` / `Monitor` and wait for explicit readiness:

```bash
ssh -i "$VAST_SSH_IDENTITY" -o StrictHostKeyChecking=no \
  -p <PORT> root@<IP> \
  'nohup vllm serve /root/DeepSeek-V4-Flash \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --kv-cache-dtype fp8 \
    > /tmp/vllm.log 2>&1 &'
```

> **2026-08-09 correction (VALIDATED):** on the dual-RTX-PRO-6000 (SM120) stack the exact
> working launch is the command in the ✅ VALIDATED section at the top of this skill —
> including `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 VLLM_ENGINE_READY_TIMEOUT_S=3600`,
> `--data-parallel-size 2 --enable-expert-parallel --disable-custom-all-reduce`,
> `--gpu-memory-utilization 0.98`, `--max-num-seqs 128` and `--download-dir /workspace/models`.

Critical flags:
| Flag | Why |
|---|---|
| `--host 0.0.0.0` | **Required.** vLLM defaults to 127.0.0.1 inside the container; vast.ai's NAT can't reach loopback. |
| `--kv-cache-dtype fp8` | **VALIDATED 2026-08-09 as REQUIRED to fit a full 256K context** on the SM120 stack (see FP8 KV caveat section for when it's optional). |
| `--max-model-len 8192` | Cap context length. B200's 183 GB VRAM handles 8K comfortably. |
| `--gpu-memory-utilization 0.95` | Leave 5% headroom for CUDA graphs. On the SM120 stack use **0.98** — CUDA graphs eat ~3.07 GiB and 0.95 OOMs at engine init. |

### Step 7 — Monitor vLLM startup (background Monitor)

Poll the log file for the `Uvicorn running` line:

```bash
ssh -i "$VAST_SSH_IDENTITY" -o StrictHostKeyChecking=no \
  -p <PORT> root@<IP> \
  'tail -f /tmp/vllm.log | grep -m1 "Uvicorn running"'
```

Record startup phases and elapsed times in the operator's sharedagents home.

### Step 8 — Discover the external port

vast.ai NAT-maps container port 8000 to a random external port. Find it:

```bash
~/.local/bin/vastai show instance <INSTANCE_ID> --raw 2>/dev/null | \
  python3 -c "
import sys, json
d = json.load(sys.stdin)
ports = d.get('ports', {})
for k, v in ports.items():
    print(f'{k} → {v[0].get(\"HostPort\",\"?\")}' if v else f'{k} → ?')
"
```

Example output: `8000/tcp → 43277`. The external URL is `http://<PUBLIC_IP>:<HostPort>`.

### Step 9 — Verify with curl

```bash
curl -s -w "\nHTTP_CODE:%{http_code}\n" -X POST \
  http://<PUBLIC_IP>:<EXTERNAL_PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": "hello world"}],
    "max_tokens": 20
  }'
```

Expected: `HTTP_CODE:200` with a JSON response containing `"choices"`.
Example plain-chat probe:

```bash
curl -s -w "\nHTTP_CODE:%{http_code}\n" -X POST \
  http://203.0.113.10:43277/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/DeepSeek-V4-Flash","messages":[{"role":"user","content":"hello world"}],"max_tokens":20}'
```

---

## 3. Gotchas

Each entry follows: **Symptom → Root cause → Fix**.

### SSH fails silently (3 attempts, no error)
- **Symptom:** `ssh root@<ssh-proxy>` hangs or returns `Permission denied` with no useful diagnostic.
- **Root cause 1:** `<ssh-proxy>` is the **proxy**, not the instance host. Direct SSH to it routes nowhere.
- **Root cause 2:** A non-default identity filename is not auto-discovered. Omitting `-i` causes auth failure.
- **Fix:** Run `vastai ssh-url <id>` to get the direct `IP:port`, then always pass `-i "$VAST_SSH_IDENTITY"`.

### Workers spin at 100% CPU, no model download (VALIDATED 2026-08-09)
- **Symptom:** On a virtualized host, workers sit at 100% CPU before any download; `nvidia-smi`
  shows GPUs at "100%" with 0 compute apps.
- **Root cause:** NCCL communicator busy-wait in `pynccl.__init__` (`cudaStreamSynchronize` spin).
- **Fix:** `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`. **NOTE: nvidia-smi util/power metrics can be
  PHANTOM on virtualized hosts — judge progress by CPU time and files, not nvidia-smi.**

### "Timed out waiting for engine core processes... Waited 600s" (VALIDATED 2026-08-09)
- **Root cause:** default `VLLM_ENGINE_READY_TIMEOUT_S` (600 s) is too short for the SM120
  kernel warmup that runs BEFORE weight download.
- **Fix:** `VLLM_ENGINE_READY_TIMEOUT_S=3600`.

### vLLM crashes on launch with KV-cache OOM
- **Symptom:** `torch.OutOfMemoryError` or `CUDA out of memory` during model load, specifically in KV cache allocation.
- **Root cause:** KV cache budget at the chosen `--max-model-len` exceeds remaining VRAM after the model loads. vLLM's error message usually includes a specific recommended `--gpu-memory-utilization` value (e.g. `0.8722`) — use that exact value, don't guess.
- **Fix ladder (try in order):**
  1. Grep the vLLM log for `gpu-memory-utilization=` recommended value, bump to that (or +0.01).
  2. Halve `--max-model-len` (1M → 512K → 256K → 128K). DeepSeek-V4-Flash on a single B200 fits at 128K with `--gpu-memory-utilization 0.95` and BF16 KV (~35 GiB KV) — or much more KV at FP8 (~17.8 GiB at 128K).
  3. Add `--kv-cache-dtype fp8` to halve KV memory — but see the FP8 KV caveat below before reaching for this.
- **VALIDATED 2026-08-09 addition:** on dual RTX PRO 6000 (SM120), the specific failure was
  "need 7.55 GiB KV, only 5.59 GiB available" at `--gpu-memory-utilization 0.95` — CUDA graphs
  eat ~3.07 GiB, so use **0.98**; sampler warmup OOMs below `--max-num-seqs 128`.

### FP8 KV cache caveat (NOT always required)
- **Status:** `--kv-cache-dtype fp8` is **optional**, not required for DeepSeek-V4. Earlier versions of this skill called it "asserted, non-negotiable" — that was wrong.
- **When to use:** When BF16 KV won't fit and you need more concurrent-request capacity or a bigger context window. **VALIDATED 2026-08-09:** on the dual-RTX-PRO-6000 SM120 stack it IS required to fit the full 262144-token context (the validated recipe uses it).
- **Risk:** FP8 KV cache has been associated with synthesis-quality degradation on DeepSeek-V4-Flash — model output can degenerate into token-level repetition garbage after tool results come back. Mechanism not fully characterized; suspected interaction between FP8 KV numerics and the model's reasoning tokens.
- **Default:** Leave KV cache at BF16 (omit `--kv-cache-dtype`). Only enable FP8 KV if you've verified output quality is unaffected on your workload.

### Synthesis degenerates after tool results (DeepSeek-V4 specific)
- **Symptom:** Tool calls fire and return real results, but the model's subsequent synthesis is repetition or nonsense (e.g. `"table acrossing table across table acrossing-table..."` over and over). Generation can appear fast; the tokens are just garbage.
- **Suspected causes:** FP8 KV cache numerics (see above); DSML fragments leaking into the tool-result message (see DSML parser bugs below); or a known DeepSeek-V4 weakness with long tool-trajectory contexts.
- **Fix:** Try in order: drop `--kv-cache-dtype fp8`; ensure the sharedagents-side parser strips raw `‹¦DSML¦...¦›` fragments from message history; lower temperature to 0; clamp `max_tokens` more aggressively.

### huggingface-hub version conflict
- **Symptom:** `ImportError` or `AttributeError` from `huggingface_hub` at vLLM startup.
- **Root cause:** Pre-baked `vllm/vllm-openai:deepseekv4-cu130` image has a stale hf-hub version that conflicts with vLLM's pinned dependency.
- **Fix:** SSH in before launch and run `pip install --no-cache-dir "huggingface-hub>=0.34.0,<1.0"`.

### curl returns "Connection refused" despite vLLM running
- **Symptom:** `Uvicorn running on http://0.0.0.0:8000` appears in logs, but external `curl` gets `Connection refused`.
- **Root cause:** vLLM bound to `127.0.0.1` (default). vast.ai's NAT can only forward traffic that reaches the container's network interface — loopback is invisible.
- **Fix:** Always pass `--host 0.0.0.0` to `vllm serve`.

### Agent hits per-turn tool-call cap
- **Symptom:** Agent reaches its configured tool-call cap in a single turn; work is incomplete.
- **Root cause:** Long-running provisioning workflows consume many tool calls (polling, SSH, file reads).
- **Fix:** Batch independent SSH probes into single `Bash` calls. Use background `Monitor` for long-running commands (vLLM launch, log tailing). Insert `sleep` generously — vLLM startup takes 17–70 min; polling every 5 s is wasteful and burns calls.

---

## 4. Cost & Teardown

### Pricing (interruptible / bid)

| GPU | VRAM | ~$/hr (bid) |
|---|---|---|
| B200 | 183 GB | $4.00 |
| H100 SXM | 80 GB | $2.00–$3.00 |
| A100 SXM | 80 GB | $1.00–$1.50 |
| RTX 6000 Ada | 48 GB | $0.50–$0.80 |
| RTX PRO 6000 WS (Blackwell SM120, 2×) | 2× 96 GB | ~$3.20 GPU + disk |

Check live pricing: `vastai search offers 'gpu_name == "B200"' -i --limit 5`.

> **VALIDATED 2026-08-09 cost notes:** disk is billed per GB/hr **from creation time** (even
> while stopped) — the validated host charged $0.0061/GB/hr (~$4.4/GB/mo, 30–90× typical);
> always check `storage_cost` before renting. At 50% load the rented server ≈ $7.10/M output
> tokens vs DeepSeek API $1.10/M output — renting is for validation, not economics.

### Teardown

**Always destroy the instance when done.** Interruptible instances keep billing until destroyed.

```bash
~/.local/bin/vastai destroy instance <INSTANCE_ID>
~/.local/bin/vastai show instances --raw   # verify: must print []
```

Recommendation: set a wall-clock alarm (e.g., `sleep 3600 && vastai destroy instance <ID>`) when
launching unattended. A forgotten B200 costs ~$100/day.
**VALIDATED pattern (2026-08-09):**
`nohup bash -c 'sleep 14400; ~/.local/bin/vastai destroy instance <ID> >/tmp/vast_selfdestruct.log 2>&1' >/dev/null 2>&1 &`
— set IMMEDIATELY after create, and always destroy + verify (`show instances --raw` → `[]`) before finishing.

---

## 5. Model-Specific Flags

vLLM is model-agnostic at the top level — any HF model works with `vllm serve <model_id>`.
The table below lists flags that specific model families require:

| Model family | Required flags | Notes |
|---|---|---|
| DeepSeek-V4 (Flash, V4) | `--kv-cache-dtype fp8` | **VALIDATED 2026-08-09:** required for full 256K ctx on 2× RTX PRO 6000 (SM120); FP8 experts, FP16 KV OOMs. Earlier "mandatory" claims on other stacks were overstated — see the FP8 KV caveat. |
| DeepSeek-V3 | `--max-model-len 8192` | Context cap if VRAM-constrained; otherwise autodetect |
| Llama-3/4 | `--max-model-len 32768` | Long-context models; tune to VRAM |
| Mixtral 8×22B | (none) | Standard vLLM launch works |
| Qwen3.6 | `--kv-cache-dtype fp8` (optional) | FP8 KV cache saves VRAM; not required |

For any new model, start with:
```bash
vllm serve <model_id> --host 0.0.0.0 --port 8000
```
Add flags only if it crashes. The error message will tell you what's missing.

---

## 6. Non-Pre-Baked Images (HF Download)

When using a generic vLLM image (`vllm/vllm-openai:v0.26.0` or similar) instead of a
pre-baked one, the model downloads from HuggingFace at first launch. Estimate:

- Download speed: typically 50–200 MB/s on vast.ai hosts
- DeepSeek-V4-Flash: 149–167 GB → 12–50 minutes extra
- Include this in the cold-start budget measured for the selected host

**VALIDATED 2026-08-09:** the xet CDN download path is fine — a suspected xet hang was ruled
out; `HF_HUB_DISABLE_XET=1` does NOT change the hang (the hang was the NCCL busy-wait).

Pre-baked images (model already in `/root/`) skip this entirely. Check the image
description on Docker Hub / vast.ai template list for pre-baked model paths.

---

## 7. Quick-Reference Command Sheet

```bash
# 1. Find a GPU
vastai search offers 'gpu_name == "B200"' -i --limit 5 --order 'score-'

# 2. Rent it
vastai create instance <OFFER_ID> --image vllm/vllm-openai:v0.26.0 \
  --disk 320 --ssh --direct --env '-p 8000:8000'

# 2b. Self-destruct alarm — set IMMEDIATELY after create (VALIDATED 2026-08-09)
nohup bash -c 'sleep 14400; ~/.local/bin/vastai destroy instance <ID> >/tmp/vast_selfdestruct.log 2>&1' >/dev/null 2>&1 &

# 3. Wait for running
vastai show instance <ID> --raw | python3 -c "import sys,json;print(json.load(sys.stdin)['actual_status'])"

# 4. Get SSH target
vastai ssh-url <ID>

# 5. Fix hf-hub (older pre-baked images only)
ssh -i "$VAST_SSH_IDENTITY" -o StrictHostKeyChecking=no -p <PORT> root@<IP> \
  'pip install --no-cache-dir "huggingface-hub>=0.34.0,<1.0"'

# 6. Launch vLLM (background) — dual RTX PRO 6000 SM120 validated recipe
ssh -i "$VAST_SSH_IDENTITY" -o StrictHostKeyChecking=no -p <PORT> root@<IP> \
  'NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 \
   nohup vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 --trust-remote-code --host 0.0.0.0 \
   --port 8000 --data-parallel-size 2 --enable-expert-parallel --disable-custom-all-reduce \
   --kv-cache-dtype fp8 --gpu-memory-utilization 0.98 --max-model-len 262144 \
   --max-num-seqs 128 --download-dir /workspace/models > /tmp/vllm.log 2>&1 &'

# 7. Wait for ready (17–70 min)
ssh ... 'tail -f /tmp/vllm.log | grep -m1 "Uvicorn running"'

# 8. Discover external port
vastai show instance <ID> --raw | python3 -c "
import sys,json;d=json.load(sys.stdin)
for k,v in d.get('ports',{}).items(): print(f'{k} → {v[0][\"HostPort\"]}')"

# 9. Verify
curl -s -w '\nHTTP_CODE:%{http_code}\n' -X POST \
  http://<IP>:<PORT>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/root/DeepSeek-V4-Flash","messages":[{"role":"user","content":"hello world"}],"max_tokens":20}'

# 10. Teardown — always destroy + verify (VALIDATED 2026-08-09)
vastai destroy instance <ID>
vastai show instances --raw   # must print []
```

---

## 8. Integration with sharedagents Agents

When an sharedagents agent invokes this skill:

1. **Use `Bash(run_in_background=true)`** for the vLLM launch (Step 6) — startup can take substantial time.
2. **Use `Monitor`** for the log tail (Step 7) — streams `Uvicorn running` when ready.
3. **Batch independent SSH calls** — `ssh ... 'cmd1 && cmd2'` rather than two separate `ssh` invocations.
4. **Plan the workflow** with `TodoWrite` before provisioning.
5. **`vastai show instance --raw` returns JSON** — parse with `python3 -c` rather than trying to `grep` formatted output.
6. **Set the self-destruct alarm right after `create instance`** and always destroy + verify
   (`show instances --raw` → `[]`) before finishing, in every outcome.

Keep run logs and proof artifacts in the operator's sharedagents home.
