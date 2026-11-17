"""Local-GPU auto-detect + model auto-provision — sharedagents's "moat" UX.

The pitch: ``curl … install.sh | sh`` → ``sharedagents`` → it finds your NVIDIA
GPU(s), picks the best known model for that GPU profile, makes sure a local
OpenAI-compatible server is serving it (preferring vLLM, falling back to
Ollama), and points the ``main`` agent at it — so you're chatting with your
own hardware in one command, no questions asked.

This module is the detection + selection + provisioning core; the first-run
wizard (:mod:`sharedagents_gateway.agent_cli`) calls :func:`auto_local_setup` before
it offers the cloud fallback.

Design notes:
- **Detect-first, provision-second.** A box that already serves a model on a
  known port (the common case on a GPU workstation) is wired up instantly via
  :func:`probe_local_server` — sharedagents doesn't re-download or relaunch anything.
- **No hard deps.** Everything shells out (``nvidia-smi``) or uses ``httpx``
  (already a core dep). vLLM / Ollama are discovered at runtime, never imported.
- **GPU profile → model table** is the seed of what will eventually live at
  sharedagents.ai. For now it's a small curated table keyed by GPU model + count,
  with a VRAM-class heuristic fallback for unknown cards.
"""

from __future__ import annotations

import getpass
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

# Ports we'll probe for an already-running OpenAI-compatible server, in the
# order most likely to be "the" local LLM. vLLM defaults to 8000; the
# qwen-stack convention puts the chat model on 8001; sglang uses 30000;
# Ollama 11434; LM Studio 1234.
_PROBE_PORTS: tuple[tuple[int, str], ...] = (
    (8000, "vllm"),
    (8001, "vllm"),
    (8005, "vllm"),
    (8002, "vllm"),
    (18000, "vllm"),
    (30000, "sglang"),
    (11434, "ollama"),
    (1234, "lmstudio"),
)


@dataclass(frozen=True)
class GpuInfo:
    """One CUDA device as reported by ``nvidia-smi``."""

    name: str  # e.g. "NVIDIA GeForce RTX 5090"
    vram_mib: int  # total memory, MiB


@dataclass(frozen=True)
class GpuProfile:
    """A box's GPUs, summarised for model selection."""

    gpus: tuple[GpuInfo, ...]

    @property
    def count(self) -> int:
        return len(self.gpus)

    @property
    def primary_name(self) -> str:
        return self.gpus[0].name if self.gpus else ""

    @property
    def total_vram_gib(self) -> float:
        return round(sum(g.vram_mib for g in self.gpus) / 1024, 1)

    @property
    def per_gpu_vram_gib(self) -> float:
        return round(self.gpus[0].vram_mib / 1024, 1) if self.gpus else 0.0

    def short_name(self) -> str:
        """Compact human label, e.g. "2×RTX 5090 (64 GiB)"."""
        n = self.primary_name.replace("NVIDIA ", "").replace("GeForce ", "").strip()
        prefix = f"{self.count}×" if self.count > 1 else ""
        return f"{prefix}{n} ({self.total_vram_gib:.0f} GiB)"


@dataclass(frozen=True)
class LocalServer:
    """An OpenAI-compatible local model server sharedagents can point at."""

    base_url: str  # ends in /v1
    model_id: str  # the served model id
    backend: str  # vllm | sglang | ollama | lmstudio | unknown
    max_model_len: int | None = None  # live ctx from /v1/models, when reported


@dataclass(frozen=True)
class ModelChoice:
    """A model to provision for a GPU profile, with its serving recipe.

    The recipe (which vLLM tool/reasoning parser) is part of "the best model"
    — it's what makes the model's tool calls + thinking actually work, and it
    differs by build: the fp8 Qwen3.6 wants ``qwen3_coder``, the AWQ build
    ``qwen3_xml`` (mirroring known-good launch recipes), and Qwen2.5
    wants ``hermes`` with no reasoning parser."""

    model_id: str  # HF repo id (vLLM) or ollama tag
    serve: str  # "vllm" | "ollama"
    context_window: int  # advertised context for the config row
    note: str = ""
    tool_parser: str = "qwen3_xml"  # vLLM --tool-call-parser ("" = omit)
    reasoning_parser: str = "qwen3"  # vLLM --reasoning-parser ("" = omit)
    # Per-GPU serving recipe — extra vLLM serve flags and env, mirroring that
    # box's live launch config. Each GPU profile owns its full config.
    extra_args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    # True for a curated table entry whose recipe is authoritative (don't apply
    # generic multi-GPU fallbacks); False for the VRAM heuristic (unlisted card).
    recipe_complete: bool = False
    # Context to serve at on a COLD launch — each GPU profile owns its own value
    # (conservative per-card KV-pool headroom; capped at context_window). NOT a
    # global knob: a card with the VRAM for it sets a bigger number here.
    launch_ctx: int = 32_768


# --------------------------------------------------------------------------
# 1. GPU detection
# --------------------------------------------------------------------------


def detect_gpus() -> list[GpuInfo]:
    """Return the local CUDA devices, or [] when there's no NVIDIA GPU.

    Shells out to ``nvidia-smi``; any failure (binary missing, no driver,
    no device) yields an empty list so callers cleanly fall back to cloud.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    gpus: list[GpuInfo] = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            vram = int(float(parts[1]))
        except ValueError:
            continue
        gpus.append(GpuInfo(name=name, vram_mib=vram))
    return gpus


def gpu_profile() -> GpuProfile | None:
    """Summarise the local GPUs, or None when there's no GPU."""
    gpus = detect_gpus()
    if not gpus:
        return None
    return GpuProfile(gpus=tuple(gpus))


# --------------------------------------------------------------------------
# 2. Probe for an already-running local server (the fast path)
# --------------------------------------------------------------------------


def probe_local_server(timeout: float = 1.5) -> LocalServer | None:
    """Find an OpenAI-compatible model server already listening locally.

    Tries each known port's ``GET /v1/models`` (Ollama's ``/v1/models``
    works too). Returns the first server advertising at least one model.
    This is what makes a GPU workstation that already runs vLLM come up
    chatting instantly — sharedagents just wires onto it.
    """
    for port, backend in _PROBE_PORTS:
        base = f"http://127.0.0.1:{port}/v1"
        try:
            r = httpx.get(f"{base}/models", timeout=timeout)
        except httpx.HTTPError:
            continue
        if r.status_code != 200:
            continue
        try:
            data = r.json()
        except ValueError:
            continue
        rows = data.get("data") if isinstance(data, dict) else None
        if not rows:
            continue
        model_id = ""
        max_len = None
        for row in rows:
            if isinstance(row, dict) and row.get("id"):
                model_id = str(row["id"])
                mml = row.get("max_model_len")
                if isinstance(mml, int) and mml > 0:
                    max_len = mml
                break
        if not model_id:
            continue
        return LocalServer(base_url=base, model_id=model_id, backend=backend, max_model_len=max_len)
    return None


# --------------------------------------------------------------------------
# 3. GPU profile → best known model (the seed of the sharedagents.ai table)
# --------------------------------------------------------------------------

# Curated table: substring matched against the GPU model name → ordered
# choices by GPU count. This is what sharedagents.ai will eventually serve. Keys are
# lowercased substrings of the nvidia-smi GPU name.
#
# Each entry maps a minimum GPU count to a ModelChoice. Selection picks the
# highest min-count entry whose count <= the box's GPU count, so a 2-GPU box
# matched against {1: …, 2: …} gets the 2-GPU choice.
#
# Qwen3.6-27B is the recommended house model — the best for these cards — and
# each GPU family wants a different DOWNLOADABLE quant of it:
#   * Ampere (RTX 3090): no fp8 tensor cores → AWQ-Int4 (cyankiwi build).
#   * Blackwell (RTX 5090 / RTX PRO 6000): fp8 tensor cores → the fp8 build.
# These ids are real HF repos so the cold-provision path downloads the right
# model (not a generic fallback). On the detect-first path sharedagents wires onto
# whatever the box already serves and only borrows the context window from here.
# The VRAM heuristic below covers any card not listed with public Qwen2.5 ids.
# Each entry carries its full serving recipe (parsers + flags + env) for that
# GPU profile, derived from a known-good launch config for that hardware.
_MODEL_TABLE: dict[str, dict[int, ModelChoice]] = {
    # 2× RTX 5090: public fp8 build, qwen3_coder parser, 262k, FlashAttn-2
    # (Blackwell has fp8 tensor cores; no custom-all-reduce disable needed).
    "5090": {
        2: ModelChoice(
            "Qwen/Qwen3.6-27B-FP8",
            "vllm",
            262_144,
            "dual-5090 (Blackwell): fp8 Qwen3.6-27B @ 256k",
            tool_parser="qwen3_coder",
            reasoning_parser="qwen3",
            env=(("NCCL_P2P_DISABLE", "1"), ("NCCL_IB_DISABLE", "1"), ("VLLM_FLASH_ATTN_VERSION", "2")),
            recipe_complete=True,
            launch_ctx=262_144,
        ),  # 256k — Blackwell fp8 has the VRAM
    },
    "rtx pro 6000": {
        1: ModelChoice(
            "Qwen/Qwen3.6-27B-FP8",
            "vllm",
            262_144,
            "RTX PRO 6000 (Blackwell): fp8 Qwen3.6-27B @ 256k",
            tool_parser="qwen3_coder",
            reasoning_parser="qwen3",
            env=(("VLLM_FLASH_ATTN_VERSION", "2"),),
            recipe_complete=True,
            launch_ctx=262_144,
        ),  # 256k — 96 GB Blackwell
    },
    # 2× RTX 3090: AWQ-Int4 build (no fp8 on Ampere), qwen3_xml parser. Works
    # WITH OR WITHOUT the NVLink bridge. Verified end-to-end on a test machine (dual RTX
    # 3090 + NVLink), 2026-06-01:
    #   * --disable-custom-all-reduce: custom AR CUDA-crashes on dual-3090
    #     (custom_all_reduce.cuh:455 'invalid argument') → vLLM uses NCCL all-reduce.
    #   * NCCL_P2P_LEVEL=NVL: run the all-reduce over NVLink when the bridge is
    #     fitted; NCCL gracefully falls back to SHM/host when it isn't — so this
    #     one env is correct with or without NVLink (no separate bridgeless recipe).
    #   * KV cache stays bf16 (vLLM default — no --kv-cache-dtype) for long-context
    #     quality; --max-num-seqs 8 for several concurrent sessions.
    #   * launch_ctx=64k (65536): this card's own served max-model-len.
    #   * LimitMEMLOCK=infinity (in the rendered unit): NCCL pins NVLink IPC
    #     buffers; systemd's 8 MB default deadlocks init (harmless without NVLink).
    "3090": {
        2: ModelChoice(
            "cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4",
            "vllm",
            96_000,
            "dual-3090 (Ampere, no fp8): Qwen3.6-27B AWQ-Int4 @ 64k",
            tool_parser="qwen3_xml",
            reasoning_parser="qwen3",
            extra_args=("--max-num-seqs", "8", "--disable-custom-all-reduce"),
            env=(("NCCL_P2P_LEVEL", "NVL"), ("NCCL_CUMEM_ENABLE", "0"), ("VLLM_WORKER_MULTIPROC_METHOD", "spawn")),
            recipe_complete=True,
            launch_ctx=65_536,
        ),
    },
    # Smaller Ampere consumer cards: a Qwen2.5 instruct (hermes parser, no
    # reasoning) with the same dual-GPU stability recipe as the 3090.
    "3060": {
        2: ModelChoice(
            "Qwen/Qwen2.5-7B-Instruct",
            "vllm",
            32_768,
            "dual-3060: 7B fits 24 GiB",
            tool_parser="hermes",
            reasoning_parser="",
            extra_args=("--disable-custom-all-reduce",),
            env=(("NCCL_P2P_DISABLE", "1"), ("NCCL_CUMEM_ENABLE", "0"), ("VLLM_WORKER_MULTIPROC_METHOD", "spawn")),
            recipe_complete=True,
        ),
    },
}


def _heuristic_choice(profile: GpuProfile) -> ModelChoice | None:
    """Pick a model by total-VRAM class when the card isn't in the table.

    Conservative, broadly-available instruct models sized to fit with room
    for KV cache. vLLM repo ids; falls back to Ollama tags in the wizard
    when vLLM isn't installed (see :func:`provision_model`).
    """
    vram = profile.total_vram_gib
    # (min total VRAM GiB, model id, note). Qwen2.5 instruct → hermes tool
    # parser, no reasoning parser (not a reasoning model).
    bands = [
        (120, "Qwen/Qwen2.5-72B-Instruct", "72B-class fits ≥120 GiB"),
        (60, "Qwen/Qwen2.5-32B-Instruct", "32B-class for 60–120 GiB"),
        (36, "Qwen/Qwen2.5-14B-Instruct", "14B-class for 36–60 GiB"),
        (20, "Qwen/Qwen2.5-7B-Instruct", "7B-class for 20–36 GiB"),
        (8, "Qwen/Qwen2.5-3B-Instruct", "3B-class for 8–20 GiB"),
    ]
    for floor, model_id, note in bands:
        if vram >= floor:
            return ModelChoice(model_id, "vllm", 32_768, note, tool_parser="hermes", reasoning_parser="")
    return None


def select_model_for_profile(profile: GpuProfile) -> ModelChoice | None:
    """Best known model for a GPU profile: curated table, then heuristic."""
    name = profile.primary_name.lower()
    for key, by_count in _MODEL_TABLE.items():
        if key in name:
            pick = None
            for min_count in sorted(by_count):
                if profile.count >= min_count:
                    pick = by_count[min_count]
            if pick:
                return pick
    return _heuristic_choice(profile)


# --------------------------------------------------------------------------
# 4. Provision (best-effort) when nothing is running
# --------------------------------------------------------------------------


def _emit(msg: str) -> None:
    """Provisioning progress → stderr, so the first-run wizard surfaces each
    step without polluting ``-p`` stdout. Never raises."""
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _provision_port() -> int:
    try:
        return int(os.environ.get("SHAREDAGENTS_PROVISION_PORT", "8000"))
    except ValueError:
        return 8000


def _server_model(base_url: str) -> str | None:
    """The first served model id at ``base_url``, or None if not ready."""
    try:
        r = httpx.get(f"{base_url}/models", timeout=2.0)
        if r.status_code != 200:
            return None
        data = r.json()
        rows = data.get("data") if isinstance(data, dict) else None
        if rows:
            for row in rows:
                if isinstance(row, dict) and row.get("id"):
                    return str(row["id"])
    except (httpx.HTTPError, ValueError):
        return None
    return None


def _hf_cache_path(model_id: str) -> str:
    """Local HF hub cache dir for ``model_id`` (where vLLM downloads weights)."""
    safe = "models--" + model_id.replace("/", "--")
    root = os.environ.get("HF_HOME")
    hub = os.path.join(root, "hub") if root else os.path.expanduser("~/.cache/huggingface/hub")
    return os.path.join(hub, safe)


def _hf_total_bytes(model_id: str) -> int | None:
    """Total download size of a HF repo (best-effort, for the progress bar)."""
    try:
        r = httpx.get(
            f"https://huggingface.co/api/models/{model_id}/tree/main",
            params={"recursive": "true"},
            timeout=10.0,
            follow_redirects=True,
        )
        if r.status_code != 200:
            return None
        total = 0
        for it in r.json():
            if isinstance(it, dict) and it.get("type") == "file":
                sz = (it.get("lfs") or {}).get("size") or it.get("size") or 0
                total += int(sz)
        return total or None
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        return None


def _dir_bytes(path: str) -> int:
    """Bytes on disk under ``path`` (symlinks not followed → blobs counted once)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _has_incomplete(cache: str | None) -> bool:
    """True if the HF cache dir has an in-progress (``*.incomplete``) download."""
    if not cache or not os.path.isdir(cache):
        return False
    return any(any(f.endswith(".incomplete") for f in files) for _r, _d, files in os.walk(cache))


def _model_cached(model_id: str) -> bool:
    """True if ``model_id``'s weights are already fully in the HF cache (a
    snapshot with weight files and no in-progress downloads) — so vLLM will
    load from disk and we can skip the download phase entirely."""
    cache = _hf_cache_path(model_id)
    if not os.path.isdir(cache) or _has_incomplete(cache):
        return False
    snap = os.path.join(cache, "snapshots")
    if not os.path.isdir(snap):
        return False
    return any(any(f.endswith((".safetensors", ".bin", ".gguf")) for f in files) for _r, _d, files in os.walk(snap))


def _progress_bar(downloaded: int, total: int | None, width: int = 26) -> str:
    gb = 1024**3
    if total and total > 0:
        frac = min(downloaded / total, 1.0)
        fill = int(frac * width)
        bar = "█" * fill + "░" * (width - fill)
        return f"  ↓ downloading model [{bar}] {frac * 100:3.0f}%  {downloaded / gb:.1f}/{total / gb:.1f} GB"
    return f"  ↓ downloading model… {downloaded / gb:.1f} GB so far"


def _wait_for_server(
    base_url: str, proc=None, timeout_s: float = 1800.0, backend: str = "vllm", progress_model: str | None = None
) -> LocalServer | None:
    """Poll ``{base}/models`` until a model is served, the launched process
    dies, or we time out. When ``progress_model`` is set, render a live
    download progress bar (weights size vs HF cache on disk) so a long cold
    download never looks stuck — then a distinct "loading into GPU" phase."""
    import time

    deadline = time.time() + timeout_s
    total = _hf_total_bytes(progress_model) if progress_model else None
    cache = _hf_cache_path(progress_model) if progress_model else None
    try:
        tty = bool(sys.stderr.isatty())
    except Exception:
        tty = False
    last_line = 0.0

    def _clear():
        if tty and progress_model:
            try:
                sys.stderr.write("\r" + " " * 72 + "\r")
                sys.stderr.flush()
            except Exception:
                pass

    while time.time() < deadline:
        mid = _server_model(base_url)
        if mid:
            _clear()
            return LocalServer(base_url, mid, backend)
        if proc is not None and proc.poll() is not None:
            _clear()
            return None  # the server exited before becoming ready
        now = time.time()
        if progress_model and cache and os.path.isdir(cache):
            downloaded = _dir_bytes(cache)
            if total and downloaded >= total * 0.99:
                msg = "  ⏳ loading model into GPU (CUDA init + graph capture)…"
            else:
                msg = _progress_bar(downloaded, total)
            if tty:
                sys.stderr.write("\r" + msg.ljust(70))
                sys.stderr.flush()
            elif now - last_line >= 15.0:
                _emit(msg.strip())
                last_line = now
        elif now - last_line >= 20.0:
            _emit("    … still provisioning (starting server)…")
            last_line = now
        time.sleep(2.0 if tty else 4.0)
    _clear()
    return None


def _module_available(mod: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _resolve_vllm_launch(model_id: str) -> list[str] | None:
    """argv prefix that launches an OpenAI-compatible vLLM server for
    ``model_id``, or None when no vLLM is reachable.

    Order: ``SHAREDAGENTS_VLLM_BIN`` → a ``vllm`` on PATH (both ``vllm serve
    <model>``) → ``python -m vllm.entrypoints.openai.api_server --model`` when
    vLLM is importable in sharedagents's own interpreter. The env knob lets a box
    that keeps vLLM in a separate venv (the common case — vLLM is NOT an
    sharedagents dependency) point sharedagents at it."""
    explicit = os.environ.get("SHAREDAGENTS_VLLM_BIN", "").strip()
    if explicit and os.path.exists(explicit):
        return [explicit, "serve", model_id]
    onpath = shutil.which("vllm")
    if onpath:
        return [onpath, "serve", model_id]
    managed = os.path.join(_managed_vllm_dir(), "bin", "vllm")
    if os.path.exists(managed):
        return [managed, "serve", model_id]
    if _module_available("vllm"):
        return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_id]
    return None


def _managed_vllm_dir() -> str:
    """Isolated venv sharedagents installs vLLM into, so vLLM's heavy CUDA/torch
    deps never collide with sharedagents's own lean environment.
    """
    return os.path.expanduser("~/.sharedagents/vllm-venv")


def _python_for_vllm() -> str | None:
    """A CPython that vLLM ships wheels for (3.10–3.12). Prefer one on PATH,
    else the running interpreter when it's in range — vLLM has no wheel for
    e.g. 3.14, so we don't try to build it from source here."""
    for cand in ("python3.12", "python3.11", "python3.10"):
        p = shutil.which(cand)
        if p:
            return p
    if (3, 10) <= sys.version_info[:2] <= (3, 12):
        return sys.executable
    return None


def _ensure_vllm_launch(model_id: str, allow_install: bool) -> list[str] | None:
    """Find a usable vLLM, INSTALLING one into an isolated venv if none exists.

    This is what lets sharedagents stand up a model on a genuinely bare GPU box —
    no vLLM, no Ollama — with a single command. The install is a large one-time
    download; gated by ``allow_install`` (off via ``SHAREDAGENTS_NO_VLLM_INSTALL=1``).
    Returns the launch argv or None."""
    launch = _resolve_vllm_launch(model_id)
    if launch is not None:
        return launch
    if not allow_install:
        return None
    venv = _managed_vllm_dir()
    vbin = os.path.join(venv, "bin", "vllm")
    if os.path.exists(vbin):
        _ensure_managed_ninja(venv)
        return [vbin, "serve", model_id]
    py = _python_for_vllm()
    if not py:
        _emit(
            "    can't install vLLM: no CPython 3.10–3.12 found (vLLM ships no "
            "wheel for this Python). Install python3.12 or point "
            "SHAREDAGENTS_VLLM_BIN at an existing vLLM."
        )
        return None
    _emit(f"  ↓ installing vLLM into {venv} (one-time, multi-GB download — this can take several minutes)…")
    try:
        os.makedirs(os.path.dirname(venv), exist_ok=True)
        subprocess.run([py, "-m", "venv", venv], timeout=180, check=True)
        pip = os.path.join(venv, "bin", "pip")
        subprocess.run([pip, "install", "-q", "--upgrade", "pip"], timeout=300)
        # `ninja` is needed at runtime to JIT-compile some model kernels (e.g.
        # Qwen3.6's Mamba/GDN layers); plain `pip install vllm` omits it.
        subprocess.run(
            [pip, "install", "vllm", "ninja"],
            timeout=float(os.environ.get("SHAREDAGENTS_VLLM_INSTALL_TIMEOUT", "3600")),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _emit(f"    vLLM install failed: {exc}")
        return None
    # Trust the entrypoint's existence over pip's return code — pip can exit
    # non-zero on harmless post-install warnings while vLLM is fully usable.
    if not os.path.exists(vbin):
        _emit("    vLLM install failed — see pip output above")
        return None
    _emit("  ✓ vLLM installed")
    return [vbin, "serve", model_id]


def _ensure_managed_ninja(venv: str) -> None:
    """Make sure ``ninja`` is in the managed vLLM venv (needed to JIT-compile
    some model kernels). Idempotent; a no-op once present."""
    if os.path.exists(os.path.join(venv, "bin", "ninja")):
        return
    pip = os.path.join(venv, "bin", "pip")
    if not os.path.exists(pip):
        return
    try:
        subprocess.run([pip, "install", "-q", "ninja"], timeout=300)
    except (OSError, subprocess.SubprocessError):
        pass


def _parsers_for_model(model_id: str) -> tuple[str, str]:
    """``(tool_parser, reasoning_parser)`` compatible with a model FAMILY.

    The parsers are a property of the model, not the GPU. Applying Qwen3's
    reasoning/tool parsers to a Qwen2.5 model makes vLLM abort at startup
    (``Qwen3ReasoningParser … could not locate think start/end tokens``), so when
    the model actually being served differs from the GPU table's choice (an
    ``SHAREDAGENTS_PROVISION_MODEL`` override, or a VRAM-heuristic swap) we re-derive
    the parsers from the model id rather than trust the table's:

    - Qwen3* (incl. Qwen3.6) → ``qwen3`` reasoning + ``qwen3_xml`` tool calls.
    - Qwen2.5 / Qwen2 → ``hermes`` tool calls, no reasoning parser.
    - anything else → no parsers (safest; vLLM still serves plain chat)."""
    m = model_id.lower()
    if "qwen3" in m:
        # qwen3_xml is accepted by generic Qwen3 + the AWQ build; the fp8 build's
        # qwen3_coder is only used when the curated table's own model matches.
        return ("qwen3_xml", "qwen3")
    if "qwen2.5" in m or "qwen2" in m:
        return ("hermes", "")
    return ("", "")


def _reconcile_choice(model_id: str, choice: ModelChoice) -> ModelChoice:
    """A :class:`ModelChoice` whose parsers fit the model ACTUALLY being served.

    When ``model_id`` matches the curated choice, the choice is authoritative and
    returned unchanged (its qwen3_coder vs qwen3_xml split is intentional). When
    it differs — an override or heuristic swap — only the tool/reasoning parsers
    are re-derived from the model family; the GPU-specific recipe (extra_args,
    env, context window, recipe_complete) is kept, since those flags are about
    the hardware/NCCL, not the model.
    """
    if model_id == choice.model_id:
        return choice
    tool, reason = _parsers_for_model(model_id)
    return replace(choice, model_id=model_id, tool_parser=tool, reasoning_parser=reason)


def _provision_model_id(choice: ModelChoice, profile: GpuProfile) -> str | None:
    """The HF repo id to actually download + serve. Honors
    ``SHAREDAGENTS_PROVISION_MODEL``, then the choice when it's a real HF id
    (``owner/name``), else the public VRAM-heuristic id — hand-quantized
    local-weight ids (no ``/``) can't be fetched from scratch."""
    pinned = os.environ.get("SHAREDAGENTS_PROVISION_MODEL", "").strip()
    if pinned:
        return pinned
    if "/" in choice.model_id:
        return choice.model_id
    h = _heuristic_choice(profile)
    return h.model_id if h else None


def provision_model(choice: ModelChoice, profile: GpuProfile) -> LocalServer | None:
    """Cold-start: download + launch a local server for this GPU profile.

    Prefers vLLM (the stated preference + the common GPU toolchain), falls back to
    Ollama (trivial to install/pull on a genuinely bare box). Returns a live
    :class:`LocalServer`, or None when no toolchain can serve — the caller
    then falls back to the cloud default. The served model is always a
    *downloadable* public id (see :func:`_provision_model_id`).
    """
    model_id = _provision_model_id(choice, profile)
    if not model_id:
        return None
    # The model actually being served may differ from the table's choice (a
    # pinned override or heuristic swap) — realign the serving recipe's parsers
    # to that model so vLLM doesn't get a parser its tokenizer can't satisfy.
    choice = _reconcile_choice(model_id, choice)
    no_install = os.environ.get("SHAREDAGENTS_NO_VLLM_INSTALL", "").strip() in ("1", "true", "yes")
    launch = _ensure_vllm_launch(model_id, allow_install=not no_install)
    if launch is not None:
        # vLLM is available — commit to it. If the launch fails we return None
        # (→ cloud fallback) rather than silently serving a different/smaller
        # model via Ollama, which would mask the failure and break "best model".
        return _launch_vllm(launch, model_id, choice, profile)
    # vLLM genuinely unavailable (couldn't install) → Ollama is the last resort
    # for a bare box, only when present.
    if shutil.which("ollama"):
        return _launch_ollama(choice)
    return None


def _log_tail(path: str, n: int = 4000) -> str:
    try:
        with open(path, "rb") as fh:
            try:
                fh.seek(-n, os.SEEK_END)
            except OSError:
                fh.seek(0)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _vllm_capability_args(choice: ModelChoice, tp: int) -> list[str]:
    """Serving flags that make the provisioned model actually work, not just
    load — the per-model recipe (mirroring known-good launch configs).

    - Tool-call + reasoning parsers from the choice (qwen3_coder for the fp8
      build, qwen3_xml for the AWQ build, hermes/none for Qwen2.5). Without the
      right tool parser sharedagents's tool calls are dead on arrival.
    - Multi-GPU on consumer cards (no reliable P2P) → disable custom all-reduce
      (paired with NCCL_P2P_DISABLE in :func:`_vllm_env`), which otherwise
      hangs at load on e.g. dual RTX 3090."""
    args: list[str] = []
    if choice.tool_parser:
        args += ["--enable-auto-tool-choice", "--tool-call-parser", choice.tool_parser]
    if choice.reasoning_parser:
        args += ["--reasoning-parser", choice.reasoning_parser]
    # Per-GPU recipe flags (this profile's own config).
    args += list(choice.extra_args)
    # Safety net for an UNLISTED multi-GPU card (heuristic, no curated recipe):
    # disable custom all-reduce. Curated entries are authoritative.
    if tp > 1 and not choice.recipe_complete and "--disable-custom-all-reduce" not in args:
        args += ["--disable-custom-all-reduce"]
    return args


def _vllm_env(tp: int, choice: ModelChoice | None = None) -> dict:
    """Env for the vLLM child: this profile's own env recipe, plus NCCL /
    multiproc fallbacks for an unlisted multi-GPU card."""
    env = dict(os.environ)
    if choice is not None:
        for k, v in choice.env:
            env[k] = v
    if tp > 1 and (choice is None or not choice.env):
        env.setdefault("NCCL_P2P_DISABLE", "1")
        env.setdefault("NCCL_CUMEM_ENABLE", "0")
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


# ---- vLLM user-systemd unit ------------------------------------------
#
# The wizard installs vLLM as a *user-instance* systemd service
# (``sharedagents-vllm.service``) under ``~/.config/systemd/user/`` — no
# sudo, no system unit, no root-owned files. ``systemctl --user`` drives
# the lifecycle and ``loginctl enable-linger`` (best-effort) makes it
# survive reboot.
#
# A reachable user systemd manager is a HARD requirement — there is no
# fallback path. If the user manager isn't reachable, the wizard refuses
# with a clear error rather than silently spawn a detached process the
# OS won't bring back after a reboot. The "install persists a reboot"
# contract is load-bearing.

_VLLM_UNIT_NAME = "sharedagents-vllm.service"

# Env keys we copy from the launching env into the unit. Tightly
# whitelisted — we don't want to bake SSH_AUTH_SOCK, GPG_*, anything
# token-shaped, etc. into a unit file. The Huggingface family in
# particular has both safe knobs (HF_HOME, HF_HUB_CACHE…) AND tokens
# (HF_TOKEN, HUGGINGFACE_HUB_TOKEN, HF_API_TOKEN); a prefix match would
# leak the tokens, so HF / HUGGINGFACE keys are listed exactly. Same
# logic for the rest: cross-cutting essentials by exact name, plus
# narrow prefixes that only match value-style knobs.
_VLLM_ENV_KEEP_EXACT: frozenset[str] = frozenset(
    {
        "PATH",
        "LD_LIBRARY_PATH",
        "TMPDIR",
        # Huggingface — cache paths and behaviour knobs ONLY. Token keys
        # are explicitly excluded; downloads should use ``huggingface-cli
        # login`` (writes the token under ``~/.cache/huggingface/token``,
        # which the unit's process picks up from disk) rather than have
        # the token baked into a world-readable unit file.
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TRANSFORMERS_OFFLINE",
    }
)
_VLLM_ENV_KEEP_PREFIX: tuple[str, ...] = (
    "VLLM_",
    "NCCL_",
    "CUDA_",
    "NVIDIA_",
    "TORCH_",
    "PYTORCH_",
    "OMP_",
    "MKL_",
)
# Hard denylist on names that look secret regardless of how they were
# whitelisted. Belt-and-braces against a future prefix being added
# without thinking through token namespaces.
_VLLM_ENV_DENY_SUBSTR: tuple[str, ...] = ("TOKEN", "SECRET", "KEY", "PASSWORD", "PASSWD")


def _vllm_user_unit_dir() -> Path:
    """``~/.config/systemd/user`` — honours ``XDG_CONFIG_HOME`` like
    systemd itself does, so test fixtures and operators with relocated
    XDG dirs both behave."""
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg) if xdg else (Path.home() / ".config")
    return base / "systemd" / "user"


def _vllm_user_unit_path() -> Path:
    return _vllm_user_unit_dir() / _VLLM_UNIT_NAME


def _user_systemd_available() -> bool:
    """True iff this host has a reachable user-instance systemd
    manager. ``systemctl --user`` exits non-zero (or its `is-system-running`
    prints ``offline``) when no user manager is running — a stripped
    container, a CI runner without logind, etc.
    """
    if not shutil.which("systemctl"):
        return False
    if not Path("/run/systemd/system").is_dir():
        return False
    try:
        cp = subprocess.run(  # noqa: S603, S607
            ["systemctl", "--user", "is-system-running"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    state = (cp.stdout or "").strip()
    if state and state != "offline":
        return True
    # Some user managers (Ubuntu 22.04 in particular) print "offline"
    # while still accepting daemon-reload + start. Use a second probe.
    try:
        cp2 = subprocess.run(  # noqa: S603, S607
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return cp2.returncode == 0


def _render_vllm_unit(cmd: list[str], env: dict[str, str], model_id: str, log_path: str) -> str:
    """Body of ``sharedagents-vllm.service`` for ``cmd`` + ``env``.

    Only PATH, LD_LIBRARY_PATH and known vLLM / CUDA / HF / NCCL keys
    from ``env`` get copied into the unit (see ``_VLLM_ENV_KEEP_*``) so
    we don't bake the launching shell's full env (and any secrets in
    it) into a unit file. ``StandardOutput=append:`` uses an absolute
    path — systemd doesn't expand ``~`` in unit values.
    """
    exec_start = " ".join(shlex.quote(p) for p in cmd)
    env_lines: list[str] = []
    for key in sorted(env):
        if key not in _VLLM_ENV_KEEP_EXACT and not any(key.startswith(p) for p in _VLLM_ENV_KEEP_PREFIX):
            continue
        # Defence in depth: even if a whitelist family ever grows, a
        # token-shaped name MUST NOT land in the unit file.
        upper = key.upper()
        if any(needle in upper for needle in _VLLM_ENV_DENY_SUBSTR):
            continue
        val = env[key]
        if val is None or val == "":
            continue
        sval = str(val)
        if "\n" in sval:  # unit files don't support multi-line values
            continue
        esc = sval.replace("\\", "\\\\").replace('"', '\\"')
        env_lines.append(f'Environment="{key}={esc}"')
    env_block = "\n".join(env_lines)
    if env_block:
        env_block += "\n"
    return (
        f"[Unit]\n"
        f"Description=sharedagents managed vLLM inference server ({model_id})\n"
        f"After=network-online.target\n"
        f"Wants=network-online.target\n"
        f"StartLimitIntervalSec=300\n"
        f"StartLimitBurst=3\n"
        f"\n"
        f"[Service]\n"
        f"Type=simple\n"
        f"ExecStart={exec_start}\n"
        f"Restart=on-failure\n"
        f"RestartSec=15\n"
        # NCCL pins locked memory for multi-GPU (NVLink) IPC buffers; systemd's
        # default LimitMEMLOCK (8 MB) deadlocks NCCL init. Interactive shells get
        # a high PAM limit, which hid this. Harmless for single-GPU.
        f"LimitMEMLOCK=infinity\n"
        f"LimitNOFILE=1048576\n"
        f"{env_block}"
        f"StandardOutput=append:{log_path}\n"
        f"StandardError=append:{log_path}\n"
        f"SyslogIdentifier=sharedagents-vllm\n"
        f"\n"
        f"[Install]\n"
        f"WantedBy=default.target\n"
    )


def _user_systemctl(argv: list[str]) -> subprocess.CompletedProcess:
    """``systemctl --user <argv>`` — no sudo, no special privileges."""
    return subprocess.run(  # noqa: S603, S607
        ["systemctl", "--user", *argv],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _install_vllm_user_unit(cmd: list[str], env: dict[str, str], model_id: str, *, log_path: str) -> dict:
    """Write ``sharedagents-vllm.service`` under the user systemd dir and
    bring it up.

    Returns ``{"ok", "error", "unit_path"}``. ``restart`` is used (not
    ``start``) so a stale unit from a previous failed attempt with a
    different ``--max-model-len`` gets bounced cleanly rather than
    reused. The caller still HTTP-polls the listener — systemd only
    reports "the ExecStart line is running", not "the server is
    accepting traffic".
    """
    unit_path = _vllm_user_unit_path()
    result = {"ok": False, "error": None, "unit_path": str(unit_path)}
    try:
        unit_path.parent.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(_render_vllm_unit(cmd, env, model_id, log_path))
        try:
            unit_path.chmod(0o644)
        except OSError:
            pass
    except OSError as e:
        result["error"] = f"could not write {unit_path}: {e}"
        return result
    cp = _user_systemctl(["daemon-reload"])
    if cp.returncode != 0:
        result["error"] = f"systemctl --user daemon-reload failed: {(cp.stderr or '').strip()}"
        return result
    # ``enable`` is a HARD requirement, not best-effort: without it the
    # unit comes up for this session but isn't pulled in by
    # ``default.target`` on next login/boot — which silently breaks the
    # "install persists a reboot" contract. If it fails we surface the
    # error rather than printing "enabled" and lying to the operator.
    cp = _user_systemctl(["enable", _VLLM_UNIT_NAME])
    if cp.returncode != 0:
        result["error"] = (
            f"systemctl --user enable failed: "
            f"{(cp.stderr or '').strip()} — refusing to start since the unit "
            f"would not survive a reboot."
        )
        return result
    # ``reset-failed`` first so a unit stuck in start-limit-hit from a
    # prior OOM-retry sequence can still be brought back.
    _user_systemctl(["reset-failed", _VLLM_UNIT_NAME])
    cp = _user_systemctl(["restart", _VLLM_UNIT_NAME])
    if cp.returncode != 0:
        result["error"] = f"systemctl --user restart failed: {(cp.stderr or '').strip()}"
        return result
    result["ok"] = True
    return result


def _stop_vllm_user_unit() -> None:
    """Best-effort stop — used between OOM-retries to free the GPU
    before re-launching at a smaller context. Swallows errors: if the
    unit was never installed (e.g. the daemon-reload step failed
    upstream) the stop is a no-op."""
    try:
        _user_systemctl(["stop", _VLLM_UNIT_NAME])
    except (OSError, subprocess.SubprocessError):
        pass


def _uninstall_vllm_user_unit() -> None:
    """Best-effort removal of the one managed vLLM user unit.

    Used when provisioning must refuse after installation (for example when
    linger cannot guarantee cold-boot startup). The scope is deliberately
    limited to :data:`_VLLM_UNIT_NAME`; no process-name matching or broad kill
    is involved.
    """
    try:
        _user_systemctl(["disable", "--now", _VLLM_UNIT_NAME])
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        _vllm_user_unit_path().unlink(missing_ok=True)
    except OSError:
        pass
    try:
        _user_systemctl(["daemon-reload"])
    except (OSError, subprocess.SubprocessError):
        pass


def _try_enable_vllm_linger() -> str:
    """Try ``loginctl enable-linger`` so the vLLM unit starts at boot.

    Returns a one-line human-readable status string the caller can
    surface to the operator. Distros that gate self-linger behind
    admin auth cannot satisfy cold-boot persistence, so the caller refuses
    provisioning. The status string is
    deliberately *not* "run ``sudo loginctl enable-linger``": the
    whole point of this codepath is to avoid sudo, so the hint is an
    admin-instruction note rather than a command an unprivileged
    operator should run.
    """
    user = getpass.getuser()
    if not shutil.which("loginctl"):
        return "loginctl missing — unit will start at next user login but not on cold boot."
    try:
        if (Path("/var/lib/systemd/linger") / user).exists():
            return "starts on boot (linger already on)."
    except OSError:
        pass
    try:
        cp = subprocess.run(  # noqa: S603, S607
            ["loginctl", "enable-linger", user],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "couldn't probe loginctl — unit will start at next user login but may not on cold boot."
    if cp.returncode == 0:
        return "starts on boot (linger enabled)."
    return "starts on next user login (admin needs to enable linger for boot-start; this is a one-time host setup)."


def _launch_vllm(launch: list[str], model_id: str, choice: ModelChoice, profile: GpuProfile) -> LocalServer | None:
    """Launch vLLM for ``model_id`` with ``choice``'s serving recipe.

    The wizard installs vLLM as a **user-instance** systemd service —
    ``sharedagents-vllm.service`` under ``~/.config/systemd/user/`` — and
    drives it with ``systemctl --user`` (no sudo, no system unit).
    ``loginctl enable-linger`` must confirm that it will survive reboot.

    A reachable user systemd manager is a hard requirement: that's
    what gives us "the install persists a reboot." If it isn't
    available (typically a stripped container without logind), the
    wizard reports a clear error rather than silently fall back to a
    detached process that the OS won't bring back after a reboot.

    One retry at a smaller context if the first attempt OOMs or exits
    before serving.
    """
    if not _user_systemd_available():
        _emit(
            "    user-instance systemd is not reachable on this host. "
            "sharedagents provisions vLLM as a `systemctl --user` service so the "
            "install persists across reboot; without a user manager we "
            "can't honour that contract. Start a user manager (e.g. log in "
            "via PAM, or `systemd-run --user --scope true` from a session) "
            "and re-run the wizard."
        )
        return None
    port = _provision_port()
    base = f"http://127.0.0.1:{port}/v1"
    log = os.path.expanduser("~/.sharedagents/vllm-server.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    tp = profile.count or 1
    timeout_s = float(os.environ.get("SHAREDAGENTS_PROVISION_TIMEOUT", "1800"))
    env = _vllm_env(tp, choice)
    # Put the vLLM venv's bin on PATH so its workers find venv console scripts
    # like `ninja` (used to JIT-compile some kernels) — a bare `<venv>/bin/vllm`
    # invocation doesn't activate the venv.
    vbin_dir = os.path.dirname(launch[0]) if (launch and os.sep in launch[0]) else ""
    if vbin_dir and os.path.isdir(vbin_dir):
        env["PATH"] = vbin_dir + os.pathsep + env.get("PATH", "")
    cap_args = _vllm_capability_args(choice, tp)
    # Conservative context on a fresh serve (headroom for KV cache); the
    # detect path reports a model's real ctx, a cold launch stays modest.
    ctx_try = [min(choice.context_window, choice.launch_ctx), 8_192]
    util_try = [0.90, 0.85]
    if _model_cached(model_id):
        _emit(f"  ✓ '{model_id}' already downloaded — launching vLLM (TP={tp}) on {profile.short_name()}…")
    else:
        _emit(
            f"  ↓ provisioning '{model_id}' on {profile.short_name()} via vLLM "
            f"(TP={tp}) — downloading weights, this can take several minutes…"
        )
    served = model_id.split("/")[-1].lower()

    for attempt, (ctx, util) in enumerate(zip(ctx_try, util_try)):
        cmd = (
            list(launch)
            + [
                "--served-model-name",
                served,
                "--port",
                str(port),
                "--tensor-parallel-size",
                str(tp),
                "--max-model-len",
                str(ctx),
                "--gpu-memory-utilization",
                str(util),
                # throughput + KV-reuse flags from a known-good serving config
                "--enable-prefix-caching",
                "--block-size",
                "32",
                "--max-num-batched-tokens",
                "2048",
            ]
            + cap_args
        )
        if attempt == 0:
            _emit(f"    installing {_VLLM_UNIT_NAME} (user-instance systemd) …")
        res = _install_vllm_user_unit(cmd, env, model_id, log_path=log)
        if not res["ok"]:
            _emit(f"    couldn't install vLLM user-unit: {res['error']}")
            return None
        srv = _wait_for_server(base, proc=None, timeout_s=timeout_s, progress_model=model_id)
        if srv:
            linger_msg = _try_enable_vllm_linger()
            if not linger_msg.startswith("starts on boot"):
                _uninstall_vllm_user_unit()
                _emit(f"    vLLM provisioning refused: the enabled user unit cannot survive a cold boot ({linger_msg})")
                return None
            _emit(f"  ✓ vLLM serving '{srv.model_id}' on {base}")
            _emit(f"    {_VLLM_UNIT_NAME} enabled — {linger_msg}")
            return srv
        # Wind down the failed attempt (free GPU memory) before retry.
        _stop_vllm_user_unit()
        oom = "out of memory" in _log_tail(log).lower()
        if attempt + 1 < len(ctx_try):
            _emit(f"    vLLM didn't come up (oom={oom}); retrying with a smaller context ({ctx_try[attempt + 1]})…")
            continue
        cause = "OOM; " if oom else ""
        _emit(f"    vLLM provisioning failed — {cause}see ~/.sharedagents/vllm-server.log")
        return None
    return None


def _launch_ollama(choice: ModelChoice) -> LocalServer | None:
    tag = _ollama_tag_for(choice.model_id)
    _emit(f"  ↓ provisioning '{tag}' via Ollama…")
    try:  # make sure the daemon is up (idempotent)
        subprocess.Popen(
            ["ollama", "serve"], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        subprocess.run(["ollama", "pull", tag], timeout=3600)
    except (OSError, subprocess.SubprocessError):
        return None
    base = "http://127.0.0.1:11434/v1"
    # Only claim success if Ollama is actually serving — never wire a dead
    # endpoint into the config.
    return _wait_for_server(base, timeout_s=120.0, backend="ollama")


def _ollama_tag_for(model_id: str) -> str:
    m = model_id.lower()
    for size in ("72b", "32b", "14b", "7b", "3b"):
        if size in m:
            return f"qwen2.5:{size}"
    return "qwen2.5:7b"


# --------------------------------------------------------------------------
# 5. Orchestrator — the single entry point the wizard calls
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalSetup:
    """Result of auto-setup: a config row + how we got there (for the UX)."""

    profile: GpuProfile
    server: LocalServer
    row: dict
    provisioned: bool  # True if we launched it, False if already running


def _model_row(server: LocalServer, context_window: int) -> dict:
    """Build a ``models[]`` row pointing at a local server (no API key)."""
    alias = server.model_id.split("/")[-1][:40] or "local"
    return {
        "alias": alias,
        "baseUrl": server.base_url,
        "id": server.model_id,
        "backend": server.backend if server.backend in ("vllm", "sglang") else "unknown",
        "contextWindow": context_window,
        "maxTokens": 0,
        "apiKeyEnv": "",
        "headers": {"X-Title": "sharedagents"},
    }


def auto_local_setup(allow_provision: bool = True) -> LocalSetup | None:
    """Find or provision a local model for this box's GPU(s).

    Returns a :class:`LocalSetup` (GPU profile + live server + ready-to-save
    config row) or None when there's no GPU or no way to serve a model — the
    caller then falls back to the cloud default. ``allow_provision=False``
    restricts to the detect-only fast path (no downloads/launches).
    """
    profile = gpu_profile()
    if profile is None:
        return None

    # Fast path: a server is already up. Prefer its LIVE max_model_len (faithful
    # to the box's running config), else the table's ctx for this GPU profile,
    # else a safe default.
    server = probe_local_server()
    if server is not None:
        if server.max_model_len:
            cw = server.max_model_len
        else:
            known = select_model_for_profile(profile)
            cw = known.context_window if known else 32_768
        return LocalSetup(profile, server, _model_row(server, cw), provisioned=False)

    if not allow_provision:
        return None

    # Cold-start: nothing serving → download + launch a model for this GPU.
    choice = select_model_for_profile(profile)
    if choice is None:
        return None
    server = provision_model(choice, profile)
    if server is None:
        return None
    # A fresh serve is launched with this profile's own conservative context
    # (see _launch_vllm); reflect that cap in the saved row, not the model's
    # advertised ctx.
    cw = min(choice.context_window, choice.launch_ctx)
    return LocalSetup(profile, server, _model_row(server, cw), provisioned=True)


def attach_setup(profile: GpuProfile) -> LocalSetup | None:
    """Detect-first only: wire onto a server already serving locally (no
    download/launch). Returns None when nothing is up. Used by the first-run
    wizard to take the instant path before deciding to provision.
    """
    server = probe_local_server()
    if server is None:
        return None
    if server.max_model_len:
        cw = server.max_model_len
    else:
        known = select_model_for_profile(profile)
        cw = known.context_window if known else 32_768
    return LocalSetup(profile, server, _model_row(server, cw), provisioned=False)


# --------------------------------------------------------------------------
# 6. Background-provisioning marker — lets the TUI show a live download bar
# --------------------------------------------------------------------------
#
# The first-run wizard launches provisioning in a detached worker
# (sharedagents_model.provision_worker) and drops a marker here; the TUI polls
# :func:`provision_status` to render a progress bar and gate chat until the
# model is serving. The worker clears the marker on success / marks an error.

_TOTAL_CACHE: dict[str, int | None] = {}


def _marker_path() -> str:
    return os.path.expanduser("~/.sharedagents/provisioning.json")


def write_provision_marker(repo_id: str, base_url: str, served: str) -> None:
    marker = {"repo": repo_id, "base_url": base_url, "served": served, "state": "provisioning", "started": time.time()}
    try:
        os.makedirs(os.path.dirname(_marker_path()), exist_ok=True)
        with open(_marker_path(), "w") as fh:
            json.dump(marker, fh)
    except OSError:
        pass


def read_provision_marker() -> dict | None:
    try:
        with open(_marker_path()) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def clear_provision_marker() -> None:
    try:
        os.remove(_marker_path())
    except OSError:
        pass


def mark_provision_error(msg: str) -> None:
    marker = read_provision_marker() or {}
    marker["state"] = "error"
    marker["error"] = str(msg)[:300]
    try:
        with open(_marker_path(), "w") as fh:
            json.dump(marker, fh)
    except OSError:
        pass


def _vllm_install_active() -> bool:
    """True while sharedagents is installing its managed vLLM venv — the venv dir
    exists but its ``vllm`` entrypoint isn't there yet (``python -m venv`` +
    ``pip install vllm`` in progress). Lets the TUI/monitor show an explicit
    "installing vLLM" phase instead of a blank "starting" during the multi-GB,
    multi-minute one-time vLLM install on a bare box."""
    venv = _managed_vllm_dir()
    try:
        return os.path.isdir(venv) and not os.path.exists(os.path.join(venv, "bin", "vllm"))
    except OSError:
        return False


def provision_status() -> dict | None:
    """Snapshot of an in-flight background provision for the TUI, or None when
    nothing is provisioning. Cheap enough to poll on a 1 s timer (one local
    socket probe + a directory walk; the repo total is fetched once + cached)."""
    marker = read_provision_marker()
    if not marker:
        return None
    repo = marker.get("repo") or ""
    base = marker.get("base_url") or ""
    server_up = bool(base and _server_model(base))
    cache = _hf_cache_path(repo) if repo else None
    downloaded = _dir_bytes(cache) if cache and os.path.isdir(cache) else 0
    if repo not in _TOTAL_CACHE:
        _TOTAL_CACHE[repo] = _hf_total_bytes(repo) if repo else None
    total = _TOTAL_CACHE.get(repo)
    if server_up:
        phase = "ready"
    elif marker.get("state") == "error":
        phase = "error"
    elif repo and _model_cached(repo):
        phase = "loading"  # already fully present → loading
    elif total and downloaded >= total * 0.99:
        phase = "loading"
    elif downloaded > 0:
        phase = "downloading"
    elif _vllm_install_active():
        phase = "installing_vllm"  # one-time vLLM install (bare box)
    else:
        phase = "starting"
    # Human-readable one-liner per phase — what the TUI/monitor surfaces.
    bar = _progress_bar(downloaded, total)
    if phase == "installing_vllm":
        bar = "↓ installing vLLM (one-time, multi-GB — several minutes)…"
    elif phase == "loading":
        bar = "⏳ loading model into GPU (CUDA init + graph capture)…"
    elif phase == "starting":
        bar = "starting server…"
    return {
        "repo": repo,
        "served": marker.get("served") or repo,
        "base_url": base,
        "state": marker.get("state"),
        "error": marker.get("error"),
        "server_up": server_up,
        "downloaded": downloaded,
        "total": total,
        "phase": phase,
        "bar": bar,
        "elapsed": time.time() - float(marker.get("started") or time.time()),
    }
