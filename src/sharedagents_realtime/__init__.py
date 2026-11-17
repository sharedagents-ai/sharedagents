"""sharedagents_realtime — bidirectional realtime voice over an OpenAI-Realtime-style
WebSocket, backed by a pluggable STT → LLM → TTS cascade.

Spec: specs/realtime-voice.md. v1 runs as a standalone process
(``python -m sharedagents_realtime``) so the production gateway is untouched.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
