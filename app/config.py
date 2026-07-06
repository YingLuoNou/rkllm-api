from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("OPENAI_API_HOST", "0.0.0.0")
    port: int = int(os.getenv("OPENAI_API_PORT", "8080"))

    model_name: str = os.getenv("OPENAI_MODEL_NAME", "qwen2.5-3b-rk3588")
    api_key: str = os.getenv("OPENAI_API_KEY", "EMPTY")
    max_input_chars: int = int(os.getenv("OPENAI_MAX_INPUT_CHARS", "12000"))
    request_timeout_sec: int = int(os.getenv("OPENAI_REQUEST_TIMEOUT_SEC", "60"))

    rkllm_library_path: str = os.getenv("RKLLM_LIBRARY_PATH", "../rkllm-ros2/lib/librkllmrt.so")
    rkllm_model_path: str = os.getenv("RKLLM_MODEL_PATH", "/data/llm/model/Qwen2.5-3B-w8a8.rkllm")
    rkllm_max_new_tokens: int = int(os.getenv("RKLLM_MAX_NEW_TOKENS", "512"))
    rkllm_max_context_len: int = int(os.getenv("RKLLM_MAX_CONTEXT_LEN", "2048"))
    rkllm_top_k: int = int(os.getenv("RKLLM_TOP_K", "1"))
    rkllm_top_p: float = float(os.getenv("RKLLM_TOP_P", "0.95"))
    rkllm_temperature: float = float(os.getenv("RKLLM_TEMPERATURE", "0.8"))
    rkllm_repeat_penalty: float = float(os.getenv("RKLLM_REPEAT_PENALTY", "1.1"))


settings = Settings()
