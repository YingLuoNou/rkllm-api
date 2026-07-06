from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "settings.yaml"


@dataclass(frozen=True)
class Settings:
    host: str
    port: int

    model_name: str
    api_key: str
    max_input_chars: int
    request_timeout_sec: int

    rkllm_library_path: str
    rkllm_model_path: str
    rkllm_max_new_tokens: int
    rkllm_max_context_len: int
    rkllm_top_k: int
    rkllm_top_p: float
    rkllm_temperature: float
    rkllm_repeat_penalty: float

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH) -> "Settings":
        data: dict = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        api_cfg = data.get("api", {})
        model_cfg = data.get("model", {})
        rkllm_cfg = data.get("rkllm", {})

        return cls(
            host=str(api_cfg.get("host", "0.0.0.0")),
            port=int(api_cfg.get("port", 8081)),
            model_name=str(model_cfg.get("name", "qwen2.5-3b-rk3588")),
            api_key=str(model_cfg.get("api_key", "EMPTY")),
            max_input_chars=int(model_cfg.get("max_input_chars", 12000)),
            request_timeout_sec=int(model_cfg.get("request_timeout_sec", 60)),
            rkllm_library_path=str(rkllm_cfg.get("library_path", "../rkllm-ros2/lib/librkllmrt.so")),
            rkllm_model_path=str(rkllm_cfg.get("model_path", "/data/llm/model/Qwen2.5-3B-w8a8.rkllm")),
            rkllm_max_new_tokens=int(rkllm_cfg.get("max_new_tokens", 512)),
            rkllm_max_context_len=int(rkllm_cfg.get("max_context_len", 2048)),
            rkllm_top_k=int(rkllm_cfg.get("top_k", 1)),
            rkllm_top_p=float(rkllm_cfg.get("top_p", 0.95)),
            rkllm_temperature=float(rkllm_cfg.get("temperature", 0.8)),
            rkllm_repeat_penalty=float(rkllm_cfg.get("repeat_penalty", 1.1)),
        )


settings = Settings.from_yaml()
