from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    Union,
    byref,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_int8,
    c_int32,
    c_uint8,
    c_uint32,
    c_void_p,
    cdll,
)


@dataclass(frozen=True)
class LLMDriverConfig:
    library_path: str = os.getenv("RKLLM_LIBRARY_PATH", "../rkllm-ros2/lib/librkllmrt.so")
    model_path: str = os.getenv("RKLLM_MODEL_PATH", "/data/llm/model/Qwen2.5-3B-w8a8.rkllm")
    model_name: str = os.getenv("OPENAI_MODEL_NAME", "qwen2.5-3b-rk3588")
    max_new_tokens: int = int(os.getenv("RKLLM_MAX_NEW_TOKENS", "512"))
    max_context_len: int = int(os.getenv("RKLLM_MAX_CONTEXT_LEN", "2048"))
    top_k: int = int(os.getenv("RKLLM_TOP_K", "1"))
    top_p: float = float(os.getenv("RKLLM_TOP_P", "0.95"))
    temperature: float = float(os.getenv("RKLLM_TEMPERATURE", "0.8"))
    repeat_penalty: float = float(os.getenv("RKLLM_REPEAT_PENALTY", "1.1"))
    request_timeout_sec: int = int(os.getenv("OPENAI_REQUEST_TIMEOUT_SEC", "60"))


class _RKLLMExtendParam(Structure):
    _fields_ = [
        ("base_domain_id", c_int32),
        ("embed_flash", c_int8),
        ("enabled_cpus_num", c_int8),
        ("enabled_cpus_mask", c_uint32),
        ("n_batch", c_uint8),
        ("use_cross_attn", c_int8),
        ("reserved", c_uint8 * 104),
    ]


class _RKLLMParam(Structure):
    _fields_ = [
        ("model_path", c_char_p),
        ("max_context_len", c_int32),
        ("max_new_tokens", c_int32),
        ("top_k", c_int32),
        ("n_keep", c_int32),
        ("top_p", c_float),
        ("temperature", c_float),
        ("repeat_penalty", c_float),
        ("frequency_penalty", c_float),
        ("presence_penalty", c_float),
        ("mirostat", c_int32),
        ("mirostat_tau", c_float),
        ("mirostat_eta", c_float),
        ("skip_special_token", c_bool),
        ("is_async", c_bool),
        ("img_start", c_char_p),
        ("img_end", c_char_p),
        ("img_content", c_char_p),
        ("extend_param", _RKLLMExtendParam),
    ]


class _RKLLMInputUnion(Union):
    _fields_ = [("prompt_input", c_char_p)]


class _RKLLMInput(Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("role", c_char_p),
        ("enable_thinking", c_bool),
        ("input_type", c_int),
        ("u", _RKLLMInputUnion),
    ]


class _RKLLMInferParam(Structure):
    _fields_ = [
        ("mode", c_int),
        ("lora_params", c_void_p),
        ("prompt_cache_params", c_void_p),
        ("keep_history", c_int),
    ]


class _RKLLMResult(Structure):
    _fields_ = [("text", c_char_p)]


class LLMDriver:
    """直接驱动 RKLLM NPU（不走 ROS，不走上游转发）。"""

    def __init__(self, config: LLMDriverConfig) -> None:
        self._cfg = config
        self._lib = cdll.LoadLibrary(self._cfg.library_path)
        self._handle = c_void_p()

        self._cv = threading.Condition()
        self._busy = False
        self._done = False
        self._error: str | None = None
        self._chunks: list[str] = []

        self._init_native()

    def _init_native(self) -> None:
        callback_type = CFUNCTYPE(c_int, POINTER(_RKLLMResult), c_void_p, c_int)

        def _callback(result_ptr: POINTER(_RKLLMResult), _userdata: c_void_p, state: int) -> int:
            with self._cv:
                if state == 0:  # RKLLM_RUN_NORMAL
                    if result_ptr and result_ptr.contents.text:
                        text = result_ptr.contents.text.decode("utf-8", errors="ignore")
                        if text:
                            self._chunks.append(text)
                elif state == 2:  # RKLLM_RUN_FINISH
                    self._done = True
                    self._cv.notify_all()
                elif state == 3:  # RKLLM_RUN_ERROR
                    self._error = "rkllm_run returned error state"
                    self._done = True
                    self._cv.notify_all()
            return 0

        self._callback_ref = callback_type(_callback)

        self._lib.rkllm_createDefaultParam.restype = _RKLLMParam
        self._lib.rkllm_init.argtypes = [POINTER(c_void_p), POINTER(_RKLLMParam), callback_type]
        self._lib.rkllm_init.restype = c_int
        self._lib.rkllm_run.argtypes = [c_void_p, POINTER(_RKLLMInput), POINTER(_RKLLMInferParam), c_void_p]
        self._lib.rkllm_run.restype = c_int
        self._lib.rkllm_destroy.argtypes = [c_void_p]
        self._lib.rkllm_destroy.restype = c_int
        if hasattr(self._lib, "rkllm_abort"):
            self._lib.rkllm_abort.argtypes = [c_void_p]
            self._lib.rkllm_abort.restype = c_int

        param = self._lib.rkllm_createDefaultParam()
        self._model_path_bytes = self._cfg.model_path.encode("utf-8")
        param.model_path = c_char_p(self._model_path_bytes)
        param.max_new_tokens = self._cfg.max_new_tokens
        param.max_context_len = self._cfg.max_context_len
        param.top_k = self._cfg.top_k
        param.top_p = self._cfg.top_p
        param.temperature = self._cfg.temperature
        param.repeat_penalty = self._cfg.repeat_penalty
        param.skip_special_token = True
        param.extend_param.base_domain_id = 0
        param.extend_param.embed_flash = 1

        ret = self._lib.rkllm_init(byref(self._handle), byref(param), self._callback_ref)
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed, ret={ret}, model_path={self._cfg.model_path}")

    @property
    def model_name(self) -> str:
        return self._cfg.model_name

    def close(self) -> None:
        if self._handle:
            self._lib.rkllm_destroy(self._handle)
            self._handle = c_void_p()

    def infer(self, prompt: str, max_tokens: int, temperature: float, timeout_sec: int | None = None) -> str:
        _ = max_tokens  # 当前 RKLLM Python 层按初始化参数生效
        _ = temperature

        if not prompt.strip():
            return ""

        with self._cv:
            if self._busy:
                raise RuntimeError("NPU busy: previous request still running")
            self._busy = True
            self._done = False
            self._error = None
            self._chunks = []

        prompt_bytes = prompt.encode("utf-8")
        role_bytes = b"user"

        llm_input = _RKLLMInput()
        llm_input.role = c_char_p(role_bytes)
        llm_input.enable_thinking = False
        llm_input.input_type = 0  # RKLLM_INPUT_PROMPT
        llm_input.prompt_input = c_char_p(prompt_bytes)

        infer_param = _RKLLMInferParam()
        infer_param.mode = 0  # RKLLM_INFER_GENERATE
        infer_param.lora_params = None
        infer_param.prompt_cache_params = None
        infer_param.keep_history = 0

        ret = self._lib.rkllm_run(self._handle, byref(llm_input), byref(infer_param), None)
        if ret != 0:
            with self._cv:
                self._busy = False
            raise RuntimeError(f"rkllm_run failed, ret={ret}")

        deadline = time.monotonic() + float(timeout_sec or self._cfg.request_timeout_sec)
        with self._cv:
            while not self._done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if hasattr(self._lib, "rkllm_abort"):
                        try:
                            self._lib.rkllm_abort(self._handle)
                        except Exception:
                            pass
                    self._busy = False
                    raise TimeoutError("rkllm inference timeout")
                self._cv.wait(timeout=remaining)

            self._busy = False
            if self._error:
                raise RuntimeError(self._error)
            return "".join(self._chunks).strip()
