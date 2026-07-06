from __future__ import annotations

import json
import time
import uuid
from typing import Iterator

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.llm_driver import LLMDriver, LLMDriverConfig
from app.schemas import ChatCompletionRequest, ModelCard, ModelListResponse

app = FastAPI(title="Web AI OpenAI-Compatible API", version="0.2.0")
templates = Jinja2Templates(directory="app/templates")

_driver = LLMDriver(
    LLMDriverConfig(
        library_path=settings.rkllm_library_path,
        model_path=settings.rkllm_model_path,
        model_name=settings.model_name,
        max_new_tokens=settings.rkllm_max_new_tokens,
        max_context_len=settings.rkllm_max_context_len,
        top_k=settings.rkllm_top_k,
        top_p=settings.rkllm_top_p,
        temperature=settings.rkllm_temperature,
        repeat_penalty=settings.rkllm_repeat_penalty,
        request_timeout_sec=settings.request_timeout_sec,
    )
)


def _check_api_key(auth_header: str | None) -> None:
    if settings.api_key in ("", "EMPTY"):
        return
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    token = auth_header.replace("Bearer ", "", 1).strip()
    if token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _build_prompt(messages: list) -> str:
    chunks: list[str] = []
    for m in messages:
        role = m.role.upper()
        content = (m.content or "").strip()
        if not content:
            continue
        chunks.append(f"[{role}]\n{content}")

    prompt = "\n\n".join(chunks)
    if len(prompt) > settings.max_input_chars:
        prompt = prompt[-settings.max_input_chars :]
    return prompt


def _to_openai_response(model_name: str, content: str) -> dict:
    now = int(time.time())
    text = content or ""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _stream_chunks(model_name: str, full_text: str) -> Iterator[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    first = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

    step = 24
    for i in range(0, len(full_text), step):
        part = full_text[i : i + step]
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    done = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/healthz")
def healthz() -> dict:
    return {
        "ok": True,
        "model": _driver.model_name,
        "engine": "rkllm-local-npu",
        "rkllm_library_path": settings.rkllm_library_path,
        "rkllm_model_path": settings.rkllm_model_path,
    }


@app.get("/driver")
def driver_page(request: Request):
    return templates.TemplateResponse("driver.html", {"request": request})


@app.get("/v1/models", response_model=ModelListResponse)
def list_models(authorization: str | None = Header(default=None)) -> ModelListResponse:
    _check_api_key(authorization)
    return ModelListResponse(data=[ModelCard(id=_driver.model_name, created=int(time.time()))])


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, authorization: str | None = Header(default=None)):
    _check_api_key(authorization)

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    model_name = req.model or settings.model_name
    prompt = _build_prompt(req.messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")

    try:
        output = _driver.infer(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            timeout_sec=settings.request_timeout_sec,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="LLM timeout") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"LLM driver error: {exc}") from exc

    if req.stream:
        return StreamingResponse(
            _stream_chunks(model_name=model_name, full_text=output),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return JSONResponse(_to_openai_response(model_name=model_name, content=output))


@app.on_event("shutdown")
def _shutdown_driver() -> None:
    _driver.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
