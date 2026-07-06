# web-ai-api（本地 RK3588 NPU 直驱 + OpenAI 兼容层）

这是一个独立的 AI 驱动服务：**直接驱动 RKLLM 动态库进行本地 NPU 推理**，
不走 `rkllm-ros2` 的 ROS 接口，也不走上游转发。

提供：

- `GET /v1/models`
- `POST /v1/chat/completions`（支持 `stream=true/false`）
- `GET /driver`（AI 驱动页）

与现有后端 `ai-fitness-web-client/services/llm_service.py` 可直接对接。

## 1) 安装

```bash
cd web-ai-api
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> RK3588 上确保本地上游模型服务已启动并开放 OpenAI 兼容接口。
> RK3588 上需保证 `librkllmrt.so` 与 `.rkllm` 模型文件可访问。

## 2) 配置文件

服务配置统一读取 `config/settings.yaml`，主要项：

- `api.host` / `api.port`
- `model.name` / `model.api_key`
- `model.max_input_chars` / `model.request_timeout_sec`
- `rkllm.library_path` / `rkllm.model_path`
- `rkllm.max_new_tokens` / `rkllm.max_context_len`
- `rkllm.top_k` / `rkllm.top_p` / `rkllm.temperature` / `rkllm.repeat_penalty`

## 3) 启动

```bash
python main.py
```

## 4) OpenAI 兼容请求示例

```bash
curl http://127.0.0.1:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"qwen2.5-3b-rk3588",
    "messages":[{"role":"user","content":"给我一句深蹲纠错建议"}],
    "max_tokens":128,
    "temperature":0.7
  }'
```

## 5) 注意事项

- 驱动页地址：`http://<host>:8081/driver`
- `stream=true` 为协议兼容流式输出（按片段回放最终结果）。
- 该服务当前为单实例串行推理；并发请求时，后到请求会收到 `NPU busy`。
