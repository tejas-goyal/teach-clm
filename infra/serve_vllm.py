"""Qwen3.5-9B behind an OpenAI-compatible vLLM server on Modal.

    uv run modal deploy infra/serve_vllm.py      # persistent; scales to zero when idle
    uv run modal run infra/serve_vllm.py         # deploy-free check: one tool-call round trip

Requests need `Authorization: Bearer $TEACHCLM_API_KEY` (Modal secret `teachclm-api-key`, local `.env`).
The context window (65536) holds the paper's settings: 32768 budget + 8192 output on
ContextBench, 28672 + 4096 on BrowseComp-Plus.
"""

import os
import subprocess

import modal

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SERVED_NAME = "qwen35-9b"
MAX_MODEL_LEN = 65536
PORT = 8000
MINUTES = 60

app = modal.App("teachclm-vllm")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.33.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache = modal.Volume.from_name("teachclm-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("teachclm-vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=os.environ.get("TEACHCLM_GPU", "H100"),
    max_containers=1,                 # one GPU at most; raise deliberately for big evals
    scaledown_window=5 * MINUTES,     # idle GPU shuts down after 5 minutes
    timeout=60 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("teachclm-api-key")],
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=20 * MINUTES)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--revision", REVISION,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--api-key", os.environ["TEACHCLM_API_KEY"],
        # Flags from the CLM harness README and the Qwen3.5 model card.
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        "--language-model-only",
    ]
    subprocess.Popen(cmd)


@app.local_entrypoint()
def check():
    """M1 exit check: a bash tool call round-trips through the server."""
    import json
    import time
    import urllib.request

    from dotenv import load_dotenv

    load_dotenv()
    url = serve.get_web_url()
    headers = {
        "Authorization": f"Bearer {os.environ['TEACHCLM_API_KEY']}",
        "Content-Type": "application/json",
    }
    tools = [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }]

    def chat(messages):
        body = json.dumps({
            "model": SERVED_NAME, "messages": messages, "tools": tools,
            "temperature": 0.7, "top_p": 0.95, "max_tokens": 1024,
        }).encode()
        req = urllib.request.Request(f"{url}/v1/chat/completions", body, headers)
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=25 * MINUTES) as resp:
            out = json.load(resp)
        return out["choices"][0]["message"], out["usage"], time.time() - t0

    messages = [{"role": "user", "content": "Use the bash tool to print the current directory."}]
    msg, usage, dt = chat(messages)
    print(f"first call: {dt:.1f}s (includes cold start) usage={usage}")
    calls = msg.get("tool_calls") or []
    assert calls, f"no tool call returned: {msg}"
    args = json.loads(calls[0]["function"]["arguments"])
    print("tool call:", calls[0]["function"]["name"], args)
    assert calls[0]["function"]["name"] == "bash" and "command" in args

    messages += [
        {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls},
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": "/app/workdir-7731"},
    ]
    msg, usage, dt = chat(messages)
    print(f"second call: {dt:.1f}s usage={usage}")
    print("answer:", (msg.get("content") or "").strip()[:300])
    assert "workdir-7731" in (msg.get("content") or ""), "tool result was not used"
    print("PASS: tool call round trip")
