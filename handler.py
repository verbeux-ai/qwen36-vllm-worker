#!/usr/bin/env python3
"""
vllm-worker dual-mode handler.

Modos:
  (default)        RunPod Serverless: sobe vLLM em background, expõe handler via runpod SDK
  --standalone     vLLM em foreground, sem SDK (uso local / vast.ai)
  --vast-bg        vLLM em foreground (replacing process), entrypoint.sh roda PyWorker em paralelo

Todas as env vars opcionais. MODEL_ID é obrigatória.
"""
import json
import os
import sys
import time
import subprocess

# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------
MODEL_ID    = os.environ["MODEL_ID"]
SERVED_NAME = os.environ.get("SERVED_NAME", MODEL_ID.split("/")[-1].lower())
HF_TOKEN    = os.environ.get("HF_TOKEN", "")

MAX_LEN   = os.environ.get("MAX_MODEL_LEN",          "32768")
MAX_SEQS  = os.environ.get("MAX_NUM_SEQS",           "12")
MAX_BATCH = os.environ.get("MAX_NUM_BATCHED_TOKENS", "65536")
GPU_UTIL  = os.environ.get("GPU_MEMORY_UTIL",        "0.95")
TP_SIZE   = os.environ.get("TENSOR_PARALLEL_SIZE",   "1")

QUANTIZATION       = os.environ.get("QUANTIZATION",       "")
KV_CACHE_DTYPE     = os.environ.get("KV_CACHE_DTYPE",     "")
MOE_BACKEND        = os.environ.get("MOE_BACKEND",        "")
REASONING_PARSER   = os.environ.get("REASONING_PARSER",   "")
TOOL_CALL_PARSER   = os.environ.get("TOOL_CALL_PARSER",   "")
GENERATION_CONFIG  = os.environ.get("GENERATION_CONFIG",  "")
SPECULATIVE_CONFIG = os.environ.get("SPECULATIVE_CONFIG", "")
MM_PROCESSOR_CACHE = os.environ.get("MM_PROCESSOR_CACHE_TYPE", "")
LIMIT_MM_PER_PROMPT = os.environ.get("LIMIT_MM_PER_PROMPT", "")
EXTRA_ARGS = os.environ.get("EXTRA_VLLM_ARGS", "")

MODEL_PORT = int(os.environ.get("MODEL_PORT", "8000"))
BASE_URL   = f"http://127.0.0.1:{MODEL_PORT}"


def resolve_hf_home():
    """
    Em prioridade: /models (pre-bake da imagem) > /runpod-volume (RunPod volume)
    > /workspace (vast.ai persistent) > default.
    """
    if os.path.isdir("/models") and os.path.exists("/models/PREBAKED"):
        return "/models"
    if os.path.isdir("/runpod-volume"):
        d = "/runpod-volume/hf-cache"
        os.makedirs(d, exist_ok=True)
        return d
    if os.path.isdir("/workspace"):
        d = "/workspace/hf-cache"
        os.makedirs(d, exist_ok=True)
        return d
    return os.environ.get("HF_HOME", "/root/.cache/huggingface")


def build_cmd():
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  MODEL_ID,
        "--served-model-name",      SERVED_NAME,
        "--max-model-len",          MAX_LEN,
        "--max-num-seqs",           MAX_SEQS,
        "--max-num-batched-tokens", MAX_BATCH,
        "--gpu-memory-utilization", GPU_UTIL,
        "--tensor-parallel-size",   TP_SIZE,
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--trust-remote-code",
        "--host", "0.0.0.0",
        "--port", str(MODEL_PORT),
    ]
    if QUANTIZATION:        cmd += ["--quantization", QUANTIZATION]
    if KV_CACHE_DTYPE:      cmd += ["--kv-cache-dtype", KV_CACHE_DTYPE]
    if MOE_BACKEND:         cmd += ["--moe_backend", MOE_BACKEND]
    if REASONING_PARSER:    cmd += ["--reasoning-parser", REASONING_PARSER]
    if TOOL_CALL_PARSER:    cmd += ["--enable-auto-tool-choice", "--tool-call-parser", TOOL_CALL_PARSER]
    if GENERATION_CONFIG:   cmd += ["--generation-config", GENERATION_CONFIG]
    if SPECULATIVE_CONFIG:  cmd += ["--speculative-config", SPECULATIVE_CONFIG]
    if MM_PROCESSOR_CACHE:  cmd += ["--mm-processor-cache-type", MM_PROCESSOR_CACHE]
    if LIMIT_MM_PER_PROMPT: cmd += ["--limit-mm-per-prompt", LIMIT_MM_PER_PROMPT]
    if EXTRA_ARGS:          cmd += EXTRA_ARGS.split()
    return cmd


def build_env():
    env = {**os.environ}
    env["HF_HOME"] = resolve_hf_home()
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    return env


def start_vllm_background():
    cmd, env = build_cmd(), build_env()
    print(f"[worker] HF_HOME={env['HF_HOME']}", flush=True)
    print(f"[worker] start: {' '.join(cmd)}", flush=True)
    subprocess.Popen(cmd, env=env)

    import requests
    for i in range(1200):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"[worker] vLLM pronto em {i}s", flush=True)
                return
        except Exception:
            pass
        if i % 30 == 0:
            print(f"[worker] aguardando vLLM... {i}s", flush=True)
        time.sleep(1)
    raise RuntimeError("vLLM não subiu em 20 minutos")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def exec_vllm_foreground():
    """Substitui o processo atual por vLLM (sem RunPod SDK)."""
    cmd, env = build_cmd(), build_env()
    print(f"[worker] HF_HOME={env['HF_HOME']}", flush=True)
    print(f"[worker] exec: {' '.join(cmd)}", flush=True)
    os.execvpe(cmd[0], cmd, env)


def run_runpod_serverless():
    import requests
    import runpod

    def _build_payload(data, stream):
        # Passthrough completo — apenas força model=SERVED_NAME
        if "messages" in data:
            url = f"{BASE_URL}/v1/chat/completions"
        else:
            url = f"{BASE_URL}/v1/completions"
        payload = {**data, "model": SERVED_NAME, "stream": stream}
        return url, payload

    def handler(job):
        data = job["input"]

        if data.get("admin_cmd") == "metrics":
            r = requests.get(f"{BASE_URL}/metrics", timeout=10)
            lines = r.text.splitlines()
            relevant = [l for l in lines if not l.startswith("#") and any(
                k in l for k in ["spec_decode", "draft", "accepted", "num_emitted", "acceptance"]
            )]
            yield {"metrics": "\n".join(relevant) or "(no spec_decode metrics found)",
                   "total_lines": len(lines)}
            return

        want_stream = data.get("stream", False)
        url, payload = _build_payload(data, want_stream)

        if want_stream:
            with requests.post(url, json=payload, stream=True, timeout=600) as r:
                for raw in r.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                    if line == "[DONE]":
                        return
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"raw": line}
            return

        r = requests.post(url, json=payload, timeout=600)
        yield r.json()

    start_vllm_background()
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--standalone" in args or "--vast-bg" in args:
        exec_vllm_foreground()
    else:
        run_runpod_serverless()
