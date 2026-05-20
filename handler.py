import json
import os
import time
import subprocess
import requests
import runpod

MODEL_ID    = os.environ.get("MODEL_ID",       "AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP")
SERVED_NAME = os.environ.get("SERVED_NAME",    "qwen3.6-27b")
HF_TOKEN    = os.environ.get("HF_TOKEN",       "")
MAX_LEN     = os.environ.get("MAX_MODEL_LEN",  "262144")
MAX_SEQS    = os.environ.get("MAX_NUM_SEQS",   "12")
MAX_BATCH   = os.environ.get("MAX_NUM_BATCHED_TOKENS", "65536")
GPU_UTIL    = os.environ.get("GPU_MEMORY_UTIL", "0.94")
PORT        = 8000
BASE_URL    = f"http://localhost:{PORT}"


def start_vllm():
    env = {
        **os.environ,
        "HF_TOKEN":               HF_TOKEN,
        "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
        "VLLM_NVFP4_GEMM_BACKEND":      "flashinfer-cutlass",
        "VLLM_USE_FLASHINFER_SAMPLER":  "1",
        "PYTORCH_CUDA_ALLOC_CONF":      "expandable_segments:True",
    }

    if os.path.isdir("/runpod-volume"):
        cache_dir = "/runpod-volume/hf-cache"
        os.makedirs(cache_dir, exist_ok=True)
        env["HF_HOME"] = cache_dir

    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                       MODEL_ID,
        "--served-model-name",           SERVED_NAME,
        "--quantization",                "modelopt",
        "--kv-cache-dtype",              "fp8_e5m2",
        "--max-model-len",               MAX_LEN,
        "--max-num-seqs",                MAX_SEQS,
        "--max-num-batched-tokens",      MAX_BATCH,
        "--gpu-memory-utilization",      GPU_UTIL,
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--reasoning-parser",            "qwen3",
        "--tool-call-parser",            "qwen3_coder",
        "--enable-auto-tool-choice",
        "--speculative-config",          '{"method":"mtp","num_speculative_tokens":3}',
        "--trust-remote-code",
        "--host",                        "0.0.0.0",
        "--port",                        str(PORT),
    ]

    print(f"[worker] start: {' '.join(cmd)}", flush=True)
    subprocess.Popen(cmd, env=env)

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


def _build_payload(data, stream):
    if "messages" in data:
        payload = {
            "model":       SERVED_NAME,
            "messages":    data["messages"],
            "max_tokens":  data.get("max_tokens",  2048),
            "temperature": data.get("temperature", 0.7),
            "top_p":       data.get("top_p",       0.9),
            "stream":      stream,
        }
        for k in ("top_k", "repetition_penalty", "stop", "tools", "tool_choice",
                  "response_format", "seed", "frequency_penalty", "presence_penalty",
                  "stream_options"):
            if k in data:
                payload[k] = data[k]
        url = f"{BASE_URL}/v1/chat/completions"
    else:
        payload = {
            "model":       SERVED_NAME,
            "prompt":      data.get("prompt", ""),
            "max_tokens":  data.get("max_tokens", 512),
            "temperature": data.get("temperature", 0.7),
            "stream":      stream,
        }
        url = f"{BASE_URL}/v1/completions"
    return url, payload


# Generator function — RunPod SDK detecta streaming via inspect.isgeneratorfunction().
# Para modo sync (stream=False), yield uma única vez; com return_aggregate_stream=True
# o SDK agrega yields em uma lista no output do /run e /status.
def handler(job):
    data = job["input"]

    # Admin: scrape vLLM /metrics pra ver acceptance rate do MTP/spec decoding
    if data.get("_admin") == "metrics":
        r = requests.get(f"{BASE_URL}/metrics", timeout=10)
        lines = r.text.splitlines()
        relevant = [l for l in lines if not l.startswith("#") and any(
            k in l for k in ["spec_decode", "draft", "accepted", "num_emitted", "acceptance"]
        )]
        yield {"metrics": "\n".join(relevant) or "(no spec_decode metrics found)"}
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


start_vllm()
runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
