import json
import os
import time
import subprocess
import requests
import runpod

# ---------------------------------------------------------------------------
# Configuração via variáveis de ambiente — sem defaults modelo-específicos
# ---------------------------------------------------------------------------

MODEL_ID    = os.environ["MODEL_ID"]   # obrigatória; erro explícito se ausente
SERVED_NAME = os.environ.get("SERVED_NAME", MODEL_ID.split("/")[-1].lower())
HF_TOKEN    = os.environ.get("HF_TOKEN", "")

# Capacidade (--max-num-seqs e --max-num-batched-tokens são flags bloqueadas)
MAX_LEN   = os.environ.get("MAX_MODEL_LEN",          "32768")
MAX_SEQS  = os.environ.get("MAX_NUM_SEQS",            "12")
MAX_BATCH = os.environ.get("MAX_NUM_BATCHED_TOKENS",  "65536")
GPU_UTIL  = os.environ.get("GPU_MEMORY_UTIL",         "0.95")
TP_SIZE   = os.environ.get("TENSOR_PARALLEL_SIZE",    "1")

# Quantização / precisão — vazios por padrão; o vLLM decide automaticamente
QUANTIZATION   = os.environ.get("QUANTIZATION",    "")  # ex: compressed-tensors, modelopt, fp8, awq
KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE",  "")  # ex: fp8

# MoE — só inclui a flag se definido
MOE_BACKEND = os.environ.get("MOE_BACKEND", "")          # ex: flashinfer_cutlass

# Parsers — todos opcionais; defina só se o modelo suportar
REASONING_PARSER  = os.environ.get("REASONING_PARSER",  "")  # ex: qwen3, deepseek_r1
TOOL_CALL_PARSER  = os.environ.get("TOOL_CALL_PARSER",  "")  # ex: qwen3_coder, llama3_json
GENERATION_CONFIG = os.environ.get("GENERATION_CONFIG", "")  # ex: vllm

# Speculative decoding — JSON string ou vazio
# ex: '{"method": "mtp", "num_speculative_tokens": 2}'
SPECULATIVE_CONFIG = os.environ.get("SPECULATIVE_CONFIG", "")

# Multimodal — opcionais
MM_PROCESSOR_CACHE  = os.environ.get("MM_PROCESSOR_CACHE_TYPE", "")  # ex: shm
# IMPORTANTE: formato JSON obrigatório, ex: '{"image": 16}'  (não image=16)
LIMIT_MM_PER_PROMPT = os.environ.get("LIMIT_MM_PER_PROMPT", "")

# Escape hatch para qualquer flag extra não coberta acima
EXTRA_ARGS = os.environ.get("EXTRA_VLLM_ARGS", "")

PORT     = 8000
BASE_URL = f"http://localhost:{PORT}"


def _build_vllm_cmd():
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
        "--port", str(PORT),
    ]
    if QUANTIZATION:        cmd += ["--quantization",          QUANTIZATION]
    if KV_CACHE_DTYPE:      cmd += ["--kv-cache-dtype",        KV_CACHE_DTYPE]
    if MOE_BACKEND:         cmd += ["--moe_backend",           MOE_BACKEND]
    if REASONING_PARSER:    cmd += ["--reasoning-parser",      REASONING_PARSER]
    if TOOL_CALL_PARSER:    cmd += ["--enable-auto-tool-choice", "--tool-call-parser", TOOL_CALL_PARSER]
    if GENERATION_CONFIG:   cmd += ["--generation-config",     GENERATION_CONFIG]
    if SPECULATIVE_CONFIG:  cmd += ["--speculative-config",    SPECULATIVE_CONFIG]
    if MM_PROCESSOR_CACHE:  cmd += ["--mm-processor-cache-type", MM_PROCESSOR_CACHE]
    if LIMIT_MM_PER_PROMPT: cmd += ["--limit-mm-per-prompt",   LIMIT_MM_PER_PROMPT]
    if EXTRA_ARGS:          cmd += EXTRA_ARGS.split()
    return cmd


def start_vllm():
    env = {
        **os.environ,
        "HF_TOKEN":               HF_TOKEN,
        "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
    }
    if os.path.isdir("/runpod-volume"):
        cache_dir = "/runpod-volume/hf-cache"
        os.makedirs(cache_dir, exist_ok=True)
        env["HF_HOME"] = cache_dir

    cmd = _build_vllm_cmd()
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
    """Passthrough completo: repassa todos os campos do input para o vLLM,
    sobrescrevendo apenas 'model' e 'stream'. Compatível com qualquer modelo."""
    if "messages" in data:
        url = f"{BASE_URL}/v1/chat/completions"
    else:
        url = f"{BASE_URL}/v1/completions"
    payload = {**data, "model": SERVED_NAME, "stream": stream}
    return url, payload


# Generator function — RunPod SDK detecta streaming via inspect.isgeneratorfunction().
# Para modo sync (stream=False), yield uma única vez; com return_aggregate_stream=True
# o SDK agrega yields em uma lista no output do /run e /status.
def handler(job):
    data = job["input"]

    # Admin: scrape vLLM /metrics (ex: acceptance rate do speculative decoding)
    if data.get("admin_cmd") == "metrics":
        r = requests.get(f"{BASE_URL}/metrics", timeout=10)
        lines = r.text.splitlines()
        relevant = [l for l in lines if not l.startswith("#") and any(
            k in l for k in ["spec_decode", "draft", "accepted", "num_emitted", "acceptance"]
        )]
        yield {"metrics": "\n".join(relevant) or "(no spec_decode metrics found)", "total_lines": len(lines)}
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
