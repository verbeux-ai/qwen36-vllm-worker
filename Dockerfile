FROM vllm/vllm-openai:v0.21.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VLLM_ATTENTION_BACKEND=FLASHINFER \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NCCL_P2P_DISABLE=1 \
    OMP_NUM_THREADS=1 \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates git git-lfs && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir runpod==1.7.7 requests==2.32.3 hf_transfer uv

# Pre-clone PyWorker (evita git clone runtime no cold-start em vast.ai)
RUN git clone --depth 1 https://github.com/vast-ai/pyworker.git /opt/vast-pyworker && \
    mkdir -p /var/log/portal

COPY prebake.py /tmp/prebake.py

ARG PREBAKE_MODEL=sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP
ARG HF_TOKEN=""

# Layer 1: metadados (config, tokenizer, *.py) — pequeno, push rápido.
RUN if [ -n "$PREBAKE_MODEL" ]; then \
      HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      python3 -u /tmp/prebake.py "$PREBAKE_MODEL" '*.json,*.txt,tokenizer*,*.py,*.md,chat_template*' \
      && echo "$PREBAKE_MODEL" > /models/PREBAKED_META; \
    fi

# Layer 2: weights (.safetensors) — grande mas em layer separado pra paralelizar push GHCR.
RUN if [ -n "$PREBAKE_MODEL" ] && [ -f /models/PREBAKED_META ]; then \
      HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      python3 -u /tmp/prebake.py "$PREBAKE_MODEL" '*.safetensors,*.safetensors.index.json' \
      && echo "$PREBAKE_MODEL" > /models/PREBAKED \
      && du -sh /models; \
    fi

WORKDIR /worker
COPY handler.py /worker/handler.py
COPY entrypoint.sh /worker/entrypoint.sh
RUN chmod +x /worker/entrypoint.sh /worker/handler.py

ENTRYPOINT ["/worker/entrypoint.sh"]
