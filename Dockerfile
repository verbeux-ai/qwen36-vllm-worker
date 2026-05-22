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
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir runpod==1.7.7 requests==2.32.3 hf_transfer

# Pre-bake do modelo na imagem (~25-30GB extra).
# Trade-off escolhido: cold-start em vast.ai cai dramaticamente quando o host já tem a imagem
# em cache. Se o build não tiver acesso (modelo gated sem token), cai pra download em runtime.
ARG PREBAKE_MODEL=AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4
ARG HF_TOKEN=""
RUN if [ -n "$PREBAKE_MODEL" ]; then \
      echo "[prebake] downloading ${PREBAKE_MODEL} into HF_HOME=/models" && \
      HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      python3 -c "from huggingface_hub import snapshot_download; \
                  snapshot_download('${PREBAKE_MODEL}', max_workers=8, \
                  allow_patterns=['*.safetensors','*.json','*.txt','tokenizer*','*.py','*.md'])" \
      && echo "${PREBAKE_MODEL}" > /models/PREBAKED \
      && du -sh /models 2>/dev/null \
      || echo "[prebake] WARN: falhou, modelo será baixado em runtime"; \
    fi

WORKDIR /worker
COPY handler.py /worker/handler.py
COPY entrypoint.sh /worker/entrypoint.sh
RUN chmod +x /worker/entrypoint.sh /worker/handler.py

ENTRYPOINT ["/worker/entrypoint.sh"]
