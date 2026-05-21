FROM vllm/vllm-openai:v0.21.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VLLM_ATTENTION_BACKEND=FLASHINFER \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NCCL_P2P_DISABLE=1 \
    OMP_NUM_THREADS=1

RUN pip install --no-cache-dir runpod==1.7.7 requests==2.32.3

WORKDIR /worker
COPY handler.py /worker/handler.py

ENTRYPOINT []
CMD ["python3", "-u", "/worker/handler.py"]
