FROM vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VLLM_USE_FLASHINFER_SAMPLER=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    NCCL_CUMEM_ENABLE=0 \
    NCCL_P2P_DISABLE=1 \
    OMP_NUM_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

RUN pip install --no-cache-dir runpod==1.7.7 requests==2.32.3

WORKDIR /worker
COPY handler.py /worker/handler.py

ENTRYPOINT []
CMD ["python3", "-u", "/worker/handler.py"]
