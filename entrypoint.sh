#!/bin/bash
# Dual-mode entrypoint: detecta RunPod Serverless vs vast.ai Serverless vs standalone.
set -e

echo "[entrypoint] env discovery: RUNPOD_POD_ID=${RUNPOD_POD_ID:-unset} VAST_CONTAINERLABEL=${VAST_CONTAINERLABEL:-unset}"

if [ -n "$RUNPOD_POD_ID" ]; then
  echo "[entrypoint] mode=runpod (serverless SDK)"
  exec python3 -u /worker/handler.py

elif [ -n "$VAST_CONTAINERLABEL" ] || [ -n "$REPORT_ADDR" ]; then
  echo "[entrypoint] mode=vast.ai (vLLM + PyWorker)"

  # PyWorker openai handler espera vLLM em 127.0.0.1:18000 e log em /var/log/portal/vllm.log
  export MODEL_PORT="${MODEL_PORT:-18000}"
  export BACKEND="${BACKEND:-openai}"
  export MODEL_NAME="${MODEL_NAME:-${SERVED_NAME:-${MODEL_ID##*/}}}"
  export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
  export SERVER_DIR="${SERVER_DIR:-/opt/vast-pyworker}"
  export ENV_PATH="${ENV_PATH:-/workspace/worker-env}"
  export MODEL_LOG="${MODEL_LOG:-/var/log/portal/vllm.log}"
  export WORKER_PORT="${WORKER_PORT:-3000}"
  export REPORT_ADDR="${REPORT_ADDR:-https://run.vast.ai}"

  mkdir -p "$(dirname "$MODEL_LOG")" "$WORKSPACE_DIR"
  : > "$MODEL_LOG"

  # vLLM em background, stdout/stderr → MODEL_LOG (PyWorker monitora esse arquivo)
  echo "[entrypoint] starting vLLM bg on :$MODEL_PORT, log=$MODEL_LOG"
  python3 -u /worker/handler.py --vast-bg >> "$MODEL_LOG" 2>&1 &
  VLLM_PID=$!
  trap "kill $VLLM_PID 2>/dev/null || true" EXIT

  # PyWorker em foreground — usa o start_server.sh pré-clonado
  echo "[entrypoint] starting PyWorker BACKEND=$BACKEND SERVER_DIR=$SERVER_DIR"
  exec bash "$SERVER_DIR/start_server.sh"

else
  echo "[entrypoint] mode=standalone (vLLM em foreground)"
  exec python3 -u /worker/handler.py --standalone
fi
