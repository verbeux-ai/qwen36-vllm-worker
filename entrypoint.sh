#!/bin/bash
# Dual-mode entrypoint: detecta RunPod Serverless vs vast.ai Serverless vs standalone.
set -e

echo "[entrypoint] env discovery: RUNPOD_POD_ID=${RUNPOD_POD_ID:-unset} VAST_CONTAINERLABEL=${VAST_CONTAINERLABEL:-unset}"

if [ -n "$RUNPOD_POD_ID" ]; then
  echo "[entrypoint] mode=runpod (serverless SDK)"
  exec python3 -u /worker/handler.py

elif [ -n "$VAST_CONTAINERLABEL" ] || [ -n "$PYWORKER_REPO" ] || [ -n "$REPORT_ADDR" ]; then
  echo "[entrypoint] mode=vast.ai (vLLM + PyWorker)"
  # vLLM em background na porta MODEL_PORT (default 8000)
  python3 -u /worker/handler.py --vast-bg &
  VLLM_PID=$!
  echo "[entrypoint] vLLM pid=$VLLM_PID, fetching pyworker..."
  # PyWorker em foreground na porta WORKER_PORT (default 3000)
  wget -qO /tmp/start_server.sh https://raw.githubusercontent.com/vast-ai/pyworker/main/start_server.sh
  chmod +x /tmp/start_server.sh
  trap "kill $VLLM_PID 2>/dev/null || true" EXIT
  exec /tmp/start_server.sh

else
  echo "[entrypoint] mode=standalone (vLLM em foreground)"
  exec python3 -u /worker/handler.py --standalone
fi
