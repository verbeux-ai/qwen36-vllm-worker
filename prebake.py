#!/usr/bin/env python3
"""
Baixa o modelo pro HF cache com observabilidade (heartbeat + tamanho do cache a cada 15s).
Roda durante o docker build pra evitar download em runtime no cold-start.
"""
import os
import sys
import time
import threading
import subprocess

# Força hf_transfer ANTES do import do huggingface_hub.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import huggingface_hub  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

REPO = sys.argv[1]
PATTERNS = sys.argv[2].split(",") if len(sys.argv) > 2 else None

print(f"[prebake] repo={REPO}", flush=True)
print(f"[prebake] patterns={PATTERNS}", flush=True)
print(f"[prebake] huggingface_hub={huggingface_hub.__version__}", flush=True)
try:
    import hf_transfer
    print(f"[prebake] hf_transfer={hf_transfer.__version__} ENABLED", flush=True)
except ImportError as e:
    print(f"[prebake] WARN: hf_transfer NOT installed ({e}) — download será LENTO", flush=True)

done = threading.Event()

def heartbeat():
    started = time.time()
    while not done.wait(15):
        try:
            r = subprocess.run(["du", "-sh", "/models"], capture_output=True, text=True, timeout=5)
            print(f"[prebake/hb t={time.time()-started:.0f}s] {r.stdout.strip()}", flush=True)
        except Exception:
            pass

t = threading.Thread(target=heartbeat, daemon=True)
t.start()

t0 = time.time()
path = snapshot_download(REPO, max_workers=8, allow_patterns=PATTERNS)
done.set()
elapsed = time.time() - t0
print(f"[prebake] DONE in {elapsed:.1f}s -> {path}", flush=True)
