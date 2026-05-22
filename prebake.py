#!/usr/bin/env python3
"""
Baixa modelo HF arquivo-por-arquivo com retry + heartbeat.

Por que não snapshot_download:
- snapshot_download() não retry per-file: se um shard stall (caso visto com AEON-7),
  o download inteiro fica travado sem progresso.
- hf_hub_download() per-file com try/except permite isolar e retry o shard problemático.

Ativa hf_transfer ANTES de importar huggingface_hub (env var precede o import).
"""
import os
import sys
import time
import fnmatch
import subprocess
import threading

# Configuração obrigatória ANTES dos imports do HF.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")  # 30s por chunk

import huggingface_hub  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = sys.argv[1]
PATTERNS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["*"]

print(f"[prebake] repo={REPO}", flush=True)
print(f"[prebake] patterns={PATTERNS}", flush=True)
print(f"[prebake] huggingface_hub={huggingface_hub.__version__}", flush=True)
try:
    import hf_transfer  # noqa: F401
    print(f"[prebake] hf_transfer={hf_transfer.__version__} ENABLED", flush=True)
except ImportError as e:
    print(f"[prebake] WARN: hf_transfer não instalado ({e}) — download será lento", flush=True)

api = HfApi()
all_files = api.list_repo_files(REPO)
selected = [f for f in all_files if any(fnmatch.fnmatch(f, p) for p in PATTERNS)]

# Tamanhos pra log
info = api.model_info(REPO, files_metadata=True)
sizes = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
total_bytes = sum(sizes.get(f, 0) for f in selected)
print(f"[prebake] {len(selected)} arquivos selecionados / {total_bytes/1e9:.2f}GB total", flush=True)

# Heartbeat reportando tamanho do /models a cada 15s.
done = threading.Event()
def heartbeat():
    started = time.time()
    while not done.wait(15):
        try:
            r = subprocess.run(["du", "-sh", "/models"], capture_output=True, text=True, timeout=5)
            print(f"[prebake/hb t={time.time()-started:.0f}s] {r.stdout.strip()}", flush=True)
        except Exception:
            pass

threading.Thread(target=heartbeat, daemon=True).start()

t0 = time.time()
for i, fname in enumerate(selected, 1):
    size_gb = sizes.get(fname, 0) / 1e9
    print(f"[prebake] [{i}/{len(selected)}] {fname} ({size_gb:.2f}GB)", flush=True)
    last_err = None
    for attempt in range(1, 6):
        try:
            f_t0 = time.time()
            hf_hub_download(repo_id=REPO, filename=fname)
            elapsed = time.time() - f_t0
            speed = (sizes.get(fname, 0) / 1e6) / elapsed if elapsed > 0 else 0
            print(f"[prebake]   ✓ done in {elapsed:.1f}s ({speed:.1f} MB/s)", flush=True)
            break
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"[prebake]   ✗ attempt {attempt}/5 failed: {type(e).__name__}: {e}", flush=True)
            print(f"[prebake]   retrying in {wait}s...", flush=True)
            time.sleep(wait)
    else:
        done.set()
        print(f"[prebake] FAILED após 5 tentativas em {fname}: {last_err}", flush=True)
        sys.exit(1)

done.set()
elapsed = time.time() - t0
print(f"[prebake] ✓ DONE em {elapsed:.1f}s ({total_bytes/1e6/elapsed:.1f} MB/s média)", flush=True)
