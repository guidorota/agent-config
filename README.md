# Local LLM server: llama-swap + llama.cpp in Docker

**Target host:** Ubuntu Server 26.04, RTX 4090 (24 GB)
**Client access:** trusted LAN, no auth, bound to `0.0.0.0:8080`
**Cache:** Docker named volume (persists across container rebuilds)

---

## 1. CUDA image tag

llama-swap publishes CUDA images at `ghcr.io/mostlygeek/llama-swap`. Tag scheme is date-based:

- **Rolling:** `unified-cuda` (rebuilt frequently; convenient but drifts)
- **Pinned:** `unified-cuda-YYYY-MM-DD` (e.g. `unified-cuda-2026-05-23`) — use this for reproducibility
- **Rootless variant:** `unified-cuda-rootless` / `unified-cuda-rootless-YYYY-MM-DD` — only needed if running Docker rootless

**Recommendation:** start with `unified-cuda-2026-05-23` (most recent pinned tag at time of writing). Bump the date when you want to update.

Confirm before pulling:

```bash
# List recent tags
curl -s "https://ghcr.io/v2/mostlygeek/llama-swap/tags/list" | jq '.tags[]' | grep cuda | sort | tail -20
```

---

## 2. Host prerequisites (Ubuntu Server 26.04)

```bash
# NVIDIA driver — should already be installed; verify
nvidia-smi

# Docker Engine (if not already installed)
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
# log out / back in for group membership to take effect

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Smoke test GPU access from a container
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

If the smoke test prints `nvidia-smi` output from inside the container, you're good.

---

## 3. Directory layout on the server

```
~/llm/
├── docker-compose.yml
└── config.yaml
```

```bash
mkdir -p ~/llm && cd ~/llm
```

---

## 4. `~/llm/config.yaml`

```yaml
healthCheckTimeout: 300

models:
  qwen3.6-27b-64k:
    cmd: |
      llama-server -hf unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL
      --port ${PORT} --host 0.0.0.0
      -ngl 99 -fa on -np 1
      -c 65536
      -ctk q8_0 -ctv q8_0
      --spec-type draft-mtp --spec-draft-n-max 2
      --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
    ttl: 600

  qwen3.6-27b-96k:
    cmd: |
      llama-server -hf unsloth/Qwen3.6-27B-MTP-GGUF:IQ4_XS
      --port ${PORT} --host 0.0.0.0
      -ngl 99 -fa on -np 1
      -c 98304
      -ctk q4_0 -ctv q4_0
      --spec-type draft-mtp --spec-draft-n-max 2
      --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
    ttl: 600

  qwen3.6-27b-128k:
    cmd: |
      llama-server -hf unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q3_K_XL
      --port ${PORT} --host 0.0.0.0
      -ngl 99 -fa on -np 1
      -c 131072
      -ctk q4_0 -ctv q4_0
      --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
    ttl: 600
```

Notes:
- `--host 0.0.0.0` on the inner `llama-server` is fine because the container's network is isolated; only port 8080 is exposed to the host.
- `${PORT}` is auto-assigned by llama-swap per backend — don't hardcode it.
- MTP is dropped at 128K because VRAM is tight; the spec-decode buffers won't fit alongside model + 128K KV cache reliably. Add it back if benchmarking shows headroom.
- `ttl: 600` unloads each model after 10 minutes idle, freeing VRAM for the next swap.

---

## 5. `~/llm/docker-compose.yml`

```yaml
services:
  llama-swap:
    image: ghcr.io/mostlygeek/llama-swap:unified-cuda-2026-05-23
    container_name: llama-swap
    restart: unless-stopped
    ports:
      - "8080:8080"   # LAN-exposed
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - hf-cache:/root/.cache/huggingface
    environment:
      - HF_HOME=/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command:
      - "--config"
      - "/app/config.yaml"
      - "--listen"
      - "0.0.0.0:8080"

volumes:
  hf-cache:
```

The `hf-cache` named volume persists model downloads across `docker compose down` / `up` cycles. Models are large (15–20 GB each) — only re-downloaded if you delete the volume.

---

## 6. Bring it up

```bash
cd ~/llm
docker compose up -d
docker compose logs -f llama-swap
```

First request to any model triggers download (slow) then load (~10–30s). Subsequent requests are fast. Switching models triggers an unload + load.

---

## 7. Verify from the client machine

Replace `<server-ip>` with the LAN IP of the Ubuntu host:

```bash
# List models llama-swap knows about
curl http://<server-ip>:8080/v1/models

# Smoke-test inference (triggers a load on first call)
curl http://<server-ip>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b-64k",
    "messages": [{"role": "user", "content": "Say hi in one word."}]
  }'
```

If the server doesn't respond from the client:
- Check firewall: `sudo ufw status` — allow 8080 from your LAN subnet if ufw is active
- Check the container is listening: `docker compose exec llama-swap ss -tlnp | grep 8080`
- Check NVIDIA passthrough: `docker compose exec llama-swap nvidia-smi`

---

## 8. opencode client config

On the client machine, `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama.cpp (LAN)",
      "options": {
        "baseURL": "http://<server-ip>:8080/v1",
        "apiKey": "sk-no-auth"
      },
      "models": {
        "qwen3.6-27b-64k":  { "name": "Qwen3.6 27B (64K)",  "limit": { "context": 65536,  "output": 32768 } },
        "qwen3.6-27b-96k":  { "name": "Qwen3.6 27B (96K)",  "limit": { "context": 98304,  "output": 32768 } },
        "qwen3.6-27b-128k": { "name": "Qwen3.6 27B (128K)", "limit": { "context": 131072, "output": 32768 } }
      }
    }
  }
}
```

Model IDs must match the keys in `config.yaml` exactly — that's the swap trigger.

In opencode: `/models` → pick a Qwen3.6 entry.

---

## 9. Day-2 operations

```bash
# View logs
docker compose logs -f llama-swap

# Restart (e.g. after editing config.yaml)
docker compose restart llama-swap

# Update to a newer image
# 1. find a newer tag with the curl command in section 1
# 2. edit docker-compose.yml
docker compose pull
docker compose up -d

# Wipe model cache (force re-download)
docker compose down
docker volume rm llm_hf-cache
docker compose up -d

# See what's currently loaded in VRAM
nvidia-smi
```

---

## Open items / things to revisit

- **96K config is untested** on this hardware — watch first run for OOM, drop `-c` if needed.
- **128K config is tight** — if it OOMs, try `UD-Q2_K_XL` (12 GB) instead of `UD-Q3_K_XL`.
- **No auth** — fine for trusted LAN. If this machine ever joins an untrusted network, put Tailscale or a reverse-proxy with basic-auth in front before exposing further.
- **TTL of 10 min** is a guess; tune based on usage patterns (longer = fewer cold-load delays, but a single model squats on VRAM longer).

