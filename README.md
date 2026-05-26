# Local LLM server: llama-swap + Hermes Agent in Docker

**Target host:** Ubuntu Server 26.04, RTX 4090 (24 GB)
**Client access:** trusted LAN, bound to `0.0.0.0`
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
├── .env                  # HERMES_API_KEY (chmod 600)
├── config.yaml           # llama-swap model configs
└── hermes-data/          # Hermes' persistent state (config, sessions, skills, .env, logs)
```

```bash
mkdir -p ~/llm/hermes-data && cd ~/llm
```

---

## 4. `~/llm/config.yaml` (llama-swap)

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

## 5. `~/llm/.env`

```bash
# Generate a random key for Hermes' API server (≥8 chars)
echo "HERMES_API_KEY=$(openssl rand -hex 24)" > .env
chmod 600 .env
```

---

## 6. `~/llm/docker-compose.yml`

No dashboard service — keep it off by default and start it ad-hoc when needed (section 10).

```yaml
services:
  llama-swap:
    image: ghcr.io/mostlygeek/llama-swap:unified-cuda-2026-05-23
    container_name: llama-swap
    restart: unless-stopped
    ports:
      - "8080:8080"   # LAN-exposed (raw model API, no auth)
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

  hermes-agent:
    image: nousresearch/hermes-agent:main
    container_name: hermes
    restart: unless-stopped
    depends_on:
      - llama-swap
    ports:
      - "8642:8642"   # LAN-exposed (OpenAI-compatible API, key-auth)
    volumes:
      - ./hermes-data:/opt/data
    environment:
      - HERMES_UID=1000
      - HERMES_GID=1000
      - API_SERVER_ENABLED=true
      - API_SERVER_HOST=0.0.0.0
      - API_SERVER_KEY=${HERMES_API_KEY}
    command: ["gateway", "run"]

volumes:
  hf-cache:
```

### Service summary

| Service | Port | Purpose |
|---------|------|---------|
| llama-swap | 8080 | Serves your local models (OpenAI-compatible, no auth) |
| hermes-agent | 8642 | Hermes gateway — tools, cron, subagents, OpenAI-compatible API with `API_SERVER_KEY` |

`hf-cache` (named volume) persists model downloads across `docker compose down`/`up`. Models are 15–20 GB each — only re-downloaded if you delete the volume.

`./hermes-data` (bind mount) holds Hermes' config, sessions, skills, memory, logs, and its own `.env` of tool API keys.

`HERMES_UID`/`HERMES_GID` are baked to `1000:1000`; if your host user differs (`id -u`/`id -g`), edit them and chown `hermes-data` to match.

---

## 7. Bring up llama-swap first

```bash
cd ~/llm
docker compose up -d llama-swap
docker compose logs -f llama-swap    # Ctrl-C once it's healthy
curl http://localhost:8080/v1/models # confirms it sees your model ids
```

Note the model id you want as Hermes' default (e.g. `qwen3.6-27b-64k`) — needed in step 8.

---

## 8. Run the Hermes setup wizard (one-off)

Hermes won't start until `hermes-data/` has a config. Run the interactive wizard, passing the same UID/GID as the compose service so the files are owned correctly from the start (the image defaults to 10000:10000, which will lock you out):

```bash
docker run -it --rm \
  -v "$PWD/hermes-data:/opt/data" \
  -e HERMES_UID=1000 -e HERMES_GID=1000 \
  nousresearch/hermes-agent:main setup
```

When the wizard asks about an LLM provider, pick **custom / OpenAI-compatible** if offered, otherwise accept defaults and fix the config below. Skip Telegram/Discord/Slack unless you actually want them.

Then edit `hermes-data/config.yaml` so the `model:` block points at llama-swap by service name (compose DNS):

```yaml
model:
  provider: custom
  model: qwen3.6-27b-64k            # must match a key from config.yaml
  base_url: http://llama-swap:8080/v1
  api_key: "none"
```

Tool API keys (Tavily, Firecrawl, ElevenLabs, etc.) live in `hermes-data/.env` — add as needed.

If you ever forgot the `-e HERMES_UID/-e HERMES_GID` flags (or ran a different image command that fell back to 10000:10000), fix ownership before starting the gateway:

```bash
sudo chown -R 1000:1000 hermes-data    # match HERMES_UID/HERMES_GID
```

---

## 9. Bring up Hermes

```bash
docker compose up -d hermes-agent
docker compose logs -f hermes-agent    # watch for "API server listening on 0.0.0.0:8642"
```

---

## 10. Verify end-to-end

```bash
# llama-swap direct
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b-64k",
    "messages": [{"role": "user", "content": "Say hi in one word."}]
  }'

# Hermes gateway → llama-swap → model
HERMES_KEY=$(grep HERMES_API_KEY .env | cut -d= -f2)
curl http://localhost:8642/health
curl http://localhost:8642/v1/chat/completions \
  -H "Authorization: Bearer $HERMES_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Say hi in one word."}]
  }'

# Chat with Hermes from the host (CLI in the container)
# Note: hermes binary lives in the project venv, not on $PATH. We also pass
# -u hermes so session files stay owned by HERMES_UID, not root.
docker exec -it -u hermes hermes /opt/hermes/.venv/bin/hermes chat
```

First request to any model triggers download (slow) then load (~10–30s); subsequent requests are fast. Switching models triggers unload + load.

---

## 11. Dashboard (on-demand, optional)

The dashboard is not in the compose file because it stores API keys with no built-in auth. Run it ad-hoc against the same data dir when you need it:

```bash
docker run --rm -it \
  -v "$PWD/hermes-data:/opt/data" \
  -p 127.0.0.1:9119:9119 \
  nousresearch/hermes-agent:main dashboard --host 0.0.0.0 --no-open
# open http://localhost:9119, Ctrl-C when done
```

For remote access, tunnel: `ssh -L 9119:localhost:9119 user@<server-ip>` then visit `http://localhost:9119`.

If you'd rather have it always-on but auth'd on the LAN, put Caddy with `basicauth` in front and bind only Caddy to `0.0.0.0:9119`.

What you give up by not running it persistently:
- Visual token/cost analytics
- Full-text search across past sessions
- Click-driven cron job management (still configurable via files)
- In-browser TUI chat (use `docker exec -it hermes hermes` instead)

Everything else (config, API keys, skills, sessions) is editable from files or the REST API on `:8642`.

---

## 12. Client access (opencode / other machines)

### URLs from the LAN

| What | URL | Auth |
|------|-----|------|
| llama-swap API | `http://<server-ip>:8080/v1` | none |
| Hermes gateway API | `http://<server-ip>:8642/v1` | `Bearer $HERMES_API_KEY` |

### opencode client config

Talk to llama-swap directly (no auth, lets you pick a model id and trigger swaps explicitly). `~/.config/opencode/opencode.json` on the client:

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

If the server doesn't respond from the client:
- Firewall: `sudo ufw status` — allow 8080 (and 8642 if you want to reach Hermes too) from your LAN subnet if ufw is active
- Container listening: `docker compose exec llama-swap ss -tlnp | grep 8080`
- NVIDIA passthrough: `docker compose exec llama-swap nvidia-smi`

---

## 13. Day-2 operations

```bash
# View logs
docker compose logs -f llama-swap
docker compose logs -f hermes-agent

# Restart a service
docker compose restart llama-swap
docker compose restart hermes-agent

# Ad-hoc Hermes chat (binary is in the venv; -u hermes preserves file ownership)
docker exec -it -u hermes hermes /opt/hermes/.venv/bin/hermes chat

# Edit Hermes config (model, providers, tool keys)
$EDITOR hermes-data/config.yaml
$EDITOR hermes-data/.env
docker compose restart hermes-agent

# Update llama-swap to a newer image
# 1. find a newer tag with the curl in section 1
# 2. edit docker-compose.yml
docker compose pull
docker compose up -d

# Update Hermes to latest :main
docker compose pull hermes-agent
docker compose up -d hermes-agent

# Wipe model cache (force re-download)
docker compose down
docker volume rm llm_hf-cache
docker compose up -d

# What's currently loaded in VRAM
nvidia-smi

# Debugging CUDA OOM
nvidia-smi dmon -s mu -d 1
```

---

## Open items / things to revisit

- **96K config is untested** on this hardware — watch first run for OOM, drop `-c` if needed.
- **128K config is tight** — if it OOMs, try `UD-Q2_K_XL` (12 GB) instead of `UD-Q3_K_XL`.
- **llama-swap has no auth** on `:8080`. Fine for trusted LAN. If the host ever joins an untrusted network, front it with Tailscale or a reverse proxy with auth.
- **Hermes API key is shared** — every client uses the same `HERMES_API_KEY`. If you need per-user auth or audit, put a proxy in front.
- **TTL of 10 min** is a guess; tune based on usage patterns (longer = fewer cold-load delays, but a single model squats on VRAM longer).
- **Dashboard is opt-in** (section 11). If you find yourself running it daily, promote it to a `profiles: ["ui"]` compose service so `docker compose --profile ui up -d` toggles it.

