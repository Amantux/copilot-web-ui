# Copilot Web UI

A self-hosted, ChatGPT-style web interface for **GitHub Copilot CLI** — inspired by [Copilot-Spawner](https://github.com/MauroDruwel/Copilot-Spawner).

Instead of a raw terminal, this gives you a clean chat UI backed by the real `copilot` binary via its `-p` non-interactive mode with JSON streaming output.

## Features

- 💬 ChatGPT-style chat interface with markdown + code highlighting
- 🔄 Persistent sessions across messages (`--continue` / `--resume`)
- 🔒 Password-protected login
- 🐳 Docker + docker-compose for thin-client always-on deployment
- 📜 In-browser session history

## Quick start (Docker — recommended)

```bash
git clone https://github.com/Amantux/copilot-web-ui
cd copilot-web-ui

# Set a password
export COPILOT_WEB_PASSWORD=your-password

# Start (mounts your host gh auth + copilot binary)
docker compose up -d
```

Open [http://localhost:8765](http://localhost:8765)

> **Requires:** GitHub Copilot CLI installed on the host (`gh copilot` should work).  
> The container mounts `~/.config/gh` and `~/.local/share/gh` read-only.

## Quick start (local Python)

```bash
git clone https://github.com/Amantux/copilot-web-ui
cd copilot-web-ui
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export COPILOT_WEB_PASSWORD=your-password
python3 app.py
```

## Always-on (systemd)

```bash
sudo cp copilot-web-ui.service /etc/systemd/system/
sudo systemctl enable --now copilot-web-ui
```

Example `copilot-web-ui.service`:

```ini
[Unit]
Description=Copilot Web UI
After=network-online.target
Wants=network-online.target

[Service]
User=YOUR_USER
WorkingDirectory=/path/to/copilot-web-ui
ExecStart=docker compose up
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `COPILOT_WEB_PASSWORD` | auto-generated | Login password (printed to logs if not set) |
| `COPILOT_WEB_SECRET` | auto-generated | Cookie signing secret (set for stable sessions across restarts) |
| `COPILOT_WEB_HOST` | `0.0.0.0` | Bind host |
| `COPILOT_WEB_PORT` | `8765` | Bind port |
| `COPILOT_WORKSPACE` | `./workspace` | Working directory for Copilot sessions |
| `COPILOT_BIN` | `copilot` | Path to the Copilot CLI binary |

## Security

- Keep behind localhost + a reverse proxy (nginx/Caddy) with TLS for remote access
- Set strong `COPILOT_WEB_PASSWORD` and `COPILOT_WEB_SECRET` values
- The container mounts gh config read-only — your token never touches the image

## License

MIT
