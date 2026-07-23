#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/director-vision}"
APP_USER="${APP_USER:-director}"

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip rsync libgl1 libglib2.0-0

if ! id "$APP_USER" >/dev/null 2>&1; then
  sudo useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
fi

sudo mkdir -p "$APP_DIR"
sudo rsync -a --delete --exclude .venv --exclude .git ./ "$APP_DIR"/
sudo chown -R "$APP_USER":"$APP_USER" "$APP_DIR"

if [ ! -f "$APP_DIR/.env" ]; then
  sudo cp "$APP_DIR/.env.example" "$APP_DIR/.env"
fi

sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

sudo cp "$APP_DIR/deploy/director-vision.service" /etc/systemd/system/director-vision.service
sudo systemctl daemon-reload
sudo systemctl enable director-vision.service
sudo systemctl restart director-vision.service

echo "Director Vision is running on http://127.0.0.1:8000"
