#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/opt/projects/chemist-agent"
AIZYNTH_DATA_DIR="${PROJECT_ROOT}/data/aizynth"
MAIN_ENV="${PROJECT_ROOT}/.env"
AIZYNTH_ENV="${PROJECT_ROOT}/.env.aizynth"
UNIT_SRC="${PROJECT_ROOT}/deploy/systemd/chemist-aizynth.service"
UNIT_DST="/etc/systemd/system/chemist-aizynth.service"
IMAGE_NAME="chemist-aizynth:latest"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "Project root not found: ${PROJECT_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for AiZynthFinder deployment on this host" >&2
  exit 1
fi

mkdir -p "${AIZYNTH_DATA_DIR}"
sudo docker build -t "${IMAGE_NAME}" -f deploy/docker/aizynth.Dockerfile .
if [[ ! -f "${AIZYNTH_DATA_DIR}/config.yml" ]]; then
  sudo docker run --rm -v "${AIZYNTH_DATA_DIR}:/data" "${IMAGE_NAME}" download_public_data /data
fi

cat > "${AIZYNTH_ENV}" <<EOF
AIZYNTH_CONFIG_PATH=/data/config.yml
AIZYNTH_DEFAULT_EXPANSION_MODEL=uspto
AIZYNTH_DEFAULT_FILTER_MODEL=uspto
AIZYNTH_DEFAULT_STOCK=zinc
EOF

python3 - "$MAIN_ENV" <<'PY'
from pathlib import Path
import sys

env_path = Path(sys.argv[1])
if not env_path.exists():
    raise SystemExit(f"Main env file not found: {env_path}")

required = {
    "RETRO_ENABLE_AIZYNTH": "true",
    "AIZYNTH_BASE_URL": "http://127.0.0.1:8052",
}

lines = env_path.read_text(encoding="utf-8").splitlines()
seen = set()
updated = []
for line in lines:
    if "=" not in line or line.lstrip().startswith("#"):
        updated.append(line)
        continue
    key, _ = line.split("=", 1)
    key = key.strip()
    if key in required:
        updated.append(f"{key}={required[key]}")
        seen.add(key)
    else:
        updated.append(line)

for key, value in required.items():
    if key not in seen:
        updated.append(f"{key}={value}")

env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
PY

sudo rm -rf "${PROJECT_ROOT}/venv-aizynth" || true

sudo cp "${UNIT_SRC}" "${UNIT_DST}"
sudo systemctl daemon-reload
sudo systemctl enable --now chemist-aizynth.service

echo "AiZynthFinder service installed."
echo "Health check: curl http://127.0.0.1:8052/health"
