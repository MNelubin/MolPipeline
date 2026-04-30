# AiZynthFinder service

This repository now includes a local AiZynthFinder microservice for the additive
retrosynthesis mode.

Files:

- `mvp/aizynth_service.py` — FastAPI wrapper exposing `/api/v1/resources` and `/api/v1/run`
- `deploy/systemd/chemist-aizynth.service` — systemd unit for `/opt/projects/chemist-agent`
- `scripts/install_aizynth_service.sh` — bootstrap script for prod install
- `requirements-aizynth.txt` — optional planner runtime dependencies

## Prod install

Run on the server:

```bash
cd /opt/projects/chemist-agent
bash scripts/install_aizynth_service.sh
```

What it does:

1. Creates `/opt/projects/chemist-agent/venv-aizynth`
2. Installs `requirements-aizynth.txt`
3. Downloads public AiZynthFinder data into `/opt/projects/chemist-agent/data/aizynth`
4. Writes `/opt/projects/chemist-agent/.env.aizynth`
5. Updates the main `.env` with:
   - `RETRO_ENABLE_AIZYNTH=true`
   - `AIZYNTH_BASE_URL=http://127.0.0.1:8052`
6. Installs and starts `chemist-aizynth.service`

## Checks

```bash
curl http://127.0.0.1:8052/health
curl http://127.0.0.1:8052/api/v1/resources
sudo systemctl status chemist-aizynth.service
```

## Main app expectation

The main API continues to call `AIZYNTH_BASE_URL` through
`mvp.services.aizynth_client`. Once the service is healthy, `/retro/sources`
will report `AiZynthFinder` as enabled and reachable.
