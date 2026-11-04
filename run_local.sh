#!/bin/bash
# Run without Docker. For local testing set PUBLIC_BASE_URL=http://localhost:8000 in .env
# (http is only accepted for localhost) and DATA_DIR=./data.
set -e
cd "$(dirname "$0")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
python3 -m pip install --user -r requirements.txt
python3 server.py
