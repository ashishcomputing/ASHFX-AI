#!/usr/bin/env bash
# Startup script for ASHFX-AI Platform
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "============================================================"
echo " Starting ASHFX-AI: Enterprise Financial Intelligence"
echo " Advanced Multi-Asset Risk & Black-Litterman Network"
echo "============================================================"

export PYTHONPATH="$DIR/backend"
/Users/admin/.venv/bin/uvicorn aladdin.api:app --host 0.0.0.0 --port 8888 --reload
