#!/usr/bin/env bash
# Startup script for Aladdin-AI Platform
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "============================================================"
echo " Starting Aladdin-AI: Enterprise Financial Intelligence"
echo " Inspired by BlackRock Aladdin Architecture"
echo "============================================================"

export PYTHONPATH="$DIR/backend"
/Users/admin/.venv/bin/uvicorn aladdin.api:app --host 0.0.0.0 --port 8888 --reload
