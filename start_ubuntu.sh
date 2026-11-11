#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python3 -m pip install -r requirements.txt
exec python3 app.py
