#!/usr/bin/env bash
# run_dashboard.sh — 一键启动 SPX 0DTE Scanner 仪表板
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/Users/peikexu/miniconda3/bin/python"
STREAMLIT="/Users/peikexu/miniconda3/bin/streamlit"

echo "========================================="
echo " SPX 0DTE Scanner Dashboard"
echo " http://localhost:8501"
echo "========================================="

"$STREAMLIT" run spx_scanner/viz/dashboard.py \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false
