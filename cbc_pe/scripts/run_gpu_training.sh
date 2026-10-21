#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <training_config.json>" >&2
    exit 2
fi

CONFIG="$1"
DATA_ROOT="/data/vserrano/cbc_pe_data"
LOG_DIR="${DATA_ROOT}/logs/training"

mkdir -p "$LOG_DIR"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

RUN_NAME="$(basename "$CONFIG" .json)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log"

echo "Training config: $CONFIG"
echo "Log file: $LOG_FILE"

# Preserve Python's exit status, even when piping through tee.
python -u scripts/train_cnn_hdf5.py \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    2>&1 | tee "$LOG_FILE"