#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="/afs/ciemat.es/user/v/vserrano/Desktop/gw/Gravitational-Waves-Lab/cbc_pe/data/"
REMOTE_HOST="ciemat-pcaecuda2"
REMOTE_DIR="/data/vserrano/cbc_pe_data/"

EXTRA_ARGS=()

if [[ "${1:-}" == "--dry-run" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi

rsync -av --progress "${EXTRA_ARGS[@]}" \
  "${LOCAL_DIR}" \
  "${REMOTE_HOST}:${REMOTE_DIR}"
