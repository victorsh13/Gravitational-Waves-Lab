#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="pcaecuda2.ciemat.es"
REMOTE_PORT="822"
REMOTE_DIR="/data/vserrano/cbc_pe_data/"
LOCAL_DIR="/afs/ciemat.es/user/v/vserrano/Desktop/gw/Gravitational-Waves-Lab/cbc_pe/data/"

EXTRA_ARGS=()

if [[ "${1:-}" == "--dry-run" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi

mkdir -p "${LOCAL_DIR}"

rsync -av --progress ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  --exclude "processed/" \
  --exclude "raw/" \
  -e "ssh -p ${REMOTE_PORT}" \
  "vserrano@${REMOTE_HOST}:${REMOTE_DIR}" \
  "${LOCAL_DIR}"
