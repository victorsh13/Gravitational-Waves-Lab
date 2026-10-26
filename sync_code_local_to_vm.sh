#!/usr/bin/env bash
set -euo pipefail

LOCAL_REPO="/afs/ciemat.es/user/v/vserrano/Desktop/gw/Gravitational-Waves-Lab/cbc_pe/"
REMOTE_HOST="ciemat-pcaecuda2"
REMOTE_REPO="/data/vserrano/gw/Gravitational-Waves-Lab/cbc_pe/"

EXTRA_ARGS=()

if [[ "${1:-}" == "--dry-run" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi

rsync -av --progress "${EXTRA_ARGS[@]}" \
  --exclude "data/" \
  --exclude "notebooks/" \
  --exclude ".claude/" \
  --exclude ".ipynb_checkpoints/" \
  --exclude "**/__pycache__/" \
  --exclude "**/*.pyc" \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "venv/" \
  --exclude "env/" \
  --exclude "*.npz" \
  --exclude "*.h5" \
  --exclude "*.hdf5" \
  --exclude "*.pt" \
  --exclude "*.pth" \
  --exclude "*.png" \
  --exclude "*.pdf" \
  "${LOCAL_REPO}" \
  "${REMOTE_HOST}:${REMOTE_REPO}"