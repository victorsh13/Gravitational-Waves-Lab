#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="vserrano"
REMOTE_HOST="pcaecuda2.ciemat.es"
REMOTE_PORT="822"

REMOTE_DIR="/data/vserrano/cbc_pe/data/"
LOCAL_DIR="/data/vserrano/cbc_pe_data/"

RSYNC_ARGS=(
  --archive
  --human-readable
  --itemize-changes
  --info=progress2
  --partial
  --prune-empty-dirs
  --exclude="processed/"
  --exclude="raw/"
  --ignore-existing
)

if [[ "${1:-}" == "--dry-run" ]]; then
  RSYNC_ARGS+=(--dry-run)
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

mkdir -p "${LOCAL_DIR}"

rsync "${RSYNC_ARGS[@]}" \
  -e "ssh -p ${REMOTE_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}" \
  "${LOCAL_DIR}"