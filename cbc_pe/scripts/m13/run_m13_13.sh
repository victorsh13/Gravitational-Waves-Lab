#!/usr/bin/env bash
# M13.13 orchestrator (local CPU): noisy val -> noisy train40k -> signal 0..3 -> merge -> consistency ->
# noisy cal -> noisy test. One log per step in $L.
# Resumable: run the same script again. A step with its .done marker is skipped; a noisy or merge step whose
# output is already published gets its marker; an interrupted noisy or signal step is relaunched with --resume.
# Do not edit m13_13_build_g3t.py between runs: --resume and merge refuse another version of it.
# A gate failure stops the chain; relaunching repeats that failure on purpose (pre-registered stop).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

export CBC_PE_DATA_ROOT=/data/vserrano/cbc_pe_data
export LAL_DATA_PATH=$CBC_PE_DATA_ROOT/external/lal_data
export OMP_NUM_THREADS=1
PY=/afs/ciemat.es/user/v/vserrano/miniconda3/envs/gw-env/bin/python
B=scripts/m13/m13_13_build_g3t.py
BASE=$CBC_PE_DATA_ROOT/processed/M13_G3_band1024_w16_500k/m13_13_g3t
GATE=$BASE/gate_taper_20261105T145635Z/gate_taper.json
L=$CBC_PE_DATA_ROOT/logs/m13
WORKERS=3
NUM_SHARDS=4

[[ -e $GATE ]] || { echo "Missing $GATE"; exit 1; }
mkdir -p "$L"
# Two chains would write the same outputs. The python children inherit the lock.
exec 9>"$L/run_m13_13.lock"
flock -n 9 || { echo "Another run_m13_13.sh is running"; exit 1; }

# run_step <step> <file that proves the step finished, or ""> <file that proves it started, or ""> <args...>
run_step() {
  local step=$1 published=$2 started=$3
  shift 3
  local marker="$L/m13_13_$step.done" log="$L/m13_13_$step.log"
  local resume=()
  if [[ -e $marker ]]; then
    echo "== $step: done, skipped"
    return 0
  fi
  if [[ -n $published && -e $published ]]; then
    echo "== $step: output already published, marked done"
    touch "$marker"
    return 0
  fi
  if [[ -n $started && -e $started ]]; then
    resume=(--resume)
  fi
  echo "== $step $(date) ${resume[*]}"
  nice -n 19 ionice -c3 "$PY" -u "$B" "$@" "${resume[@]}" >> "$log" 2>&1
  touch "$marker"
}

# noisy <split> <memmap directory> <split name in the memmap>
noisy() {
  run_step "noisy_$1" "$BASE/$2/X_$3.npy" "$BASE/work_noisy_$1/meta.json" \
    noisy --split "$1" --workers "$WORKERS" --gate-json "$GATE"
}

noisy val fp16_memmap val
noisy train40k fp16_memmap train
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  run_step "signal_shard$i" "" "$BASE/signal/$(printf 'signal_shard%02dof%02d.h5' "$i" "$NUM_SHARDS")" \
    signal --shard "$i" --num-shards "$NUM_SHARDS" --workers "$WORKERS" --gate-json "$GATE"
done
run_step merge "$BASE/memmap/manifest.json" "" merge --num-shards "$NUM_SHARDS"
run_step consistency "" "" consistency
noisy cal fp16_memmap_heldout cal
noisy test fp16_memmap_heldout test
echo "== end $(date)"
