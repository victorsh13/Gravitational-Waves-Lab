#!/usr/bin/env bash
# M15 G4 orchestrator (local CPU): sample -> gate-taper -> noisy val -> noisy train40k -> signal 0..3 ->
# noisy cal -> noisy test -> materialize R -> consistency R -> materialize F -> consistency F. One log per step in $L.
# Resumable as run_m13_13.sh: run the same script again. A step with its .done marker is skipped; a step whose
# output is already published gets its marker; an interrupted noisy or signal step is relaunched with --resume.
# Do not edit m15_build_g4.py or m13_13_build_g3t.py between runs: --resume and materialize refuse other versions.
# A gate failure stops the chain. sample and gate-taper keep their FAIL report and refuse to run again, and every
# later step requires their PASS, so relaunching repeats the stop on purpose (pre-registered).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

export CBC_PE_DATA_ROOT=/data/vserrano/cbc_pe_data
export LAL_DATA_PATH=$CBC_PE_DATA_ROOT/external/lal_data
export OMP_NUM_THREADS=1
PY=/afs/ciemat.es/user/v/vserrano/miniconda3/envs/gw-env/bin/python
B=scripts/m15/m15_build_g4.py
BASE=$CBC_PE_DATA_ROOT/processed/M15_G4
GATE=$BASE/gate_taper/gate_taper.json
L=$CBC_PE_DATA_ROOT/logs/m15
WORKERS=${WORKERS:-2}
NUM_SHARDS=4

mkdir -p "$L"
# Two chains would write the same outputs. The python children inherit the lock.
exec 9>"$L/run_m15_g4.lock"
flock -n 9 || { echo "Another run_m15_g4.sh is running"; exit 1; }

# run_step <step> <file that proves the step finished, or ""> <file that proves it started, or ""> <args...>
run_step() {
  local step=$1 published=$2 started=$3
  shift 3
  local marker="$L/m15_g4_$step.done" log="$L/m15_g4_$step.log"
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

# noisy <split>
noisy() {
  run_step "noisy_$1" "$BASE/noisy_union/X_$1.npy" "$BASE/work_noisy_$1/meta.json" \
    noisy --split "$1" --workers "$WORKERS" --gate-json "$GATE"
}

# sample and gate-taper: no published shortcut, so that a FAIL report is never marked done.
run_step sample "" "" sample
run_step gate_taper "" "" gate-taper --workers "$WORKERS"
noisy val
noisy train40k
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  run_step "signal_shard$i" "" "$BASE/signal/$(printf 'signal_shard%02dof%02d.h5' "$i" "$NUM_SHARDS")" \
    signal --shard "$i" --num-shards "$NUM_SHARDS" --workers "$WORKERS" --gate-json "$GATE"
done
noisy cal
noisy test
for T in R F; do
  run_step "materialize_$T" "$BASE/$T/materialize_manifest.json" "" \
    materialize --target "$T" --num-shards "$NUM_SHARDS" --gate-json "$GATE"
  run_step "consistency_$T" "" "" consistency --target "$T"
done
echo "== end $(date)"
