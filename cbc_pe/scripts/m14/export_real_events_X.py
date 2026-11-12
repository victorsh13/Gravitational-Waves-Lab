"""Export the processed inputs X of the 9 real events of M13.13 (local, CPU).

The processing is not rewritten: the code cells of notebooks/m13/M13_13_real_events.ipynb that set the
parameters (2), paths and imports (4), reuse the M13.7 processing (6), process the events (8), load the
models (10) and run the M13.7 reproduction gate (12) are executed here from the notebook source, as that
notebook does with M13.7. Their sha256 and those of the M13.7 cells are stored. The M13.7 cells must have
the same sha256 as in the M13.13 real-event run (REAL_JSON), and the processing contract must be the same.

Output (never overwritten):
    <data_root>/results/M13_3/M14_5/real_events/events_X_<UTC>.npz
X (N, 3, 16384) float32 is the network input (z-scored once per sample and detector, as in M13.7);
no further input normalization must be applied.

Usage (from cbc_pe):
    python scripts/m14/export_real_events_X.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REAL_EVENTS_NOTEBOOK = PROJECT_ROOT / "notebooks/m13/M13_13_real_events.ipynb"
CELLS = (2, 4, 6, 8, 10, 12)          # parameters, paths, M13.7 reuse, events, models, M13.7 gate
REAL_EVENTS_STAMP = "20261110T130749Z"  # M13.13 real-event run (same events and processing code)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def x_sha256(X: np.ndarray) -> str:
    """sha256 of the C-ordered float32 bytes of X (the same definition is used by the prediction script)."""
    return sha256_bytes(np.ascontiguousarray(X, dtype=np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()

    os.chdir(PROJECT_ROOT)  # cell 4 finds the project root from the working directory
    cells = json.loads(REAL_EVENTS_NOTEBOOK.read_text())["cells"]
    ns = {"__name__": "m13_13_real_events"}
    executed = {}
    for i in CELLS:
        cell = cells[i]
        if cell["cell_type"] != "code":
            raise ValueError(f"M13.13 cell {i} is not code")
        source = "".join(cell["source"])
        executed[f"cell_{i}"] = sha256_bytes(source.encode())
        print(f"--- M13_13_real_events cell {i} ---", flush=True)
        exec(compile(source, f"M13_13_real_events_cell_{i}", "exec"), ns)

    require, read_json, file_sha256 = ns["require"], ns["read_json"], ns["file_sha256"]
    DATA_ROOT = ns["DATA_ROOT"]

    # Same M13.7 processing code and contract as the M13.13 real-event run.
    real_json_path = DATA_ROOT / "results/M13_3/M13_13/real_events" / f"M13_13_real_events_{REAL_EVENTS_STAMP}.json"
    real_json = read_json(real_json_path)
    for key, entry in real_json["provenance"]["reused_m13_7_cells"].items():
        require(ns["reused_sources"][key]["sha256"] == entry["sha256"], f"M13.7 {key} differs from {real_json_path.name}")
    dump_default = lambda o: o.item() if hasattr(o, "item") else str(o)   # as the M13.13 JSON writer
    processing_json = json.loads(json.dumps(ns["processing_contract"], default=dump_default))
    require(processing_json == real_json["processing"], "Processing contract differs from the M13.13 run")
    require(real_json["provenance"]["cohort_sha256"] == file_sha256(ns["EVENT_LIST_PATH"]), "Cohort differs from M13.13 run")

    inputs, cohort = ns["inputs"], ns["cohort"]
    events = list(inputs)
    require(events == cohort.event.tolist() and set(events) == set(real_json["events"]) and len(events) == 9,
            "Events differ from the 9 of M13.13")
    X = np.concatenate([np.asarray(inputs[e], dtype=np.float32) for e in events], axis=0)
    require(X.shape == (9, 3, 16384) and np.isfinite(X).all(), f"Unexpected X {X.shape}")
    # Exactly what enters the network: predict_real_with_embeddings casts X to float32.
    for e, x in zip(events, X):
        require(np.array_equal(x[None], np.asarray(inputs[e], dtype=np.float32)), f"{e}: X row differs")

    norm = ns["train_cfg"]["input_normalization"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = DATA_ROOT / "results/M13_3/M14_5/real_events"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"events_X_{stamp}.npz"

    metadata = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), run_stamp=stamp, script=str(Path(__file__).resolve()),
        source_notebook=str(REAL_EVENTS_NOTEBOOK), source_notebook_sha256=file_sha256(REAL_EVENTS_NOTEBOOK),
        executed_m13_13_cells=executed, reused_m13_7_cells=ns["reused_sources"],
        m13_13_real_events_json=dict(path=str(real_json_path), sha256=file_sha256(real_json_path)),
        cohort=dict(path=str(ns["EVENT_LIST_PATH"]), sha256=file_sha256(ns["EVENT_LIST_PATH"])),
        processing=processing_json, processed=ns["processed_provenance"],
        input_normalization=dict(applied=True, mode="per_sample_per_detector_zscore", eps=float(ns["INPUT_EPS"]),
                                 contract=norm, note="X is the network input; do not normalize again"),
        gate_m13_7=dict(model=ns["GATE_TAG"], tolerance=ns["GATE_TOLERANCE"],
                        max_abs_delta=ns["gate_max_abs_delta"], passed=True,
                        checkpoint_sha256=ns["gate_provenance"]["checkpoint_sha256"]),
        x_sha256=x_sha256(X), x_shape=list(X.shape), x_dtype="float32",
        versions=dict(python=sys.version, numpy=np.__version__, torch=ns["torch"].__version__))
    with out_path.open("xb") as f:
        np.savez(f, X=X, event_names=np.array(events), gps=cohort.gps_time.to_numpy(dtype=float),
                 detectors=np.array(ns["DETECTORS"]), catalog=cohort["catalog"].astype(str).to_numpy(),
                 x_sha256=np.array(metadata["x_sha256"]),
                 metadata_json=np.array(json.dumps(metadata, default=dump_default)))
    print("Events:", events)
    print("X:", X.shape, X.dtype, "sha256", metadata["x_sha256"])
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
