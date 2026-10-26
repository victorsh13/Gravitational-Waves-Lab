#!/usr/bin/env python3
"""M13.0c: whitening-filter duration sweep on 64-row G3 pilots, with a predefined choice rule.

For each whitening_max_filter_duration in DURATIONS_S (lowpass 1024 Hz, context
per side = corrupted margin + 128 samples), run generate_m13_G3_whitening.py
--pilot 64 sequentially, then measure residual spectral structure of the
model input X with Welch PSDs. Writes sweep.csv, summary.json and a figure to
<data_root>/results/M13_0c_whitening/ (aborts if it exists).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import dataset_processed_dir, resolve_data_root
from scripts.generate_m11_final_G1 import DETECTORS, require, sha256_file

GENERATOR = PROJECT_ROOT / "scripts/m13/generate_m13_G3_whitening.py"
BASE_G1_H5 = "M11_final_G1_500k/m11_final_G1_500k.h5"
RESULTS_NAME = "M13_0c_whitening"
DURATIONS_S = (0.5, 2.0, 4.0, 8.0)
LOWPASS_HZ = 1024.0
PILOT_ROWS = 64
SAMPLE_RATE = 4096
# Same slack as G1: at 0.5 s, corrupted margin 1536 + 128 = 1664 samples per side.
CONTEXT_SLACK_SAMPLES = 128
WELCH_NPERSEG = 4096
LOW_BAND = (40.0, 500.0)
HIGH_BAND = (520.0, 1000.0)
PEAK_BAND = (40.0, 1000.0)
# Predefined criterion (docs/experiments/M13_0c_whitening.md, section 4).
FLOOR_RANGE = (0.8, 1.25)
PEAK_MAX = 10.0
MIN_ROW_FRACTION = 0.95


def dataset_id(duration):
    return f"M13_G3_wsweep_{duration:g}s"


def context_samples(spec_g1, duration):
    """Context per side = SignalProcessor.corrupted_margin_samples_per_side() + slack."""
    from src.config import SimulationConfig
    from src.processing import SignalProcessor
    sim = dict(spec_g1["simulation"])
    # Provisional context only so that a long filter passes validation; the
    # corrupted margin depends on the filter settings, not on the context.
    provisional = int(math.ceil(duration * SAMPLE_RATE))
    sim["processing_context_start_samples"] = provisional
    sim["processing_context_end_samples"] = provisional
    sp = dict(spec_g1["signal_processor"], lowpass_frequency=LOWPASS_HZ,
              whitening_max_filter_duration=float(duration))
    processor = SignalProcessor(config=SimulationConfig(**sim), **sp)
    return processor.corrupted_margin_samples_per_side() + CONTEXT_SLACK_SAMPLES


def m13_config(duration, context):
    return {
        "experiment_id": dataset_id(duration), "dataset_id": dataset_id(duration),
        "base_g1_h5": BASE_G1_H5,
        "overrides": {
            "signal_processor.lowpass_frequency": LOWPASS_HZ,
            "signal_processor.whitening_max_filter_duration": float(duration),
            "simulation.processing_context_start_samples": int(context),
            "simulation.processing_context_end_samples": int(context),
        },
    }


def run_pilot(config_path, pilot_path, args):
    cmd = [sys.executable, str(GENERATOR), "--config", str(config_path), "--pilot", str(PILOT_ROWS)]
    if args.data_root is not None:
        cmd += ["--data-root", str(args.data_root)]
    if args.allow_version_mismatch:
        cmd.append("--allow-version-mismatch")
    if pilot_path.exists():
        # The generator verifies attributes and only fills rows with status 0.
        print(f"Pilot exists, resuming: {pilot_path}", flush=True)
        cmd.append("--resume")
    print("Running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def band_mask(f, band):
    return (f >= band[0]) & (f < band[1])


def analyse_pilot(path):
    from scipy.signal import welch
    with h5py.File(path, "r") as h:
        require(bool((h["status"][:] == 1).all()), f"Pilot incomplete: {path}")
        X = h["X"][:].astype(np.float64)
        generated = int(h.attrs.get("pilot_generated_rows", 0))
        seconds = float(h.attrs.get("pilot_generation_seconds", np.nan))
    f, psd = welch(X, fs=SAMPLE_RATE, nperseg=WELCH_NPERSEG, axis=-1)  # (rows, detectors, freqs)
    floor_ratio = (np.median(psd[..., band_mask(f, HIGH_BAND)], axis=-1)
                   / np.median(psd[..., band_mask(f, LOW_BAND)], axis=-1))
    peak = psd[..., band_mask(f, PEAK_BAND)]
    peak_ratio = peak.max(axis=-1) / np.median(peak, axis=-1)
    seconds_per_row = seconds / generated if generated else np.nan
    return f, np.median(psd, axis=0), floor_ratio, peak_ratio, seconds_per_row


def plot_median_psds(path, f, median_psds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(DETECTORS), 1, figsize=(9, 9), sharex=True)
    for k, (ax, ifo) in enumerate(zip(axes, DETECTORS)):
        for duration, psd in median_psds.items():
            ax.semilogy(f, psd[k], lw=0.8, label=f"{duration:g} s")
        for band in (LOW_BAND, HIGH_BAND):
            ax.axvspan(*band, color="0.9", zorder=0)
        ax.set_xlim(20, 1100)
        ax.set_ylabel(f"{ifo} median PSD of X")
    axes[0].legend(title="whitening filter", ncol=len(median_psds), fontsize=8)
    axes[-1].set_xlabel("Frequency [Hz]")
    fig.suptitle(f"M13.0c: median Welch PSD over {PILOT_ROWS} pilot rows (lowpass {LOWPASS_HZ:g} Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args()

    root = resolve_data_root(cli_data_root=args.data_root)
    out_dir = root / "results" / RESULTS_NAME
    if out_dir.exists():
        raise FileExistsError(f"Results directory exists: {out_dir}")
    with h5py.File(root / "processed" / BASE_G1_H5, "r") as h:
        spec_g1 = json.loads(h.attrs["config_json"])

    contexts = {duration: context_samples(spec_g1, duration) for duration in DURATIONS_S}
    assert contexts[0.5] == 1664, f"Context at 0.5 s must be 1664 samples (G1 slack); got {contexts[0.5]}"
    print("G1 context per side: "
          f"{spec_g1['simulation']['processing_context_start_samples']}/"
          f"{spec_g1['simulation']['processing_context_end_samples']} samples; sweep contexts: "
          + ", ".join(f"{d:g} s -> {c}" for d, c in contexts.items()), flush=True)

    (out_dir / "configs").mkdir(parents=True)
    pilots = {}
    for duration in DURATIONS_S:
        config_path = out_dir / "configs" / f"{dataset_id(duration)}.json"
        config_path.write_text(json.dumps(m13_config(duration, contexts[duration]), indent=2) + "\n")
        pilots[duration] = dataset_processed_dir(root, dataset_id(duration)) / f"pilot_{PILOT_ROWS}.h5"
        run_pilot(config_path, pilots[duration], args)

    records, median_psds, passes = [], {}, {}
    f = None
    for duration in DURATIONS_S:
        f, median_psds[duration], floor_ratio, peak_ratio, seconds_per_row = analyse_pilot(pilots[duration])
        detector_pass = []
        for k, ifo in enumerate(DETECTORS):
            floor_ok = (floor_ratio[:, k] >= FLOOR_RANGE[0]) & (floor_ratio[:, k] <= FLOOR_RANGE[1])
            peak_ok = peak_ratio[:, k] <= PEAK_MAX
            ok = bool(floor_ok.mean() >= MIN_ROW_FRACTION and peak_ok.mean() >= MIN_ROW_FRACTION)
            detector_pass.append(ok)
            records.append({
                "duration_s": duration, "context_samples_per_side": contexts[duration],
                "detector": ifo, "n_rows": len(floor_ratio),
                "floor_ratio_median": float(np.median(floor_ratio[:, k])),
                "floor_ratio_q05": float(np.quantile(floor_ratio[:, k], 0.05)),
                "floor_ratio_q95": float(np.quantile(floor_ratio[:, k], 0.95)),
                "frac_floor_in_range": float(floor_ok.mean()),
                "peak_ratio_median": float(np.median(peak_ratio[:, k])),
                "peak_ratio_max": float(np.max(peak_ratio[:, k])),
                "frac_peak_gt10": float(1.0 - peak_ok.mean()),
                "seconds_per_row": seconds_per_row,
                "detector_pass": ok,
            })
        passes[duration] = all(detector_pass)

    table = pd.DataFrame.from_records(records)
    table.to_csv(out_dir / "sweep.csv", index=False)
    plot_median_psds(out_dir / "median_psd.png", f, median_psds)
    passing = [duration for duration in DURATIONS_S if passes[duration]]
    chosen = min(passing) if passing else "NONE"
    summary = {
        "experiment": "M13.0c whitening sweep",
        "criterion": (f"shortest duration such that, in all of {list(DETECTORS)}, floor_ratio in "
                      f"[{FLOOR_RANGE[0]}, {FLOOR_RANGE[1]}] for >= {MIN_ROW_FRACTION:.0%} of rows "
                      f"and peak_ratio <= {PEAK_MAX:g} for >= {MIN_ROW_FRACTION:.0%} of rows; else NONE"),
        "floor_ratio": f"median PSD in {list(HIGH_BAND)} Hz / median PSD in {list(LOW_BAND)} Hz",
        "peak_ratio": f"max PSD in {list(PEAK_BAND)} Hz / median PSD in {list(PEAK_BAND)} Hz",
        "welch_nperseg": WELCH_NPERSEG, "lowpass_frequency_hz": LOWPASS_HZ, "pilot_rows": PILOT_ROWS,
        "context_samples_per_side": {f"{d:g}": c for d, c in contexts.items()},
        "passes": {f"{d:g}": passes[d] for d in DURATIONS_S},
        "chosen_whitening_max_filter_duration_s": chosen,
        "pilots": {f"{d:g}": str(p) for d, p in pilots.items()},
        "generator_sha256": sha256_file(GENERATOR),
        "sweep_sha256": sha256_file(__file__),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string(index=False), flush=True)
    print(f"Chosen whitening_max_filter_duration: {chosen}", flush=True)
    print(f"Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
