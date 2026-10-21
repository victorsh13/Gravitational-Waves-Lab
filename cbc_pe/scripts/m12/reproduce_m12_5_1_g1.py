#!/usr/bin/env python3
"""Opt-in historical G1 reproduction gate. Only train rows 0, 1, 5830.

Fixed thresholds: allclose(atol=rtol=1e-5), relative L2 <=1e-6;
GPS metadata tolerance 1 microsecond. These are numerical checks, not
statistical confidence statements. A failed receipt cannot authorize a pilot.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.m12 import m12_5_common as c
from src.paths import resolve_data_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly generate three train rows")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("Reproduction is opt-in: add --run")
    root = resolve_data_root(cli_data_root=args.data_root)
    out = c.output_directory(root, args.output_dir)
    report = dict(schema="M12.5.1-gate-v1", passed=False, rows=list(c.ROWS),
                  tolerances=dict(atol=c.ATOL, rtol=c.RTOL, relative_l2=c.REL_RMSE),
                  code=c.code_identity(), checks=[], psd_interval_hashes={})
    try:
        import h5py
        import numpy as np
        from scripts.m12.run_m12_5_1_pilot import prepare_signal, observation
        g, spec, rows, stored, files, provenance = c.load_frozen(root)
        report["provenance"] = provenance
        g.initialize_physics(spec)
        with h5py.File(provenance["dataset"]["path"], "r") as h:
            for i in c.ROWS:
                strains = g.read_psd_intervals(rows[i], files)
                report["psd_interval_hashes"][str(i)] = {
                    d: c.array_digest(strains[d].numpy()) for d in c.DETECTORS}
                regenerated, metadata = g.build_g1_row(
                    rows[i], long_strains=strains, psd_cache={})
                check = c.compare_x(regenerated, h["X"][i])
                check["metadata"] = c.metadata_comparison(metadata, stored[i])
                # Test the pilot's factored orchestration against the historical
                # path, not merely historical code against itself.
                psds = g.get_block_psds(rows[i], long_strains=strains, psd_cache={})
                injector, signal, pilot_meta = prepare_signal(g, rows[i], psds)
                rng = np.random.default_rng(int(rows[i]["gaussian_seed"]))
                seeds = [int(rng.integers(0, 2**32 - 1)) for _ in c.DETECTORS]
                adapted, _ = observation(g, rows[i], psds, injector, signal, seeds)
                check["pilot_adapter"] = c.compare_x(adapted, regenerated)
                check["pilot_metadata"] = c.metadata_comparison(pilot_meta, stored[i])
                check["row_id"] = i
                check["passed"] = check["passed"] and all(
                    item["passed"] for item in check["metadata"].values()
                ) and check["pilot_adapter"]["passed"] and all(
                    item["passed"] for item in check["pilot_metadata"].values())
                report["checks"].append(check)
                print(f"train row {i}: passed={check['passed']}", flush=True)
        report["passed"] = all(item["passed"] for item in report["checks"])
        c.require(report["passed"], "G1 reproduction failed; pilot is blocked")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        c.write_json(out / "gate.json", report)


if __name__ == "__main__":
    main()
