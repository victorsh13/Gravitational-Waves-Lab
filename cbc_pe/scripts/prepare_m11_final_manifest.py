#!/usr/bin/env python3
"""Prepare M11-final source/assignment metadata only; never generate strain.

Port of M11_6_training_domain_comparison.ipynb, cells 165–187, using the
accepted B2 bank unchanged. Actual placement GPS requires a waveform and is
intentionally deferred; reference GPS, its policy and RNG seeds are frozen.
The JSON is a preparation contract, not input to a historical generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import dataset_processed_dir, resolve_data_root, resolve_processed_artifact
from src.sampling import ParameterSampler, PriorConfig

SPLITS = ("train", "val", "cal", "test")
PARAMETERS = (
    "mass_1", "mass_2", "spin_1z", "spin_2z", "inclination", "ra", "dec",
    "polarization_angle", "chirp_mass", "total_mass", "chi_eff",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_bank(config, data_root):
    """Fail closed if the accepted bank changes; do not redownload or rebuild it."""
    bank = config["bank"]
    tables = {}
    for key, name in bank["files"].items():
        path = resolve_processed_artifact(
            data_root=data_root, dataset_id=bank["dataset_id"],
            file_name=name, allow_legacy_flat=False,
        )
        require(hashlib.sha256(path.read_bytes()).hexdigest() == bank["sha256"][key],
                f"Frozen bank checksum mismatch: {path}")
        tables[key] = pd.read_csv(path)
    groups, blocks, crops = (tables[k] for k in ("groups", "blocks", "crops"))
    for table, column, count in (
        (groups, "file_group_id", bank["environment_count"]),
        (blocks, "block_id", bank["block_count"]),
        (crops, "noise_crop_id", bank["crop_count"]),
    ):
        require(len(table) == count and table[column].is_unique, f"Invalid {column} bank")
        require(set(table["split"]) == set(SPLITS), "Missing bank split")
        require(table.groupby("file_group_id")["split"].nunique().eq(1).all(),
                "Environment shared across splits")
    require(blocks["passes_basic_qc"].eq(True).all(), "Unaccepted block")
    group_splits = groups.set_index("file_group_id")["split"]
    require(blocks["file_group_id"].map(group_splits).eq(blocks["split"]).all(),
            "Block/group split mismatch")
    block_lookup = blocks.set_index("block_id")
    for column in ("file_group_id", "split", "psd_start", "psd_end"):
        require(crops["block_id"].map(block_lookup[column]).eq(crops[column]).all(),
                f"Crop/block mismatch: {column}")
    require(np.allclose(crops.processing_end - crops.processing_start, 4.8125,
                        rtol=0, atol=1e-6), "Invalid processing interval")
    require(np.allclose(crops.psd_end - crops.psd_start, 128, rtol=0, atol=1e-6),
            "Invalid PSD interval")
    require((crops.processing_start >= crops.psd_end + 8).all(), "PSD guard violation")
    # Same temporal-support audit as the M11.6 preproduction split notebook.
    supports = list(blocks.itertuples(index=False))
    for i, a in enumerate(supports):
        for b in supports[i + 1:]:
            if a.split != b.split:
                require(min(a.block_end, b.block_end) <= max(a.block_start, b.block_start),
                        "Cross-split temporal overlap")
    return crops


def sample_assignments(crops, counts, seed):
    """M11.6 hierarchy and RNG draw order, with cached groups instead of filtering per row."""
    rng = np.random.default_rng(seed)
    indices = []
    for split in SPLITS:
        selected = crops[crops["split"] == split]
        groups = np.array(sorted(selected.file_group_id.unique()), dtype=int)
        blocks = {}
        crop_indices = {}
        for group in groups:
            d = selected[selected.file_group_id == group]
            blocks[group] = np.array(sorted(d.block_id.unique()))
            for block in blocks[group]:
                crop_indices[group, block] = d[d.block_id == block].index.to_numpy()
        for _ in range(counts[split]):
            group = int(rng.choice(groups))
            block = rng.choice(blocks[group])
            available = crop_indices[group, block]
            indices.append(available[int(rng.integers(0, len(available)))])
    return crops.loc[indices].reset_index(drop=True)


def build_manifest(config, crops, counts):
    seeds = config["generation"]["seeds"]
    sampler = ParameterSampler(
        prior_config=PriorConfig.from_dict(config["parameter_sampler"]),
        rng=np.random.default_rng(seeds["sources"]),
    )
    rows = []
    for _ in range(sum(counts.values())):
        p = sampler.sample_one()
        rows.append({**{name: float(getattr(p, name)) for name in PARAMETERS},
                     "reference_distance_mpc": float(p.distance)})
    sources = pd.DataFrame(rows)
    n = len(sources)
    sources["source_index"] = np.arange(n)
    sources["manifest_index"] = np.arange(n)
    sources["source_id"] = [f'{config["output"]["source_id_prefix"]}{i:06d}' for i in range(n)]
    sources["target_network_snr"] = np.random.default_rng(seeds["snr_targets"]).uniform(
        *config["generation"]["target_network_snr_range"], size=n,
    )
    assignments = sample_assignments(crops, counts, seeds["environment_assignment"])
    manifest = pd.concat([sources, assignments], axis=1)
    # This is provisional, NOT the actual geocentric time later saved with strain.
    manifest["reference_geocentric_time"] = manifest["center_gps"]
    manifest["geocentric_time_policy"] = "m11_6_random_contained_reproject"
    manifest["placement_seed"] = seeds["placement_base"] + manifest.source_index
    manifest["gaussian_seed"] = seeds["gaussian_base"] + manifest.source_index
    manifest["experiment_id"] = config["experiment_id"]
    manifest["config_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(manifest.source_id.is_unique, "Duplicate source ID")
    require(manifest["split"].value_counts().to_dict() == counts, "Wrong split counts")
    require(manifest.groupby("file_group_id")["split"].nunique().eq(1).all(),
            "Environment leakage")
    require(manifest.target_network_snr.between(10, 25).all(), "Invalid SNR target")
    require((manifest.mass_1 >= manifest.mass_2).all(), "Unordered masses")
    require(manifest.reference_distance_mpc.eq(1000).all(), "Wrong reference distance")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]
                        / "configs/generation/generate_m11_final_G1_500k.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--validate-only", action="store_true",
                        help="Check the complete bank and 16 sources per split in memory; write nothing.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    counts = config["generation"]["split_counts"]
    require(counts == dict(zip(SPLITS, (400000, 40000, 30000, 30000))),
            "Unexpected final split counts")
    require(sum(counts.values()) == config["generation"]["num_samples"] == 500000,
            "Expected 500000 sources")
    require(config["simulation"]["snr_high_frequency_cutoff"] == 512.0,
            "M11-final requires an explicit 512 Hz SNR upper cutoff")
    data_root = resolve_data_root(cli_data_root=args.data_root,
                                  config_data_root=config.get("data_root"))
    output = dataset_processed_dir(data_root, config["output"]["dataset_id"]) / config["output"]["manifest_file"]
    if output.exists() and not args.validate_only:
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    crops = load_bank(config, data_root)
    run_counts = {split: 16 for split in SPLITS} if args.validate_only else counts
    manifest = build_manifest(config, crops, run_counts)
    print(f"Validated bank: {crops.file_group_id.nunique()} environments, "
          f"{crops.block_id.nunique()} blocks, {len(crops)} crops")
    print("Manifest split counts:", manifest["split"].value_counts().to_dict())
    print("Output manifest:", output)
    if args.validate_only:
        print("Validation only: no files written; actual GPS deferred to waveform placement.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects a frozen manifest even if another job starts concurrently.
    with output.open("x") as handle:
        manifest.to_csv(handle, index=False)
    print("Wrote metadata only. No strain, scaler, training or prediction artifacts generated.")


if __name__ == "__main__":
    main()
