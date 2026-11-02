#!/usr/bin/env python3
"""M13.7: locate (and optionally download) GWTC-3 LVK posterior samples for the M11_14 cohort.

Cohort: the frozen M10/M11_14 candidate list
    <data_root>/results/<M10 dataset_id>/real_events_m10_500k_lvk_comparison/
        multi_event_all_catalog_candidates.csv
(GWTC-3-confident, 14 events), restricted to events with H1+L1+V1 as in M11_14. No catalog query.

Source: Zenodo record 8177023 (GWTC-3 PE data release, v2), via https://zenodo.org/api/records/8177023.
Per event the file `IGWN-GWTC3p0-v2-<event>_PEDataRelease_mixed_cosmo.h5` is selected:
  - "mixed": samples combined over the waveform models (IMRPhenomXPHM + SEOBNRv4PHM); this is the
    default analysis behind the GWTC-3 headline values (the GWOSC best values used in M10/M11).
  - "cosmo": distance prior uniform in comoving volume and source-frame time, as in the catalog
    tables; the "nocosmo" files use a Euclidean distance prior and are not used.
The h5 files are pesummary-format; the detector-frame chirp_mass/total_mass and chi_eff are read later
in the notebook (M13_7 section E), not here.

Default: --dry-run (table event -> file -> size, and total; nothing is written).
--download: resumable download (HTTP Range on <file>.part) with md5 check against the Zenodo checksum to
<data_root>/external/gwtc3_pe/. A complete file is renamed into place only after the checksum matches;
an existing final file is verified and never overwritten. Writes manifest_<UTC stamp>.json (mode 'x').
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import resolve_data_root
from src.real_data.catalog import has_required_detectors

ZENODO_RECORD = 8177023
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
FILE_VARIANT = "mixed_cosmo"
DETECTORS = ("H1", "L1", "V1")
M10_TRAIN_CONFIG = PROJECT_ROOT / "configs/experiments/train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json"
M11_TRAIN_CONFIG = PROJECT_ROOT / "configs/experiments/train_m11_final_G1_500k.json"
OUT_SUBDIR = Path("external/gwtc3_pe")
CHUNK = 1 << 20


def read_json(path):
    return json.loads(Path(path).read_text())


def cohort_path(data_root):
    dataset_id = read_json(M10_TRAIN_CONFIG)["dataset"]["dataset_id"]
    return (data_root / "results" / dataset_id / "real_events_m10_500k_lvk_comparison"
            / "multi_event_all_catalog_candidates.csv")


def load_cohort(path):
    events = pd.read_csv(path)
    if events.event.duplicated().any():
        raise SystemExit("Duplicate cohort events")
    if len(events) != 14:
        raise SystemExit(f"M10 frozen candidate cohort changed: {len(events)} events")
    keep = events.detectors.map(lambda d: has_required_detectors(d, DETECTORS))
    dropped = events.event[~keep].tolist()
    if dropped:
        print(f"Excluded (not {'+'.join(DETECTORS)}): {dropped}")
    return events[keep].reset_index(drop=True)


def zenodo_files():
    r = requests.get(ZENODO_API, timeout=60)
    r.raise_for_status()
    record = r.json()
    files = {}
    for f in record["files"]:
        algo, _, digest = f["checksum"].partition(":")
        if algo != "md5":
            raise SystemExit(f"Unexpected checksum type {f['checksum']!r} for {f['key']}")
        files[f["key"]] = dict(size=int(f["size"]), md5=digest, url=f["links"]["self"])
    meta = dict(record_id=record.get("id"), doi=record.get("doi"),
                version=record.get("metadata", {}).get("version"), title=record.get("metadata", {}).get("title"))
    return meta, files


def select_file(event, files):
    """Return the unique mixed_cosmo key for `event` (GWOSC name, optional -vN suffix), or None."""
    base = re.sub(r"-v\d+$", "", str(event))
    hits = [k for k in files if k.endswith(f"{base}_PEDataRelease_{FILE_VARIANT}.h5")]
    if len(hits) > 1:
        raise SystemExit(f"Ambiguous files for {event}: {hits}")
    return hits[0] if hits else None


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(key, info, out_dir):
    final, part = out_dir / key, out_dir / (key + ".part")
    if final.exists():
        if md5sum(final) != info["md5"]:
            raise SystemExit(f"{final} exists with wrong md5; not overwriting (inspect it manually)")
        print(f"  ok (already present) {key}")
        return "present"
    done = part.stat().st_size if part.exists() else 0
    if done > info["size"]:
        raise SystemExit(f"{part} larger than expected; remove it manually")
    if done < info["size"]:
        headers = {"Range": f"bytes={done}-"} if done else {}
        with requests.get(info["url"], headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            if done and r.status_code != 206:
                print(f"  server ignored Range for {key}; restarting from 0")
                done = 0
            with open(part, "ab" if done else "wb") as fh:
                for chunk in r.iter_content(CHUNK):
                    fh.write(chunk)
    if part.stat().st_size != info["size"]:
        raise SystemExit(f"Incomplete {part}: {part.stat().st_size} / {info['size']} bytes; rerun to resume")
    if md5sum(part) != info["md5"]:
        raise SystemExit(f"md5 mismatch for {part}; remove it manually and rerun")
    part.rename(final)
    print(f"  ok (downloaded) {key}")
    return "downloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default: list files, write nothing")
    mode.add_argument("--download", action="store_true")
    args = parser.parse_args()

    data_root = resolve_data_root(cli_data_root=args.data_root,
                                  config_data_root=read_json(M11_TRAIN_CONFIG)["data_root"])
    events = load_cohort(cohort_path(data_root))
    meta, files = zenodo_files()
    print(f"Zenodo record {meta['record_id']} (version {meta['version']}, doi {meta['doi']}): {meta['title']}")

    rows = [dict(event=e, file=select_file(e, files)) for e in events.event]
    table = pd.DataFrame(rows)
    table["size_MB"] = table.file.map(lambda k: files[k]["size"] / 1e6 if k else float("nan"))
    with pd.option_context("display.max_colwidth", None, "display.width", 200):
        print(table.to_string(index=False, float_format="%.1f"))
    missing = table.event[table.file.isna()].tolist()
    print(f"Total: {table.file.notna().sum()} files, {table.size_MB.sum() / 1e3:.2f} GB; missing: {missing}")

    if not args.download:
        print("Dry run: nothing written. Use --download to fetch.")
        return

    out_dir = data_root / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {out_dir}")
    status = {}
    for key in table.file.dropna():
        status[key] = download(key, files[key], out_dir)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest = dict(created_utc=stamp, zenodo=meta, variant=FILE_VARIANT, cohort_csv=str(cohort_path(data_root)),
                    detectors=list(DETECTORS), missing_events=missing,
                    files=[dict(event=r.event, key=r.file, size=files[r.file]["size"], md5=files[r.file]["md5"],
                                status=status[r.file]) for r in table.dropna(subset=["file"]).itertuples()])
    with open(out_dir / f"manifest_{stamp}.json", "x") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest: {out_dir / f'manifest_{stamp}.json'}")
    if missing:
        raise SystemExit(f"No {FILE_VARIANT} file found for: {missing}")


if __name__ == "__main__":
    main()
