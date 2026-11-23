"""CLI de gwsim.

  python -m gwsim generate --config C --output F.h5 [--workers N] [--resume] [--n-samples N] [--psd-pack P]
  python -m gwsim info F.h5

Con --workers N > 1 las filas se reparten en bloques (agrupados por bloque de ruido, para reutilizar las
PSD) entre N procesos creados con fork; solo el proceso padre escribe en el HDF5.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

CHUNK_ROWS = 8
_STATE = {}  # fijado en el padre antes del fork


def _run_chunk(indices):
    from .recipe import simulate_row
    from .io import STATUS_DONE, STATUS_FAILED
    rows, noise, opts = _STATE["rows"], _STATE["noise"], _STATE["opts"]
    mode = opts.mode
    results = []
    for i in indices:
        row = rows[i]
        try:
            psds = noise.psds(row["block_id"], opts.physics.config)
            outputs, meta = simulate_row(
                row, row["gps"], psds, row["placement_seed"], row["gaussian_seed"],
                target_snr=row["target_network_snr"] if mode == "snr" else None,
                distance=row["distance_input"] if mode == "distance" else None,
                opts=opts)
            results.append((i, STATUS_DONE, outputs, meta, ""))
        except Exception as exc:  # una fila que falla no para la generación
            from .physics.clickfree import AlignmentError
            message = str(exc) if isinstance(exc, AlignmentError) else f"{type(exc).__name__}: {exc}"
            results.append((i, STATUS_FAILED, None, {}, message))
    return results


def _chunks(rows, pending):
    """Bloques de como mucho CHUNK_ROWS filas del mismo bloque de ruido, en orden de bloque e índice."""
    order = sorted(pending, key=lambda i: (str(rows[i]["block_id"]), i))
    chunks, current = [], []
    for i in order:
        if current and (len(current) == CHUNK_ROWS or rows[i]["block_id"] != rows[current[0]]["block_id"]):
            chunks.append(current)
            current = []
        current.append(i)
    if current:
        chunks.append(current)
    return chunks


def cmd_generate(args):
    import h5py
    import numpy as np
    from .config import ConfigError, config_json, load_config
    from .io import (STATUS_DONE, STATUS_FAILED, STATUS_PENDING, create_output, read_plan_rows,
                     verify_for_resume, write_results)
    from .noise import make_noise_source
    from .priors import build_plan
    from .recipe import Physics, RowOptions

    try:
        cfg, warnings = load_config(args.config, {"n_samples": args.n_samples, "psd_pack": args.psd_pack})
    except ConfigError as exc:
        sys.exit(f"Error de configuración: {exc}")
    for w in warnings:
        print(f"AVISO: {w}", flush=True)
    if args.workers < 1:
        sys.exit("--workers debe ser >= 1")
    output = Path(args.output)
    text = config_json(cfg)
    noise = make_noise_source(cfg["noise"])
    physics = Physics(cfg["simulation"], cfg["signal_processor"])

    if args.resume:
        if not output.exists():
            sys.exit(f"--resume: no existe {output}")
    else:
        if output.exists():
            sys.exit(f"{output} ya existe; usa --resume para continuar o elige otra salida")
        plan = build_plan(cfg, noise)
        output.parent.mkdir(parents=True, exist_ok=True)
        create_output(output, plan, cfg["outputs"], physics.config.length,
                      {"config_json": text, "warnings": json.dumps(warnings, ensure_ascii=False)})
        print(f"Creado {output}: {len(plan)} filas", flush=True)

    with h5py.File(output, "r+") as h:
        try:
            verify_for_resume(h, text)
        except ValueError as exc:
            sys.exit(str(exc))
        status = h["status"][:]
        pending = np.flatnonzero(status == STATUS_PENDING)
        n = len(status)
        if pending.size == 0:
            print("Nada pendiente.", flush=True)
            return
        rows = {r["row_index"]: r for r in read_plan_rows(h, pending)}
        _STATE.update(rows=rows, noise=noise,
                      opts=RowOptions(physics=physics, mode=cfg["mode"], outputs=tuple(cfg["outputs"]),
                                      click_free=cfg["click_free"]))
        chunks = _chunks(rows, pending.tolist())
        done = int((status == STATUS_DONE).sum())
        failed = int((status == STATUS_FAILED).sum())
        print(f"{pending.size} filas pendientes de {n}; {args.workers} proceso(s)", flush=True)
        t0 = time.monotonic()

        def consume(results):
            nonlocal done, failed
            write_results(h, results)
            for i, s, _, _, err in results:
                if s == STATUS_DONE:
                    done += 1
                else:
                    failed += 1
                    print(f"  fila {i} FALLIDA: {err}", flush=True)
            elapsed = time.monotonic() - t0
            print(f"{done + failed}/{n} (hechas {done}, fallidas {failed}); {elapsed:.0f} s", flush=True)

        if args.workers == 1:
            for chunk in chunks:
                consume(_run_chunk(chunk))
        else:
            with mp.get_context("fork").Pool(args.workers) as pool:
                for results in pool.imap_unordered(_run_chunk, chunks):
                    consume(results)
    print(f"Terminado: {output} (hechas {done}, fallidas {failed})", flush=True)


def cmd_info(args):
    import h5py
    import numpy as np
    with h5py.File(args.file, "r") as h:
        cfg = json.loads(h.attrs["config_json"])
        status = h["status"][:]
        print(f"{args.file}: gwsim {h.attrs['gwsim_version']}, creado {h.attrs['created_utc']}")
        print(f"  modo {cfg['mode']}, ruido {cfg['noise']['source']}, click_free {cfg['click_free']}, "
              f"seed {cfg['seed']}")
        print(f"  filas {len(status)}: hechas {int((status == 1).sum())}, fallidas {int((status == 2).sum())}, "
              f"pendientes {int((status == 0).sum())}")
        for name in h.attrs["outputs"].split(","):
            print(f"  {name}: {h[name].shape} {h[name].dtype}")
        ok = status == 1
        if ok.any():
            for name in ("total_mass", "final_network_snr", "final_distance"):
                v = h[name][:][ok]
                if np.isfinite(v).any():
                    print(f"  {name}: mín {np.nanmin(v):.4g}, mediana {np.nanmedian(v):.4g}, máx {np.nanmax(v):.4g}")
        for w in json.loads(h.attrs.get("warnings", "[]")):
            print(f"  AVISO: {w}")
        errors = h["error"].asstr()[:][status == 2]
        for i, err in list(zip(np.flatnonzero(status == 2), errors))[:10]:
            print(f"  fila {i}: {err}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m gwsim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Genera (o continúa) un fichero HDF5")
    gen.add_argument("--config", required=True, type=Path)
    gen.add_argument("--output", required=True, type=Path)
    gen.add_argument("--workers", type=int, default=1)
    gen.add_argument("--resume", action="store_true", help="Rehace las filas con status = 0")
    gen.add_argument("--n-samples", type=int, help="Sustituye a n_samples de la configuración")
    gen.add_argument("--psd-pack", type=Path, help="Sustituye a noise.psd_pack de la configuración")
    info = sub.add_parser("info", help="Resumen de un fichero generado")
    info.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        cmd_generate(args)
    else:
        cmd_info(args)


if __name__ == "__main__":
    main()
