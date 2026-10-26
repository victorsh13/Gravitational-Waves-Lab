"""Crea la config del piloto de 16 s a partir de la de 8 s del barrido M13.0c.

Contexto por lado = ceil(16*4096/2) + 2*256 (FIR) + 128 (holgura de G1) = 33408 muestras.
Uso: python make_16s_config.py <configs_dir>
"""
import json
import sys
from pathlib import Path

NEW = {
    "whitening_max_filter_duration": 16.0,
    "processing_context_start_samples": 33408,
    "processing_context_end_samples": 33408,
}
OLD_ID, NEW_ID = "M13_G3_wsweep_8s", "M13_G3_wsweep_16s"

cfg_dir = Path(sys.argv[1])
src, dst = cfg_dir / f"{OLD_ID}.json", cfg_dir / f"{NEW_ID}.json"
if dst.exists():
    raise SystemExit(f"{dst} ya existe")
cfg = json.loads(src.read_text())
hits = {k: 0 for k in NEW}
n_id = 0


def walk(o):
    global n_id
    if isinstance(o, dict):
        for k, v in o.items():
            base = k.split(".")[-1]
            if base in NEW:
                hits[base] += 1
                o[k] = NEW[base]
            elif isinstance(v, str) and OLD_ID in v:
                o[k] = v.replace(OLD_ID, NEW_ID)
                n_id += 1
            else:
                walk(v)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            if isinstance(v, str) and OLD_ID in v:
                o[i] = v.replace(OLD_ID, NEW_ID)
                n_id += 1
            else:
                walk(v)


walk(cfg)


def leftovers(o, path=""):
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            out += leftovers(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out += leftovers(v, f"{path}[{i}]")
    elif o == 17024 or (isinstance(o, str) and OLD_ID in o):
        out.append(path)
    return out


left = leftovers(cfg)
if left:
    raise SystemExit(f"Quedan valores de 8 s en {left}. No se escribe nada.")
bad = {k: n for k, n in hits.items() if n < 1}
if bad or n_id == 0:
    raise SystemExit(f"Estructura inesperada: claves {hits}, sustituciones de id {n_id}. No se escribe nada.")
dst.write_text(json.dumps(cfg, indent=2))
print(f"Escrito {dst}; claves {hits}; sustituciones de id {n_id}")
