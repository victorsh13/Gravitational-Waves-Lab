"""M13.0c (complemento): fraccion de varianza de X en lineas espectrales residuales.

Solo lectura. Lee las 64 filas de cada piloto del barrido y las 64 primeras de G1.
Uso:
  python m13_0c_line_power.py --data-root /data/vserrano/cbc_pe_data
  python m13_0c_line_power.py --data-root ... --durations 8s 16s --no-g1 --out-name line_power_16s
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

FS = 4096.0
NPERSEG = 4096  # 1 Hz, igual que el barrido
DETS = ("H1", "L1", "V1")
BANDS = {"40_500": (40.0, 500.0), "40_1000": (40.0, 1000.0)}
NULL_Q = 0.999  # cuantil del cociente P/mediana en ruido blanco puro
N_NULL = 2000
SEED = 12345


def psd_rows(x):
    """x: (n, 3, 16384) -> f, P (n, 3, nf)."""
    f, p = welch(x, fs=FS, nperseg=NPERSEG, axis=-1)
    return f, p


def null_threshold(n_samples):
    """Umbral T: cuantil NULL_Q de P/mediana_banda en ruido blanco con el mismo Welch."""
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((N_NULL, n_samples))
    f, p = welch(x, fs=FS, nperseg=NPERSEG, axis=-1)
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        r = p[:, m] / np.median(p[:, m], axis=1, keepdims=True)
        out[name] = float(np.quantile(r, NULL_Q))
    return out


def line_metrics(f, p, thr):
    """Por fila y detector: fraccion de la varianza total de X en bins de linea.

    floor = mediana de P en la banda; bins de linea = P > T*floor;
    exceso = sum(P - floor) en esos bins; fraccion = exceso / sum(P sobre todo f).
    """
    total = p.sum(axis=-1)  # varianza total (salvo constante comun)
    res = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        pb = p[..., m]
        floor = np.median(pb, axis=-1, keepdims=True)
        is_line = pb > thr[name] * floor
        excess = np.where(is_line, pb - floor, 0.0).sum(axis=-1)
        res[name] = {
            "line_fraction": excess / total,
            "n_line_bins": is_line.sum(axis=-1),
            "peak_ratio": (pb / floor).max(axis=-1),
            "peak_freq": f[m][np.argmax(pb, axis=-1)],
        }
    return res


def _top_freqs(pf, k=5):
    v, c = np.unique(np.round(pf), return_counts=True)
    o = np.argsort(-c)[:k]
    return [f"{v[i]:.0f}Hz x{c[i]}" for i in o]


def summarize(label, res):
    rows = []
    for band, d in res.items():
        for j, det in enumerate(DETS):
            lf = d["line_fraction"][:, j]
            rows.append({
                "dataset": label, "band": band, "detector": det,
                "line_frac_median": float(np.median(lf)),
                "line_frac_q95": float(np.quantile(lf, 0.95)),
                "line_frac_max": float(lf.max()),
                "frac_rows_line_frac_le_0.02": float(np.mean(lf <= 0.02)),
                "n_line_bins_median": float(np.median(d["n_line_bins"][:, j])),
                "peak_ratio_median": float(np.median(d["peak_ratio"][:, j])),
                "top_peak_freqs_Hz": _top_freqs(d["peak_freq"][:, j]),
            })
    return rows


def main():
    import h5py
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--durations", nargs="+", default=["0.5s", "2s", "4s", "8s"])
    ap.add_argument("--no-g1", action="store_true")
    ap.add_argument("--out-name", default="line_power")
    a = ap.parse_args()
    root = Path(a.data_root)
    out_dir = root / "results" / "M13_0c_whitening" / a.out_name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} ya existe; no se sobrescribe.")

    sources = {}
    if not a.no_g1:
        sources["G1_0.5s_lp512"] = root / "processed/M11_final_G1_500k/m11_final_G1_500k.h5"
    for d in a.durations:
        sources[f"G3_{d}_lp1024"] = root / f"processed/M13_G3_wsweep_{d}/pilot_64.h5"

    thr = None
    all_rows = []
    for label, path in sources.items():
        with h5py.File(path, "r") as h:
            x = h["X"][: a.n].astype(np.float64)
        if thr is None:
            thr = null_threshold(x.shape[-1])
            print("Umbral nulo (P/mediana, cuantil %.3f):" % NULL_Q, thr)
        f, p = psd_rows(x)
        res = line_metrics(f, p, thr)
        if label.startswith("G1"):
            res = {"40_500": res["40_500"]}  # G1 tiene paso bajo a 512 Hz
        all_rows += summarize(label, res)

    # Referencia: el mismo metodo sobre ruido blanco puro (sesgo del estimador).
    rng = np.random.default_rng(SEED + 1)
    xw = rng.standard_normal((a.n, 3, 16384))
    f, p = psd_rows(xw)
    all_rows += summarize("white_noise_null", line_metrics(f, p, thr))

    df = pd.DataFrame(all_rows)
    out_dir.mkdir(parents=True)
    df.to_csv(out_dir / "line_power.csv", index=False)
    (out_dir / "null_threshold.json").write_text(json.dumps(thr, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.drop(columns=["top_peak_freqs_Hz"]).to_string(index=False))
        print()
        print(df[["dataset", "band", "detector", "top_peak_freqs_Hz"]].to_string(index=False))
    print(f"\nResultados: {out_dir}")


if __name__ == "__main__":
    main()
