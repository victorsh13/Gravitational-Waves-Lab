"""Fuentes de ruido: paquete de PSDs empíricas de O3b (psd_pack) o PSD analíticas (analytic).

Ambas devuelven, por bloque, {"snr": {ifo: PSD}, "proc": {ifo: PSD}}:
  - "snr": Δf = 1/duration (0,25 Hz), para la SNR óptima;
  - "proc": Δf = 1/processing_duration, para el ruido gaussiano y el blanqueado.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .physics.steps import DETECTORS, estimate_block_psds_for_detector, require

PACK_FORMAT_VERSION = 1
# Claves del .npz que escribe tools/maintenance/export_psd_pack.py.
PACK_KEYS = ("format_version", "detectors", "psd", "delta_f", "block_id", "split", "file_group_id",
             "psd_start", "psd_end", "crop_noise_crop_id", "crop_block_id", "crop_center_gps", "crop_split")
PACK_SAMPLE_RATE_HZ = 4096.0
ANALYTIC_PSD_MODELS = {"H1": "aLIGOZeroDetHighPower", "L1": "aLIGOZeroDetHighPower", "V1": "AdvVirgo"}
ANALYTIC_BLOCK_ID = "analytic"


class PsdPackNoise:
    """PSD de Welch por bloque y detector (antes de interpolar) y tabla de recortes de ruido."""

    source = "psd_pack"

    def __init__(self, path, splits):
        path = Path(path)
        require(path.is_file(), f"psd_pack not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            missing = [k for k in PACK_KEYS if k not in data]
            require(not missing, f"{path}: not a gwsim PSD pack (missing {missing})")
            arrays = {k: data[k] for k in PACK_KEYS}
        require(int(arrays["format_version"]) == PACK_FORMAT_VERSION,
                f"{path}: pack format {int(arrays['format_version'])}, expected {PACK_FORMAT_VERSION}")
        require(tuple(str(d) for d in arrays["detectors"]) == DETECTORS, f"{path}: detector order is not {DETECTORS}")
        self.path = path
        self.psd = arrays["psd"]                                   # (n_blocks, 3, n_freq) float64
        self.delta_f = arrays["delta_f"].astype(np.float64)        # (n_blocks,)
        self.block_id = np.array([str(b) for b in arrays["block_id"]])
        self.block_split = np.array([str(s) for s in arrays["split"]])
        self.block_group = arrays["file_group_id"].astype(np.int64)
        require(self.psd.dtype == np.float64 and self.psd.ndim == 3 and self.psd.shape[1] == 3,
                f"{path}: psd must be float64 (n_blocks, 3, n_freq)")
        self.block_index = {b: k for k, b in enumerate(self.block_id)}
        require(len(self.block_index) == len(self.block_id), f"{path}: duplicate block_id")

        available = sorted(set(self.block_split))
        splits = list(splits)
        bad = [s for s in splits if s not in available]
        require(not bad, f"noise.splits {bad} not in the pack (available: {available})")
        crop_split = np.array([str(s) for s in arrays["crop_split"]])
        keep = np.isin(crop_split, splits)
        require(keep.any(), f"No crops for splits {splits}")
        self.splits = splits
        self.crop_id = np.array([str(c) for c in arrays["crop_noise_crop_id"]])[keep]
        self.crop_block = np.array([str(b) for b in arrays["crop_block_id"]])[keep]
        self.crop_gps = arrays["crop_center_gps"].astype(np.float64)[keep]
        self.crop_split = crop_split[keep]
        require(all(b in self.block_index for b in self.crop_block), f"{path}: crop with unknown block_id")
        self.crop_group = self.block_group[[self.block_index[b] for b in self.crop_block]]
        self._cache_key = None
        self._cache = None

    # Copiado de scripts/prepare_m11_final_manifest.py:sample_assignments @ 4b34677; cambios: un solo
    # sorteo sobre la unión de los splits elegidos (n filas) en vez de un bucle por split con counts;
    # trabaja con arrays del paquete en vez del DataFrame de recortes; devuelve índices de recorte.
    def sample_assignments(self, n, rng):
        """Jerarquía: entorno (file_group_id) -> bloque -> recorte, uniforme en cada nivel."""
        groups = np.array(sorted(np.unique(self.crop_group)), dtype=int)
        blocks = {}
        crop_indices = {}
        for group in groups:
            in_group = self.crop_group == group
            blocks[group] = np.array(sorted(np.unique(self.crop_block[in_group])))
            for block in blocks[group]:
                crop_indices[group, block] = np.flatnonzero(in_group & (self.crop_block == block))
        indices = []
        for _ in range(int(n)):
            group = int(rng.choice(groups))
            block = rng.choice(blocks[group])
            available = crop_indices[group, block]
            indices.append(available[int(rng.integers(0, len(available)))])
        return np.asarray(indices, dtype=np.int64)

    def assign(self, n, rng):
        idx = self.sample_assignments(n, rng)
        return self.crops(idx)

    def crops(self, idx):
        return {"block_id": self.crop_block[idx], "noise_crop_id": self.crop_id[idx],
                "noise_split": self.crop_split[idx], "gps": self.crop_gps[idx]}

    def crops_by_id(self, crop_ids):
        lookup = {c: k for k, c in enumerate(self.crop_id)}
        bad = [c for c in crop_ids if str(c) not in lookup]
        require(not bad, f"noise_crop_id not in the selected splits of the pack: {bad[:5]}")
        return self.crops(np.array([lookup[str(c)] for c in crop_ids], dtype=np.int64))

    def psds(self, block_id, config):
        """PSD "snr" y "proc" de un bloque, interpoladas y recortadas como estimate_raw_offsource_psd."""
        from pycbc.types import FrequencySeries
        key = (str(block_id), config)
        if self._cache_key == key:
            return self._cache
        require(float(config.sampling_frequency) <= PACK_SAMPLE_RATE_HZ,
                "The PSD pack reaches 2048 Hz: sampling_frequency must be <= 4096 Hz")
        k = self.block_index[str(block_id)]
        out = {"snr": {}, "proc": {}}
        for j, ifo in enumerate(DETECTORS):
            welch = FrequencySeries(np.array(self.psd[k, j], dtype=np.float64), delta_f=float(self.delta_f[k]))
            result = estimate_block_psds_for_detector(config, welch_psd=welch)
            out["snr"][ifo] = result["snr"]
            out["proc"][ifo] = result["proc"]
        self._cache_key, self._cache = key, out
        return out


class AnalyticNoise:
    """aLIGOZeroDetHighPower (H1, L1) y AdvVirgo (V1), como src/noise.py, con el Δf de cada PSD."""

    source = "analytic"

    def __init__(self, gps_range):
        lo, hi = (float(x) for x in gps_range)
        require(lo < hi, "noise.gps_range must be [min, max] with min < max")
        self.gps_range = (lo, hi)
        self._cache_key = None
        self._cache = None

    def assign(self, n, rng):
        n = int(n)
        return {"block_id": np.array([ANALYTIC_BLOCK_ID] * n), "noise_crop_id": np.array([""] * n),
                "noise_split": np.array([""] * n), "gps": rng.uniform(*self.gps_range, size=n)}

    def psds(self, block_id, config):
        from .physics.noise import NoiseModel
        if self._cache_key == config:
            return self._cache
        model = NoiseModel(config)
        out = {"snr": {ifo: model.get_psd(ifo, length=config.length) for ifo in DETECTORS},
               "proc": {ifo: model.get_psd(ifo, length=config.processing_length) for ifo in DETECTORS}}
        self._cache_key, self._cache = config, out
        return out


def make_noise_source(noise_cfg):
    if noise_cfg["source"] == "psd_pack":
        return PsdPackNoise(noise_cfg["psd_pack"], noise_cfg["splits"])
    return AnalyticNoise(noise_cfg["gps_range"])
