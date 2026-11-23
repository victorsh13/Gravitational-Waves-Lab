"""gwsim: señales de coalescencias binarias (CBC) en ruido gaussiano coloreado.

Uso: python -m gwsim generate --config C --output F.h5 [--workers N] [--resume]
     python -m gwsim info F.h5
     python -m gwsim selftest
Desde Python: gwsim.load, gwsim.estimate_size; from gwsim.recipe import simulate_row, Physics, RowOptions
(ver docs/API.md).
"""
__version__ = "0.1.0"


def load(path, outputs=("X",), only_ok=True):
    """Dict de arrays con las salidas pedidas y todos los metadatos (por defecto, solo filas con status == 1)."""
    from .io import load as _load
    return _load(path, outputs=outputs, only_ok=only_ok)


def estimate_size(config, n):
    """Bytes previstos del HDF5 con n filas, según las salidas pedidas (sin comprimir, float32)."""
    from .io import estimate_size as _estimate_size
    return _estimate_size(config, n)
