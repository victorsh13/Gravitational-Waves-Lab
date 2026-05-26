from __future__ import annotations
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "cbc_pe/data/raw"


"""
Data structure:
------------------------------------------
X:
    signal with shape (n_samples, n_detectors, signal_length)
y:
    labels with shape (n_samples, n_detectors). (Chirp mass, total mass, chi_eff)
injection_times:
    injection times of the signal with shape (n_samples,)
network_snrs: 
    network SNR of the signal (n_samples,)
detector_names: 
    list of detector names (["H1", "L1", "V1"])
parameters: 
    "params_x" used to build the gw signal.
"""


data = np.load(DATA_DIR/ "cbc_100_m1_20_m2_20.npz")

# Print the keys of the data dictionary
print(data.files)

# Extract the keys that start with "params_" into a dictionary
params = {
    k.replace("params_", ""): data[k]
    for k in data.files
    if k.startswith("params_")
}

signals = data["X"]
labels = data["y"]

distances = params["distance"]
print(signals.shape) ## (100, 3, 16384) # 100 samples, 3 detectors ("H1", "L1", "V1"), 16384 sample length.
print(labels.shape) ## (100, 3) # 100 samples, 3 labels (chirp mass, total mass, chi_eff).
print(distances.shape) ## (100,) # 100 samples.






