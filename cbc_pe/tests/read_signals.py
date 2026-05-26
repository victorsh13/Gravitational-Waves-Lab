from pathlib import Path
import h5py
import numpy as np

path = Path("/scratch/vserrano/cbc_pe_data/processed/bbh_processed_32s_m1-20_m2-20_n2.h5")

with h5py.File(path, "r") as f:
    print("X shape:", f["X"].shape)
    print("duration:", f.attrs["duration"])
    print("sampling_frequency:", f.attrs["sampling_frequency"])
    print("length:", f.attrs["length"])

    print("mass_1:", f["parameters/mass_1"][:])
    print("mass_2:", f["parameters/mass_2"][:])
    print("spin_1z:", f["parameters/spin_1z"][:])
    print("spin_2z:", f["parameters/spin_2z"][:])

    print("total_mass:", f["parameters/total_mass"][:])
    print("chirp_mass:", f["parameters/chirp_mass"][:])
    print("chi_eff:", f["parameters/chi_eff"][:])

    print("network_snr:", f["snr/network"][:])


