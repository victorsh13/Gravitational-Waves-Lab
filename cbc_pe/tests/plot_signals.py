from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt

path = Path("/scratch/vserrano/cbc_pe_data/processed/gw_only/bbh_gw_only_32s_m1-20_m2-20_n100_01.h5")
path = Path("/scratch/vserrano/cbc_pe_data/processed/gw_only/bbh_gw_only_32s_m1-20_m2-20_n100_01.h5")

with h5py.File(path, "r") as f:
    
    print(list(f.keys()))
    print("X:", f["X"].shape)
    #print("mass_1:", f["parameters/mass_1"][:])
    #print("mass_2:", f["parameters/mass_2"][:])

    print("generation keys", list(f["generation"].keys()))
    print("placement keys:", list(f["placement"].keys()))
    print("windowing keys:", list(f["windowing"].keys()))
    print("projection keys:", list(f["projection"].keys()))
    print("injection H1 keys:", list(f["injection/H1"].keys()))
    print("snr keys:", list(f["snr"].keys()))

    strain = f["X"]
    print(len(strain[0][0]))
    time = np.linspace(0, 32, len(strain[0][0]))
    
    for i in range(1):
       plt.figure(figsize=(14,16))
       plt.plot(time ,strain[1][i])
       plt.xlim(0,32)
       plt.show()


