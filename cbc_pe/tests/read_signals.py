from pathlib import Path
import h5py

path = Path("/scratch/vserrano/cbc_pe_data/processed/gw_only/bbh_gw_only_32s_m1-20_m2-20_n100_01.h5")

with h5py.File(path, "r") as f:
    print(list(f.keys()))

    print("X:", f["X"].shape)
    print("mass_1:", f["parameters/mass_1"][:])
    print("mass_2:", f["parameters/mass_2"][:])

    print("placement keys:", list(f["placement"].keys()))
    print("windowing keys:", list(f["windowing"].keys()))
    print("projection keys:", list(f["projection"].keys()))
    print("injection H1 keys:", list(f["injection/H1"].keys()))
    print("snr keys:", list(f["snr"].keys()))

    print("snr/network:", f["snr/network"][:])
    print("snr/initial_network:", f["snr/initial_network"][:])
    print("snr/target_network:", f["snr/target_network"][:])
    print("distance_before:", f["snr/distance_before_rescale"][:])
    print("distance_after:", f["snr/distance_after_rescale"][:])

    #print("H1 signal_start_index:", f["injection/H1/signal_start_index"][:])
    #print("H1 overlap_start_index_strain:", f["injection/H1/overlap_start_index_strain"][:])

