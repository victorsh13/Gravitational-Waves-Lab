#!/usr/bin/env python3
"""Generate only the frozen M11.15B embedding caches on the GPU VM.

Run in cbc_torch: python scripts/generate_m11_15b_latent_embeddings.py --all
Existing complete caches are validated and skipped. --force regenerates selected
caches. Valid partial synthetic caches resume missing cross terms; incomplete
real-event caches still require --force.
No diagonal inference, training, Mondrian fitting or latent analysis is performed.
G0 uses processed/M11_15B_temp/G0_test30k_model_input_source.h5 when present,
otherwise the original full G0 dataset. Both sources require one input z-score.
"""
import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic-cross", action="store_true", help="Generate E10(G1) and E11(G0)")
    parser.add_argument("--real-events", action="store_true", help="Generate both encoders for the frozen nine events")
    parser.add_argument("--all", action="store_true", help="Generate both cache artifacts")
    parser.add_argument("--force", action="store_true", help="Regenerate selected caches, including invalid/incomplete ones")
    parser.add_argument("--batch-size", type=int, default=128, help="CUDA inference batch size (default: 128)")
    args = parser.parse_args(argv)
    if not (args.synthetic_cross or args.real_events or args.all):
        parser.error("Select --synthetic-cross, --real-events or --all")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    want_cross = args.synthetic_cross or args.all
    want_real = args.real_events or args.all
    from pathlib import Path
    import sys, json, hashlib, os
    import numpy as np
    if want_real:
        import pandas as pd
    if want_cross:
        import h5py

    root = Path(__file__).resolve().parents[1]
    if not (root / "src/paths.py").is_file():
        raise RuntimeError("Open from cbc_pe")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.paths import resolve_project_root, resolve_data_root
    PROJECT_ROOT = resolve_project_root()
    DATA_ROOT = resolve_data_root(config_data_root="/data/vserrano/cbc_pe_data")
    if DATA_ROOT.resolve() != Path("/data/vserrano/cbc_pe_data").resolve():
        raise ValueError("DATA_ROOT differs from frozen paths")
    G0_ID = "bbh_processed_4s_seobnrv4opt_snr10-25_n500_000"
    G1_ID = "M11_final_G1_500k"
    if want_cross:
        DATASETS = {"G0": DATA_ROOT / "processed" / G0_ID / f"{G0_ID}.h5",
                    "G1": DATA_ROOT / "processed" / G1_ID / "m11_final_G1_500k_trainio.h5"}
        COMPACT_G0 = DATA_ROOT / "processed/M11_15B_temp/G0_test30k_model_input_source.h5"
        G0_SOURCE = "compact_test30k" if COMPACT_G0.is_file() else "full_dataset"
        if G0_SOURCE == "compact_test30k":
            DATASETS["G0"] = COMPACT_G0
        SPLITS = {d: DATA_ROOT / "processed" / name /
                  f"{name}_splits_train400000_val40000_cal30000_test30000_seed123.npz"
                  for d,name in [("G0",G0_ID),("G1",G1_ID)]}
    CHECKPOINTS = {
     "E10": DATA_ROOT / "models/checkpoints" / G0_ID / f"{G0_ID}_SimpleCNN_ResidualDilated_M10_inputzscore_resdilated_emb64_d124_train400k_MSELoss_seed123_checkpoint.pt",
     "E11": DATA_ROOT / "models/checkpoints" / G1_ID / "M11_final_G1_500k_SimpleCNN_ResidualDilated_M11_final_G1_inputzscore_resdilated_emb64_d124_train400k_MSELoss_seed123_checkpoint.pt"}
    if want_cross:
        DIAGONALS = {
         "G0": DATA_ROOT / "results" / G0_ID / "m10_inputzscore_500k_cal_test_predictions_embeddings.npz",
         "G1": DATA_ROOT / "results" / G1_ID / "m11_final_G1_500k_cal_test_predictions_embeddings.npz"}
    if want_real:
        REAL_DIR = DATA_ROOT / "results/M11_final_G1_500k/real_events/multi_event"
        STATUS_PATH = REAL_DIR / "event_status.csv"
        REAL_PROVENANCE_PATH = REAL_DIR / "provenance.json"
        GEN_PATH = PROJECT_ROOT / "configs/generation/generate_500k_bbh_4s.json"
    RESULT_ROOT = DATA_ROOT / "results/M11_final_G1_500k/M11_15B_latent_domain_geometry"
    if want_cross:
        CROSS_CACHE = RESULT_ROOT / "latent_cross_embeddings.npz"
    if want_real:
        REAL_CACHE = RESULT_ROOT / "real_event_embeddings.npz"
    LABELS = ["chirp_mass", "total_mass", "chi_eff"]

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def fingerprint(path, content=False):
        path = Path(path)
        stat = path.stat()
        result = dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if content:
            result["sha256"] = sha256(path)
        return result

    def array_hash(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

    def valid_embedding(a, n):
        return a.shape == (n,64) and np.isfinite(a).all()

    def save_npz(path, arrays):
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)

    required_paths = list(CHECKPOINTS.values())
    if want_cross:
        required_paths += [*DATASETS.values(), *SPLITS.values(), *DIAGONALS.values()]
    if want_real:
        required_paths += [STATUS_PATH, REAL_PROVENANCE_PATH, GEN_PATH]
    for path in required_paths:
        require(path.is_file(), f"Missing frozen input: {path}")
    if want_cross:
        idx, labels_std, diagonal = {}, {}, {}
        for domain,encoder in [("G0","E10"),("G1","E11")]:
            with np.load(DIAGONALS[domain], allow_pickle=False) as z:
                diagonal[f"emb_{encoder}_{domain}"] = z["emb_test"].copy()
                idx[domain] = z["idx_test"].copy()
                labels_std[domain] = z["y_test"].copy()
                require(z["label_names"].tolist() == LABELS, "Label order changed")
            require(valid_embedding(diagonal[f"emb_{encoder}_{domain}"],30000), "Diagonal shape/finite")
            require(labels_std[domain].shape == (30000,3) and np.isfinite(labels_std[domain]).all(), "Test labels")
            require(idx[domain].shape == (30000,) and np.issubdtype(idx[domain].dtype,np.integer)
                    and len(np.unique(idx[domain])) == 30000 and (idx[domain]>=0).all(), "Test indices")
            with np.load(SPLITS[domain], allow_pickle=False) as z:
                keys = [k for k in ("idx_test","test_idx","test_indices","test") if k in z.files]
                require(len(keys) == 1, f"Ambiguous test split keys: {z.files}")
                require(np.array_equal(np.sort(z[keys[0]]),np.sort(idx[domain])), "Test membership changed")
            with h5py.File(DATASETS[domain],"r") as h:
                if domain == "G0" and G0_SOURCE == "compact_test30k":
                    require("X" in h and "original_idx_test" in h, "Compact G0 missing X/original_idx_test")
                    require(h["X"].shape == (30000,3,16384), "Compact G0 X shape")
                    require(h["X"].dtype == np.dtype("float32"), "Compact G0 X must be float32")
                    require(h["original_idx_test"].shape == (30000,)
                            and np.issubdtype(h["original_idx_test"].dtype,np.integer)
                            and np.array_equal(h["original_idx_test"][:],idx["G0"]),
                            "Compact G0 original_idx_test differs from frozen artifact order")
                else:
                    require(h["X"].shape[1:] == (3,16384) and idx[domain].max()<len(h["X"]), "Dataset dimensions")
    if want_real:
        status = pd.read_csv(STATUS_PATH)
        require(not status.event.duplicated().any(), "Duplicate events")
        cohort = status.loc[status.processing_status.eq("success")].copy().reset_index(drop=True)
        event_names = cohort.event.astype(str).tolist()
        require(len(event_names) == 9, "Frozen cohort must contain exactly nine successful events")
        require("GW170814" not in event_names, "M11.14 cohort changed: inspect before proceeding")
        real_provenance = json.loads(REAL_PROVENANCE_PATH.read_text())
        nominal_policy = dict(psd_window=[-1024.0,-640.0], center_offset=0.0,
                              psd_segment_duration=8.0, fallback_windows=[], zscore_eps=1e-6)
        require(real_provenance["nominal_policy"] == nominal_policy, "M11.14 nominal policy changed")
    if want_cross:
        contract = dict(version=1, labels=LABELS, latent_dimension=64,
            g0_source=G0_SOURCE,
            datasets={d:fingerprint(p) for d,p in DATASETS.items()},
            splits={d:fingerprint(p,True) for d,p in SPLITS.items()},
            diagonals={d:fingerprint(p) for d,p in DIAGONALS.items()},
            checkpoints={e:fingerprint(p,True) for e,p in CHECKPOINTS.items()},
            test_index_sha256={d:array_hash(v) for d,v in idx.items()},
            diagonal_sha256={k:array_hash(v) for k,v in diagonal.items()},
            normalization={"G0":"zscore once eps=1e-6 before E11", "G1":"already normalized; feed E10 directly"})
        contract_json = json.dumps(contract,sort_keys=True)
    RESULT_ROOT.mkdir(parents=True,exist_ok=True)

    models = {}
    def get_model(encoder):
        import torch
        from src.models.network import SimpleCNN_ResidualDilated
        require(torch.cuda.is_available(), "Missing cached inference requires a CUDA GPU")
        if encoder not in models:
            checkpoint = torch.load(CHECKPOINTS[encoder],map_location="cpu")
            config = checkpoint["model_config"]
            require(config["label_names"] == LABELS and config["signal_length"] == 16384
                    and config["n_detectors"] == 3, "Checkpoint contract")
            model = SimpleCNN_ResidualDilated(**config["model_kwargs"])
            model.load_state_dict(checkpoint["model_state_dict"])
            models[encoder] = (model.to("cuda").eval(), checkpoint["y_mean"], checkpoint["y_std"])
        return models[encoder]

    def infer_embeddings(encoder, inputs):
        from src.real_data.inference import predict_real_with_embeddings
        model,mean,std = get_model(encoder)
        _, _, embedding = predict_real_with_embeddings(model,np.asarray(inputs,dtype=np.float32),
                                                        "cuda",mean,std,batch_size=args.batch_size)
        require(valid_embedding(embedding,len(inputs)), "Invalid inferred embedding")
        return embedding.astype(np.float32)

    def cross_inference(encoder,domain):
        from src.models.dataset import normalize_input_per_sample_per_detector_zscore
        require((encoder,domain) in [("E10","G1"),("E11","G0")], "Never recompute diagonals")
        result = np.empty((30000,64),dtype=np.float32)
        with h5py.File(DATASETS[domain],"r") as h:
            for start in range(0,30000,args.batch_size):
                indices = idx[domain][start:start+args.batch_size]
                if domain == "G0" and G0_SOURCE == "compact_test30k":
                    # Compact rows already follow the frozen artifact's idx_test order.
                    inputs = h["X"][start:start+len(indices)]
                else:
                    order = np.argsort(indices)
                    # h5py fancy indexing needs increasing IDs; restore artifact order.
                    inputs = np.asarray(h["X"][indices[order]],dtype=np.float32)[np.argsort(order)]
                # Validate raw values before normalization, without a second full pass.
                require(np.isfinite(inputs).all(), f"Nonfinite {domain} source batch at row {start}")
                if domain == "G0":
                    inputs = np.stack([normalize_input_per_sample_per_detector_zscore(sample,eps=1e-6)
                                       for sample in inputs])
                # G1 is already normalized: no second z-score.
                require(np.isfinite(inputs).all(), "Nonfinite synthetic input")
                result[start:start+len(indices)] = infer_embeddings(encoder,inputs)
        return result

    if want_real:
        DETECTORS = ["H1","L1","V1"]
        GWOSC_CACHE = DATA_ROOT / "gwosc_cache"
        NOMINAL_PSD = (-1024.0,-640.0)
        CENTER_OFFSET = 0.0
        def prepare_nominal(event, gps, available_detectors, catalog):
            require(set(str(available_detectors).split(",")) >= set(DETECTORS), "missing_required_detectors")
            raw, sources = {}, []
            for detector in DETECTORS:
                url = get_event_detector_url(event_name=event, detector=detector, catalog=catalog, sample_rate=4096)
                require(url is not None, f"missing_url: {event}/{detector}")
                filename = url.rsplit("/", 1)[-1]
                locations = [GWOSC_CACHE / filename, GWOSC_CACHE / "notebooks_root" / filename,
                             GWOSC_CACHE / "m11" / filename]
                local = next((p for p in locations if p.is_file()), locations[0])
                if not local.is_file():
                    GWOSC_CACHE.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(url, local)
                strain = read_gwosc_hdf5_as_pycbc_timeseries(local)
                require(float(strain.sample_rate) == 4096, f"sample_rate: {detector}")
                raw[detector] = strain
                sources.append(dict(detector=detector, url=url, cached_file=str(local)))
            center = float(gps) + CENTER_OFFSET
            ok, reason = event_window_is_available_and_finite(raw, center, final_duration=4.0,
                context_start_samples=1664, context_end_samples=1664, sampling_frequency=4096)
            require(ok, f"Invalid event window: {reason}")
            selected = select_valid_psd_window(raw, center, preferred_window=NOMINAL_PSD, candidate_windows=())
            X_pre, _, _, _, meta = build_real_input_like_training(raw_strains=raw, detectors=DETECTORS,
                center_time=center, processor=processor, expected_detector_order=DETECTORS,
                final_duration=4.0, final_length=16384, processing_length=19712,
                context_start_samples=1664, context_end_samples=1664, sampling_frequency=4096,
                psd_delta_f=4096/19712, psd_target_flength=9857,
                psd_start_offset=selected[0], psd_end_offset=selected[1], psd_segment_duration=8.0)
            X = normalize_input_per_sample_per_detector_zscore(X_pre[0], eps=1e-6)[None, :, :]
            require(X.shape == (1,3,16384) and np.isfinite(X).all(), "Invalid nominal input")
            diagnostics = pd.DataFrame([dict(event=event, detector=detector,
                pre_zscore_std=float(X_pre[0,j].std()),
                pre_zscore_rms=float(np.sqrt(np.mean(X_pre[0,j].astype(float)**2))),
                post_zscore_mean=float(X[0,j].mean()), post_zscore_std=float(X[0,j].std()),
                finite=bool(np.isfinite(X_pre[0,j]).all() and np.isfinite(X[0,j]).all()))
                for j,detector in enumerate(DETECTORS)])
            return X, diagnostics, dict(event=event, gps=float(gps), sources=sources,
                psd_window=list(selected), center_offset=CENTER_OFFSET, zscore_count=1, processing=meta)

        real_contract = dict(model_contract=dict(version=1, labels=LABELS, latent_dimension=64,
            checkpoints={e:fingerprint(p,True) for e,p in CHECKPOINTS.items()}),
            cohort=fingerprint(STATUS_PATH,True), nominal_provenance=fingerprint(REAL_PROVENANCE_PATH,True),
            generation_config=fingerprint(GEN_PATH,True), events=event_names, policy=nominal_policy)
        real_contract_json = json.dumps(real_contract,sort_keys=True)

    def load_cache(path, validator):
        try:
            with np.load(path, allow_pickle=False) as archive:
                cached = {key: archive[key].copy() for key in archive.files}
            validator(cached)
            return cached
        except Exception as error:
            raise ValueError(f"Invalid cache {path}: {error}. Run the GPU script with --force to regenerate.") from error

    def validate_cross(cached, allow_partial=False):
        required = {"emb_E10_G0", "emb_E10_G1", "emb_E11_G0", "emb_E11_G1",
                    "idx_G0_test", "idx_G1_test", "label_names", "provenance_json"}
        if allow_partial:
            required -= {"emb_E10_G1", "emb_E11_G0"}
        require(required <= cached.keys(), f"Missing keys: {required - cached.keys()}")
        require(cached["provenance_json"].item() == contract_json, "Cross provenance mismatch")
        require(cached["label_names"].tolist() == LABELS, "Cached label order")
        for domain in ("G0", "G1"):
            require(np.array_equal(cached[f"idx_{domain}_test"], idx[domain]), "Cached test order")
            for encoder in ("E10", "E11"):
                key = f"emb_{encoder}_{domain}"
                if key in cached:
                    require(valid_embedding(cached[key], 30000), "Cached shape/finite")
        for key, value in diagonal.items():
            require(np.array_equal(cached[key], value), "Existing diagonal changed")

    if want_real:
        def validate_real(cached):
            required = {"event_names", "emb_E10_real", "emb_E11_real", "provenance_json", "event_metadata_json"}
            require(required <= cached.keys(), f"Missing keys: {required - cached.keys()}")
            cached_contract = json.loads(cached["provenance_json"].item())
            # Older real caches embedded synthetic provenance; only model identity
            # is relevant to real inference. No synthetic artifacts are accessed.
            if "synthetic_contract" in cached_contract and "model_contract" not in cached_contract:
                previous = cached_contract.pop("synthetic_contract")
                cached_contract["model_contract"] = {key:previous[key] for key in
                    ("version", "labels", "latent_dimension", "checkpoints")}
            require(cached_contract == real_contract, "Real provenance mismatch")
            require(cached["event_names"].tolist() == event_names, "Incomplete or changed real cohort/order")
            for encoder in ("E10", "E11"):
                require(valid_embedding(cached[f"emb_{encoder}_real"], len(event_names)), "Real shape/finite")
            records = json.loads(cached["event_metadata_json"].item())
            require(len(records) == len(event_names), "Real metadata length")
            require([record["event"] for record in records] == event_names, "Real metadata event order")

    cross = real_cache = None
    if want_cross and CROSS_CACHE.is_file() and not args.force:
        cross = load_cache(CROSS_CACHE, lambda cached: validate_cross(cached, allow_partial=True))
        for encoder, domain in [("E10", "G1"), ("E11", "G0")]:
            if f"emb_{encoder}_{domain}" in cross:
                print(f"Reusing {encoder}({domain}) from cache", flush=True)
    if want_real and REAL_CACHE.is_file() and not args.force:
        real_cache = load_cache(REAL_CACHE, validate_real)
        print(f"Valid cache; skipping real inference: {REAL_CACHE}", flush=True)
    missing_cross = want_cross and (cross is None or any(
        key not in cross for key in ("emb_E10_G1", "emb_E11_G0")))
    if missing_cross or (want_real and real_cache is None):
        import torch
        require(torch.cuda.is_available(), "Missing/replaced caches require CUDA in cbc_torch on the GPU VM")
    if missing_cross:
        if cross is None:
            cross = {**diagonal, "idx_G0_test":idx["G0"], "idx_G1_test":idx["G1"],
                     "label_names":np.array(LABELS), "provenance_json":np.array(contract_json)}
        for encoder, domain in [("E10", "G1"), ("E11", "G0")]:
            if f"emb_{encoder}_{domain}" in cross:
                continue
            print(f"Generating {encoder}({domain}): 30000 samples", flush=True)
            cross[f"emb_{encoder}_{domain}"] = cross_inference(encoder, domain)
            save_npz(CROSS_CACHE, cross)
        validate_cross(cross)
        print(f"Saved {CROSS_CACHE}", flush=True)
    if want_real and real_cache is None:
        real_cache = dict(event_names=np.array([],dtype=str), emb_E10_real=np.empty((0,64),np.float32),
                          emb_E11_real=np.empty((0,64),np.float32),
                          provenance_json=np.array(real_contract_json), event_metadata_json=np.array("[]"))
        metadata = []
        import urllib.request
        from src.config import SimulationConfig
        from src.processing import SignalProcessor
        from src.models.dataset import normalize_input_per_sample_per_detector_zscore
        from src.real_data.catalog import get_event_detector_url
        from src.real_data.gwosc_utils import read_gwosc_hdf5_as_pycbc_timeseries
        from src.real_data.signal_processing import build_real_input_like_training, event_window_is_available_and_finite
        from src.real_data.psd import select_valid_psd_window
        gen_cfg = json.loads(GEN_PATH.read_text())
        sim_config = SimulationConfig(duration=4.0,processing_context_start_samples=1664,
                                      processing_context_end_samples=1664)
        processor = SignalProcessor(config=sim_config,**gen_cfg["signal_processor"])
        require(not processor.apply_standardization, "Zscore must occur once outside processor")
        require(json.loads(json.dumps(processor.metadata())) == real_provenance["preprocessing"], "Frozen processor changed")
        for _,row in cohort.iterrows():
            print(f"Generating real pair: {row.event}", flush=True)
            inputs, _, meta = prepare_nominal(row.event,row.gps_time,row.detectors,row["catalog"])
            pair = {encoder:infer_embeddings(encoder,inputs) for encoder in ("E10","E11")}
            for encoder in pair:
                key = f"emb_{encoder}_real"
                real_cache[key] = np.concatenate([real_cache[key],pair[encoder]],axis=0)
            real_cache["event_names"] = np.append(real_cache["event_names"],str(row.event))
            metadata.append(meta)
            real_cache["event_metadata_json"] = np.array(json.dumps(metadata))
            save_npz(REAL_CACHE,real_cache)
        validate_real(real_cache)
        print(f"Saved {REAL_CACHE}", flush=True)
    models.clear()
    if "torch" in sys.modules:
        sys.modules["torch"].cuda.empty_cache()


if __name__ == "__main__":
    main()
