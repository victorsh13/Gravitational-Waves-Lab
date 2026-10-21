"""Small frozen-G1 contract for the NPE experiment; no physics or patch-MLP imports."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from src.paths import resolve_processed_artifact

ROOT = Path(__file__).resolve().parents[2]
DATASET = 'M11_final_G1_500k'
LABELS = ['chirp_mass', 'total_mass', 'chi_eff']
PINNED = {
    'split_file': '1abf607c44cc4c55ca196de046874137e274c88cf221143cd936d645e97b97d2',
    'label_stats_file': 'd0891bf1c130186fb4e7af442e74952a03e4d7d5e0cc240f78d8772e4421855f',
    'baseline': '586b95f2cb2cfaf150fbdd05760f50f47a95a555451fcb46d89492b7d00ed88b'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with open(path, 'x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def code_identity():
    names = ['src/models/g1_npe.py', 'scripts/m12/g1_npe_contract.py',
             'scripts/m12/run_m12_5_2_npe.py', 'src/models/hdf5_batch_dataset.py',
             'src/paths.py', 'configs/experiments/train_m11_final_G1_500k.json']
    return {n: digest(ROOT/n) for n in names}


def load_contract(root):
    cfg = json.loads((ROOT/'configs/experiments/train_m11_final_G1_500k.json').read_text())
    require(cfg['dataset']['dataset_id'] == DATASET and
            cfg['input_normalization'] == dict(enabled=False, already_normalized=True,
                mode='per_sample_per_detector_zscore', eps=1e-6), 'Frozen input config changed')
    paths = {k: resolve_processed_artifact(data_root=root, dataset_id=DATASET,
             file_name=cfg['dataset'][k]) for k in ('dataset_file','split_file','label_stats_file')}
    for key in ('split_file','label_stats_file'):
        require(digest(paths[key]) == PINNED[key], 'Frozen identity mismatch: '+key)
    with np.load(paths['split_file']) as a:
        splits = {s: np.sort(a[s+'_idx']) for s in ('train','val','cal','test')}
    for s, n in dict(train=400000,val=40000,cal=30000,test=30000).items():
        require(splits[s].shape == (n,) and splits[s].dtype.kind in 'iu', 'Bad split '+s)
    require(np.array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(500000)),
            'Incomplete/overlapping split IDs')
    with np.load(paths['label_stats_file']) as a:
        require(a['label_names'].tolist() == LABELS and
                np.array_equal(np.sort(a['train_idx']), splits['train']), 'Wrong scaler provenance')
        mean, std = a['y_mean'].astype('float32'), a['y_std'].astype('float32')
    require(mean.shape == std.shape == (3,) and np.isfinite(mean).all() and
            np.isfinite(std).all() and (std>0).all(), 'Invalid historical scaler')
    attrs = dict(config_sha256='cf9562725bb88ee68b325beb3e44b41bfb5327a6d1a946cd1d9577bab6bd9467',
        manifest_sha256='544579ef7270675b80caee34cfe440d9b767b02ada3fd5f4de2e8e0f066b864e',
        labels=','.join(LABELS), detector_order='H1,L1,V1',
        input_normalization='per_sample_per_detector_zscore', input_zscore_epsilon=1e-6,
        storage_layout_variant='trainio')
    with h5py.File(paths['dataset_file'], 'r') as h:
        require(h['X'].shape == (500000,3,16384) and h['X'].dtype == np.dtype('float32') and
                h['y'].shape == (500000,3), 'G1 shape/dtype mismatch')
        require(all(h.attrs.get(k) == v for k,v in attrs.items()), 'G1 header mismatch')
        require(np.all(h['status'][:] == 1), 'Incomplete G1 generation')
        membership, sources = h['split'].asstr()[:], h['source_id'].asstr()[:]
        env = h['file_group_id'][:].astype(str)
        require(len(np.unique(sources)) == 500000, 'Repeated source IDs')
        environments = {}
        for s, ids in splits.items():
            require(np.all(membership[ids] == s), 'HDF5 split mismatch')
            environments[s] = set(env[ids])
        names = list(splits)
        require(all(not environments[a] & environments[b] for i,a in enumerate(names)
                    for b in names[i+1:]), 'Environment leakage')
    stat = paths['dataset_file'].stat()
    identity = dict(version='M12.5.2-NPE-v1', pinned=PINNED, header=attrs,
        dataset_size=stat.st_size, dataset_mtime_ns=stat.st_mtime_ns,
        code=code_identity(), labels=LABELS)
    # mtime is a mutation sentinel within a run, not a portable content hash.
    return paths, splits, mean, std, identity
