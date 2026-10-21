"""User-operated G1 X-only joint NPE training and validation inference."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.paths import resolve_data_root
from src.models.hdf5_batch_dataset import HDF5BatchIterableDataset
from src.models.g1_npe import G1Posterior, mass_bounds
from scripts.m12.g1_npe_contract import (
    DATASET, LABELS, PINNED, require, digest, write_json, load_contract)

TRAIN = dict(batch_size=64, epochs=100, patience=20, lr=3e-4, weight_decay=3e-4)


def loader(path, ids, train, seed, workers, batch=64):
    ds = HDF5BatchIterableDataset(path, ids, batch_size=batch, drop_last=False,
        seed=seed, shuffle_batches=train, input_normalization=None)
    kw = dict(batch_size=None, num_workers=workers, pin_memory=True)
    if workers:
        kw.update(prefetch_factor=2, persistent_workers=False)
    return DataLoader(ds, **kw)


def finite(value):
    require(bool(torch.isfinite(value).all()), 'Nonfinite value; abort without dropping rows')


def train(args, paths, splits, mean, std, identity, out, model):
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN['lr'], weight_decay=TRAIN['weight_decay'])
    loaders = {s: loader(paths['dataset_file'], splits[s], s=='train', args.seed,
                         args.workers) for s in ('train','val')}
    best, stale, history, checkpoints = float('inf'), 0, [], {}
    start = time.monotonic()
    for epoch in range(TRAIN['epochs']):
        row = dict(epoch=epoch+1)
        for split, dl in loaders.items():
            training = split == 'train'
            model.train(training)
            dl.dataset.set_epoch(epoch)
            total, count, boundary = 0., 0, 0
            with torch.set_grad_enabled(training):
                for x, y in dl:
                    x, y = x.to(args.device), y.to(args.device)
                    lp, clipped = model.log_prob(y, model.encode(x))
                    loss = -lp.mean()
                    if training:
                        opt.zero_grad()
                        loss.backward()
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                finite(parameter.grad)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        opt.step()
                    total += float(loss.detach())*len(y)
                    count += len(y)
                    boundary += int(clipped.sum())
            require(count == len(splits[split]), 'Loader omitted or repeated rows')
            row[split+'_nll_physical'] = total/count
            row[split+'_boundary_adjustments'] = boundary
        # Standardized density differs by this fixed Jacobian only.
        row['val_nll_standardized'] = row['val_nll_physical']-float(np.log(std+np.float32(1e-8)).sum())
        row['elapsed_seconds'] = time.monotonic()-start
        history.append(row)
        improved = row['val_nll_physical'] < best
        stale = 0 if improved else stale+1
        if improved:
            best = row['val_nll_physical']
            name = 'epoch_%03d.pt' % (epoch+1)
            with (out/name).open('xb') as f:
                torch.save(dict(state=model.state_dict(), identity=identity, seed=args.seed,
                                training=TRAIN, epoch=epoch+1, y_mean=mean, y_std=std), f)
            checkpoints[name] = digest(out/name)
            best_name = name
        print(json.dumps(row), flush=True)
        if stale >= TRAIN['patience']:
            break
    require(paths['dataset_file'].stat().st_mtime_ns == identity['dataset_mtime_ns'],
            'Dataset changed during training')
    write_json(out/'history.json', history)
    write_json(out/'complete.json', dict(complete=True, identity=identity, seed=args.seed,
        training=TRAIN, checkpoint=best_name, checkpoints=checkpoints,
        history_sha256=digest(out/'history.json'),
        cuda_peak_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0))


def summaries(draw, truth, lp_draw, lp_truth):
    # Draws are IID from the learned flow; this does not measure approximation error.
    n = draw.shape[1]
    chunks = draw.reshape(len(draw), 8, n//8, 3)
    means = chunks.mean(2)
    variances = chunks.var(2, ddof=1)
    ranks = (draw < truth[:,None,:]).sum(1)
    return dict(mean=draw.mean(1), variance=draw.var(1,ddof=1),
        mean_mcse=means.std(1,ddof=1)/np.sqrt(8),
        variance_mcse=variances.std(1,ddof=1)/np.sqrt(8),
        block_mean=means, block_variance=variances,
        half_mean=draw[:,:n//2].mean(1), half_variance=draw[:,:n//2].var(1,ddof=1),
        quantiles=np.quantile(draw,[.025,.05,.1,.25,.5,.75,.9,.95,.975],axis=1).transpose(1,0,2),
        ranks=ranks, joint_density_rank=(lp_draw < lp_truth[:,None]).sum(1),
        draws=np.full(len(draw), n, dtype=np.int64))


def infer(args, paths, splits, mean, std, identity, out, model, root):
    receipt = json.loads((args.checkpoint.parent/'complete.json').read_text())
    require(receipt['complete'] and
            receipt['checkpoints'].get(args.checkpoint.name) == digest(args.checkpoint),
            'Incomplete or changed checkpoint')
    saved = torch.load(args.checkpoint, map_location='cpu')  # trusted local artifact only
    # Portable identity: timestamps may change after a user-operated transfer.
    portable = lambda x: {k:v for k,v in x.items() if k != 'dataset_mtime_ns'}
    require(portable(saved['identity']) == portable(identity) == portable(receipt['identity']) and
            saved['training'] == TRAIN and saved['seed'] == receipt['seed'], 'Checkpoint provenance mismatch')
    require(np.array_equal(saved['y_mean'],mean) and np.array_equal(saved['y_std'],std), 'Scaler mismatch')
    model.load_state_dict(saved['state'], strict=True)
    model.eval()
    ids = splits['val']
    baseline_path = root/'results'/DATASET/'m11_final_G1_500k_val_predictions_embeddings.npz'
    require(digest(baseline_path) == PINNED['baseline'], 'Frozen M11 prediction hash mismatch')
    with np.load(baseline_path) as b:
        order = np.argsort(b['idx_val'])
        require(np.array_equal(b['idx_val'][order],ids) and b['label_names'].tolist()==LABELS and
                np.array_equal(b['y_mean'],mean) and np.array_equal(b['y_std'],std), 'M11 alignment/scaler mismatch')
        baseline, baseline_truth = b['pred_val'][order], b['y_val'][order]
    require(np.isfinite(baseline).all(), 'Nonfinite M11 output')
    # Train-only compact labels for prior diagnostics; never inputs to the encoder.
    with h5py.File(paths['dataset_file'],'r') as h:
        yt = h['y'][splits['train']].astype('float64')
        require(np.isfinite(yt).all(), 'Nonfinite train labels')
        metadata = dict(source_id=h['source_id'].asstr()[ids],
                        environment=h['file_group_id'][ids].astype(str),
                        snr=h['target_network_snr'][ids])
    rng = np.random.default_rng(125220)
    prior = yt[rng.choice(len(yt),8192,replace=False)]
    scale = std.astype('float64')
    loader_scale = (std+np.float32(1e-8)).astype('float64')
    center = ((yt-mean)/scale).mean(0)
    precision = np.linalg.pinv(np.cov((yt-mean)/scale,rowvar=False))
    dist = np.einsum('ni,ij,nj->n',(yt-mean)/scale-center,precision,(yt-mean)/scale-center)
    payload = dict(idx_val=ids, m11_pred_val=baseline, label_names=np.array(LABELS),
        y_mean=mean,y_std=std,train_label_cuts=np.quantile(yt,[.1,.9],axis=0),
        train_center=center,train_precision=precision,train_distance_cuts=np.quantile(dist,[.8,.95,.99]),
        prior_mean=yt.mean(0),prior_variance=yt.var(0,ddof=1),
        prior_quantiles=np.quantile(yt,[.025,.05,.1,.25,.5,.75,.9,.95,.975],axis=0),**metadata)
    accum, doubled, truths, nll, adjusted = {}, {}, [], [], []
    offset, support_count = 0, 0
    torch.manual_seed(args.sample_seed)
    with torch.no_grad():
        for x,y in loader(paths['dataset_file'],ids,False,0,0,batch=32):
            x,y = x.to(args.device),y.to(args.device)
            ctx = model.encode(x)
            lp, boundary = model.log_prob(y,ctx)
            truth = y.cpu().numpy().astype('float64')
            require(np.allclose((truth-mean)/loader_scale,baseline_truth[offset:offset+len(y)],atol=2e-6,rtol=0),
                    'M11 truth/row alignment failed')
            # First 1024 sorted validation IDs are the fixed budget-doubling panel.
            n = args.draws*2 if offset < 1024 else args.draws
            draws, logps = [], []
            for start in range(0,n,256):
                sample = model.sample(ctx,min(256,n-start))
                lo,hi = mass_bounds(sample[...,1])
                valid = ((sample[...,0]>lo)&(sample[...,0]<hi)&(sample[...,1]>10)&
                         (sample[...,1]<180)&(sample[...,2]>-1)&(sample[...,2]<1))
                support_count += int((~valid).sum())
                require(bool(valid.all()), 'Support/sigmoid saturation failure')
                ctx_rep = ctx[:,None,:].expand(len(y),sample.shape[1],128).reshape(-1,128)
                slp,_ = model.log_prob(sample.reshape(-1,3),ctx_rep)
                draws.append(sample.cpu().numpy())
                logps.append(slp.reshape(len(y),-1).cpu().numpy())
            draw, logp = np.concatenate(draws,1),np.concatenate(logps,1)
            result = summaries(draw[:,:args.draws],truth,logp[:,:args.draws],lp.cpu().numpy())
            for k,v in result.items(): accum.setdefault(k,[]).append(v)
            if offset < 1024:
                for k,v in summaries(draw,truth,logp,lp.cpu().numpy()).items():
                    doubled.setdefault('double_'+k,[]).append(v)
            # Joint energy-score comparison uses paired IID samples, not independent marginals.
            a,b = draw[:,:args.draws//2]/scale,draw[:,args.draws//2:args.draws]/scale
            es = np.linalg.norm(a-truth[:,None,:]/scale,axis=-1).mean(1)-.5*np.linalg.norm(a-b,axis=-1).mean(1)
            pa,pb = prior[:4096]/scale,prior[4096:]/scale
            pes = np.linalg.norm(pa[None,:,:]-truth[:,None,:]/scale,axis=-1).mean(1)-.5*np.linalg.norm(pa-pb,axis=-1).mean()
            accum.setdefault('energy_score',[]).append(es)
            accum.setdefault('prior_energy_score',[]).append(pes)
            truths.append(truth); nll.append(-lp.cpu().numpy()); adjusted.append(boundary.cpu().numpy())
            offset += len(y)
            if offset%1024==0: print('validation rows',offset,flush=True)
    require(offset==40000 and paths['dataset_file'].stat().st_mtime_ns==identity['dataset_mtime_ns'],
            'Incomplete inference or mutated dataset')
    payload.update({k:np.concatenate(v) for k,v in {**accum,**doubled}.items()})
    payload.update(y_physical=np.concatenate(truths),nll_physical=np.concatenate(nll),
                   boundary_adjusted=np.concatenate(adjusted),double_ids=ids[:1024])
    payload.update(pred_val=(payload['mean']-mean)/scale,
        y_val=(payload['y_physical'].astype('float32')-mean)/(std+np.float32(1e-8)),
        variance_standardized=payload['variance']/scale**2,
        mean_mcse_standardized=payload['mean_mcse']/scale,
        variance_mcse_standardized=payload['variance_mcse']/scale**2)
    for k,v in payload.items():
        if np.asarray(v).dtype.kind in 'fc': require(np.isfinite(v).all(),'Nonfinite archive: '+k)
    with (out/'posterior.npz').open('xb') as f: np.savez_compressed(f,**payload)
    write_json(out/'complete.json',dict(complete=True,identity=identity,split='val',
        domain='historical held-out validation environment mixture; validation-selected',
        checkpoint_sha256=digest(args.checkpoint),training_seed=saved['seed'],epoch=saved['epoch'],
        sample_seed=args.sample_seed,draws=args.draws,support_violations=support_count,
        baseline_sha256=digest(baseline_path),archive_sha256=digest(out/'posterior.npz')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','infer'])
    p.add_argument('--data-root'); p.add_argument('--tag',required=True)
    p.add_argument('--checkpoint',type=Path); p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,choices=[123,124],default=123)
    p.add_argument('--sample-seed',type=int,default=125221)
    p.add_argument('--draws',type=int,default=4096)
    p.add_argument('--workers',type=int,choices=[0,1,2,4],default=2)
    a=p.parse_args()
    require((a.mode=='infer')==(a.checkpoint is not None),'Checkpoint required only for inference')
    require(a.draws>=512 and a.draws%256==0,'Draw budget must be >=512 and divisible by 256')
    require(a.tag and all(c.isalnum() or c in '_-' for c in a.tag),'Invalid unique tag')
    root=resolve_data_root(cli_data_root=a.data_root)
    paths,splits,mean,std,identity=load_contract(root)
    out=(root/'results'/DATASET/'M12_5_2_NPE'/a.tag).resolve()
    require(ROOT not in out.parents and out!=ROOT,'Artifacts must be outside Git')
    out.mkdir(parents=True,exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    model=G1Posterior().to(a.device)
    write_json(out/'started.json',dict(complete=False,identity=identity,args={k:str(v) for k,v in vars(a).items()},
        parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__))
    if a.mode=='train': train(a,paths,splits,mean,std,identity,out,model)
    else: infer(a,paths,splits,mean,std,identity,out,model,root)

if __name__=='__main__': main()
