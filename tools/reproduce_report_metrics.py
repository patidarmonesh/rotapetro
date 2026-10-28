"""Verify reported point estimates directly from archived OOF predictions.

Does not run training, overwrite result files, or claim to regenerate bootstrap CIs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['quartz', 'feldspar', 'mica', 'pyroxene', 'opaque_material']
ARMS = ['static_k1', 'unordered_k12', 'angle_aware_k12']


def metric(df):
    d = df.loc[df.state.eq('singleton')].copy()
    assert all(len(x) == 1 for x in d.allowed), 'Invalid singleton target'
    assert not d.region_id.duplicated().any(), 'Repeated OOF region'
    truth = d.allowed.map(lambda x: CLASSES.index(x[0])).to_numpy()
    pred = d.y_pred.map(CLASSES.index).to_numpy()
    counts = d.specimen.value_counts()
    weights = d.specimen.map(lambda s: 1 / counts[s]).to_numpy()
    cm = np.bincount(5 * truth + pred, weights=weights, minlength=25).reshape(5, 5)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(5), where=denom != 0)
    return float(f1.mean()), len(d), d.specimen.nunique()


def main():
    n_fits = 0
    for run in ['baseline', 'regularized']:
        folder = ROOT / 'results' / run
        summary = json.loads((folder / 'evaluation_summary.json').read_text())
        signatures = {}
        estimates = {}
        print(f'\n{run.upper()}')
        for arm in ARMS:
            files = sorted((folder / arm).glob('fold_*/seed_*/predictions.parquet'))
            assert files, f'No predictions for {run}/{arm}'
            n_fits += len(files)
            chunks = []
            for file in files:
                d = pd.read_parquet(file)
                assert set(d.arm) == {arm}
                assert set(d.fold) == {file.parent.parent.name}
                assert set(d.seed.astype(int)) == {int(file.parent.name.split('_')[1])}
                probs = d[['probs_' + c for c in CLASSES]].to_numpy()
                assert np.isfinite(probs).all() and (probs >= 0).all()
                assert np.allclose(probs.sum(1), 1, atol=2e-5)
                assert np.array_equal(np.array(CLASSES)[probs.argmax(1)], d.y_pred.to_numpy())
                chunks.append(d)
            all_rows = pd.concat(chunks, ignore_index=True)
            values = []
            for seed, d in all_rows.groupby('seed'):
                assert d.fold.nunique() == 5
                assert not d.region_id.duplicated().any()
                assert d.groupby('specimen').fold.nunique().max() == 1
                sig = d[['region_id', 'specimen', 'fold', 'state']].copy()
                sig['allowed'] = d.allowed.map(lambda x: '|'.join(x))
                sig = sig.sort_values('region_id').reset_index(drop=True)
                if seed in signatures:
                    assert sig.equals(signatures[seed]), f'Target/identity mismatch: {arm}/{seed}'
                else:
                    signatures[seed] = sig
                f1, nr, ns = metric(d)
                assert (nr, ns) == (581, 77)
                values.append(f1)
            mean = float(np.mean(values))
            saved = summary['point_estimates'][arm]['macro_f1_mean']
            assert abs(mean - saved) <= 5.1e-5, (arm, mean, saved)
            estimates[arm] = mean
            print(f'{arm:20s} F1={mean:.6f}  seeds={len(values)}  singleton regions=581 specimens=77')
        print(f'Angle-aware minus static: {estimates[ARMS[2]] - estimates[ARMS[0]]:+.8f}')
    assert n_fits == 120, n_fits
    print('\nPASS: 120 fit prediction files; matching targets across arms; probabilities valid; primary point estimates reproduced.')
    print('This verifies archived prediction arithmetic, not independent mineral truth or full training reproducibility.')


if __name__ == '__main__':
    main()
