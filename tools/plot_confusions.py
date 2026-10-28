"""Render specimen-weighted confusion summaries from saved test predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['quartz', 'feldspar', 'mica', 'pyroxene', 'opaque_material']
ARMS = ['static_k1', 'unordered_k12', 'angle_aware_k12']
fig, axs = plt.subplots(2, 3, figsize=(10.8, 7.2), layout='constrained')
records = []
for row, run in enumerate(['baseline', 'regularized']):
    for col, arm in enumerate(ARMS):
        d = pd.concat([pd.read_parquet(p) for p in sorted((ROOT/'results'/run/arm).glob('fold_*/seed_*/predictions.parquet'))])
        matrices = []
        for seed, ds in d[d.state.eq('singleton')].groupby('seed'):
            y = ds.allowed.map(lambda x: CLASSES.index(x[0])).to_numpy()
            yp = ds.y_pred.map(CLASSES.index).to_numpy()
            counts = ds.specimen.value_counts()
            w = ds.specimen.map(lambda s: 1/counts[s]).to_numpy()
            cm = np.bincount(5*y+yp, weights=w, minlength=25).reshape(5,5)
            matrices.append(cm/cm.sum(axis=1, keepdims=True))
        cm = np.mean(matrices, axis=0)
        ax = axs[row,col]
        ax.imshow(cm, vmin=0, vmax=1, cmap='Blues')
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f'{100*cm[i,j]:.0f}',ha='center',va='center',color='white' if cm[i,j]>.55 else '#203a43',fontsize=10)
                records.append(dict(experiment=run,arm=arm,target=CLASSES[i],prediction=CLASSES[j],row_fraction=float(cm[i,j])))
        ax.set_xticks(range(5),['Q','F','M','P','O']); ax.set_yticks(range(5),['Q','F','M','P','O'])
        ax.set_title(('Initial' if row==0 else 'Regularised')+' | '+['Static','Unordered','Angle-aware'][col],fontsize=11)
        ax.set_xlabel('Predicted family')
        if col==0: ax.set_ylabel('Weak target family')
fig.suptitle('Row-normalised confusion (%) | mean across seeds\nQ: quartz  F: feldspar  M: mica  P: pyroxene  O: opaque',fontsize=12)
fig.savefig(ROOT/'report/images/confusions.png',dpi=190)
pd.DataFrame(records).to_csv(ROOT/'evidence/confusion_fractions.csv',index=False)
