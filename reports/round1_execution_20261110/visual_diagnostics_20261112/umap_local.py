"""Read exported frozen-common-checkpoint features only; no network or labels used in fitting."""
import sys,json
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'local_vendor'))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import umap
X=np.load(P/'COMMON_FEATURES.npz')['features'];labels=json.loads((P/'FEATURE_LABELS.json').read_text());assert len(X)==len(labels) and X.shape[1]==128 and np.isfinite(X).all()
z=StandardScaler().fit_transform(X);u=umap.UMAP(n_neighbors=15,min_dist=.1,metric='euclidean',random_state=0,n_jobs=1).fit_transform(z);groups=sorted({r['domain'] for r in labels});cent=[];cent2=[];fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
for g in groups:
 mask=np.array([r['domain']==g for r in labels]);ax.scatter(u[mask,0],u[mask,1],s=24,alpha=.75,label=f'{g}: {mask.sum()} windows / {len({labels[i]["person"] for i in np.flatnonzero(mask)})} people');cent.append(z[mask].mean(0));cent2.append(u[mask].mean(0))
ax.legend(fontsize=9);ax.set_title('One common fold0 checkpoint, same feature coordinates\nExploratory UMAP; some public subjects were training data');ax.set_xlabel('UMAP 1 (arbitrary units)');ax.set_ylabel('UMAP 2 (arbitrary units)');fig.savefig(P/'UMAP.png',dpi=160);plt.close(fig)
result=dict(groups=groups,centroid_distance_standardized128=cdist(cent,cent).tolist(),centroid_distance_umap2=cdist(cent2,cent2).tolist(),n_windows=len(X),umap_version=umap.__version__,parameters=dict(n_neighbors=15,min_dist=.1,random_state=0),limitations='Common checkpoint includes public training subjects; windows correlated; BTC chosen as calibrated-error extremes. No causal domain-shift test or OOF performance claim. Scaling uses all visualization pixels/features, no labels.');(P/'UMAP_RESULT.json').write_text(json.dumps(result,indent=2));np.savez_compressed(P/'UMAP_COORDINATES.npz',u=u,standardized=z);print(json.dumps(result))
