from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import strlearn
from ffm import FFM
from utils import get_gt, get_drfs
import numpy as np

stream = strlearn.streams.StreamGenerator(n_chunks=400, chunk_size=200, 
                                          n_features=2, n_informative=2, 
                                          n_redundant=0, n_repeated=0,
                                          n_clusters_per_class=2,
                                          n_drifts=3, random_state=332)

drifts_gt = get_gt(400,3)
drifts_moments = get_drfs(400,3)

accs = []

chunks_idx = [30, 120, 240, 360]
chunks_dist = []

clf = MLPClassifier()
for chunk_id in range(400):
    X, y = stream.get_chunk()
    
    if chunk_id in chunks_idx:
        chunks_dist.append([X, y])
    
    if chunk_id>0:
        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        accs.append(acc)
    
    
    clf.partial_fit(X,y,[0,1])    

# Plot concept clusters
fig, ax = plt.subplots(3,1,figsize=(10,5))

cols = plt.cm.coolwarm(np.linspace(0,1,4))

ax[1].plot(accs, c='black')
ax[1].set_xticks(drifts_moments)
# ax[2].scatter(np.arange(400), clusters, c=cols[clusters])
ax[2].scatter(np.arange(400), drifts_gt, c=cols[drifts_gt])
ax[2].set_xticks(drifts_moments)

ax[1].grid(ls=':')
ax[2].grid(ls=':')

ax[1].set_xlim(0,400)
ax[2].set_xlim(0,400)

ax[1].set_ylabel('accuracy')
ax[2].set_ylabel('concept identifier')
# ax[1].set_xlabel('chunk of a concept change')
ax[2].set_xlabel('chunk index / concept change')

ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].spines['left'].set_visible(False)

ax1 = plt.subplot(3,4,1)
# pca1 = PCA(n_components=2).fit_transform(chunks_dist[0][0])
pca1 = chunks_dist[0][0]
ax1.scatter(pca1[:,0], pca1[:,1], c=chunks_dist[0][1], cmap='coolwarm', s=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(ls=':')
ax1.set_title('chunk: %i' % chunks_idx[0])

ax2 = plt.subplot(3,4,2)
# pca2 = PCA(n_components=2).fit_transform(chunks_dist[1][0])
pca2 = chunks_dist[1][0]
ax2.scatter(pca2[:,0], pca2[:,1], c=chunks_dist[1][1], cmap='coolwarm', s=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(ls=':')
ax2.set_title('chunk: %i' % chunks_idx[1])

ax3 = plt.subplot(3,4,3)
# pca3 = PCA(n_components=2).fit_transform(chunks_dist[2][0])
pca3 = chunks_dist[2][0]
ax3.scatter(pca3[:,0], pca3[:,1], c=chunks_dist[2][1], cmap='coolwarm', s=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(ls=':')
ax3.set_title('chunk: %i' % chunks_idx[2])

ax4 = plt.subplot(3,4,4)
# pca4 = PCA(n_components=2).fit_transform(chunks_dist[3][0])
pca4 = chunks_dist[3][0]
ax4.scatter(pca4[:,0], pca4[:,1], c=chunks_dist[3][1], cmap='coolwarm', s=10)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(ls=':')
ax4.set_title('chunk: %i' % chunks_idx[3])

for aa in [ax1, ax2, ax3, ax4]:
    aa.set_xlabel('feature 0')
    aa.set_ylabel('feature 1')
    
ax[2].set_yticks([0,1,2,3])

plt.tight_layout()
plt.savefig('foo.png')
