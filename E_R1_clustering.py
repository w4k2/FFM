## Compare variance, entropy, energy (magnitude/amplitude) as a frequency selection mechanism

from sklearn import clone
import strlearn
from tabulate import tabulate
from ffm import FFM
import numpy as np
from tqdm import tqdm
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS, AffinityPropagation, AgglomerativeClustering, Birch, BisectingKMeans, FeatureAgglomeration, KMeans, MeanShift, SpectralBiclustering, SpectralClustering, SpectralCoclustering
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

### PART 1 -- Cluster bes representation from R1_div

# from utils import get_gt

# res = np.load('res/e_r1_div.npy')
# print(res.shape) #(100, 3, 1000, 8)

# res = res[:,0] # variance
# print(res.shape) #(100, 1000, 8) == reps x chunks x metafeatures

# gt = get_gt(1000, 3)

# metrics = ['NMI', 'Rand', 'Completness', 'Homogenity']
# clustering = [
#     AffinityPropagation(),
#     AgglomerativeClustering(n_clusters=4),
#     Birch(n_clusters=4),
#     BisectingKMeans(n_clusters=4),
#     DBSCAN(),
#     KMeans(n_clusters=4),
#     MeanShift(),
#     OPTICS(),
#     SpectralClustering(n_clusters=4)
# ]

# res_all = np.zeros((100, len(clustering), len(metrics)))
# pbar = tqdm(total=100*len(clustering))

# for rep in range(100):
#     samples = res[rep]
#     samples[np.isinf(samples)] = 0
#     samples[np.isnan(samples)] = 0
#     samples_std = StandardScaler().fit_transform(samples)
    
#     for c_id, clustering_alg in enumerate(clustering):
#         alg = clone(clustering_alg)
#         try:
#             clusters_std = alg.fit_predict(samples_std)
#         except:
#             alg.fit(samples_std)
#             clusters_std = alg.labels_

#         for m_id, m in enumerate([normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score]):
#             res_all[rep, c_id, m_id] = m(gt, clusters_std)
        
#         pbar.update(1)
#         np.save('res/e_r1_cluster.npy', res_all)
   

### PART 2 -- ANALYZE metrics

cluster_names = [
    'AP', 'AC', 'B', 'KNM', 'DBS',
    'KM', 'MS', 'OPT', 'SC'
]
metrics = ['NMI', 'Rand', 'Completness', 'Homogenity']

res_all = np.load('res/e_r1_cluster.npy')
print(res_all.shape) # 100, 9, 4 = reps, clustering, metrics

res_m = np.round(np.mean(res_all, axis=0),3).astype('object')
res_s = np.round(np.std(res_all, axis=0),3).astype('object')

res_m = np.column_stack((cluster_names, res_m))
res_s = np.column_stack((cluster_names, res_s))

print(tabulate(res_m, headers=cluster_names))
print(tabulate(res_s, headers=cluster_names))


### PLOT

fig, ax = plt.subplots(1,4,figsize=(10,3), sharex=True, sharey=True)

for m_id, m in enumerate(metrics):
    ax[m_id].boxplot(res_all[:,:,m_id])
    ax[m_id].set_xticks(np.arange(1,10), cluster_names, rotation=45)
    ax[m_id].set_title(m)
    ax[m_id].spines['top'].set_visible(False)
    ax[m_id].spines['right'].set_visible(False)
    ax[m_id].grid(ls=':')
    # ax[m_id].set_ylim(0.9,1)
    
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig_r1/exp_cluster.png')
plt.savefig('fig_r1/exp_cluster.pdf')

# exit()
mask = np.array([0,1,1,1,0,1,0,0,1]).astype(bool)
res_all = res_all[:,mask]
cluster_names = np.array(cluster_names)[mask]

fig, ax = plt.subplots(1,4,figsize=(10,3), sharex=True, sharey=True)

for m_id, m in enumerate(metrics):
    ax[m_id].boxplot(res_all[:,:,m_id])
    ax[m_id].set_xticks(np.arange(1,len(cluster_names)+1), cluster_names, rotation=45)
    ax[m_id].set_title(m)
    ax[m_id].spines['top'].set_visible(False)
    ax[m_id].spines['right'].set_visible(False)
    ax[m_id].grid(ls=':')
    # ax[m_id].set_ylim(0.9,1)
    
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig_r1/exp_cluster_z.png')
plt.savefig('fig_r1/exp_cluster_z.pdf')



