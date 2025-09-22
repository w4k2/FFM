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

### PART 0-- Experiemnt with gradual drift

# np.random.seed(3997)

# # Stream params
# n_chunks = 1000
# n_drifts = 5
# percent_informative = 0.3

# chunk_size = 256
# dim = 128

# # Experiment params
# reps = 100
# rs = np.random.randint(100, 100000, reps)

# results = np.full((reps, n_chunks, 8), np.nan)
# pbar = tqdm(total=reps)

# # Experiment
# for _rs_id, _rs in enumerate(rs):
#     stream = strlearn.streams.StreamGenerator(n_chunks=n_chunks,
#                     chunk_size=chunk_size,
#                     n_drifts=n_drifts,
#                     n_features=dim,
#                     n_informative=int(percent_informative*dim),
#                     random_state=_rs,
#                     concept_sigmoid_spacing=5)

#     ffm = FFM(n=8)
    
#     ffm.describe(stream, div='var')
#     results[_rs_id] = ffm.mean_fft_all[:,ffm.arg_div]

#     pbar.update(1)    
#     np.save('res/e_r1_grad.npy', results)

# exit()

# PART 1 --clustering

# from utils import get_gt

# res = np.load('res/e_r1_grad.npy')
# print(res.shape) #(100, 1000, 8) == reps x chunks x metafeatures

# gt = get_gt(1000, 3)

# metrics = ['NMI', 'Rand', 'Completness', 'Homogenity']
# clustering = [
#     AffinityPropagation(),
#     AgglomerativeClustering(n_clusters=6),
#     Birch(n_clusters=6),
#     BisectingKMeans(n_clusters=6),
#     DBSCAN(),
#     KMeans(n_clusters=6),
#     MeanShift(),
#     OPTICS(),
#     SpectralClustering(n_clusters=6)
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
#         np.save('res/e_r1_cluster_grad.npy', res_all)
   

### PART 2 -- ANALYZE metrics

cluster_names = [
    'AP', 'AC', 'B', 'BKM', 'DBS',
    'KM', 'MS', 'OPT', 'SC'
]
metrics = ['NMI', 'Rand', 'Completness', 'Homogenity']

res_all = np.load('res/e_r1_cluster_grad.npy')
print(res_all.shape) # 100, 9, 4 = reps, clustering, metrics

res_m = np.round(np.mean(res_all, axis=0),3).astype('object')
res_s = np.round(np.std(res_all, axis=0),3).astype('object')

print(np.min(res_all))

res_m = np.column_stack((cluster_names, res_m))
res_s = np.column_stack((cluster_names, res_s))

print(tabulate(res_m, headers=metrics))
print(tabulate(res_s, headers=metrics))


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
plt.savefig('fig_r1/exp_cluster_grad.png')
plt.savefig('fig_r1/exp_cluster_grad.pdf')
