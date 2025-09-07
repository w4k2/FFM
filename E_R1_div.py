## Compare variance, entropy, energy (magnitude/amplitude) as a frequency selection mechanism

import strlearn
from tabulate import tabulate
from ffm import FFM
import numpy as np
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

### PART 1 -- EXPERIEMENT

# np.random.seed(3997)

# # Stream params
# n_chunks = 1000
# n_drifts = 3
# percent_informative = 0.3

# chunk_size = 256
# dim = 64

# # Experiment params
# reps = 100
# rs = np.random.randint(100, 100000, reps)

# results = np.full((reps, 3, n_chunks, 8), np.nan)
# pbar = tqdm(total=reps)

# # Experiment
# for _rs_id, _rs in enumerate(rs):
#     stream = strlearn.streams.StreamGenerator(n_chunks=n_chunks,
#                     chunk_size=chunk_size,
#                     n_drifts=n_drifts,
#                     n_features=dim,
#                     n_informative=int(percent_informative*dim),
#                     random_state=_rs)

#     ffm = FFM(n=8)
    
#     ffm.describe(stream, div='var')
#     results[_rs_id, 0] = ffm.mean_fft_all[:,ffm.arg_div]
#     stream.reset()
    
#     ffm.describe(stream, div='ent') 
#     results[_rs_id, 1] = ffm.mean_fft_all[:,ffm.arg_div]
#     stream.reset()

#     ffm.describe(stream, div='eng')  
#     results[_rs_id, 2] = ffm.mean_fft_all[:,ffm.arg_div]

#     pbar.update(1)    
#     np.save('res/e_r1_div.npy', results)
        

### PART 2 -- ANALYZE

from utils import get_gt

res = np.load('res/e_r1_div.npy') # FFM

res_all = np.zeros((100, 3, 4))
gt = get_gt(1000, 3)

strategies = ['Variance', 'Entropy', 'Energy']
metrics = ['NMI', 'Rand', 'Completness', 'Homogenity']
      
for rep in range(100):
    for strategy_id in range(len(strategies)):
        
        samples = res[rep, strategy_id]
        samples[np.isinf(samples)] = 0
        samples[np.isnan(samples)] = 0
        samples_std = StandardScaler().fit_transform(samples)
        clusters_std = KMeans(n_clusters=4).fit_predict(samples_std)
        
        for m_id, m in enumerate([normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score]):
            res_all[rep, strategy_id, m_id] = m(gt, clusters_std)

print(res_all.shape) # 100, 3, 4 = reps, strategy, metrics

res_m = np.round(np.mean(res_all, axis=0),3).astype('object')
res_s = np.round(np.std(res_all, axis=0),3).astype('object')

res_m = np.column_stack((strategies, res_m))
res_s = np.column_stack((strategies, res_s))

print(tabulate(res_m, headers=metrics))
print(tabulate(res_s, headers=metrics))


### PART 3 -- PLOT

fig, ax = plt.subplots(1,4,figsize=(10,3), sharex=True, sharey=True)

for m_id, m in enumerate(metrics):
    ax[m_id].boxplot(res_all[:,:,m_id])
    ax[m_id].set_xticks([1,2,3], strategies)
    ax[m_id].set_title(m)
    ax[m_id].spines['top'].set_visible(False)
    ax[m_id].spines['right'].set_visible(False)
    ax[m_id].grid(ls=':')
    ax[m_id].set_ylim(0.9,1)
    
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig_r1/exp_div.png')
plt.savefig('fig_r1/exp_div.pdf')



