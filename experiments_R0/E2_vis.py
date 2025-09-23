import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils import get_gt

res_a = np.load('res/e2_a.npy') # CED
res_c = np.load('res/e2_c.npy') # ICI
res_d = np.load('res/e2_d.npy') # FFM
res_e = np.load('res/e2_e.npy') # PCA

res_all = np.zeros((4, 10, 3, 4))
gt = get_gt(1000, 3)

for res_id, res in enumerate([res_a, res_c, res_d, res_e]):
        
    for rep in range(10):
        for drift in range(3):
            
            samples = res[rep, drift]
            samples[np.isinf(samples)] = 0
            samples[np.isnan(samples)] = 0
            samples_std = StandardScaler().fit_transform(samples)
            clusters_std = KMeans(n_clusters=4).fit_predict(samples_std)
            
            for m_id, m in enumerate([normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score]):
                res_all[res_id, rep, drift, m_id] = m(gt, clusters_std)

mean_res_all = np.mean(res_all, axis=1) # 5, 3, 4

labels = ['CED', 'ICI', 'FFM', 'PCA']
cols = plt.cm.coolwarm(np.linspace(0,1,4))

fig, ax = plt.subplots(3,4,figsize=(10,5), sharex=True, sharey=True)

for drf_id, drf in enumerate(['Sudden','Gradual','Incremental']):
    ax[drf_id,0].set_ylabel('%s drift' % drf)
    
    for metric_id, metric in enumerate(['NMI', 'Adjusted Rand', 'Completeness', 'Homogeneity']):
        
        ax[drf_id, metric_id].bar(np.arange(4), mean_res_all[:,drf_id,metric_id], width=0.35, color = plt.cm.coolwarm(mean_res_all[:,drf_id,metric_id]))
        ax[drf_id, metric_id].grid(ls=':')
        ax[drf_id, metric_id].spines['top'].set_visible(0)
        ax[drf_id, metric_id].spines['right'].set_visible(0)
        
        
        ax[drf_id, metric_id].set_xticks(np.arange(4), labels)

        if drf_id==0:
            ax[drf_id, metric_id].set_title(metric)

            
fig.align_ylabels()

plt.tight_layout()
plt.savefig('foo.png') 
plt.savefig('vis_E2.png')
plt.savefig('vis_E2.pdf')
        