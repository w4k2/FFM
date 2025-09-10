import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils import get_gt
np.random.seed(92882)

res_a = np.load('res/e2_a.npy') # CED
res_c = np.load('res/e2_c.npy') # ICI
res_d = np.load('res/e2_d.npy') # FFM
res_di = np.load('res/e_r1_incr.npy') # iFFM
res_e = np.load('res/e2_e_8.npy') # PCA
res_ei = np.load('res/e_r1_pca_incr.npy') # iPCA

cluster_reps = 10
rs = np.random.randint(100,10000,cluster_reps)

res_all = np.zeros((6, 10, cluster_reps, 3, 4))
gt = get_gt(500, 5)

for res_id, res in enumerate([res_a, res_c, res_d, res_di, res_e, res_ei]):
        
    for rep in range(10):
        for drift in range(3):
            
            samples = res[rep, drift]
            samples[np.isinf(samples)] = 0
            samples[np.isnan(samples)] = 0
            samples_std = StandardScaler().fit_transform(samples)
            
            for cr_id in range(cluster_reps):
                clusters_std = KMeans(n_clusters=6, random_state=rs[cr_id]).fit_predict(samples_std)
                
                for m_id, m in enumerate([normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score]):
                    try:
                        res_all[res_id, rep, cr_id, drift, m_id] = m(gt, clusters_std)
                    except:
                        pass

res_all = np.mean(res_all, axis=2)
mean_res_all = np.mean(res_all, axis=(1)) # 5, 3, 4
print(res_all.shape)
# exit()

labels = ['CED', 'ICI', 'FFM', 'iFFM', 'PCA', 'iPCA']
cols = plt.cm.coolwarm(np.linspace(0,1,4))

fig, ax = plt.subplots(3,4,figsize=(12,8), sharex=True, sharey=True)

for drf_id, drf in enumerate(['Sudden','Gradual','Incremental']):
    ax[drf_id,0].set_ylabel('%s drift' % drf)
    
    for metric_id, metric in enumerate(['NMI', 'Adjusted Rand', 'Completeness', 'Homogeneity']):
        
        bplot = ax[drf_id, metric_id].boxplot(res_all[:,:,drf_id,metric_id].T, 
                                              patch_artist=True)
        
        colors = plt.cm.coolwarm(mean_res_all[:,drf_id,metric_id])
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        
        for line in bplot['medians']:
            line.set_color('gray')
                    
        ax[drf_id, metric_id].grid(ls=':')
        ax[drf_id, metric_id].spines['top'].set_visible(0)
        ax[drf_id, metric_id].spines['right'].set_visible(0)
        
        
        ax[drf_id, metric_id].set_xticks(np.arange(1,7), labels)

        if drf_id==0:
            ax[drf_id, metric_id].set_title(metric)

            
fig.align_ylabels()

plt.tight_layout()
plt.savefig('foo.png') 
plt.savefig('vis_E2_R1.png')
plt.savefig('vis_E2_R1.pdf')
        