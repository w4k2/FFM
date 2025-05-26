import numpy as np
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score, normalized_mutual_info_score
from tabulate import tabulate

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

print(res_all.shape) # 4, 10, 3, 4 = method, reps, drift, metrics

labels = ['CED', 'ICI', 'FFM', 'PCA']
drifts = ['Sudden','Gradual','Incremental']
metrics = ['NMI', 'Rand', 'Completness', 'Homogenity']

all_metric_rows = []

for m_id, m in enumerate(metrics):
    all_metric_rows.append([m])

    for drf_id, drf in enumerate(drifts):

        row = []
        row.append('%s' % (drf))
        
        aa =  res_all[:, :, drf_id, m_id].swapaxes(0,1) # 10, 4
        print(aa.shape)
        
        mean_aa = np.nanmean(aa, axis=0)
        std_aa = np.nanstd(aa, axis=0)
        
        # ttest
        alpha = 0.05
        
        t_stat_all = np.full((4,4), np.nan)
        pval_all = np.full((4,4), np.nan)
        
        for i in range(4):
            for j in range(4):
                t_stat, pval = ttest_ind(aa[:,i], aa[:,j])
                print(t_stat, pval)
                t_stat_all[i,j] = t_stat
                pval_all[i,j] = pval
                
        significant = pval_all<alpha
        better = t_stat_all>0 
        
        significantly_better = significant*better                

        for method_id, method in enumerate(labels):
            row.append('%.3f (%.3f)' % (mean_aa[method_id], std_aa[method_id]))
        
        all_metric_rows.append(row)

        row = ['',]
        for method_id, method in enumerate(labels):
            b = np.argwhere(significantly_better[method_id])
            row.append(', '.join('%s' % _ for _ in b))
            print(row)
        all_metric_rows.append(row)
    

print(tabulate(all_metric_rows, tablefmt='latex'))
            


