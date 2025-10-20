import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from ffm import FFM
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score
import matplotlib.pyplot as plt

dir = 'experiments_R2/datasets'

files = os.listdir(dir)
try:
    files.remove('.DS_Store')
except:
    pass

files = np.sort(files)
files2 = []
# print(files)

data_info = []
for f_id, f in enumerate(files):
    data = np.loadtxt('%s/%s' % (dir, f), delimiter=',')
    if data.shape[1]>=9:
        print(files2)
        files2.append(f)
        continue
    u, c = np.unique(data[:,-1], return_counts=True)
    data_info.append([f, len(data), data.shape[1]-1, len(u), np.round(np.min(c/len(data)),3)])
    
data_info = np.array(data_info)
np.savetxt('experiments_R2/data_info.csv', data_info, delimiter='\t', fmt="%s")

files = np.array(files2)
print(len(files))

# ___ experiments
metrics = [normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score]
results = np.zeros((len(files),10,4,2))

for f_id, f in enumerate(files):
    data = np.loadtxt('%s/%s' % (dir, f), delimiter=',')
    X, y = data[:,:-1], data[:,-1]
    
    order = np.argsort(y)
    X, y = X[order], y[order]
    
    # establish chunk size
    if len(y)<200:
        chunk_size = 10
    elif len(y)<500:
        chunk_size = 25
    elif len(y)<1000:
        chunk_size = 50
    else:
        chunk_size = 100
        
    n_chunks = len(y)//chunk_size
    print(n_chunks)
    
    gt = []    
    for ch_id in range(n_chunks):
        end = (ch_id+1)*chunk_size
        gt.append(int(y[end-1]))
    
    for rep in range(10):
        
        # FFM
        ffm = FFM(n=4)
        ffm.describe_data(X[:(len(y)//2)], chunk_size=chunk_size)
        cids = ffm.cluster_data(X, chunk_size, np.max(gt)+1)
        for metric_id, m in enumerate(metrics):
            results[f_id, rep, metric_id, 0] = m(gt, cids)
        
        #iPCA
        mean_chunks = []
        for i in range(n_chunks):
            _chunk_X = X[i*chunk_size:(i+1)*chunk_size]
            mean_chunks.append(np.mean(_chunk_X, axis=0))
            
        pca = PCA(n_components=4)
        pca.fit(mean_chunks[:(len(y)//2)])
        meta = pca.transform(mean_chunks)
        samples_std = StandardScaler().fit_transform(meta)
        cids = KMeans(n_clusters=np.max(gt)+1).fit_predict(samples_std)
        for metric_id, m in enumerate(metrics):
            results[f_id, rep, metric_id, 1] = m(gt, cids)

mean_res = np.round(np.mean(results, axis=1),3)
rows = []
for f_id, f in enumerate(files):
    rows.append([f, 
                 mean_res[f_id,0,0], mean_res[f_id,0,1],
                 mean_res[f_id,1,0], mean_res[f_id,1,1],
                 mean_res[f_id,2,0], mean_res[f_id,2,1],
                 mean_res[f_id,3,0], mean_res[f_id,3,1],
                 ])
    

rows = np.array(rows)
np.savetxt('experiments_R2/results_benchmarks.csv', rows, delimiter='\t', fmt="%s")

# ______

fig, ax = plt.subplots(1,4,figsize=(10,3))
for metric_id, metric in enumerate(['NMI', 'Adjusted Rand', 'Completeness', 'Homogeneity']):
        
        bplot = ax[metric_id].boxplot(mean_res[:,metric_id], 
                                              patch_artist=True)
        
        colors = plt.cm.coolwarm(np.mean(mean_res[:,metric_id], axis=0))
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        
        for line in bplot['medians']:
            line.set_color('gray')
                    
        ax[metric_id].grid(ls=':')
        ax[metric_id].spines['top'].set_visible(0)
        ax[metric_id].spines['right'].set_visible(0)
        
        
        ax[metric_id].set_xticks(np.arange(1,3), ['iFFM', 'iPCA'], rotation=45)
        ax[metric_id].set_title(metric)

            
fig.align_ylabels()

plt.tight_layout()
plt.savefig('foo.png') 
