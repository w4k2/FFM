import numpy as np
from tabulate import tabulate
from ffm import FFM
import os
from strlearn.streams import ARFFParser
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

files = os.listdir('insects')
files.remove('.DS_Store')
print(files)

chunk_size=[100,500,50,200,50,100,500,100,500,500]
n=8

metrics = [silhouette_score, calinski_harabasz_score, davies_bouldin_score]
res = np.zeros((len(files), len(metrics)))

rows = []

for f_id, f in enumerate(files):
    
    data = []
    stream = ARFFParser('insects/%s' % f, chunk_size=chunk_size[f_id], n_chunks=10000000)
    while(1):
        try:
            data.extend(stream.get_chunk()[0])
        except:
            break
    
    print(len(data))
    print(data[0].shape)
        
    ffm = FFM(n=n)
    ffm.describe_data(data, chunk_size[f_id])
    
    rep = ffm.mean_fft_all[:,ffm.arg_div]
    print(rep.shape)
    
    rep = StandardScaler().fit_transform(rep)
    
    # first identify number of clusters
    clusters = np.load('insects_data_streams/%i_ci.npy' % f_id)
    for m_id, m in enumerate(metrics):
        res[f_id, m_id] = m(rep, clusters)
        
    
    rows.append([ '%s' % files[f_id].split('_norm.')[0], 
                 '%i' % len(np.unique(clusters)),
                 '%.3f' % (res[f_id,0]),
                 '%.3f' % (res[f_id,1]),
                 '%.3f' % (res[f_id,2])
                 ])

print(tabulate(rows, tablefmt='latex'))
