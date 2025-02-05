import numpy as np
from ffm import FFM
import os
from strlearn.streams import ARFFParser
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

files = os.listdir('insects')

print(files)
chunk_size=[100,500,50,200,50,100,500,100,500,500]
n=5

search = np.arange(4,9)
metrics = [silhouette_score, calinski_harabasz_score, davies_bouldin_score]
res = np.zeros((len(files), 10, len(search), len(metrics)))

for f_id, f in enumerate(files):
    
    data = []
    stream = ARFFParser('insects/%s' % f, chunk_size=chunk_size[f_id], n_chunks=10000000)
    while(1):
        try:
            data.extend(stream.get_chunk()[0])
        except:
            break
       
    # print(f) 
    # print(len(data))
    # print(data[0].shape)
    # continue
    
    print(len(data))
    print(data[0].shape)
        
    ffm = FFM(n=n)
    ffm.describe_data(data, chunk_size[f_id])
    
    rep = ffm.mean_fft_all[:,ffm.arg_var]
    print(rep.shape)
    
    rep = StandardScaler().fit_transform(rep)
    
    # first identify number of clusters
    for i in range(10):
        for s_id, s in enumerate(search):
            clusters = KMeans(n_clusters=s).fit_predict(rep)
            for m_id, m in enumerate(metrics):
                res[f_id, i, s_id, m_id] = m(rep, clusters)
        
np.save('res/e4.npy', res)
    
