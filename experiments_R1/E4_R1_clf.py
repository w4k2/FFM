import numpy as np
from sklearn.neural_network import MLPClassifier
from ffm import FFM
import os
from strlearn.streams import ARFFParser
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, silhouette_score

files = os.listdir('insects')
files.remove('.DS_Store')
print(files)
# exit()

print(files)
chunk_size=[100,500,50,200,50,100,500,100,500,500]
n = 8

for f_id, f in enumerate(files):
    
    data = []
    data_y = []
    stream = ARFFParser('insects/%s' % f, chunk_size=chunk_size[f_id], n_chunks=10000000)
    while(1):
        try:
            X, y = stream.get_chunk()
            data.extend(X)
            data_y.extend(y)
        except:
            break
    
    print(len(data))
    print(data[0].shape)
    print(np.unique(data_y))
    # exit()
    
    
    ffm = FFM(n=n)
    ffm.describe_data(data, chunk_size[f_id])
    
    rep = ffm.mean_fft_all[:,ffm.arg_div]
    print(rep.shape)
    
    rep = StandardScaler().fit_transform(rep)
    
    # first identify number of clusters
    scores_reps = []
    for i in range(10):
        search = np.arange(4,9)
        scores = []
        
        for s in search:
            clusters = KMeans(n_clusters=s).fit_predict(rep)
            scores.append(silhouette_score(rep, clusters))
        
        scores_reps.append(scores)
    
    scores_reps = np.array(scores_reps)
    print(scores_reps.shape)
    scores_reps = np.mean(scores_reps, axis=0)
    best = search[np.argmax(scores_reps)]
    print(best)
    
    # cluster
    clusters = KMeans(n_clusters=best).fit_predict(StandardScaler().fit_transform(rep))
    print(clusters)
    
    clusters_2 = np.copy(clusters)

    mapping_src = []
    for i in clusters:
        if i not in mapping_src:
            mapping_src.append(i)
            
    for i_id, i in enumerate(mapping_src):
        clusters_2[clusters==i] = i_id
    
    clusters = clusters_2
    
    ###
    chunks_X = []
    chunks_y = []
    n_chunks = len(data)//chunk_size[f_id]
    for chunk_id in range(n_chunks):
        s = chunk_id*chunk_size[f_id]
        e = (chunk_id+1)*chunk_size[f_id]
        X_chunk = data[s:e]
        y_chunk = data_y[s:e]
        chunks_X.append(X_chunk)
        chunks_y.append(y_chunk)
    
    # save clusters and data
    chunks_X = np.array(chunks_X)
    chunks_y = np.array(chunks_y)
    print(chunks_X.shape)
    print(chunks_y.shape)
    print(clusters.shape)
    np.save('insects_data_streams/%i_X.npy' % f_id, chunks_X)
    np.save('insects_data_streams/%i_y.npy' % f_id, chunks_y)
    np.save('insects_data_streams/%i_ci.npy' % f_id, clusters)
    