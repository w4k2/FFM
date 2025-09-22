from sklearn.decomposition import PCA
import strlearn
from ffm import FFM
import numpy as np
from tqdm import tqdm
import time

np.random.seed(3997)

# Stream params
n_chunks = np.linspace(100, 1000, 6).astype(int)
chunk_size = np.linspace(50, 500, 8).astype(int)
dims = np.linspace(8, 512, 10).astype(int)

n_drifts = 0

# Experiment params
reps = 50

results = np.full((reps, len(n_chunks), len(chunk_size), len(dims)), np.nan)
pbar = tqdm(total=reps*len(n_chunks)*len(chunk_size)*len(dims))

# Experiment
for n_ch_id, n_ch in enumerate(n_chunks):
    for ch_s_id, ch_s in enumerate(chunk_size):
        for d_id, d in enumerate(dims):
                
            stream = strlearn.streams.StreamGenerator(
                n_chunks=n_ch,
                chunk_size=ch_s,
                n_drifts=n_drifts,
                n_features=d,
                n_informative=d//3,
                random_state=1232)
            
            data = []
            for chunk_id in range(n_ch):
                data.extend(stream.get_chunk()[0])
            
            data = np.array(data)
            print(data.shape)
        
            for rep_id in range(reps):
            
                pca = PCA(n_components=8)
                t0 = time.time()
                
                n_chunks = len(data)//ch_s
                chunk_means = []
                for chunk_id in range(n_chunks):
                    X = data[chunk_id*ch_s : (chunk_id+1)*ch_s]     
                    mean_chunk = np.mean(X, axis=0)
                    chunk_means.append(mean_chunk)
                rep = pca.fit_transform(chunk_means)
                t = time.time() - t0
            
                results[rep_id, n_ch_id, ch_s_id, d_id] = t
                
                pbar.update(1)    
                np.save('res/e_r1_scale_pca.npy', results)
