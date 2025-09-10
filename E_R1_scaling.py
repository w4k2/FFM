import strlearn
from ffm import FFM
import numpy as np
from tqdm import tqdm
import time

np.random.seed(3997)

# Stream params
n_chunks = np.linspace(100, 1000, 6).astype(int)
chunk_size = np.linspace(50, 256, 6).astype(int)
dims = np.linspace(8, 256, 8).astype(int)

n_drifts = 0

# Experiment params
reps = 10

results = np.full((reps, len(n_chunks), len(chunk_size), len(dims)), np.nan)
pbar = tqdm(total=reps*len(n_chunks)*len(chunk_size)*len(dims))

# Experiment
for rep_id in range(reps):
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
    
                ffm = FFM(n=5)

                t0 = time.time()
                rep = ffm.describe(stream)
                t = time.time() - t0
                # print(rep.shape)
                # print(t)
                # exit()
            
                results[rep_id, n_ch_id, ch_s_id, d_id] = t
                
                pbar.update(1)    
                np.save('res/e_r1_scale.npy', results)
