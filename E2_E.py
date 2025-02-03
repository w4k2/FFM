from sklearn.decomposition import PCA
import strlearn
import numpy as np
from tqdm import tqdm

"""
Experiment 2: Compare with different metafeatures on single data stream type


A. Meta-Feature-based Concept Evolution Detection framework on Feature Streams (Guo et al., 2023),  
    mean, standard deviation, correlation, skewness, and kurtosis

C. On metafeatures ability of implicit concept identification
    int, nre, c1, f1.mean, cl_c.mean, cl_c.sd, cl_ent,
    j_ent.mean, j_ent.sd, mi.mean, mi.sd, wn.mean,
    g_m.sd, mean.mean, mean.sd, med.mean, t_m.mean
    
D. FFM

E. PCA from original features

"""

np.random.seed(3997)

# Stream params
n_chunks = 1000
n_drifts = 3
percent_informative = 0.3

chunk_size = 256
dim = 64
drift_params = [
    {'incremental':False,
     'concept_sigmoid_spacing':999},
    {'incremental':False,
     'concept_sigmoid_spacing':5},
    {'incremental':True,
     'concept_sigmoid_spacing':5},
]

# Experiment params
reps = 10
rs = np.random.randint(100, 100000, reps)

results = np.full((reps, len(drift_params), n_chunks, 2), np.nan)
pbar = tqdm(total=reps*3*1000)

# Experiment
for dp_id, dp in enumerate(drift_params):
    for _rs_id, _rs in enumerate(rs):
                    
        stream = strlearn.streams.StreamGenerator(n_chunks=n_chunks,
                        chunk_size=chunk_size,
                        n_drifts=n_drifts,
                        n_features=dim,
                        n_informative=int(percent_informative*dim),
                        random_state=_rs,
                        **dp)
    
        meta_all = []
        for i in range(n_chunks):
            X, y = stream.get_chunk()
            
            meta = np.mean(X, axis=0)
            
            meta_all.append(meta)
            pbar.update(1)    

            
        meta_all = np.array(meta_all)
        meta_all = PCA(n_components=2).fit_transform(meta_all)
        
        results[_rs_id, dp_id] = meta_all

        np.save('res/e2_e.npy', results)