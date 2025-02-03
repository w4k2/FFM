from sklearn.discriminant_analysis import StandardScaler
import strlearn
from ffm import FFM
from utils import get_gt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

"""
Experiment 0: Evaluate descriptor param: n
"""

np.random.seed(3997)

# Stream params
n_chunks = 500
dim = 500
percent_informative = 0.3

chunk_size = [50, 100, 200]
n_drifts = [1, 3, 5, 7, 9]

# Descriptor params 
ns = [1,2,4,8,16]

# Experiment params
reps = 10
rs = np.random.randint(100, 100000, reps)

metrics = [normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score]

results = np.full((reps, len(n_drifts), len(chunk_size), len(ns), len(metrics)), np.nan)
pbar = tqdm(total=reps*len(n_drifts)*len(chunk_size)*len(ns))


# Experiment
for _rs_id, _rs in enumerate(rs):
    for _nd_id, _nd in enumerate(n_drifts):
        gt = get_gt(n_chunks, _nd)

        for _chs_id, _chs in enumerate(chunk_size):
        
            stream = strlearn.streams.StreamGenerator(n_chunks=n_chunks,
                                            chunk_size=_chs,
                                            concept_sigmoid_spacing=999,
                                            n_drifts=_nd,
                                            n_features=dim,
                                            n_informative=int(percent_informative*dim),
                                            incremental=False,
                                            recurring=False,
                                            random_state=_rs)


            for _n_id, _n in enumerate(ns):
                    
                ffm = FFM(n=_n)
                stream.reset()
                ffm.describe(stream)

                rep = ffm.mean_fft_all[:,ffm.arg_var]   
                # normalize
                rep = StandardScaler() .fit_transform(rep)        
                
                #cluster with k-means
                clusters = KMeans(n_clusters=_nd+1).fit_predict(rep)
                print(clusters[:10], gt[:10])
                                                
                for metric_id, metric in enumerate(metrics):
                    results[_rs_id, _nd_id, _chs_id, _n_id, metric_id] = metric(clusters, gt)

                print(results[_rs_id, _nd_id, _chs_id, _n_id])
                pbar.update(1)
                np.save('res/e0.npy', results)
