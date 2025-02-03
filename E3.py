from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
import strlearn
import numpy as np
from ffm import FFM
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from tqdm import tqdm


'''
Experiment 3: Discover number of concepts

'''

np.random.seed(28837)

reps = 10
rs = np.random.randint(100, 100000, reps)

n_concepts = [2,4,6,8,10]
chunk_size = [100,200,400]
n_features = 500

metrics = [davies_bouldin_score, silhouette_score, calinski_harabasz_score]
n_consiedred_concepts = np.arange(2,12)

res = np.full((reps, len(chunk_size), len(n_concepts), len(n_consiedred_concepts), len(metrics)), np.nan)
pbar = tqdm(total=reps*len(n_concepts)*len(chunk_size))


for _rs_id, _rs in enumerate(rs):
    for ch_s_id, ch_s in enumerate(chunk_size):

        for n_concepts_id, n_c in enumerate(n_concepts):
                    
            stream = strlearn.streams.StreamGenerator(
                n_chunks=500,
                chunk_size=ch_s,
                concept_sigmoid_spacing=999,
                n_drifts=(n_c-1),
                n_features=n_features,
                n_informative=int(0.3*n_features),
                incremental=False,
                recurring=False,
                random_state=_rs)

            ffm = FFM(n=16)

            ffm.describe(stream)
            rep = ffm.mean_fft_all[:,ffm.arg_var]
            rep = StandardScaler().fit_transform(rep)

            for n_considered_id, n_considered in enumerate(n_consiedred_concepts):
                
                clusters = KMeans(n_clusters=n_considered).fit_predict(rep)
                
                for metric_id, metric in enumerate(metrics):
                    res[_rs_id, ch_s_id, n_concepts_id, n_considered_id, metric_id] = metric(rep, clusters)

            pbar.update(1)
            print(res[_rs_id, ch_s_id, n_concepts_id])
            np.save('res/e3.npy', res)               
            
pbar.close()
