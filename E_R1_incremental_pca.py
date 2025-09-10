## Compare variance, entropy, energy (magnitude/amplitude) as a frequency selection mechanism

from sklearn.decomposition import PCA
import strlearn
from tabulate import tabulate
from ffm import FFM
import numpy as np
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from scipy.stats import entropy
from scipy.ndimage import median, gaussian_filter1d
from utils import get_drfs, get_gt


"""
Incremental Frequency filtering metadescriptor
"""

class iFFM:
    def __init__(self, n, chunk_size):
        self.n = n
        self.chunk_size = chunk_size
        self.arg_div = None
                
    def fit(self, X_chunks):
            
        # divide into chunks

        mean_fft_all = []
        for X in X_chunks:       
            mean_chunk = np.mean(X, axis=0)
            fft_signal = np.fft.fft(mean_chunk)[:len(mean_chunk)//2]
            mean_fft_all.append(fft_signal.real)
                
        s_div = np.var(np.array(mean_fft_all), axis=0)
        self.arg_div = np.flip(np.argsort(s_div))[:self.n]
    
        return self
        
    def describe(self, X_chunks):
        
        mean_fft_all = []
        for X in X_chunks:       
            mean_chunk = np.mean(X, axis=0)
            fft_signal = np.fft.fft(mean_chunk)[:len(mean_chunk)//2]
            mean_fft_all.append(fft_signal.real)
            
        mean_fft_all = np.array(mean_fft_all)
        return mean_fft_all[:, self.arg_div]
        

##### Experiment online

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

results = np.full((reps, len(drift_params), n_chunks, 8), np.nan)
pbar = tqdm(total=reps*3)

n_chunks_to_filter = 100

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
    
        iffm = iFFM(n=8, chunk_size=chunk_size)

        X_fit = []
        X_describe = []
        for chunk_id in range(n_chunks):
            X, y = stream.get_chunk()
            print(chunk_id)
            
            X_describe.append(X)
            
            if chunk_id<n_chunks_to_filter:
                X_fit.append(X)
        
        X_fit = np.array(X_fit)
        X_describe = np.array(X_describe)
        
        print(X_fit.shape)
        print(X_describe.shape)
                    
        X_fit_mean = np.mean(X_fit, axis=1)
        X_describe_mean = np.mean(X_describe, axis=1)
        pca = PCA(n_components=8).fit(X_fit_mean)
        rep = pca.transform(X_describe_mean)
        
        results[_rs_id, dp_id] = rep
        
        pbar.update(1)    
        np.save('res/e_r1_pca_incr.npy', results)
