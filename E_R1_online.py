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
from scipy.ndimage import median
from utils import get_gt

"""
Online Frequency filtering metadescriptor
"""

class OFFM:
    def __init__(self, n, chunk_size, n_chunks_to_filter):
        self.n = n
        self.chunk_size = chunk_size
        self.n_chunks_to_filter = n_chunks_to_filter
        
        self.samples = []
        self.arg_div = None
        
        self.buffer = []
        
    def fit(self, sample):
        
        self.samples.append(sample)
        if len(self.samples)>=self.chunk_size*self.n_chunks_to_filter:
            
            # divide into chunks
            X_chunks = np.array(self.samples).reshape(self.n_chunks_to_filter, self.chunk_size, -1) # order

            mean_fft_all = []
            for X in X_chunks:       
                mean_chunk = np.mean(X, axis=0)
                fft_signal = np.fft.fft(mean_chunk)[:len(mean_chunk)//2]
                mean_fft_all.append(fft_signal.real)
                  
            s_div = np.var(np.array(mean_fft_all), axis=0)
            self.arg_div = np.flip(np.argsort(s_div))[:self.n]
            
            self.samples.pop(0) # remove first
        
        return self
        
    def describe(self, sample):
        
        if len(self.buffer)>=self.chunk_size:
            self.buffer.pop(0) # remove first
        
        self.buffer.append(sample)
        
        if self._check_is_fitted():
            mean_buffer = np.mean(np.array(self.buffer), axis=0)
            # print('mean_buffer.shape', mean_buffer.shape)
        
            fft_signal = np.fft.fft(mean_buffer)[:len(mean_buffer)//2]
            fft_signal = fft_signal.real
            
            return fft_signal[self.arg_div]
        
        else:
            return None
    
    def _check_is_fitted(self):
        if self.arg_div is not None:
            return True
        return False
    
    # def describe_stream(self, X):
        
    #     n_chunks = len(X)//self.chunk_size
    #     X_chunks = X_chunks.reshape(n_chunks, self.chunk_size, -1)
        
    #     for X in X_chunks:       
    #         mean_chunk = np.mean(X, axis=0)
    #         fft_signal = np.fft.fft(mean_chunk)[:len(mean_chunk)//2]
    #         self.mean_fft_all.append(fft_signal.real)
                
    #     self.mean_fft_all = np.array(self.mean_fft_all)
        
    #     return self.mean_fft_all[self.arg_div]
    
# -----------------------------------------------------------------------
np.random.seed(3997)

# Stream params
n_chunks = 500
n_drifts = 3
percent_informative = 0.3

chunk_size = 64
dim = 32

stream = strlearn.streams.StreamGenerator(n_chunks=n_chunks,
                    chunk_size=chunk_size,
                    n_drifts=n_drifts,
                    n_features=dim,
                    n_informative=int(percent_informative*dim),
                    random_state=34323)

offm = OFFM(n=8, chunk_size=64, n_chunks_to_filter=100)

reps = []
components = []
    
for chunk_id in range(n_chunks):
    X, y = stream.get_chunk()
    print(chunk_id)
    
    for sample in X:
        
        offm.fit(sample)
        rep = offm.describe(sample)
        
        if rep is not None:
            reps.append(rep)
            components.append(offm.arg_div)
        
print(reps)
reps = np.array(reps)

# divide into chunks
n_chunks = len(reps)//chunk_size

reps = reps[:n_chunks*chunk_size].reshape(n_chunks, chunk_size, 8)
reps = np.mean(reps, axis=1)

components = np.array(components)
components = components[:n_chunks*chunk_size].reshape(n_chunks, chunk_size, 8)
components = np.mean(components, axis=1)

print(components.shape)
print(reps.shape)

pca_features = PCA(n_components=2).fit_transform(reps)
drift_gt = get_gt(500,3)[100:]

fig, ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.ravel()

ax[0].scatter(pca_features[:,0], pca_features[:,1], c=drift_gt)

for a in range(4):
    ax[1].plot(reps[:,a])
    
for a in range(4):
    ax[2].plot(components[:,a])

plt.tight_layout()
plt.savefig('foo.png')
    
   