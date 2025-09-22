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
from scipy.ndimage import median, gaussian_filter1d, median_filter
from utils import get_drfs, get_gt


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
    
    
# -----------------------------------------------------------------------
np.random.seed(3997)

# Stream params
n_chunks = 500
n_drifts = 5
percent_informative = 0.3

chunk_size = 128
dim = 32

stream = strlearn.streams.StreamGenerator(n_chunks=n_chunks,
                    chunk_size=chunk_size,
                    n_drifts=n_drifts,
                    n_features=dim,
                    n_informative=int(percent_informative*dim),
                    random_state=34323)

n_chunks_to_filter = 100
offm = OFFM(n=8, chunk_size=chunk_size, n_chunks_to_filter=n_chunks_to_filter)

reps = []
components = []
    
for chunk_id in range(n_chunks):
    X, y = stream.get_chunk()
    print(chunk_id)
    
    for sample in X:
        
        # Strategy 1 -- offline phase
        # if offm._check_is_fitted() == False:
        #     offm.fit(sample)
            
        # Strategy 2 -- incremental fit 
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

drift_gt = get_gt(500,n_drifts)[n_chunks_to_filter:]

fig, ax = plt.subplots(1,3,figsize=(12,4))
cols = plt.cm.viridis(np.linspace(0,0.9,4))
for a in range(4):
    # ax[0].plot(gaussian_filter1d(reps[:,a],5), c=cols[a])
    ax[0].plot(median_filter(reps[:,a],25)+(3*a), 
               c=cols[a], alpha=0.95, label='metafeature %i' % a)

ax[0].set_xticks(get_drfs(500,n_drifts)-n_chunks_to_filter)
ax[0].grid(ls=':')
ax[0].set_xlim(-100,400)
ax[0].set_title('FFM representation')
ax[0].set_xlabel('chunk')
ax[0].set_ylabel('adjusted representation value')
ax[0].legend(ncols=2, loc='upper left', frameon=False)
ax[0].set_ylim(-2,14)

for a in range(1):
    # ax[1].plot(gaussian_filter1d(components[:,a],5), c=cols[a])
    # ax[1].plot(median_filter(components[:,a],5), c=cols[a])
    ax[1].scatter(np.arange(len(components[:,0])), components[:,a], c='black', s=5)

ax[1].set_xticks(get_drfs(500,n_drifts)-n_chunks_to_filter)
ax[1].grid(ls=':')
ax[1].set_xlim(-100,400)
ax[1].set_title('most informative frequency')
ax[1].set_xlabel('chunk')
ax[1].set_ylabel('frequency')

pca_rep = PCA(n_components=2).fit_transform(reps)
pca_comp = PCA(n_components=2).fit_transform(components)
ax[2].scatter(pca_rep[:,0], pca_rep[:,1], c=drift_gt, s=10)
# ax[2].scatter(pca_comp[:,0], pca_comp[:,1], c=drift_gt)
ax[2].grid(ls=':')
ax[2].set_title('principal components')
ax[2].set_xlabel('component 0')
ax[2].set_ylabel('component 1')

for aa in ax:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig_r1/online-sample.png')
plt.savefig('fig_r1/online-sample.pdf')
    
   