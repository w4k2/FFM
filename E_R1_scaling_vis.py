import strlearn
from ffm import FFM
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

np.random.seed(3997)

# Stream params
n_chunks = np.linspace(100, 1000, 6).astype(int)
chunk_size = np.linspace(50, 256, 6).astype(int)
dims = np.linspace(8, 256, 8).astype(int)

res = np.load('res/e_r1_scale.npy')
mean_res = np.mean(res, axis=0)
print(res.shape) # (10, 6, 6, 8) = reps, chunks, chunk size, dims

fig, ax = plt.subplots(1,3,figsize=(12,4), sharex=False, sharey=True)

ax[0].boxplot(res[:,:,0,0])
ax[0].set_xlabel('n_chunks')
ax[0].set_xticks(np.arange(len(n_chunks))+1, n_chunks)
ax[0].set_ylabel('time')

ax[1].boxplot(res[:,0,:,0])
ax[1].set_xlabel('chunk_size')
ax[1].set_xticks(np.arange(len(chunk_size))+1, chunk_size)
# ax[1].set_ylabel('time')

ax[2].boxplot(res[:,0,0,:])
ax[2].set_xlabel('dimensionality')
ax[2].set_xticks(np.arange(len(dims))+1, dims)
# ax[2].set_ylabel('time')

for aa in ax:
    aa.grid(ls=':')
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    
plt.tight_layout()
plt.savefig('foo.png')