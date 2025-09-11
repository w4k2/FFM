import strlearn
from ffm import FFM
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

np.random.seed(3997)

# Stream params
n_chunks = np.linspace(100, 1000, 6).astype(int)
chunk_size = np.linspace(50, 500, 8).astype(int)
dims = np.linspace(8, 512, 10).astype(int)

res = np.load('res/e_r1_scale.npy')*1000

print(res.shape) # (10, 6, 6, 8) = reps, chunks, chunk size, dims

fig, ax = plt.subplots(1,3,figsize=(12,4), sharex=False, sharey=True)

res1 = res.swapaxes(0,1).reshape(len(n_chunks), -1).T
mean_res1 = np.mean(res1, axis=0)
mean_res1 /= 1.2*np.max(mean_res1) 

bplot = ax[0].boxplot(res1, patch_artist=True)
colors = plt.cm.coolwarm(mean_res1)
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

for line in bplot['medians']:
    line.set_color('gray')
    
ax[0].set_xlabel('number of chunks')
ax[0].set_xticks(np.arange(len(n_chunks))+1, n_chunks)
ax[0].set_ylabel('time [ms]')

res2 = res.swapaxes(0,2).reshape(len(chunk_size), -1).T
mean_res2 = np.mean(res2, axis=0)
mean_res2 /= 1.2*np.max(mean_res2) 

bplot = ax[1].boxplot(res2, patch_artist=True)
colors = plt.cm.coolwarm(mean_res2)
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

for line in bplot['medians']:
    line.set_color('gray')
ax[1].set_xlabel('chunk size')
ax[1].set_xticks(np.arange(len(chunk_size))+1, chunk_size)
# ax[1].set_ylabel('time')

res3 = res.swapaxes(0,3).reshape(len(dims), -1).T
mean_res3 = np.mean(res3, axis=0)
mean_res3 /= 1.2*np.max(mean_res3) 

bplot = ax[2].boxplot(res3, patch_artist=True)
colors = plt.cm.coolwarm(mean_res3)
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

for line in bplot['medians']:
    line.set_color('gray')
ax[2].set_xlabel('data dimensionality')
ax[2].set_xticks(np.arange(len(dims))+1, dims)
# ax[2].set_ylabel('time')

ax[0].set_ylim(0,100)

for aa in ax:
    aa.grid(ls=':')
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig_r1/scaling.png')
plt.savefig('fig_r1/scaling.pdf')