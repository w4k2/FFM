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
res_ref = np.load('res/e_r1_scale_pca.npy')*1000

print(res.shape) # (10, 6, 6, 8) = reps, chunks, chunk size, dims
mean_res = np.nanmean(res, axis=0)
mean_res_ref = np.nanmean(res_ref, axis=0)

fig, ax = plt.subplots(1,3,figsize=(12,4), sharex=False, sharey=True)
cols = plt.cm.coolwarm(np.linspace(0,1,len(chunk_size)))

chs_id = -1
chs = chunk_size[chs_id]
nch_id = -1
nch = n_chunks[nch_id]

ax[0].plot(mean_res[:,chs_id,-1], 
               label='FFM | chunk size = %i' % chs,
               color=cols[chs_id])
ax[0].plot(mean_res_ref[:,chs_id,-1], 
                label='PCA | chunk size = %i' % chs,
               color=cols[chs_id], ls=':')
ax[0].scatter(np.arange(len(n_chunks)), mean_res[:,chs_id,-1], 
               color=cols[chs_id], marker="$\u25EF$")
    
ax[0].set_xlabel('number of chunks')
ax[0].set_xticks(np.arange(len(n_chunks)), n_chunks)
ax[0].set_ylabel('time [ms]')
ax[0].legend(frameon=False)

cols = plt.cm.coolwarm(np.linspace(0,1,len(n_chunks)))
ax[1].plot(mean_res[nch_id,:,-1], 
            label='FFM | n chunks = %i' % nch,
            color=cols[nch_id])
ax[1].scatter(np.arange(len(chunk_size)), mean_res[nch_id,:,-1], 
            color=cols[nch_id], marker="$\u25EF$")
ax[1].plot(mean_res_ref[nch_id,:,-1], 
            label='PCA | n chunks = %i' % nch, 
            color=cols[nch_id], ls=':')

ax[1].set_xlabel('chunk size')
ax[1].set_xticks(np.arange(len(chunk_size)), chunk_size)
ax[1].legend(frameon=False)
# ax[1].set_ylabel('time')

cols = plt.cm.coolwarm(np.linspace(0,1,len(n_chunks)))
ax[2].plot(mean_res[nch_id,-1,], 
            label='FFM | n chunks = %i' % nch,
            color=cols[nch_id])
ax[2].plot(mean_res_ref[nch_id,-1,],
            label='PCA | n chunks = %i' % nch, 
            color=cols[nch_id], ls=':')
ax[2].scatter(np.arange(len(dims)), mean_res[nch_id,-1,:], 
            color=cols[nch_id], marker="$\u25EF$")
    
ax[2].set_xlabel('data dimensionality')
ax[2].set_xticks(np.arange(len(dims)), dims)
ax[2].legend(frameon=False)
# ax[2].set_ylabel('time')

# ax[0].set_ylim(0,80)

for aa in ax:
    aa.grid(ls=':')
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig_r1/scaling2.png')
plt.savefig('fig_r1/scaling2.pdf')