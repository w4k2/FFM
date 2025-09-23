import matplotlib.pyplot as plt
import numpy as np

"""
Experiment 0: Evaluate descriptor param: n 
-- Visualize results
"""

np.random.seed(3997)

# Stream params
chunk_size = [50, 100, 200]
n_drifts = [1, 3, 5, 7, 9]

# Descriptor params 
ns = [1,2,4,8,16]

# Experiment params
reps = 10
metrics = ['NMI', 'Adjusted Rand', 'Completeness', 'Homogeneity']

results = np.load('res/e0.npy')
print(results.shape) # (10, 5, 3, 5, 4)

mean_res = np.nanmean(results, axis=0) # drifts, chunk_size, ns, metrics

fig, ax = plt.subplots(1,3,figsize=(10,4), sharex=True, sharey=True)
cols = plt.cm.coolwarm(np.linspace(0,1,5))
cols[2][:3] -=0.15
for chunk_size_id in range(3):
    aa = ax[chunk_size_id]
    aa.set_title('chunk size: %i' % chunk_size[chunk_size_id])
    
    for n_drifts_id in range(5):
        
        data = mean_res[n_drifts_id, chunk_size_id, :, 0]

        aa.plot(np.arange(len(ns)), data, color = cols[n_drifts_id], lw=0.5)
        aa.scatter(np.arange(len(ns)), data, color = cols[n_drifts_id], label = 'drifts: %i' % n_drifts[n_drifts_id], s=20)
        
        aa.set_xticks(np.arange(len(ns)), ns)
        
        aa.spines['top'].set_visible(False)
        aa.spines['right'].set_visible(False)
        aa.grid(ls=':')
        
ax[0].set_ylabel('NMI')
ax[0].legend(frameon=False, loc='upper left')

ax[1].set_xlabel('number of frequency components')

        
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('vis_E1.png')
plt.savefig('vis_E1.pdf')
        
