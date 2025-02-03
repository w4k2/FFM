from matplotlib import pyplot as plt
import numpy as np


'''
Experiment 3: Discover number of concepts -- visualize results (heatmap)

'''

n_concepts = [2,4,6,8,10]
chunk_size = [100,200,400]

n_consiedred_concepts = np.arange(2,12)

res = np.load('res/e3.npy') # 10, 3, 5, 10, 2
res_mean = np.nanmean(res, axis=0)

fig, ax = plt.subplots(1,3,figsize=(8,4), sharex=True, sharey=True)

for ch_s_id, ch_s in enumerate(chunk_size):
    
    rr = res_mean[ch_s_id, :,:,1].T
    
    ax[ch_s_id].imshow(rr, aspect='auto', cmap='coolwarm', interpolation='bessel')
    if ch_s_id==0:
        ax[ch_s_id].set_ylabel('considered number of concepts')
    
    ax[ch_s_id].set_xlabel('true number of concepts')
    ax[ch_s_id].set_title('chunk size: %i' % ch_s)
    ax[ch_s_id].scatter(np.arange(len(n_concepts)), 
                                [0,2,4,6,8], marker='x', c='black', s=20)
    ax[ch_s_id].scatter(np.arange(len(n_concepts)), 
                                np.argmax(rr,axis=0), marker='o', c='red', s=100, alpha=0.5)
        
for aa in ax.ravel():
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')
    aa.set_yticks(np.arange(len(n_consiedred_concepts)), n_consiedred_concepts)
    aa.set_xticks(np.arange(len(n_concepts)), n_concepts)
    

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('vis_E3.png')
plt.savefig('vis_E3.pdf')
        