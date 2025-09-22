from matplotlib import pyplot as plt
import numpy as np


'''
Experiment 3: Discover number of concepts -- visualize results (heatmap)

'''

n_concepts = [2,4,6,8,10]
chunk_size = [100,200,400]

n_consiedred_concepts = np.arange(2,12)

res500 = np.load('res/e3.npy') # 10, 3, 5, 10, 2
res100 = np.load('res/e3_r1.npy') # 10, 3, 5, 10, 2
res500_mean = np.nanmean(res500, axis=0)
res100_mean = np.nanmean(res100, axis=0)

fig, ax = plt.subplots(2,3,figsize=(8,4), sharex=True, sharey=True)

for res_mean_id, res_mean in enumerate([res100_mean, res500_mean]):
    for ch_s_id, ch_s in enumerate(chunk_size):
        
        rr = res_mean[ch_s_id, :,:,1].T
        
        ax[res_mean_id, ch_s_id].imshow(rr, aspect='auto', cmap='coolwarm', interpolation='nearest', vmin=0.9*np.min(rr), vmax = 1.1*np.max(rr))
        if ch_s_id==0:
            ax[res_mean_id, ch_s_id].set_ylabel('%i features\nconsidered concepts' % [100,500][res_mean_id])
        

        ax[res_mean_id, ch_s_id].scatter(np.arange(len(n_concepts)), 
                                    [0,2,4,6,8], marker='x', c='black', s=20)
        ax[res_mean_id, ch_s_id].scatter(np.arange(len(n_concepts)), 
                                    np.argmax(rr,axis=0), marker='o', c='red', s=100, alpha=0.5)

        if res_mean_id==1:
            ax[res_mean_id, ch_s_id].set_xlabel('true concepts')
        else:
            ax[res_mean_id, ch_s_id].set_title('chunk size: %i' % ch_s)       
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
        