from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from ffm import FFM
import strlearn
import matplotlib.pyplot as plt
import numpy as np

# Stream params
n_chunks = 500
chunk_size = 200
n_drifts = 3
dim = 250

stream = strlearn.streams.StreamGenerator(n_chunks=n_chunks,
                                            chunk_size=chunk_size,
                                            concept_sigmoid_spacing=999,
                                            n_drifts=n_drifts,
                                            n_features=dim,
                                            n_informative=int(0.3*dim),
                                            incremental=False,
                                            recurring=False,
                                            random_state=None)


## Przetarzanie
n=5
ffm = FFM(n=n)
ffm.describe(stream)
ffm.visualize()

print(ffm.mean_fft_all.shape)
aa = ffm.mean_fft_all[:, ffm.arg_var]
print(aa.shape)

## Plot
fig, ax = plt.subplots(n,n,figsize=(7,7), sharey=True, sharex=True)
gt = KMeans(n_clusters=4).fit_predict(StandardScaler().fit_transform(aa))

for i in range(n):
    for j in range(n):
        
        ax[i,j].scatter(aa[:,i], aa[:,j], c=gt, cmap='coolwarm', s=5)
        
        ax[i,j].spines['top'].set_visible(False)
        ax[i,j].spines['right'].set_visible(False)
        ax[i,j].grid(ls=':')
        
        if i==0:
            ax[-1,j].set_xlabel('freq = %i' % ffm.arg_var[j])
                
        if j==0:
            ax[i,j].set_ylabel('freq = %i' % ffm.arg_var[i])
    
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('vis_clusters.png')
plt.savefig('vis_clusters.pdf')