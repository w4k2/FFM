from ffm import FFM
import strlearn
import matplotlib.pyplot as plt
import numpy as np

    
# Stream params
n_chunks = 20
chunk_size = 500
n_drifts = 1
dim = 50

stream = strlearn.streams.StreamGenerator(n_chunks=n_chunks,
                                            chunk_size=chunk_size,
                                            concept_sigmoid_spacing=5,
                                            n_drifts=n_drifts,
                                            n_features=dim,
                                            n_informative=int(0.3*dim),
                                            incremental=False,
                                            recurring=False,
                                            random_state=None)


## Przetarzanie
ffm = FFM(n=16)
ffm.describe(stream)
ffm.visualize()

print(ffm.mean_fft_all.shape)
aa = ffm.mean_fft_all[:, ffm.arg_var]
print(aa.shape)

## Plot
fig, ax = plt.subplots(2,10,figsize=(10,2.5), sharex=1, sharey=1)
ax = ax.ravel()

vmin = np.mean(ffm.chunk_convs) - 2*np.std(ffm.chunk_convs)
vmax = np.mean(ffm.chunk_convs) + 2*np.std(ffm.chunk_convs)

for i in range(n_chunks):
    ax[i].imshow(ffm.chunk_convs[i], cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='none')
    
ax[0].set_ylabel('concept 0')
ax[10].set_ylabel('concept 1')
    
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('vis_freqs.png')
plt.savefig('vis_freqs.pdf')