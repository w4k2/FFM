from ffm import FFM
import strlearn
import matplotlib.pyplot as plt
import numpy as np

    
# Stream params
n_chunks = 100
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
fig, ax = plt.subplots(1,8,figsize=(10,2), sharex=1, sharey=1)

vmin = np.mean(ffm.chunk_convs) - 2*np.std(ffm.chunk_convs)
vmax = np.mean(ffm.chunk_convs) + 2*np.std(ffm.chunk_convs)

n_chunks_vis = np.linspace(0,100-1,8).astype(int)
for i_id, i in enumerate(n_chunks_vis):
    ax[i_id].imshow(ffm.chunk_convs[i], cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='none')
    ax[i_id].set_title('chunk %i' % i)
        
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('vis_freqs.png')
plt.savefig('vis_freqs.pdf')