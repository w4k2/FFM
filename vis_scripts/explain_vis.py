import strlearn
import matplotlib.pyplot as plt
import numpy as np


class FFM:
    def __init__(self, n):
        self.n=n
        
    def describe(self, stream):    
        self.mean_fft_all = []
        
        while chunk := stream.get_chunk():
            X, y = chunk
            mean_chunk = np.mean(X, axis=0)
                        
            fft_signal = np.fft.fft(mean_chunk)[:len(mean_chunk)//2]
            self.mean_fft_all.append(fft_signal.real)
  
        self.mean_fft_all = np.array(self.mean_fft_all)
        
        print(self.mean_fft_all.shape) #20 x 12 (chunks x n_features//2)

        var = np.var(self.mean_fft_all, axis=0)
        self.arg_var = np.flip(np.argsort(var))[:self.n]
        
        self.arg_var = np.sort(self.arg_var)
        
        print(var.shape) # 12 (n_features//2)
        print(self.arg_var.shape) # 8 (n)
        print(self.arg_var) #posortowane od najwiekszej
        # exit()
            
        return self
    
    
    def visualize(self):
        
        self.chunk_convs = []
        self.chunk_convs_base = []
        
        for chunk_fft in self.mean_fft_all:
            i_convs = []
            i_convs_base = []

            for freq in self.arg_var:
            
                mask = np.zeros((self.mean_fft_all.shape[1]))
                mask[freq] = 1
                
                filtered = chunk_fft*mask
                filtered_base = mask
                i_filtered = np.fft.ifft(filtered).real[:self.n]
                i_filtered_base = np.fft.ifft(filtered_base).real[:self.n]
            
                i_convs.append(i_filtered)
                i_convs_base.append(i_filtered_base)
        
            i_convs = np.array(i_convs)            
            i_convs_base = np.array(i_convs_base)            
            self.chunk_convs.append(i_convs)
            self.chunk_convs_base.append(i_convs_base)
    
        self.chunk_convs = np.array(self.chunk_convs)
        self.chunk_convs_base = np.array(self.chunk_convs_base)
        
        return self.chunk_convs

# Stream params
n_chunks = 20
chunk_size = 1000
n_drifts = 1
dim = 64

stream = strlearn.streams.StreamGenerator(n_chunks=n_chunks,
                                            chunk_size=chunk_size,
                                            concept_sigmoid_spacing=999,
                                            n_drifts=n_drifts,
                                            n_features=dim,
                                            n_informative=dim,
                                            n_redundant=0,
                                            n_repeated=0,
                                            incremental=False,
                                            recurring=False,
                                            random_state=5678)


## Przetarzanie
_n = 32

ffm = FFM(n=_n)
ffm.describe(stream)
ffm.visualize()

print(ffm.mean_fft_all.shape)
aa = ffm.mean_fft_all[:, ffm.arg_var]
print(ffm.arg_var)

vis = ffm.chunk_convs
vis_base = ffm.chunk_convs_base

n_chuks_vis = 2
chunks_vis = np.linspace(0,n_chunks-1,n_chuks_vis).astype(int)

n_show = 8

## Plot
fig, ax = plt.subplots(1,2,figsize=(10,4))

for n in range(n_chuks_vis):
    vis_a = vis[chunks_vis[n]]
    vis_ab = vis_base[chunks_vis[n]]
    vis_a[n_show:] = 0
    ax[1].imshow(vis_a,
                cmap='coolwarm',
                interpolation='none', aspect='auto')

    y0 = []
    y1 = []
    cols = plt.cm.coolwarm(np.linspace(0,1,n_show))
    cols[:,:3] -= 0.2
    cols = np.clip(cols, 0,1)
    
    for i, f in enumerate(ffm.arg_var[:n_show]):
        offset1 = (i*0.3)

        ax[0].plot(vis_ab[i]-offset1, color = 'black', ls=':')
        ax[0].plot(vis_a[i]-offset1, color = cols[i])
        y1.append(-offset1)
    
    ax[0].set_yticks(y1, ffm.arg_var[:n_show])
    ax[0].set_xlim(0,_n-1)
    ax[0].set_xticks(np.arange(_n)[::2])
    ax[0].set_ylabel('discreete frequency')
    ax[0].set_xlabel('sample')
    
    ax[1].set_ylabel('discreete frequency')
    ax[1].set_xlabel('sample')
    
    break
    
for aa in ax.ravel():
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('vis_explain.png')
plt.savefig('vis_explain.pdf')