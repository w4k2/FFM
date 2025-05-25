import strlearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits


class FFM:
    def __init__(self, n):
        self.n=n
        
    def describe(self, stream):    
        self.mean_fft_all = []
        
        chunk_size = 10
        n_chunks = stream.shape[0]//chunk_size
        for chunk in range(n_chunks):
            start = chunk*chunk_size
            end = (chunk+1)*chunk_size

            X = stream[start:end]
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
X, y = load_digits(return_X_y=True)
order = np.argsort(y)
X = X[order]

## Przetarzanie
_n = 16

ffm = FFM(n=_n)
ffm.describe(X)
ffm.visualize()

aa = ffm.mean_fft_all[:, ffm.arg_var]
print(ffm.arg_var)

aa = ffm.mean_fft_all
print(aa.shape) # (179, 32)
aa = aa[:,ffm.arg_var]
print(aa.shape)

## reconstruct
reconstruction = []

for a in aa:
    rec = np.zeros(64)
    
    for freq_id, strength in enumerate(a):
        freq = ffm.arg_var[freq_id]
        
        mask = np.zeros(64)
        mask[freq] = 1
        
        filtered = mask*strength
        i_filtered = np.fft.ifft(filtered).real
        rec += i_filtered
        
        print(freq_id, strength)
    
    print(rec)
    reconstruction.append(rec.reshape(8,8))
    
print(reconstruction[0])
args = np.linspace(0,178,100).astype(int)

fig, ax = plt.subplots(10,10,figsize=(10,10), sharex=True, sharey=True)
ax = ax.ravel()

for i in range(100):
    ax[i].imshow(reconstruction[args[i]], cmap='binary')

plt.tight_layout()
plt.savefig('foo.png')