import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

chunk_size = 20


class FFM:
    def __init__(self, n):
        self.n=n
        
    def describe(self, stream):
        self.mean_fft_all = []
        
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
_n = 8

ffm = FFM(n=_n)
ffm.describe(X)
vs = ffm.visualize()

aa = ffm.mean_fft_all[:, ffm.arg_var]
print(ffm.arg_var)

aa = ffm.mean_fft_all
print(aa.shape) # (179, 32)
aa = aa[:,ffm.arg_var]
print(aa.shape)

## reconstruct
reconstruction_iters = []
reconstruction_iters2 = []

offset = 13

rec = np.zeros(64)

for freq_id, strength in enumerate(aa[0]):
    freq = ffm.arg_var[freq_id]
    
    mask = np.zeros(64)
    mask[freq] = 1
    
    filtered = mask*strength
    i_filtered = np.fft.ifft(filtered).real
    
    i_filtered_base = np.fft.ifft(mask).real
    
    rec += i_filtered
    reconstruction_iters.append(i_filtered)

rec2 = np.zeros(64)
for freq_id, strength in enumerate(aa[offset]):
    freq = ffm.arg_var[freq_id]
    
    mask = np.zeros(64)
    mask[freq] = 1
    
    filtered = mask*strength
    i_filtered = np.fft.ifft(filtered).real
    
    i_filtered_base = np.fft.ifft(mask).real
    
    rec2 += i_filtered
    reconstruction_iters2.append(i_filtered)


fig, ax = plt.subplots(3, 3, figsize=(11,6), width_ratios=[0.7, 0.15, 0.15])

cols = plt.cm.coolwarm(np.linspace(0,1,_n))
cols[:,:3] -= 0.2
cols = np.clip(cols, 0,1)

blue, red = plt.cm.coolwarm([0.0,1.0])

for i in range(chunk_size):
    ax[0,0].plot(X[i], color=red, lw=0.5, alpha=.7)
    ax[0,0].plot(X[(offset*chunk_size)+i], ls=':', color=blue, lw=0.5, alpha=0.7)
ax[0,0].set_title('original features')
ax[0,0].set_ylabel('feature value')

ax[1,0].plot(rec, color=red)
ax[1,0].plot(rec2, color=blue, ls=':')
ax[1,0].set_title('reconstructed features')
ax[1,0].set_ylabel('feature value')

s = 1.3
y=[]
for i in range(_n):
    y.append(-s*i)
    # ax[2,0].plot(reconstruction_iters[i]-s*i, color=cols[i])
    ax[2,0].plot(reconstruction_iters[i]-s*i, color=red)
    ax[2,0].plot(reconstruction_iters2[i]-s*i, color=blue, ls=':')

ax[2,0].set_yticks(y, ffm.arg_var)
ax[2,0].set_title('frequency components')
ax[2,0].set_ylabel('descreete frequency')

ax[0,1].imshow(np.mean(X[:chunk_size],axis=0).reshape(8,8), cmap='Reds')
ax[0,2].imshow(np.mean(X[(offset*chunk_size):(offset+1)*chunk_size],axis=0).reshape(8,8), cmap='Blues')
ax[1,1].imshow(rec.reshape(8,8), cmap='Reds')
ax[1,2].imshow(rec2.reshape(8,8), cmap='Blues')

ax[2,1].imshow(vs[0], cmap='coolwarm')
ax[2,2].imshow(vs[offset], cmap='coolwarm')

for aa in ax[:,0]:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')
    aa.set_xlim(0,64)

ax[0,1].set_title('chunk 0')
ax[0,2].set_title('chunk 13')
ax[0,1].set_ylabel('mean in chunk')
ax[1,1].set_ylabel('reconstruction')
ax[2,1].set_ylabel('FFM visualization')

fig.align_ylabels()
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('vis_zero.png')
plt.savefig('vis_zero.pdf')