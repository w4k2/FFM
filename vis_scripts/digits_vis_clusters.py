from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

chunk_size = 3

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
                
        print(var.shape) # 12 (n_features//2)
        print(self.arg_var.shape) # 8 (n)
        print(self.arg_var) #posortowane od najwiekszej
            
        return self
    
    
    def visualize(self):
        
        s_arg_var = np.sort(self.arg_var)
        
        self.chunk_convs = []
        self.chunk_convs_base = []
        
        for chunk_fft in self.mean_fft_all:
            i_convs = []
            i_convs_base = []

            for freq in s_arg_var:
            
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
X = X[y<4]
y = y[y<4]

order = np.argsort(y)
X = X[order]

## Przetarzanie
n = 4

ffm = FFM(n=n)
ffm.describe(X)
vs = ffm.visualize()

aa = ffm.mean_fft_all[:, ffm.arg_var]
print(aa.shape) # 89, 8


## Plot
fig, ax = plt.subplots(n,n,figsize=(7,7), sharey=True, sharex=True)
gt = KMeans(n_clusters=4).fit_predict(StandardScaler().fit_transform(aa))

for i in range(n):
    for j in range(n):
        
        ax[i,j].scatter(aa[:,i], aa[:,j], c=gt, 
                        cmap='coolwarm', s=3)
        
        ax[i,j].spines['top'].set_visible(False)
        ax[i,j].spines['right'].set_visible(False)
        ax[i,j].grid(ls=':')
        
        if i==0:
            ax[-1,j].set_xlabel('freq = %i' % ffm.arg_var[j])
                
        if j==0:
            ax[i,j].set_ylabel('freq = %i' % ffm.arg_var[i])
    
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('vis_clusters_digits.png')
plt.savefig('vis_clusters_digits.pdf')