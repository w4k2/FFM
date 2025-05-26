import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler

"""
Frequency filtering metadescriptor
"""

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
                
        var = np.var(self.mean_fft_all, axis=0)
        self.arg_var = np.flip(np.argsort(var))[:self.n]
            
        return self.mean_fft_all[:,self.arg_var]
    
    def describe_data(self, data, chunk_size):    
        self.mean_fft_all = []
        
        n_chunks = len(data)//chunk_size
        
        for chunk_id in range(n_chunks):
            X = data[chunk_id*chunk_size : (chunk_id+1)*chunk_size]
                        
            mean_chunk = np.mean(X, axis=0)
            fft_signal = np.fft.fft(mean_chunk)[:len(mean_chunk)//2]
            self.mean_fft_all.append(fft_signal.real)
               
        self.mean_fft_all = np.array(self.mean_fft_all)
                
        var = np.var(self.mean_fft_all, axis=0)
        self.arg_var = np.flip(np.argsort(var))[:self.n]
            
        return self.mean_fft_all[:,self.arg_var]
    
    def cluster(self, c):
        samples_std = StandardScaler().fit_transform(self.mean_fft_all[:,self.arg_var])
        return KMeans(n_clusters=c).fit_predict(samples_std)
            
    
    def visualize(self):
        
        self.chunk_convs = []
        for chunk_fft in self.mean_fft_all:
            i_convs = []

            for freq in np.sort(self.arg_var):
            
                mask = np.zeros((self.mean_fft_all.shape[1]))
                mask[freq] = 1
                
                filtered = chunk_fft*mask
                i_filtered = np.fft.ifft(filtered).real[:self.n]
            
                i_convs.append(i_filtered)
        
            i_convs = np.array(i_convs)            
            self.chunk_convs.append(i_convs)
    
        self.chunk_convs = np.array(self.chunk_convs)
        
        return self.chunk_convs

    