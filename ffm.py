import numpy as np

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
                        
            fft_all = []
            for X_i in X:
                fft_signal = np.fft.fft(X_i)[:len(X_i)//2] # len//2?
                fft_all.append(fft_signal)
            
            fft_all = np.array(fft_all)
            mean_fft = np.mean(fft_all, axis=0)
                
            self.mean_fft_all.append(mean_fft.real)
  
        self.mean_fft_all = np.array(self.mean_fft_all)
                
        var = np.var(self.mean_fft_all, axis=0)
        self.arg_var = np.flip(np.argsort(var))[:self.n]
            
        return self
    
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

    