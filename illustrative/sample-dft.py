## Compare variance, entropy, energy (magnitude/amplitude) as a frequency selection mechanism

from sklearn.decomposition import PCA
import strlearn
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

np.random.seed(3997)

# Stream params
n_chunks = 5
n_drifts = 0
percent_informative = 0.3

chunk_size = 128
dim = 32

stream = strlearn.streams.StreamGenerator(n_chunks=n_chunks,
                    chunk_size=chunk_size,
                    n_drifts=n_drifts,
                    n_features=dim,
                    n_informative=int(percent_informative*dim),
                    random_state=2232)

for chunk_id in range(n_chunks):
    X, y = stream.get_chunk()
    print(chunk_id)
    
    mean_chunk = np.mean(X, axis=0)
    fft_signal_all = np.fft.fft(mean_chunk)
    
    fft_signal = fft_signal_all[:len(mean_chunk)//2].real
                
    # s_div = np.var(np.array(mean_fft_all), axis=0)
    # self.arg_div = np.flip(np.argsort(s_div))[:self.n]
            
    fig, ax = plt.subplots(1,3,figsize=(12,3))
    ax[0].scatter(np.arange(dim), mean_chunk, c='black')
    ax[0].set_xlim(0,dim)
    ax[0].set_xlabel('feature index')
    ax[0].set_ylabel('feature value')
    ax[0].set_ylim(-1.5,1.5)

    ax[1].scatter(np.arange(dim), fft_signal_all.real, label='real part', c='blue')
    # ax[1].plot(np.arange(dim), gaussian_filter1d(fft_signal_all.real,1), c='blue', alpha=0.5)
    ax[1].scatter(np.arange(dim), fft_signal_all.imag, label='imaginary part', c='red', alpha=0.5)
    # ax[1].plot(np.arange(dim), gaussian_filter1d(fft_signal_all.imag,1), c='red', alpha=0.5)
    ax[1].set_xlim(0,dim)
    ax[1].set_xlabel('discrete frequency')
    ax[1].set_ylabel('component value')
    ax[1].legend(ncols=2, frameon=False)
    ax[1].set_ylim(-3,5)
    
    sel = np.array([0,2,3,8,13])
    nonsel = np.array([1,4,5,6,7,9,10,11,12,14,15])
    fft_signal_sel = fft_signal[sel]
    fft_signal_nonsel = fft_signal[nonsel]
    ax[2].scatter(np.arange(dim//2)[sel], fft_signal_sel, c='blue', label='selected ($\mathcal{R}_n$) ')
    ax[2].scatter(np.arange(dim//2)[nonsel], fft_signal_nonsel, c='gray', label='discarded')
    ax[2].set_xlim(0,dim//2)
    ax[2].set_xlabel('discrete frequency')
    ax[2].set_ylabel('component value')
    ax[2].legend(ncols=2, frameon=False, loc='upper center')
    ax[2].set_ylim(-3,5)
    
    ax[0].set_title('$\\bar{\mathcal{X}_n}$', fontsize=15)
    ax[1].set_title('$\mathcal{F}_n$', fontsize=15)
    ax[2].set_title('$\mathcal{F}_n\'$', fontsize=15)

    ax[0].set_xticks(np.arange(32)[::2])
    ax[1].set_xticks(np.arange(32)[::2])
    ax[2].set_xticks(np.arange(16)[::2])

    for aa in ax:
        aa.spines['top'].set_visible(False)
        aa.spines['right'].set_visible(False)
        aa.grid(ls=':')
    
    plt.tight_layout()
    plt.savefig('foo.png')
    plt.savefig('dft_frequencies.png')
    plt.savefig('dft_frequencies.pdf')
       
    exit()