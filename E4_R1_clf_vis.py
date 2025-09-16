import numpy as np
from sklearn import clone
from sklearn.neural_network import MLPClassifier
from ffm import FFM
import os
from strlearn.streams import ARFFParser
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, silhouette_score
import time 
from scipy.ndimage import gaussian_filter1d

files = os.listdir('insects')
files.remove('.DS_Store')
print(files)

clfs = ['MLP', 'GNB']
cols2 = plt.cm.coolwarm([0.0,1.0])
s = 1

for f_id, f in enumerate(files):
    
    clusters = np.load('insects_data_streams/%i_ci.npy' % f_id)
    results = np.load('res/r1_e4_clf_%i.npy' % f_id)
       
    for clf_id in range(2): 
        fig, ax = plt.subplots(3,1,figsize=(7,4), sharex=True, height_ratios=[2,2,1])
        plt.suptitle('%s | Classifier: %s' % (f.split('.')[0].replace('-', ' | ').replace('_', ' ').replace('norm', ''),clfs[clf_id]))
        
        ax[0].plot(gaussian_filter1d(results[:,clf_id,0],s), label='common', c=cols2[0], alpha=0.7)
        ax[0].plot(gaussian_filter1d(results[:,clf_id,1],s), label='dedicated', c=cols2[1], alpha=0.7)
        ax[0].set_ylabel('incremental\nBAC')
        ax[0].legend(ncols=2, loc='upper left', frameon=False)
        ax[0].set_ylim(0,1.01)
        
        ax[1].plot(gaussian_filter1d(results[:,clf_id,2],s), label='common', c=cols2[0], alpha=0.7)
        ax[1].plot(gaussian_filter1d(results[:,clf_id,3],s), label='dedicated', c=cols2[1], alpha=0.7)
        ax[1].set_ylabel('first chunk\nBAC')
        ax[1].set_ylim(0,1.01)

        cols = plt.cm.coolwarm(np.linspace(0,1,len(np.unique(clusters))))
        ax[2].scatter(np.arange(len(clusters)), clusters, c=cols[clusters],s=10)
        ax[2].set_ylabel('identified\nconcept')

        for aa in ax:
            aa.set_xlim(0,len(clusters))
            aa.grid(ls=':')
            aa.spines['top'].set_visible(False)
            aa.spines['right'].set_visible(False)
        
        aa.set_xlabel('chunk')
        plt.tight_layout()
        plt.savefig('foo.png')
        plt.savefig('fig_r1/clf_%i_%s.png' % (f_id, clfs[clf_id]))
        plt.savefig('fig_r1/clf_%i_%s.pdf' % (f_id, clfs[clf_id]))
    
        # exit()
        time.sleep(2)
                    
    