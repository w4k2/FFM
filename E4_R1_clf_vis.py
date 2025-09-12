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

files = os.listdir('insects')
files.remove('.DS_Store')
print(files)

clfs = ['MLP', 'GNB']

for f_id, f in enumerate(files):
    
    clusters = np.load('insects_data_streams/%i_ci.npy' % f_id)
    results = np.load('res/r1_e4_clf_%i.npy' % f_id)
       
    for clf_id in range(2): 
        fig, ax = plt.subplots(3,1,figsize=(10,5))
        plt.suptitle('%s | Classifier: %s' % (f.split('.')[0].replace('-', ' | ').replace('_', ' ').replace('norm', ''),clfs[clf_id]))
        
        ax[0].plot(results[:,clf_id,0], label='common', c='k')
        ax[0].plot(results[:,clf_id,1], label='dedicated', c='r')
        ax[0].set_ylabel('incremental\nBAC')
        ax[0].legend(ncols=2, loc='upper left')
        ax[0].set_ylim(0,1.01)
        
        ax[1].plot(results[:,clf_id,2], label='common', c='k')
        ax[1].plot(results[:,clf_id,3], label='dedicated', c='r')
        ax[1].set_ylabel('first chunk\nBAC')
        ax[1].set_ylim(0,1.01)


        cols = plt.cm.coolwarm(np.linspace(0,1,len(np.unique(clusters))))
        ax[2].scatter(np.arange(len(clusters)), clusters, c=cols[clusters])
        ax[2].set_ylabel('identified concept')

        for aa in ax:
            aa.set_xlim(0,len(clusters))
            aa.grid(ls=':')
            aa.spines['top'].set_visible(False)
            aa.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('foo.png')
        plt.savefig('fig_r1/clf_%i_%s.png' % (f_id, clfs[clf_id]))
        plt.savefig('fig_r1/clf_%i_%s.pdf' % (f_id, clfs[clf_id]))
    
        # exit()
        time.sleep(2)
                    
    