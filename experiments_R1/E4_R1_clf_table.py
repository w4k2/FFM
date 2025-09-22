import numpy as np
from sklearn import clone
from sklearn.neural_network import MLPClassifier
from tabulate import tabulate
from ffm import FFM
import os
from strlearn.streams import ARFFParser
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
import time 

files = os.listdir('insects')
files.remove('.DS_Store')
print(files)

clfs = ['MLP', 'GNB']

rows = []
# fname | GNB incremental (common, dedicated) | GNB first (common, dedicated) | MLP incremental (common, dedicated) | MLP first (common, dedicated) |
for f_id, f in enumerate(files):
    
    clusters = np.load('insects_data_streams/%i_ci.npy' % f_id)
    results = np.load('res/r1_e4_clf_%i.npy' % f_id)
    
    print(results.shape) #(799, 2, 4)
    av_results = np.nanmean(results, axis=0)

    rows.append([ '%s' % files[f_id].split('_norm.')[0], 
                 '%.3f' %  (av_results[1,0]),
                 '%.3f' %  (av_results[1,1]),
                 '%.3f' %  (av_results[1,2]),
                 '%.3f' %  (av_results[1,3]),
                 '%.3f' %  (av_results[0,0]),
                 '%.3f' %  (av_results[0,1]),
                 '%.3f' %  (av_results[0,2]),
                 '%.3f' %  (av_results[0,3]),
                 ])

print(tabulate(rows, tablefmt='latex'))
