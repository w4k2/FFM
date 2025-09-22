import numpy as np
from sklearn import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from ffm import FFM
import os
from strlearn.streams import ARFFParser
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, silhouette_score

files = os.listdir('insects')
files.remove('.DS_Store')
print(files)


base_clfs = [MLPClassifier(random_state=567),
            GaussianNB()]

for f_id, f in enumerate(files):
    
    chunks_X = np.load('insects_data_streams/%i_X.npy' % f_id)
    chunks_y = np.load('insects_data_streams/%i_y.npy' % f_id)
    clusters = np.load('insects_data_streams/%i_ci.npy' % f_id)
    
    results = np.full((chunks_X.shape[0], len(base_clfs), 4), np.nan)
    
    for base_clf_id, bclf in enumerate(base_clfs):
        ### Incremental learning
        
        ## Common classifier
        clf = clone(bclf)
        for chunk_id in range(chunks_X.shape[0]):
            X, y = chunks_X[chunk_id], chunks_y[chunk_id]
                        
            try:
                pred = clf.predict(X)
                results[chunk_id, base_clf_id, 0] = balanced_accuracy_score(y, pred)
                
                clf.partial_fit(X,y,np.unique(chunks_y))
            except:
                clf.partial_fit(X,y,np.unique(chunks_y))
                
                
        ## Dedicated classifiers
        clfs = [clone(bclf) for i in np.unique(clusters)]
        
        for chunk_id in range(chunks_X.shape[0]):
            X, y = chunks_X[chunk_id], chunks_y[chunk_id]
            
            currenct_concept = clusters[chunk_id]
            
            try:
                pred = clfs[currenct_concept].predict(X)
                results[chunk_id, base_clf_id, 1] = balanced_accuracy_score(y, pred)
                
                clfs[currenct_concept].partial_fit(X,y,np.unique(chunks_y))
                                
            except:
                clfs[currenct_concept].partial_fit(X,y,np.unique(chunks_y))


        ### No update
        
        ## Common classifier
        clf = clone(bclf)
        for chunk_id in range(chunks_X.shape[0]):
            X, y = chunks_X[chunk_id], chunks_y[chunk_id]
                        
            try:
                pred = clf.predict(X)
                results[chunk_id, base_clf_id, 2] = balanced_accuracy_score(y, pred)
            except:
                clf.fit(X,y)
                
                
        ## Dedicated classifiers
        clfs = [clone(bclf) for i in np.unique(clusters)]
        
        for chunk_id in range(chunks_X.shape[0]):
            X, y = chunks_X[chunk_id], chunks_y[chunk_id]
            
            currenct_concept = clusters[chunk_id]
            
            try:
                pred = clfs[currenct_concept].predict(X)
                results[chunk_id, base_clf_id, 3] = balanced_accuracy_score(y, pred)                                
            except:
                clfs[currenct_concept].fit(X,y)

                
        np.save('res/r1_e4_clf_%i.npy' % f_id, results)
                
    