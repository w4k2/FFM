import numpy as np
from ffm import FFM
import os
from strlearn.streams import ARFFParser
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

files = os.listdir('insects')

print(files)
chunk_size=[100,500,50,200,50,100,500,100,500,500]
n=5

for f_id, f in enumerate(files):
    
    data = []
    stream = ARFFParser('insects/%s' % f, chunk_size=chunk_size[f_id], n_chunks=10000000)
    while(1):
        try:
            data.extend(stream.get_chunk()[0])
        except:
            break
       
    # print(f) 
    # print(len(data))
    # print(data[0].shape)
    # continue
    
    print(len(data))
    print(data[0].shape)
        
    ffm = FFM(n=n)
    ffm.describe_data(data, chunk_size[f_id])
    
    rep = ffm.mean_fft_all[:,ffm.arg_var]
    print(rep.shape)
    
    rep = StandardScaler().fit_transform(rep)
    
    # first identify number of clusters
    scores_reps = []
    for i in range(10):
        search = np.arange(4,9)
        scores = []
        
        for s in search:
            clusters = KMeans(n_clusters=s).fit_predict(rep)
            scores.append(silhouette_score(rep, clusters))
        
        scores_reps.append(scores)
    
    scores_reps = np.array(scores_reps)
    print(scores_reps.shape)
    scores_reps = np.mean(scores_reps, axis=0)
    best = search[np.argmax(scores_reps)]
    print(best)
    
    # cluster
    clusters = KMeans(n_clusters=best).fit_predict(StandardScaler().fit_transform(rep))
    print(clusters)
    
    clusters_2 = np.copy(clusters)

    mapping_src = []
    for i in clusters:
        if i not in mapping_src:
            mapping_src.append(i)
            
    for i_id, i in enumerate(mapping_src):
        clusters_2[clusters==i] = i_id
    
    clusters = clusters_2
    
    fig, ax = plt.subplots(n+1,n, figsize=(6,8), sharex=True, sharey=True)
    axx = plt.subplot(n+1,1,6)
    
    plt.suptitle('%s| %i concepts' % (f.split('.')[0].replace('-', ' | ').replace('_', ' ').replace('norm', ''), best))
    
    vmin, vmax = np.min(rep), np.max(rep)
    
    for i in range(n):
        for j in range(n):
            ax[i,j].scatter(rep[:,i], rep[:,j], c=clusters, s=3, cmap='coolwarm', alpha=0.5)
            ax[i,j].grid(ls=':')
            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['right'].set_visible(False)
            
            if i==0:
                ax[i,j].set_title('freq = %i' % ffm.arg_var[j], fontsize=10)
            #     ax[i,j].set_xticks([-2.5,0,2.5])
            #     ax[i,j].set_xlim(vmin,vmax)
            # else:
            #     ax[i,j].set_xticks([-2.5,0,2.5],[])
            #     ax[i,j].set_xlim(vmin,vmax)
                
            if j==0:
                ax[i,j].set_ylabel('freq = %i' % ffm.arg_var[i])
                # ax[i,j].set_yticks([-2.5,0,2.5])
                # ax[i,j].set_ylim(vmin,vmax)
            # else:
                # ax[i,j].set_yticks([-2.5,0,2.5],[])
                # ax[i,j].set_ylim(vmin,vmax)

    axx.scatter(np.arange(len(clusters)), clusters, c=clusters, cmap='coolwarm', s=10)
    axx.grid(ls=':')
    axx.spines['top'].set_visible(False)
    axx.spines['right'].set_visible(False)
    axx.set_xlim(0,len(clusters))
    axx.set_yticks(np.arange(np.max(clusters)+1))
    axx.set_xlabel('chunk')
    axx.set_ylabel('concept')
                
    for aa in ax[-1,:]:
        aa.set_xticks([])
        aa.set_yticks([])
        aa.spines['right'].set_visible(False)

            
    plt.tight_layout()
    plt.savefig('foo.png')
    plt.savefig('fig_insects/%i_%s.png' % (f_id, f.split('.')[0]))
    plt.savefig('fig_insects/%i_%s.pdf' % (f_id, f.split('.')[0]))
    
    # exit()