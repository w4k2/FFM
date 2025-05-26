from matplotlib import pyplot as plt
import strlearn
from ffm import FFM
from utils import get_gt
import numpy as np

stream = strlearn.streams.StreamGenerator(n_chunks=400, chunk_size=200, 
                                          n_features=15, n_informative=4, n_redundant=10, n_repeated=0,
                                          n_drifts=3, random_state=5678)

# Describe data
ffm = FFM(n=8)
des = ffm.describe(stream)

# Representation of size (n_chunks x n frequency components)
print(des.shape)

# Selected frequencies
print(ffm.arg_var)

# Cluster into concepts
clusters = ffm.cluster(c=4)
print(clusters)

# Visualize as images
vis = ffm.visualize()
print(vis.shape)
print(vis[0])

# Plot concept clusters
fig, ax = plt.subplots(4,4,figsize=(7,7))
for i in range(4):
    for j in range(4):
        ax[i,j].scatter(des[:,i], des[:,j], 
                        c=get_gt(400,3), cmap='coolwarm', 
                        s=7)
plt.tight_layout()
plt.savefig('clusters.png')

# Plot stream visualization
chunks = np.linspace(0,400-1,10).astype(int)
fig, ax = plt.subplots(1,10,figsize=(10,1.5))
for i in range(10):
    ax[i].imshow(vis[chunks[i]], cmap='coolwarm')
    ax[i].set_title('chunk %i' % chunks[i])
    
plt.tight_layout()
plt.savefig('visualization.png')
