from matplotlib import pyplot as plt
import numpy as np
import strlearn
from ffm import FFM

'''
Experiment 3: Discover number of concepts -- visualize results (data chunk vis)

'''

np.random.seed(28837)

reps = 10
rs = np.random.randint(100, 100000, reps)

n_concepts = 6
chunk_size = 400
n_features = 500
            
stream = strlearn.streams.StreamGenerator(
    n_chunks=500,
    chunk_size=chunk_size,
    concept_sigmoid_spacing=999,
    n_drifts=(n_concepts-1),
    n_features=n_features,
    n_informative=int(0.3*n_features),
    incremental=False,
    recurring=False,
    random_state=rs[8])

ffm = FFM(n=16)

ffm.describe(stream)
ffm.visualize()
vis = ffm.chunk_convs

print(vis.shape)

vmin = np.mean(vis) - 2*np.std(vis)
vmax = np.mean(vis) + 2*np.std(vis)

concept_starts = [0, 50, 150, 250, 350, 450]
   
fig, ax = plt.subplots(6, 10, figsize=(10,7), sharex=True, sharey=True)

for concept_start_id, cs in enumerate(concept_starts):
    ax[concept_start_id, 0].set_ylabel('concept %i' % concept_start_id)

    for i in range(10):
        ax[concept_start_id, i].imshow(vis[cs+1+i], cmap='coolwarm', vmin=vmin, vmax=vmax)


plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('vis_E3_3.png')
plt.savefig('vis_E3_3.pdf')
        