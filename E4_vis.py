import numpy as np
from tabulate import tabulate
import os

files = os.listdir('insects')

search = np.arange(4,9)
metrics = ['SIL', 'CH', 'DB']
    
res = np.load('res/e4.npy')
res_mean = np.mean(res, axis=1)
res_std = np.std(res, axis=1)

argamx_sil = np.argmax(res_mean[:,:,0], axis=1).flatten()
print(argamx_sil.shape)

res_best = []
for i_id, i in enumerate(argamx_sil):
    res_best.append(res_mean[i_id,i])
res_best = np.array(res_best)

res_best_std = []
for i_id, i in enumerate(argamx_sil):
    res_best_std.append(res_std[i_id,i])
res_best_std = np.array(res_best_std)

print(res_best_std.shape)

rows = []
for i in range(10):
    rows.append([ '%s' % files[i].split('_norm.')[0], 
                 '%.3f (%.3f)' % (res_best[i,0], res_best_std[i,0]),
                 '%.3f (%.3f)' % (res_best[i,1], res_best_std[i,1]),
                 '%.3f (%.3f)' % (res_best[i,2], res_best_std[i,2])
                 ])

print(tabulate(rows, tablefmt='latex'))
