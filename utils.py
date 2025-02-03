import numpy as np
                   
def get_gt(n_chunks, n_drifts):
    off = (n_chunks/n_drifts)//2
    drfs = np.linspace(0, n_chunks, n_drifts+1) + off
    gt = np.array([np.argwhere(i<drfs)[0][0] for i in range(n_chunks)])
    return gt

def get_drfs(n_chunks, n_drifts):
    off = (n_chunks/n_drifts)//2
    drfs = np.linspace(0, n_chunks, n_drifts+1) + off
    return (drfs[:-1]).astype(int)