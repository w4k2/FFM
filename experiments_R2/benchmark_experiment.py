import math
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from ffm import FFM
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import chi2

dir = 'experiments_R2/datasets'

np.random.seed(2137)

files = os.listdir(dir)
try:
    files.remove('.DS_Store')
except:
    pass

files = np.sort(files)
files2 = []
# print(files)

data_info = []
for f_id, f in enumerate(files):
    data = np.loadtxt('%s/%s' % (dir, f), delimiter=',')
    if data.shape[1]>=9:
        print(files2)
        files2.append(f)
        continue
    u, c = np.unique(data[:,-1], return_counts=True)
    data_info.append([f, len(data), data.shape[1]-1, len(u), np.round(np.min(c/len(data)),3)])
    
data_info = np.array(data_info)
np.savetxt('experiments_R2/data_info.csv', data_info, delimiter='\t', fmt="%s")

files = np.array(files2)
print(len(files))

# ___ experiments
reps = 20
metrics = [normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score]
results = np.zeros((len(files),reps,4,2))

for f_id, f in enumerate(files):
    data = np.loadtxt('%s/%s' % (dir, f), delimiter=',')
    X, y = data[:,:-1], data[:,-1]
    
    order = np.argsort(y)
    X, y = X[order], y[order]
    
    # establish chunk size
    if len(y)<200:
        chunk_size = 10
    elif len(y)<500:
        chunk_size = 25
    elif len(y)<1000:
        chunk_size = 50
    else:
        chunk_size = 100
        
    n_chunks = len(y)//chunk_size
    print(n_chunks)
    
    gt = []    
    for ch_id in range(n_chunks):
        end = (ch_id+1)*chunk_size
        gt.append(int(y[end-1]))
    
    for rep in range(reps):
        
        # FFM
        ffm = FFM(n=4)
        ffm.describe_data(X[:(len(y)//2)], chunk_size=chunk_size)
        cids = ffm.cluster_data(X, chunk_size, np.max(gt)+1)
        for metric_id, m in enumerate(metrics):
            results[f_id, rep, metric_id, 0] = m(gt, cids)
        
        #iPCA
        mean_chunks = []
        for i in range(n_chunks):
            _chunk_X = X[i*chunk_size:(i+1)*chunk_size]
            mean_chunks.append(np.mean(_chunk_X, axis=0))
            
        pca = PCA(n_components=4)
        pca.fit(mean_chunks[:(len(y)//2)])
        meta = pca.transform(mean_chunks)
        samples_std = StandardScaler().fit_transform(meta)
        cids = KMeans(n_clusters=np.max(gt)+1).fit_predict(samples_std)
        for metric_id, m in enumerate(metrics):
            results[f_id, rep, metric_id, 1] = m(gt, cids)

mean_res = np.round(np.mean(results, axis=1),3)
rows = []
for f_id, f in enumerate(files):
    rows.append([f, 
                 mean_res[f_id,0,0], mean_res[f_id,0,1],
                 mean_res[f_id,1,0], mean_res[f_id,1,1],
                 mean_res[f_id,2,0], mean_res[f_id,2,1],
                 mean_res[f_id,3,0], mean_res[f_id,3,1],
                 ])
    

rows = np.array(rows)
np.savetxt('experiments_R2/results_benchmarks.csv', rows, delimiter='\t', fmt="%s")

# ______

fig, ax = plt.subplots(1,4,figsize=(10,3))
for metric_id, metric in enumerate(['NMI', 'Adjusted Rand', 'Completeness', 'Homogeneity']):
        
        bplot = ax[metric_id].boxplot(mean_res[:,metric_id], widths=0.6, 
                                              patch_artist=True)
        
        colors = plt.cm.coolwarm(np.mean(mean_res[:,metric_id], axis=0))
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        
        for line in bplot['medians']:
            line.set_color('gray')
                    
        ax[metric_id].grid(ls=':')
        ax[metric_id].spines['top'].set_visible(0)
        ax[metric_id].spines['right'].set_visible(0)
        
        
        ax[metric_id].set_xticks(np.arange(1,3), ['iFFM', 'iPCA'], rotation=45)
        ax[metric_id].set_title(metric)

            
fig.align_ylabels()

plt.tight_layout()
plt.savefig('foo.png') 

exit()
#---- stat

def friedman_test(X, alpha=0.05):
    N = X.shape[0]
    k = X.shape[1]
    ranks = k + 1 - rankdata(X, axis=1)
    stat = (12 / N*k*(k+1)) * np.sum(np.sum(ranks, axis=0)**2) - 3*N*(k+1)
    chi = chi2.ppf(1 - alpha, k-1)
    return np.mean(ranks, axis=0), stat >= chi


def compute_CD(avranks, n):
    k = len(avranks)
    q = [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
        2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
        3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
        3.426041, 3.458425, 3.488685, 3.517073,
        3.543799]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    return cd

def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

def lloc(l, n):
    """
    List location in list of list structure.
    Enable the use of negative locations:
    -1 is the last element, -2 second last...
    """
    if n < 0:
        return len(l[0]) + n
    else:
        return n
    
def mxrange(lr):
    """
    Multiple xranges. Can be used to traverse matrices.
    This function is very slow due to unknown number of
    parameters.

    >>> mxrange([3,5])
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    >>> mxrange([[3,5,1],[9,0,-3]])
    [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

    """
    if not len(lr):
        yield ()
    else:
        # it can work with single numbers
        index = lr[0]
        if isinstance(index, int):
            index = [index]
        for a in range(*index):
            for b in mxrange(lr[1:]):
                yield tuple([a] + list(b))


# ----

def graph_ranks(avranks, names, cd, width=6, textspace=1, reverse=False, title=None, color='r'):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
    """
    
    width = float(width)
    textspace = float(textspace)

    sums = avranks
    tempsort = sorted([(a, i) for i, a in enumerate(sums)], reverse=reverse)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [names[x] for x in sortidx]

    lowv = min(1, int(math.floor(min(ssums))))
    highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4
    k = len(sums)
    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    def get_lines(sums, hsd):
        # get all pairs
        lsums = len(sums)
        allpairs = [(i, j)
                    for i, j in mxrange([[lsums], [lsums]]) if j > i]
        # remove not significant
        notSig = [(i, j) for i, j in allpairs
                    if abs(sums[i] - sums[j]) <= hsd]
        # keep only longest

        def no_longer(ij_tuple, notSig):
            i, j = ij_tuple
            for i1, j1 in notSig:
                if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                    return False
            return True

        longest = [(i, j) for i, j in notSig if no_longer((i, j), notSig)]

        return longest

    lines = get_lines(ssums, cd)
    linesblank = 0.2 + 0.2 + (len(lines) - 1) * 0.1

    # add scale
    distanceh = 0.25
    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig, ax = plt.subplots(1,1,figsize=(width, height))
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=0.7)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom")

    k = len(ssums)

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=0.7)
        text(textspace - 0.2, chei, nnames[i], ha="right", va="center")

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=0.7)
        text(textspace + scalewidth + 0.2, chei, nnames[i],
             ha="left", va="center")

    # upper scale
    if not reverse:
        begin, end = rankpos(lowv), rankpos(lowv + cd)
    else:
        begin, end = rankpos(highv), rankpos(highv - cd)

    line([(begin, distanceh), (end, distanceh)], linewidth=0.7, color=color)
    line([(begin, distanceh + bigtick / 2),
            (begin, distanceh - bigtick / 2)],
            linewidth=0.7, color=color)
    line([(end, distanceh + bigtick / 2),
            (end, distanceh - bigtick / 2)],
            linewidth=0.7, color=color)
    text((begin + end) / 2, distanceh - 0.05, "CD",
            ha="center", va="bottom", color=color)

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2
        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                    (rankpos(ssums[r]) + side, start)],
                    linewidth=2.5, color=color)
            start += height

    draw_lines(lines)
    
    return fig
    
# _____
metric_names = ['NMI', 'Adjusted Rand', 'Completeness', 'Homogeneity']
print(mean_res.shape) # (10, 6)
    # should be reps x methods
for metric_id, metric in enumerate(metric_names):
    results_metric = mean_res[:,metric_id]
    print(results_metric.shape)
    print(friedman_test(results_metric)) # ALL true

    ranks = []
    for r in range(results_metric.shape[0]):
        # rangi dla replikacji r
        rep_res = results_metric[r]
        print(rep_res.shape)
        ranks.append(rankdata(rep_res).tolist())
    ranks = np.array(ranks)
    print(ranks)

    av_ranks = np.mean(ranks, axis=0)
    print(av_ranks)
    # exit()
    cd = compute_CD(av_ranks, results_metric.shape[0])

    fig = graph_ranks(av_ranks, ['iFFM', 'iPCA'], cd=cd, width=6, 
                    textspace=1.1, 
                    title='Metric: %s' % metric, 
                    color=plt.cm.coolwarm(.9))
    plt.tight_layout()

    plt.savefig("foo.png", dpi=300)
    exit()
