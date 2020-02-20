import numpy as np
from numpy import linalg

def argmaxsort(m):
    ams    = m.argmax(axis = 0) # position of max for every gene
#    amsort = ams.argsort()      # ordering of those positions
#    breaks       = np.where(np.diff(ams[amsort])!=0)[0] # indexes where it switches
#    amsortSplit  =  np.split(amsort, breaks+1) # groups of indexes with common argmax
    amsortSplit = [[] for i in range(m.shape[0])]
    for j in range(m.shape[1]):
        amsortSplit[ams[j]].append(j)

    amsortSplit = [np.array(k) for k in amsortSplit]
    subOrders   = [np.argsort(m[i,amsortSplit[i]]) if amsortSplit[i].shape[0]>0 else np.array([]) for i in range(m.shape[0])] # subsort within each group
    sortOrder2  = np.hstack([amsortSplit[i][subOrders[i]] for i in range(len(subOrders)) if amsortSplit[i].shape[0]>0]) # recombine into one matrix
    
    return sortOrder2, list(map(len, amsortSplit))

def list2multiDict(l):
    d = {}
    for k,v in l:
        if k in d:
            d[k] += [v]
        else:
            d[k] = [v]
    return d

def multiDictMerge(lA, lB):
    # lA has structure: keyA:[valA1, valA2, ...]
    # lB has structure: keyB:[valB1, valB2, ...]
    # where keyB can be valA1, valA2 ...
    # output has structure: keyA: [valB1, valB2, ...]
    d = {}
    
    for kA,vsA in lA.items():
        #vsA   = list.copy(lA[kA])
        d[kA] = list.copy(lB.get(vsA[0], []))
#        d[kA] = list.copy(lB[vsA[0]]) # assumes no null keys
        for vA in vsA[1:]:
            d[kA] += list.copy(lB.get(vA,[]))
        #print('i = {0} dlens = {1}'.format(i, [len(d[k]) for k in d]))

    return d

def reverseDictList(d1):
    d2 = {}
    for k,vs in d1.items():
        for v in vs:
            d2[v] = d2.get(v, [])
            d2[v].append(k)
    return d2

flatten2d = lambda x: [i for j in x for i in j] 

def pArgsort(seq):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    return sorted(range(len(seq)), key=seq.__getitem__)

cipv = lambda x,y: stats.t.interval(y, len(x)-1, loc=np.mean(x),  scale = stats.sem(x))
ci95 = lambda x: stats.t.interval(0.95, len(x)-1, loc=np.mean(x), scale = stats.sem(x))

def venn3_split(sets):
    '''
    input: [A, B, C] where they're all python sets
    output: [A, B, C, AB, AC, BC, ABC] where XY is set union of X and Y
    '''
    set111 = sets[0].intersection(sets[1]).intersection(sets[2])
    set110 = sets[0].intersection(sets[1]).difference(sets[2])
    set101 = sets[0].intersection(sets[2]).difference(sets[1])
    set011 = sets[1].intersection(sets[2]).difference(sets[0])
    set100 = sets[0].difference(sets[1]).difference(sets[2])
    set010 = sets[1].difference(sets[0]).difference(sets[2])
    set001 = sets[2].difference(sets[0]).difference(sets[1])
    return [set100, set010, set001, set110, set101, set011, set111]

# source: https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
# NOTE: CURRENTLY ONLY TESTED WHEN RESAMPLING HAS SIZE 1
def bootstrappedCI(mu, samps, ciWidth):
    sampMus = np.mean(samps, axis=1) if len(samps.shape) > 1 else samps
    diffs   = sampMus - mu
    order   = np.argsort(diffs)
    pvalHf  = (1-ciWidth)/2
    idx0    = int(np.floor(pvalHf*len(diffs)))
    idx1    = int(np.ceil((1-pvalHf)*len(diffs)))
    return (mu - diffs[order][idx1], mu - diffs[order][idx0])

def set_axs_fontsize(ax,fs,inc_leg=False):
    items = [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if inc_leg:
        items += ax.legend().get_texts()
    for itm in items:
        itm.set_fontsize(fs)

def clean_nanmax(s):
    if np.all(np.isnan(s)):
        return np.nan
    elif s.shape[0] == 0:
        return 0
    else:
        return np.nanmax(s)

divz = lambda x,y : np.divide(x, y, out=np.zeros_like(x), where=y!=0)
meanor0 = lambda x: np.nanmean(x) if x.shape[0]>0 else 0
#maxor0 = lambda x: np.nan if np.all(x) np.nanmax(x) if x.shape[0]>0 else 0

meanOrZero = lambda l: np.nanmean(l) if len(l) > 0 else 0

def logCenPca(dat,minval):
    lgdats   = np.log10(dat.T + minval)#[tpmThIdxs]
    ngenes   = lgdats.shape[0]
    mus      = lgdats.mean(axis=1)
    lgdatsfz = (lgdats.T-mus)#/sigs

    gpca    = linalg.svd(lgdatsfz, full_matrices = False)
    eigs    = gpca[1]**2/ngenes
    pcs     = lgdatsfz.dot(gpca[2].T)
    return pcs, lgdatsfz, gpca[2]
