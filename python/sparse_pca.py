import numpy as np
from numpy import linalg as linalg




def spca_err_j_xtx(xtx,aj,bj,lambdaj):
    # error function for one column, given xT.x
    # typically not used because its too big of a matrix to carry around
    return -2*aj.T.dot(xtx).dot(bj)+np.inner(bj,bj)+lambdaj*np.sum(np.abs(bj))

def spca_err_j(x,aj,bj,lambdaj):
    # error function for one column, given x
    return -2*aj.T.dot(x.T).dot(x).dot(bj)+np.inner(bj,bj)+lambdaj*np.sum(np.abs(bj))

def spca_err_all(x,A,B,lambdas,k):
    return [spca_err_j(x,A[:,j],B[:,j],lambdas[j]) for j in range(k)]

def spca_err_tot(x,A,B,lambdas,k):
    return np.sum(np.array(scpa_err_all(x,A,B,lamdbas,k)))

def sparsity(v):
    if len(v.shape) > 1:
        return [np.where(v[:,j]!=0)[0] for j in range(v.shape[1])]
    else:
        return np.where(v != 0)[0]


def minimizeAB(x, ainit, binit, lambdas, nIter, errFreq):

    # Algorithm 1 (with eqn 4.3) in Zou, Hastie, Tibshirani, 2006 

    A = ainit
    B = binit
    nvec = ainit.shape[1]
    
    for i in range(nIter):
        
        # error reporting
        if i % errFreq == 0 or i == nIter - 1:
            err = spca_err_all(x, A, B, lambdas,nvec)
            errstr = '[' + ','.join(['{0:.3e}'.format(k) for k in err])+']'
            print('iter {0}, errors = {1} ==> total error = {2:.6e}'.format(i,errstr, np.sum(np.array(err))));

        # loop over eigenvectors
        for j in range(nvec):
            axtxJ  = A[:,j].dot(x.T).dot(x) 
            B[:,j] = np.maximum(np.abs(axtxJ)-lambdas[j]/2,0)*np.sign(axtxJ)
        
        bsvd = linalg.svd(x.T.dot(x.dot(B)),full_matrices=False) # not really sure this order of operations is ok...
        A    = bsvd[0].dot(bsvd[2]) 
    
    return (A,B)
