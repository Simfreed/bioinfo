import sparse_pca as spca
import csv
import os
import numpy as np
from numpy import linalg as linalg
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("directory",    type=str,   help="output directory")
parser.add_argument("matfile",      type=str,   help="matrix on which to do sparse pca")
parser.add_argument("--sigfile",    type=str,   help="lists of matrix rows for each signal", default = '')
parser.add_argument("--lams",       type=float, help="lowest and highest lambdas",           default=[100,500], nargs = 2)
parser.add_argument("--nlams",      type=int,   help="number of lambdas",                    default=2)
parser.add_argument("--nsamps",     type=int,   help="number of samples per lambda",         default=2)
parser.add_argument("--sampsize",   type=int,   help="number of columns per sample",         default=17)
parser.add_argument("--niter",      type=int,   help="number of sparse pca iters",           default=100)
parser.add_argument("--pfreq",      type=int,   help="frequency of spca print statements",   default=1000)
parser.add_argument("--seed",       type=int,   help="seed for sampling matrix",             default=0)
parser.add_argument("--replace",    type=bool,  help="sample with replacement?",             default=False)

args = parser.parse_args()

nsamps     = args.nsamps 
sampSize   = args.sampsize
lams       = np.linspace(args.lams[0],args.lams[1],args.nlams)
outdir     = '{0}/lam{1}-{2}_n{3}_sz{4}_seed{5}'.format(args.directory, args.lams[0], args.lams[1], nsamps, sampSize, args.seed)

if args.replace:
    outdir = outdir + '_wr'

dat        = np.loadtxt(args.matfile)

np.random.seed(args.seed)

nsig = 0
if args.sigfile:
    sigs       = open(args.sigfile, 'r').readlines()
    sigGenes   = [ np.array(l.rstrip().split('\t'), dtype=int) for l in sigs ]
    nsig       = len(sigGenes)

print('nsig = {0}'.format(nsig))
nrow, ncol = dat.shape

if not os.path.exists(outdir):
    os.makedirs(outdir)


######## RUN SPCA ##########
niter     = args.niter 
printfreq = args.pfreq

nnz_mat  = np.zeros((nsig, nsamps, lams.shape[0]))
ncor_mat = np.zeros_like(nnz_mat)

for i in range(nsamps):
    print('{0}th sample'.format(i))
    x         = dat[np.random.choice(nrow, sampSize, replace=args.replace)]
    xsvd      = linalg.svd(x, full_matrices=False)
    xpcVecs   = xsvd[2].T
    B         = np.ones(xpcVecs.shape)
    
    for j in range(len(lams)):
        print('\tlambda = {0}'.format(lams[j]))
        lambdas             = lams[j]*np.ones(sampSize)
        anew,bnew           = spca.minimizeAB(x, xpcVecs, B, lambdas, niter, printfreq)
        sgenes              = spca.sparsity(bnew[:, 0:nsig]) 
        lens                = [ k.shape[0] for k in sgenes ]
        print('lens = {0}'.format(lens))
        ncor                = [ np.intersect1d( sigGenes[k] , sgenes[k] ).shape[0] for k in range(nsig) ]
        nnz_mat[:, i, j]    = np.array(lens)
        ncor_mat[:, i, j]   = np.array(ncor)
        print('(i,j) = ({0},{1}); nnzmat = \n{2}'.format(i, j, nnz_mat))

np.save('{0}/nnz_mat.npy'.format(outdir),  nnz_mat)
np.save('{0}/ncor_mat.npy'.format(outdir), ncor_mat)
