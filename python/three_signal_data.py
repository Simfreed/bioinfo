import sparse_pca as spca
import csv
import os
import numpy as np
from numpy import linalg as linalg
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("directory",type=str,   help="output directory")
parser.add_argument("--ps",     type=float, help="percentages of data for signal",      default=[0.6, 0.1, 0.01], nargs = 3)
parser.add_argument("--nrow", type=int,   help="number of rows",    default=30000)
parser.add_argument("--nsamp",  type=int,   help="number of samples (ncol = nsamp*nrep)",    default=6)
parser.add_argument("--nrep",   type=int,   help="number replicates of columns (ncol = nsamp*nrep)",    default=3)
parser.add_argument("--seed",   type=int,   help="number of columns per sample",    default=0)

args = parser.parse_args()

nrow = args.nrow
ncol    = args.nrep*args.nsamp
nsamp      = args.nsamp
nrep    = args.nrep
outdir  = '{0}/three_sig_{1}x{2}_seed{3}'.format(args.directory, nrow, ncol, args.seed)
samps       = np.arange(nsamp)
all_samps   = np.hstack([samps for i in range(nrep)])

if not os.path.exists(outdir):
    os.makedirs(outdir)

##################################
####### CREATE FAKE DATA #########
##################################


#data funcs
lin   = lambda m,b,x: m*x+b
parab = lambda a,b,x: a*(x-b)**2

# initialize matrix to noise
temp = 0.5
dat  = temp*np.random.lognormal(size=(nrow,ncol)) 

# add linear signal to fraction of data
linfrac         = args.ps[0]
linGenes        = np.random.choice(nrow, int(nrow*linfrac), replace = False)
dat[linGenes]   = dat[linGenes] + lin(1, (nsamp-1)/2, all_samps)

# add parabolic signal to smaller fraction of data
parabfrac       = args.ps[1]
parabGenes      = np.random.choice(nrow, int(nrow*parabfrac), replace=False)
dat[parabGenes] = dat[parabGenes] + parab(1, (nsamp-1)/2, all_samps)

# add constant offset to smallest fraction of data
constfrac       = args.ps[2]
constGenes      = np.random.choice(nrow, int(nrow*constfrac), replace = False)
dat[constGenes] = dat[constGenes] + np.hstack([lin(0,1000,samps),np.zeros(nsamp),np.zeros(nsamp)])

sigGenes  = [linGenes, parabGenes, constGenes]
nsig      = len(sigGenes)

# SAVE METADATA
np.savetxt('{0}/mat.tsv'.format(outdir), dat.T)

with open('{0}/sigGenes.tsv'.format(outdir),'w') as f:
    wr = csv.writer(f, delimiter='\t')
    wr.writerows(sigGenes)

# zscore, log
ldat  = np.log10(dat)
mus   = ldat.mean(axis=1)
sigs  = ldat.std(axis=1,ddof=1)
ldatz = (ldat.T-mus)/sigs

np.savetxt('{0}/mat_log_zsc.tsv'.format(outdir), ldatz)
