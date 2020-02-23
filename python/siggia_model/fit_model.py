import numpy as np
import scipy as sc
import emcee

from scipy.optimize import minimize, basinhopping, dual_annealing
from numpy import linalg as linalg

import matplotlib
matplotlib.use('Agg')

from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import corner

import time

import argparse 

from multiprocessing import Pool
from multiprocessing import cpu_count

import os

# possibly important for parallelization
os.environ["OMP_NUM_THREADS"] = "1"

# this should force matplotlib to not require a display

parser = argparse.ArgumentParser()

parser.add_argument("--dir",          type=str,   help="output subdirectory", default='mcmc{0}'.format(time.strftime('%m-%d-%y_%H:%M')))
parser.add_argument("--niter",        type=int,   help="number of iterations for dual_annealing", default=10)
parser.add_argument("--nwalkers",     type=int,   help="mcmc nwalkers",    default=32)
parser.add_argument("--chain_len",    type=int,   help="mcmc len",    default=100)
parser.add_argument("--thin",         type=int,   help="output chain every k timesteps",    default=1)
parser.add_argument("--seed",         type=int,   help="the seed of course",    default=0)
parser.add_argument("--showProgress", type=bool,  help="show mcmc progress",    default=False)

args = parser.parse_args()

# directories
topdir  = '/home/slf3348'
datdir  = '{0}/xenopus/data'.format(topdir)
outdir  = '{0}/{1}'.format(datdir,args.dir)
os.makedirs(outdir, exist_ok = True)

# record args
f = open("{0}/args.txt".format(outdir),"w")
f.write( str(args) )
f.close()

logfile = open("{0}/log.txt".format(outdir), "w")

sqrt3over3 = np.sqrt(3) / 3.

# likelihood function settings
nper = 100
nt   = 100
dt   = 1
tau  = 10

# prior priors
ndim = 9 # number of parameters to fit
mag_sigma = 5
mag_sigma2 = mag_sigma**2
dff_scale  = 0.01
maxx = 2
maxr0 = maxx**2

# model input
ncond  = 3 
nstg   = 6
nrep   = 3

seed      = args.seed 
niter     = args.niter
nwalkers  = args.nwalkers
chain_len = args.chain_len

initGaus       = np.array([0,10,5]) #0*e-((x-10)^2)/5^2
initPos        = np.array([0,0])
initDiff       = np.array([0.0001])
initLogErrFrac = np.array([-1])

tbounds      = [0,nt]
magbounds    = [-100,100] 
posbounds    = [-2,2]
dffbounds    = [0,10]
logerrbounds = [-10, 1]


logfile.write("initializing functions")
# HERE BEGINS FUNCTIONS
def f(r):
    return 2*r + np.vstack([-2*r[:,0]*r[:,1] , r[:,1]**2 - r[:,0]**2]).T

def sigma1(f):
    nrm = np.linalg.norm(f,axis=1)
    #return np.tanh(nrm)*f/nrm
    return (np.tanh(nrm)*np.divide(f.T, nrm, out=np.zeros_like(f.T), where=nrm!=0)).T

def rdot0(r, tau):
    return 1./tau * (sigma1(f(r)) - r)

def getBasins(rs):
    basins = np.zeros((rs.shape[0],3))
    
    inb0   = rs[:,1] > sqrt3over3*np.abs(rs[:,0])
    inb1   = ~inb0 & (rs[:,0]>0)
    inb2   = ~(inb0 | inb1)
    
    basins[inb0,0] = 1
    basins[inb1,1] = 1
    basins[inb2,2] = 1
    return basins

##############################################################
##############################################################
# calculate fixed points
r0s = np.array([[[0, 0.1]], [[0.1, -0.1]], [[-0.1, -0.1]]]);
dt = 1;
ntfp = 10000;
fpsL = []
tau = 100

for r0 in r0s:
    r = r0
    for t in range(ntfp):
        r += dt * rdot0(r, tau)
    fpsL.append(r)
fps = np.array(fpsL)[:,0]
##############################################################
##############################################################

def rdot2(r, noise, tau, l0, l1):
    return 1/tau * (sigma1(f(r) + np.outer(l0,fps[0]) + np.outer(l1,fps[2])) - r) + noise

def getSigSeriesG(sts, nt, a, mu, sig):

    # sts has shape T x M
    # the rest are numbers
    gaus        = a*np.exp(-(np.arange(nt)-mu)**2/sig**2)
    nper        = np.int(nt/sts.shape[0])
    stsRepeated = np.vstack([np.repeat(sts,nper,axis=0),np.zeros((nt-nper*6,sts.shape[1]))])
    return (stsRepeated.T*gaus).T

def fullTraj(sts0, sts1, sigParams0, sigParams1, r0, noises, nt, dt, tau, npts=6):
    
    # sts = on/off-ness of bmp at each of the T stages -- should be T x M -- currently T = 6
    # sigParams = parameters for getSigSeries function
    # r0 = initial position on fate landscape 1x2
    # noises = noise at each timestep for each data point --> nt x M
    # nt = number of timesteps (integer)
    # dt = length of timesteps (float)
    # tau = timescale (float)
    
    l0s = getSigSeriesG(sts0, nt, *sigParams0)
    l1s = getSigSeriesG(sts1, nt, *sigParams1) # nt x M
    
    #r   = r0 * np.ones((sts0.shape[1], r0.shape[0])) # M x 2 -- assumes r0 is same for all inputs
    rs  = np.zeros((sts0.shape[1], nt, r0.shape[0])) # M x nt x 2
    rs[:,0] = r0
    for t in range(0, nt-1):
        rs[:,t+1] = rs[:,t] + dt*rdot2(rs[:,t], noises[t], tau, l0s[t], l1s[t])
    
    tidxs = np.array(np.around(np.linspace(0,nt-1,npts)), dtype='int')
    return np.array([getBasins(rs[:,t]) for t in tidxs]) # should vectorize getBasins...

def fullTrajD(sts0, sts1, sigParams0, sigParams1, r0, dff, nt, dt, tau, seed=None, npts=6):
    np.random.seed(seed)
    noises = np.sqrt(2*dff)*np.random.normal(size=(nt,sts0.shape[1],2))
    return fullTraj(sts0, sts1, sigParams0, sigParams1, r0, noises, nt, dt, tau, npts)

def getTrajBasinProbabilities(sts, sigParams0, sigParams1, r0, dff, nt, dt, tau, 
                              nper = 100, seed=None, npts=6):
    trajBasins  = fullTrajD(np.repeat(sts[0:6], nper, axis=1), 
                            np.repeat(sts[6:12], nper, axis=1), 
                            sigParams0, sigParams1, r0, dff, nt, dt, tau, seed, npts)
    trajBasinsS = np.array(np.split(trajBasins, range(nper, trajBasins.shape[1], nper), axis=1))
    return np.mean(trajBasinsS, axis=2)#.reshape(-1)

def log_likelihood(theta, x, y, yerr):
    #a0, a1, log_f = theta
    a0, a1, a2, b0, b1, b2, x0, y0, dff = theta

    model  = getTrajBasinProbabilities(x, [a0, a1, a2], [b0, b1, b2], np.array([x0, y0]), dff, nt, dt, tau, nper)[:,:,:2]
    sigma2 = yerr ** 2 #+ model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 ) #+ np.log(sigma2))

def log_prior(theta):
    a0, a1, a2, b0, b1, b2, x0, y0, dff = theta
    tvars     = np.array([a1,a2,b1,b2])
    initdist  = np.linalg.norm([x0,y0])
    magvars   = np.array([a0,b0])

    if np.any(tvars < 0) or np.any(tvars >= nt) or dff < 0 or initdist > maxr0:
        return -np.inf

    return np.sum(-magvars**2/mag_sigma2) - dff / dff_scale

def random_parameter_set(iseed = seed):
    np.random.seed(iseed)
    
    a0,b0       = np.random.normal(loc=0, scale = mag_sigma, size = 2)
    
    a1,a2,b1,b2 = np.random.uniform(low=0,high=nt,size=4)
    #x0,y0       = np.random.uniform(low=-maxx, high=maxx, size=2)
    x0,y0       = np.random.normal(loc = 0 , scale = maxx/10, size=2)
    dff         = np.random.exponential(scale = dff_scale)
    
    return np.array([a0,a1,a2,b0,b1,b2,x0,y0,dff])

def log_probability(theta, x, y, yerr):
    
    lp = log_prior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(theta, x, y, yerr)

nll = lambda *args: -log_likelihood(*args)

logfile.write("\nloading log reg output")
# model output: 

#preds   = np.load('{0}/log_reg_pca_preds.npy'.format(outdir))
# arranged as c0_rep0_t0, c0_rep0_t1, c0_rep0_t2...c0_rep1_t0, c0_rep1_t1,...,c0_rep2_t5,c1_rep0_t0,...,c2_rep2_t5

#yerr = 0.02 + 0.02 * np.random.rand(*preds.shape) # completely made up!
#predss  = np.load('{0}/log_reg_pca_predss.npy'.format(datdir))

predsS  = np.load('{0}/log_reg_pca_predss.npy'.format(datdir))

y    = np.mean(predsS,axis=1)[:,:,:2]
yerr = np.std(predsS,axis = 1)[:,:,:2]


bmpOn = np.vstack([ np.ones(6) , np.zeros(6) , np.ones(6)])
tgfOn = np.vstack([ np.zeros(6), np.zeros(6) , np.ones(6)])

x = np.hstack([bmpOn,tgfOn]).T
            
logfile.write('\nhammer time')
#pos = np.array([random_parameter_set() for i in range(nwalkers)])
pos  = random_parameter_set() + np.abs(1e-4*np.random.randn(nwalkers, ndim))
#print((pos,pos.shape))

# Set up the backend
# Don't forget to clear it in case the file already exists
filename = "{0}/sampler_backend.h5".format(outdir)
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

# Initialize the sampler
start = time.time()
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr), pool=pool, backend=backend)
    sampler.run_mcmc(pos, chain_len, progress=args.showProgress);

end     = time.time()
elapsed = end - start

logfile.write('\nsampling with {0} walkers for {1} steps took {2} seconds (niter = {3})\n'.format(nwalkers, chain_len, elapsed, niter))

samples = sampler.get_chain(thin=args.thin)
np.save('{0}/samples.npy'.format(outdir),samples)

fig, axes = plt.subplots(ndim, figsize=(10, 20), sharex=True)

labels = ["a0", "a1","a2","b0","b1","b2","x0","y0","Diff"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('{0}/parameter_chains.png'.format(outdir),bbox_inches="tight")

flat_samples = sampler.get_chain(discard=50, thin=15, flat=True)
myguess = np.array([100,0,10,100,0,10,100,0,10,0,0,0.0001,-1])
fig = corner.corner(
    flat_samples, labels=labels, truths=myguess#[m_true, b_true, np.log(f_true)]
);
plt.savefig('{0}/corner_plot.png'.format(outdir),bbox_inches="tight")

try:
    tau = sampler.get_autocorr_time()
    logfile.write(tau)
except emcee.autocorr.AutocorrError:
    logfile.write('\nnot enough data to compute autocorr')

