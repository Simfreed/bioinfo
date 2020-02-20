import numpy as np
import scipy as sc
import emcee

from scipy.optimize import minimize, basinhopping, dual_annealing
from numpy import linalg as linalg

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import corner

import time


# directories
datdir  = '/Users/simonfreedman/cqub/xenopus/data'
plotdir = '/Users/simonfreedman/cqub/xenopus/plots'

sqrt3over3 = np.sqrt(3) / 3.

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
nt = 10000;
fpsL = []
tau = 100

for r0 in r0s:
    r = r0
    for t in range(nt):
        r += dt * rdot0(r, tau)
    fpsL.append(r)
fps = np.array(fpsL)[:,0]
##############################################################
##############################################################

def rdot(r, noise, tau, l0, l1, l2):
    return 1/tau * (sigma1(f(r) + np.outer(l0,fps[0]) + np.outer(l1,fps[1]) 
                           + np.outer(l2,fps[2])) - r) + noise

def norm0to1(x):
    minx = np.amin(x)
    return (x - minx)/(np.max(x) - minx)

def getSigSeriesG(sts, nt, a, mu, sig):
    
    # sts has shape T x M
    # the rest are numbers
    gaus        = a*np.exp(-(np.arange(nt)-mu)**2/sig**2)
    nper        = np.int(nt/sts.shape[0])
    stsRepeated = np.vstack([np.repeat(sts,nper,axis=0),np.zeros((nt-nper*6,sts.shape[1]))])
    return (stsRepeated.T*gaus).T
    
def finalpos(sts0, sts1, sts2, sigParams0, sigParams1, sigParams2, r0, noises, nt, dt, tau):
    
    # sts = on/off-ness of bmp at each of the T stages -- should be T x M -- currently T = 6
    # sigParams = parameters for getSigSeries function
    # r0 = initial position on fate landscape 1x2
    # noises = noise at each timestep for each data point --> nt x M
    # nt = number of timesteps (integer)
    # dt = length of timesteps (float)
    # tau = timescale (float)
    
    l0s = getSigSeriesG(sts0, nt, *sigParams0)
    l1s = getSigSeriesG(sts1, nt, *sigParams1) # nt x M
    l2s = getSigSeriesG(sts2, nt, *sigParams2) # nt x M
    
    r   = r0 * np.ones((sts0.shape[1], r0.shape[0])) # M x 2 -- assumes r0 is same for all inputs
    
    for t in range(nt):
        r += dt*rdot(r, noises[t], tau, l0s[t], l1s[t], l2s[t])
    #return r.reshape(-1)
    return getBasins(r) #.reshape(-1)

def finalposD(sts0, sts1, sts2, sigParams0, sigParams1, sigParams2, r0, dff, nt, dt, tau, seed=None):
    np.random.seed(seed)
    noises = np.sqrt(2*dff)*np.random.normal(size=(nt,sts0.shape[1],2))
    return finalpos(sts0, sts1, sts2, sigParams0, sigParams1, sigParams2, r0, noises, nt, dt, tau)

def getBasinProbabilities(sts0, sts1, sts2, sigParams0, sigParams1, sigParams2, r0, 
                          dff, nt, dt, tau, ensembleSize = 100, seed=None):
    finalPoss  = finalposD(sts0, sts1, sts2, sigParams0, sigParams1, sigParams2, r0, dff, nt, dt, tau, seed)
    finalPossS = np.array(np.split(finalPoss, range(nper, finalPoss.shape[0], nper)))
    return np.mean(finalPossS, axis=1)#.reshape(-1)

# likelihood function
nper = 100
nt   = 100
dt   = 1
tau  = 10
seed = 42
def basinProbFittingFunc(sts, a0, a1, a2, b0, b1, b2, c0, c1, c2, x0, y0, dff):
    return getBasinProbabilities(np.repeat(sts[0:6],   nper, axis=1), 
                                 np.repeat(sts[6:12],  nper, axis=1),
                                 np.repeat(sts[12:18], nper, axis=1),
                                 [a0, a1, a2], [b0, b1, b2], [c0, c1, c2], 
                                 np.array([x0, y0]), dff, 
                                 nt, dt, tau, nper, seed)[:,0:2] 
    #probably only need to fit two values because probabilities are bounded

# designed in accordance with emcee's curve fitting tutorial: 
# https://emcee.readthedocs.io/en/stable/tutorials/line/
# variance is underestimated by some fractional amount f
def log_likelihood(theta, x, y, yerr):
    #a0, a1, log_f = theta
    a0, a1, a2, b0, b1, b2, c0, c1, c2, x0, y0, dff, log_f = theta

    model  = basinProbFittingFunc(x, a0, a1, a2, b0, b1, b2, c0, c1, c2, x0, y0, dff)    
    sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))  # not sure what to use as yerr

mag_sigma2 = 25
dff_scale  = 1
maxr0      = 5

def log_prior(theta):
    a0, a1, a2, b0, b1, b2, c0, c1, c2, x0, y0, dff, log_f = theta

    tvars     = np.array([a1,a2,b1,b2,c1,c2])
    initdist  = np.linalg.norm([x0,y0])
    magvars   = np.array([a0,b0,c0])
    
    if np.any(tvars < 0) or np.any(tvars >= nt) or log_f >= 1 or log_f < -10 or dff < 0 or initdist > maxr0:
        return -np.inf
    
    return np.sum(-magvars**2/mag_sigma2) - dff / dff_scale

def log_probability(theta, x, y, yerr):
    
    lp = log_prior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(theta, x, y, yerr)

nll = lambda *args: -log_likelihood(*args)


# model output: 
preds  = np.load('{0}/log_reg_pca_preds.npy'.format(datdir))
predss  = np.load('{0}/log_reg_pca_predss.npy'.format(datdir))
# arranged as c0_rep0_t0, c0_rep0_t1, c0_rep0_t2...c0_rep1_t0, c0_rep1_t1,...,c0_rep2_t5,c1_rep0_t0,...,c2_rep2_t5

# model input
ncond  = 3 
nstg   = 6
nrep   = 3
nper   = 100

exps = np.zeros((preds.shape[0],ncond*nstg))
idx  = 0
for i in np.arange(ncond):
    for j in np.arange(nrep):
        for t in np.arange(nstg):
            exps[idx, i*nstg:(i*nstg+t+1)]=1
            idx += 1
            
yerr = 0.02 + 0.02 * np.random.rand(*preds.shape) # completely made up!
            
            
#alternatively, could use average over replicates and standard deviation as err? 
predssMu = np.mean(predss,axis=2).reshape(3,18).T #i, j, k= p(cond), traj, stage
yerrMu   = np.std(predss,axis = 2).reshape(3,18).T
expsMu   = np.zeros((predssMu.shape[0],ncond*nstg))
idx      = 0
for i in np.arange(ncond):
    for t in np.arange(nstg):
        expsMu[idx, i*nstg:(i*nstg+t+1)]=1
        idx += 1

initGaus       = np.array([0,10,5]) #0*e-((x-10)^2)/5^2
initPos        = np.array([0,0])
initDiff       = np.array([0.0001])
initLogErrFrac = np.array([-1])

tbounds      = [0,nt]
magbounds    = [-100,100] 
posbounds    = [-2,2]
dffbounds    = [0,10]
logerrbounds = [-10, 1]

idxs = np.arange(18)

xs    = expsMu[idxs].T
ys    = predssMu[idxs,0:2]
yerrs = yerrMu[idxs,0:2]

initial = np.array(np.hstack([initGaus, initGaus, initGaus, initPos, initDiff, initLogErrFrac])) #+ np.random.randn(13)
niter = 50
print('before fitting\n')
print( basinProbFittingFunc(xs, *initial[0:-1]), nll(initial, xs, ys, yerrs) )
start = time.time()
soln    = dual_annealing(nll, x0=initial, args=(xs, ys, yerrs), bounds = [magbounds, tbounds, tbounds,
                                                                          magbounds, tbounds, tbounds,
                                                                          magbounds, tbounds, tbounds,
                                                                          posbounds, posbounds,
                                                                          dffbounds, logerrbounds], maxiter = niter)
end = time.time()
elapsed = end - start

print('{0} iterations took {1} seconds\n'.format(niter, elapsed))
print(basinProbFittingFunc(xs, *soln.x[0:-1]),soln)

nwalkers  = 32
chain_len = 500
ndim     = len(initial)
pos      = soln.x + np.abs(1e-4 * np.random.randn(nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xs, ys, yerrs))
sampler.run_mcmc(pos, chain_len, progress=True);

samples = sampler.get_chain()
np.save('{0}/mcmc_samples.npy'.format(datdir),samples)

fig, axes = plt.subplots(ndim, figsize=(10, 20), sharex=True)

labels = ["a0", "a1","a2","b0","b1","b2","c0","c1","c2","x0","y0","Diff","log_f"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('{0}/mcmc_parameter_chains.jpg'.format(plotdir),bbox_inches="tight")

flat_samples = sampler.get_chain(discard=50, thin=15, flat=True)
myguess = np.array([100,0,10,100,0,10,100,0,10,0,0,0.0001,-1])
fig = corner.corner(
    flat_samples, labels=labels, truths=myguess#[m_true, b_true, np.log(f_true)]
);
plt.savefig('{0}/mcmc_corner_plot.jpg'.format(plotdir),bbox_inches="tight")

tau = sampler.get_autocorr_time()
print(tau)

