import three_well as w3
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
parser.add_argument("--err_scale",    type=float,  help="scale of error",    default=0.1)
parser.add_argument("--dff_scale",    type=float,  help="scale of diffusion",    default=0.01)
parser.add_argument("--backend",        type=str,    help="backend file name, within dir", default="sampler_backend")
parser.add_argument("--reload_backend", type=bool,    help="backend to start from", default=False)

args = parser.parse_args()

seed      = args.seed 
niter     = args.niter
nwalkers  = args.nwalkers
chain_len = args.chain_len


# directories
topdir  = '/projects/p30129/simonf/out' #/home/slf3348'
datdir  = '{0}/xenopus/data/siggia_mcmc'.format(topdir)
outdir  = '{0}/{1}'.format(datdir,args.dir)
os.makedirs(outdir, exist_ok = True)

# record args
f = open("{0}/args.txt".format(outdir),"a+")
f.write( str(args) )
f.close()

logfile = open("{0}/log.txt".format(outdir), "a+")

# Load the training data
predsS  = np.load('{0}/log_reg_pca_predss.npy'.format(datdir))

bmpOn = np.vstack([ np.ones(6) , np.zeros(6) , np.ones(6)])
tgfOn = np.vstack([ np.zeros(6), np.zeros(6) , np.ones(6)])

x = np.hstack([np.repeat(bmpOn,nrep,axis=0),np.repeat(tgfOn,nrep,axis=0)]).T
y = predsS.reshape((9,6,3))[:,:,:2]

# initialize the model
default_params = ['dt', 'tau']
fixed_params   = {'nt':100, 'nper':100}
prior_scales   = {'yerr':[0.0005], 'diff':[0.05], 'a0':[0,5], 'b0':[0,5]}

myw3 = w3.ThreeWell(set_param_dict = fixed_params, default_value_params = default_params,
        unset_param_prior_scale_dict = prior_scales, seed = args.seed)


logfile.write('\nhammer time')
ndim = myw3.ntheta
pos  = myw3.random_parameter_set() + np.abs(1e-4*np.random.randn(nwalkers, ndim))

# Set up the backend
# Don't forget to clear it in case the file already exists
filename = "{0}/{1}.h5".format(outdir, args.backend)
backend  = emcee.backends.HDFBackend(filename)

# Initialize the sampler
start = time.time()

if args.reload_backend:
    logfile.write("Initial size: {0}".format(backend.iteration))
    pos=None
else:
    backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, myw3.log_probability, args=(x, y), pool=pool, backend=backend)
    sampler.run_mcmc(pos, chain_len, progress=args.showProgress);

end     = time.time()
elapsed = end - start

logfile.write('\nsampling with {0} walkers for {1} steps took {2} seconds (niter = {3})\n'.format(nwalkers, chain_len, elapsed, niter))
logfile.write("Final size: {0}".format(backend.iteration))

samples = sampler.get_chain(thin=args.thin)
np.save('{0}/samples.npy'.format(outdir),samples)

fig, axes = plt.subplots(ndim, figsize=(10, 20), sharex=True)

labels = ["a0", "a1","a2","b0","b1","b2","x0","y0","Diff","yvar"]
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
    logfile.write(str(tau))
except emcee.autocorr.AutocorrError:
    logfile.write('\nnot enough data to compute autocorr')

logfile.close()
