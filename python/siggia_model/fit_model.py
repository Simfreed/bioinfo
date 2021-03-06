import three_well as w3
import numpy as np
import scipy as sc
import emcee
import re

from scipy.optimize import minimize, basinhopping, dual_annealing
from numpy import linalg as linalg

import matplotlib
matplotlib.use('Agg') # this should force matplotlib to not require a display

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

moveList =[ 
        emcee.moves.StretchMove(), 
        emcee.moves.WalkMove(),
        emcee.moves.KDEMove(),
        emcee.moves.DEMove(),
        emcee.moves.DESnookerMove(),
]


parser = argparse.ArgumentParser()

parser.add_argument("--dir",        type=str,   help="output subdirectory", default='mcmc{0}'.format(time.strftime('%m-%d-%y_%H:%M')))
parser.add_argument("--nwalkers",   type=int,   help="mcmc nwalkers",    default=32)
parser.add_argument("--chain_len",  type=int,   help="mcmc len",    default=100)
parser.add_argument("--thin",       type=int,   help="output chain every k timesteps",    default=1)
parser.add_argument("--seed",       type=int,   help="the seed of course",    default=0)
parser.add_argument("--backend",    type=str,   help="backend file name, within dir", default="sampler_backend")

parser.add_argument("--prev_backend_file", type=str, help="backend filename to continue from, but not (necessarily) to rewrite", 
        default='')

parser.add_argument('--reload_backend', dest='reload_backend', help="start from backend file in the same dir (and rewrite all those files)", 
        action='store_true')
parser.add_argument("--show_progress",   dest='show_progress',   help="show mcmc progress",    action='store_true')
parser.add_argument("--run_in_serial",   dest='run_in_serial',   help="flag-- if true, don't use multiprocessing",    action='store_true')

parser.add_argument("--moves",          type=int,   nargs='+',    help=str({i:moveList[i] for i in range(len(moveList))}), default =[0])
parser.add_argument("--move_probs",     type=float, nargs='+', help="probabilities of selected moves", default=[1])

parser.add_argument("--fixed_params",       type=str, nargs='+', help="list of params to provide fixed values for",     default = [])
parser.add_argument("--default_params",     type=str, nargs='+', help="list of params to use default values for",       default = [])
parser.add_argument("--prior_type_params",  type=str, nargs='+', help="list of params to change the default prior for", default = [])
parser.add_argument("--prior_scale_params", type=str, nargs='+', help="list of params to use default values for",       default = [])

parser.add_argument("--fixed_values",   type=float, nargs='+', help="values of fixed params",     default = [])
parser.add_argument("--prior_types",    type=int,   nargs='+', help="prior types, 0: uniform, 1: gaussian, 2: exponential, 3: integer", default = [])
parser.add_argument("--prior_scales",   type=str,   nargs='+', 
        help="list of prior scales-- format: comma between two numbers for the same param, space between numbers for different params", 
        default = [])
parser.add_argument("--model_mode", type=str, help='''type of model to use; e.g, 1d2w_s means 1 dimensional model with 2 wells and soft boundaries.
        currently supported: [1d2w_s, 1d2w_h, 2d3w_s, 2d3w_h, 2d4w_s, 2d4w_h, 2d3w_S_s, 2d3w_S_h]''')
parser.add_argument("--rdot_depth",      type=float, help="b parameter for rdot function", default = -1)

parser.add_argument("--init_pos_file",   type=str, help="dict with sampling initial position data", default = '')

args = parser.parse_args()

seed      = args.seed 
nwalkers  = args.nwalkers
chain_len = args.chain_len

# directories
topdir  = '/projects/p30129/simonf/out' #/home/slf3348'
topdir  = '/projects/p31095/simonf/out' #/home/slf3348'
datdir  = '{0}/xenopus/data/siggia_mcmc'.format(topdir)
outdir  = '{0}/{1}'.format(datdir,args.dir)
os.makedirs(outdir, exist_ok = True)
logfile = open("{0}/log.txt".format(outdir), "a+")

# record args
np.save('{0}/args.npy'.format(outdir), args) 

# Load the training data

ndim    = int(re.search('\dd',args.model_mode).group()[0])
nprobs  = int(re.search('\dw',args.model_mode).group()[0])
predsS  = np.load('{0}/log_reg_pca_preds{1}.npy'.format(datdir, nprobs))
#nrep    = predsS.shape[1] 

bmpOn = np.vstack([ np.ones(6) , np.zeros(6) , np.ones(6)])
tgfOn = np.vstack([ np.zeros(6), np.zeros(6) , np.ones(6)])

#x = np.hstack([np.repeat(bmpOn,nrep,axis=0),np.repeat(tgfOn,nrep,axis=0)]).T
#y = predsS.reshape((9,6,nprobs))[:,:,:(nprobs-1)]

if ndim == 1:
    x = bmpOn[0:2].T
else:
    x = np.hstack([bmpOn,tgfOn]).T

y = predsS[:,:,:,:-1]

print('\n\nx.shape={0}\n\n'.format(x.shape))
print('\n\nx={0}\n\n'.format(x))
print('\n\ny.shape={0}\n\n'.format(y.shape))
print('\n\ny={0}\n\n'.format(y))


# initialize the model
# prior_scales   = {'yerr':[0.0005], 'diff':[0.05], 'a0':[0,5], 'b0':[0,5], 'a2':[0,10], 'b2':[0,10]}
default_params   = args.default_params 
prior_scales     = [[float(vi) for vi in v.split(',')] for v in args.prior_scales]
fixed_param_dict = {k:v for k,v in zip(args.fixed_params, args.fixed_values)} 
prior_scale_dict = {k:v for k,v in zip(args.prior_scale_params, prior_scales)}
prior_type_dict  = {k:v for k,v in zip(args.prior_type_params, args.prior_types)}

if args.rdot_depth > 0:
    rdot_b = args.rdot_depth
elif nprobs == 3:
    rdot_b = 2
else:
    rdot_b = 2./3.

myw3   = w3.ThreeWell(set_param_dict = fixed_param_dict,   default_value_params = default_params,
        unset_param_prior_scale_dict = prior_scale_dict, unset_param_prior_type_dict = prior_type_dict, seed = args.seed,
        mode = args.model_mode, rdot_depth = rdot_b)

labels = myw3.get_theta_labels()

print('fitting params: {0}'.format(labels))
print('fixed params: {0}'.format(myw3.get_fixed_params()))
print('\npriors on sampling params:\n')

logfile.write('fitting params: {0}'.format(labels))
logfile.write('fixed params: {0}'.format(myw3.get_fixed_params()))
logfile.write('\npriors on sampling params:\n')

np.save('{0}/fixed_params.npy'.format(outdir), myw3.get_fixed_params()) 
np.save('{0}/fit_params.npy'.format(outdir),   myw3.get_theta_labels()) 

for k,v in myw3.get_prior_info().items():
    print('{0}:{1}'.format(k,v))
    logfile.write('{0}:{1}'.format(k,v))

logfile.write('\nhammer time')
ndim = myw3.ntheta
#pos  = myw3.random_parameter_set() + np.abs(1e-4*np.random.randn(nwalkers, ndim))

#########################################################
# initialization of the sampler...
# possibilities: 
# (a) new sampling (==> new backend)
#    (i)  random position  (args.init_pos_file = '')
#    (ii) guessed position (args.init_pos_file = '<path_to_dict.npy>')
# (b) continued sampling
#    (i) re-using same h5 file (args.reload_backend = True)
#    (ii) new h5 file (args.prev_backend_file = True, args.reload_backend = False)

# Initialize the sampler
start = time.time()

# Set up the backend
backend  = emcee.backends.HDFBackend("{0}/{1}.h5".format(outdir, args.backend))

if args.reload_backend:
    logfile.write("Initial size: {0}".format(backend.iteration))
    pos=None
else:
    # clear it
    backend.reset(nwalkers, ndim)
    
    if args.prev_backend_file:
        prev_backend = emcee.backends.HDFBackend(args.prev_backend_file)
        pos          = prev_backend.get_last_sample().coords[0:nwalkers]
        logfile.write("Starting from iteration: {0}".format(prev_backend.iteration))
    else:
        if args.init_pos_file:
            params_guess = np.load('{0}/{1}'.format(datdir, args.init_pos_file), allow_pickle=True).item()
            theta_guess = myw3.make_theta(params_guess)
        else:
            theta_guess = myw3.random_parameter_set()
        
        print('\ninitializing near (if possible)\n')
        logfile.write('\ninitializing near (if possible)\n')
        for k,v in zip(labels,theta_guess):
            print('{0}:{1}'.format(k,v))
            logfile.write('{0}:{1}'.format(k,v))

        pos = theta_guess + np.abs(1e-5*np.random.randn(nwalkers, ndim))
    for i in range(pos.shape[0]):
        pos[i] = myw3.resample_pos(pos[i])
    print('\npos.shape={0}'.format(pos.shape))

################################################################

sampling_moves = tuple(((moveList[m], p) for (m,p) in zip(args.moves, args.move_probs)))
print('using the following moves / probabilities: \n{0}'.format(sampling_moves))
logfile.write('using the following moves / probabilities: \n{0}'.format(sampling_moves))
if args.run_in_serial:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, myw3.log_probability, args=(x, y), backend=backend, moves = sampling_moves)
    sampler.run_mcmc(pos, chain_len, progress=args.show_progress);
else:
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, myw3.log_probability, args=(x, y), pool=pool, backend=backend, moves = sampling_moves)
        sampler.run_mcmc(pos, chain_len, progress=args.show_progress);

end     = time.time()
elapsed = end - start

logfile.write('\nsampling with {0} walkers for {1} steps took {2} seconds)\n'.format(nwalkers, chain_len, elapsed))
logfile.write("Final size: {0}".format(backend.iteration))

samples = sampler.get_chain(thin=args.thin)
np.save('{0}/samples.npy'.format(outdir),samples)

fig, axes = plt.subplots(ndim, figsize=(10, 20), sharex=True)

for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('{0}/parameter_chains.png'.format(outdir),bbox_inches="tight")

flat_samples = sampler.get_chain(discard=50, thin=15, flat=True)
#myguess = np.array([100,0,10,100,0,10,100,0,10,0,0,0.0001,-1])
fig = corner.corner(flat_samples, labels=labels);
plt.savefig('{0}/corner_plot.png'.format(outdir),bbox_inches="tight")

try:
    tau = sampler.get_autocorr_time()
    logfile.write(str(tau))
except emcee.autocorr.AutocorrError:
    logfile.write('\nnot enough data to compute autocorr')

logfile.close()
