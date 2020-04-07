import three_well as w3
import numpy as np
import scipy as sc
import emcee

from scipy.optimize import minimize, basinhopping, dual_annealing
from numpy import linalg as linalg

from multiprocessing import Pool
from multiprocessing import cpu_count

import os
from importlib import reload

from autograd import grad, jacobian, hessian
import autograd.numpy as np

# directories
datdir  = '/Users/simonfreedman/cqub/xenopus/data'
outdir = '/Users/simonfreedman/cqub/xenopus/plots'

stagestrs = ['9','10','10.5','11','12','13']

# Load the training data
predsS  = np.load('{0}/log_reg_pca_preds4.npy'.format(datdir))
nrep    = predsS.shape[1]

bmpOn = np.vstack([ np.ones(6) , np.zeros(6) , np.ones(6)])
tgfOn = np.vstack([ np.zeros(6), np.zeros(6) , np.ones(6)])

x = np.hstack([np.repeat(bmpOn,nrep,axis=0),np.repeat(tgfOn,nrep,axis=0)]).T
y = predsS.reshape((9,6,4))[:,:,:3]

xtest = x[:,::3]

b=2
rdotf = lambda r,tau,tilt: w3.rdot4(r,tau,tilt, b)
basinf = w3.getBasins4
#rdot = lambda r,tau,tilt: w3.rdot(r,tau,tilt)
# for rdot3, still doesn't work
nt=4000
pd = {
    'nt':nt,
    'dt':0.1,
    'tau':200,
    'nper':100,
    'lag':int(0.01*nt),
    'diff':0.02,
    'xpos':0,
    'ypos':0,
    'a0':3,
    'a3':11*np.pi/6,
    'b0':6,
    'b1':0.7*nt,
    'b2':0.4*nt,
    'b3':np.pi/2,
    'c0':9,
    'c1':0.7*nt,
    'c2':0.4*nt,
    'c3':7*np.pi/6,
    'yerr':0.01
}

myw3    = w3.ThreeWell(set_param_dict={k:pd[k] for k in ['nt','dt','lag','nper','yerr']},
                      unset_param_prior_scale_dict={'diff':[-4,-1],'a0':[0,5],'b0':[0,25],'c0':[0,50]},
                      rdot_idx = 2)

f1 = lambda x: x
f2 = lambda x: 10.**x
thTransform = [f1]*myw3.ntheta
for k in myw3.log_param_list:
    pd[k] = np.log10(pd[k])

for i in range(myw3.ntheta):
    if myw3.theta_idxs[i] in myw3.log_param_idxs:
        thTransform[i] = f2

theta00 = myw3.make_theta(pd)
params0 = myw3.get_params(theta00)
#thparams = [myw3.get_params(th) for th in ths]

nstg = 6

# this function encapsulates which parameters I'm trying to fit, and which are permanent,
# so I'd need a new version of this every time I want to make a jacobian / hessian for
# a different set of parameters

# however, as long as I'm keeping nt constant, I shouldn't need to make this for both test and train separately...
# should also be able to alias this for different inputs...
trajPosNpts  = lambda th, x, nstg: w3.fullTrajD(np.repeat(x[0:6],  pd['nper'], axis=1),
                                       np.repeat(x[6:12], pd['nper'], axis=1),
                                       th[4:6], th[6:10], th[10:14], th[2:4], th[1],
                                       pd['nt'], pd['dt'], th[0], pd['lag'], nstg,
                                 rdotf = rdotf, basinf = w3.getBasins4C)

trajFull = lambda x, pd: w3.fullTrajPosD(x[0:6], x[6:12],
                     [pd['a0'], pd['a3']],
                     [pd['b{0}'.format(i)] for i in range(4)],
                     [pd['c{0}'.format(i)] for i in range(4)],
                     np.array([pd['xpos'],pd['ypos']]), pd['diff'],
                     int(pd['nt']), pd['dt'], pd['tau'], int(pd['lag']), rdotf = rdotf)


def basinProbsTh(th, inp = x, npts = nstg):
    th2         = np.array([thTransform[i](th[i]) for i in range(len(th))])
    basins      = trajPosNpts(th2, inp, npts)
    basinsS     = np.array(np.split(basins, range(int(pd['nper']), basins.shape[1],
                                                  int(pd['nper'])),
                                    axis=1))
    return np.mean(basinsS, axis=2)

trajPosNptsTest  = lambda th: trajPosNpts(th, xtest, pd['nt'])
basinProbsThTest = lambda th: basinProbsTh(th, xtest, pd['nt'])

log_likelihood = lambda th: -0.5*np.sum((y - basinProbsTh(th)[:,:,:3]) ** 2 / pd['yerr'] )

dll = jacobian(log_likelihood)

def log_probability(th):

    lp = myw3.log_prior(th)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(th)

def nll(th):
    return -log_likelihood(th)

def nlpost(th):
    return -log_probability(th)

dnll     = jacobian(nll)
dnlpost  = jacobian(nlpost)
ddnlpost = hessian(nlpost)

time_bnds = [1e-5,pd['nt']]
ang_bnds = [0,2*np.pi]
bounds_dict = {
    'tau':[10,500],
    'diff':[1e-5,1e-1],
    'xpos':[-0.5,0.5],
    'ypos':[-0.5,0.5],
    'a0':[-10,10],
    'a3':ang_bnds,
    'b0':[-50,50],
    'b1':time_bnds,
    'b2':time_bnds,
    'b3':ang_bnds,
    'c0':[-100,100],
    'c1':time_bnds,
    'c2':time_bnds,
    'c3':ang_bnds,
    'yerr':[1e-8,1]
}

mybounds = [
          [10,500], #tau
          [1e-4,1e-1], #diff
          [-0.5,0.5], [-0.5,0.5], #xpos, ypos
          [-10,10], #a0
          ang_bnds, #a3
          [-50, 50], #b0
          time_bnds, #b1
          time_bnds, #b2
          ang_bnds, #b3
          [-100, 100], #c0
          time_bnds, #c1
          time_bnds, #c2
          ang_bnds #b3
          #[1e-8,1], #yerr
         ]

boundsarr = np.array(mybounds)
eps = 1e-8
for i in range(myw3.ntheta):
    if myw3.theta_prior_types[i] == 0:
        boundsarr[i] = myw3.theta_prior_scales[i]+np.array([[eps,-eps]])

print('fitting')
fit_soln_all = minimize(nlpost, x0=theta00, jac = dnlpost, hess = ddnlpost, 
                         bounds  = boundsarr, method='trust-constr')
np.save('{0}/tc_fit_x.npy'.format(datdir),fit_soln_all.x)
np.save('{0}/tc_fit.npy'.format(datdir),  fit_soln_all)

print('done')

