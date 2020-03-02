import numpy as np
import scipy as sc

from numpy import linalg as linalg

import os

class ThreeWell():
    
    ''' 
    variable params (not usually fixed):

    nt: the number of time steps in the model
    tau: timescale, still not sure why necessary
    lag_time: initial lag length, before tilting effects things ?
    diff: diffusion constant
    x0, y0: initial position
    a0, a1, a2, ath: characterization of gaussian signal for bmp: a0*e^(-(t-a1)^2/a2^2)*{cos(ath), sin(ath)}
    b0, b1, b2, bth: same as "a" but for tgf-beta
    yerr: error of each data point;
        if err_mode = 0:
            same for every data point
        if err_mode = 1:
            variable for every experiment
        elif err_mode = 2:
            variable for every data point

    any of these can be fixed too


    hyper params (fixed):

    dff_scale: scale for exponentially distributed diffusion coeff
    err_scale: scale for exponentially distributed err
    tilt_scale: scale for gaussian distributed tilt magnitude (mean 0)
    pos_scale: scale for gaussian distributed initial positions (mean 0)

    what is a ThreeWell object
        it knows which params can vary
            for those params, it knows prior functions to sample them from
        it knows which params are fixed
            for those params it has exact values
        it can run a trajectory
        it can score a trajectory
        it can compute the error function (log likelihood) with respect to input data, output data, and error
        it can compute the log_prior on the variable params
        it can (therefore) compute the posterior probability 
    

    '''
    param_list = ['nt','dt','tau','diff','xpos','ypos','a0','a1','a2','ath','b0','b1','b2','bth','nper','seed']
    param2idx  = {param_list[i]:i for i in range(len(param_list))}

    def __init__(self, param_dict, param_prior_func_dict, param_prior_scale_dict, fit_angs = True):
        
        '''
            model_params       = values for all the parameters
            theta_idxs         = indexes of model_params for parameters that go into theta
            theta_prior_types  = types of prior for each theta: 0 = uniform, 1 = gaussian, 2 = exponential
            theta_prior_scales = parameters for the priors of each theta: 
                                        for uniform,     should be: [lower, upper]
                                        for gaussian,    should be: [mu, sigma]
                                        for exponential, should be: [tau]
        '''

        nparams = 15
        np.random.seed(seed)

        self.model_params       = np.zeros(nparams)
        self.theta_idxs         = []
        self.theta_prior_types  = []
        self.theta_prior_scales = []

        for k,i in param2idx.items():
            if param_dict.get(k,None):
                self.model_params[i] = param_dict[k]
            else:
                prior_func = param_prior_func_dict.get(k,0)
                prior_sc   = param_prior_scale_dict.get(k,np.array([0,1]))

                self.theta_idxs.append(i)
                self.theta_prior_types.append(prior_func)
                self.theta_prior_scales.append(prior_sc)
        
        self.ntheta = len(self.theta_idxs)
        param_inits = self.random_parameter_set()
        for i in range(self.ntheta):
            self.model_params[self.theta_idxs[i]] = param_inits[i]

        if not fit_angs:
            self.model_params[param2idx['ath']] = np.pi/2
            self.model_params[param2idx['bth']] = 7*np.pi/6

    # end initialization
    
    
    log_prior_uniform = lambda x, lower, upper: 0 if lower < x < upper else -np.inf
    log_prior_gauss   = lambda x, mu, sigsq: -(x-mu)**2/sigsq
    log_prior_exp     = lambda x, sc: -x/sc

    log_prior_funcs   = [log_prior_uniform, log_prior_gauss, log_prior_exp]
    sampling_funcs = [np.random.uniform, np.random.normal, np.random.exponential]

    def log_prior(self, theta):
        prior_tot = 0

        for i in range(self.ntheta):
            prior_tot += log_prior_funcs[self.theta_prior_types[i]](theta[i], *self.theta_prior_scales[i])

        return prior_tot

    def random_parameter_set(self):
        
        return [sampling_funcs[self.theta_prior_types[i]](*self.theta_prior_scales[i]) for i in range(self.ntheta)]

    def log_likelihood(self, theta, x, y):
        
        for i in range(self.ntheta):
            self.model_params[self.theta_idxs[i]] = theta[i]
            model  = getTrajBasinProbabilities(x) #[:,:,:2]
    
        return -0.5*np.sum((y - model) ** 2 / self.model_params[-1] )

    def log_probability(self, theta, x, y):
        
        lp = self.log_prior(theta)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_likelihood(theta, x, y)

    def nll(self, theta, x, y):
        return -self.log_likelihood(theta, x, y)
    
    def getTrajBasinProbabilities(self, x, theta = self.model_params):

        '''
        theta indexes:
        nt, dt, tau, diff: 0,1,2,3
        x,y: 4,5
        a0,a1,a2,ath: 6,7,8,9
        b0,b1,b2,bth: 10,11,12,13
        nper: 14
        seed: 15
        yerr: 16        
        '''
        
        trajBasins  = fullTrajD(np.repeat(x[0:self.nstg],           theta[14], axis=1), 
                                np.repeat(x[self.nstg:2*self.nstg], theta[14], axis=1), 
                                theta[6:10], theta[10:14], theta[4:6], 
                                theta[3], theta[0], theta[1], theta[2], theta[15], self.nstg)
        
        trajBasinsS = np.array(np.split(trajBasins, range(nper, trajBasins.shape[1], theta[14]), axis=1))
        return np.mean(trajBasinsS, axis=2)


    # model setup functions
    def f(r):
        return 2*r + np.vstack([-2*r[:,0]*r[:,1] , r[:,1]**2 - r[:,0]**2]).T

    def sigma1(f):
        nrm = np.linalg.norm(f,axis=1)
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

    def rdot2(r, noise, tau, l0, l1, v0, v1):
        return 1/tau * (sigma1(f(r) + np.outer(l0,v0) + np.outer(l1,v1)) - r) + noise

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
        
        l0s = getSigSeriesG(sts0, nt, *sigParams0[0:3])
        l1s = getSigSeriesG(sts1, nt, *sigParams1[0:3]) # nt x M
        v0  = np.array([np.cos(sigParams0[3]), np.sin(sigParams0[3])])
        v1  = np.array([np.cos(sigParams1[3]), np.sin(sigParams1[3])])

        rs      = np.zeros((sts0.shape[1], nt, r0.shape[0])) # M x nt x 2
        rs[:,0] = r0

        for t in range(0, nt-1):
            rs[:,t+1] = rs[:,t] + dt*rdot2(rs[:,t], noises[t], tau, l0s[t], l1s[t], v0, v1)
        
        tidxs = np.array(np.around(np.linspace(0,nt-1,npts+1)), dtype='int')[1:]
        return np.array([getBasins(rs[:,t]) for t in tidxs]) # should vectorize getBasins...

    def fullTrajD(sts0, sts1, sigParams0, sigParams1, r0, dff, nt, dt, tau, seed=None, npts=6):
        np.random.seed(seed)
        noises = np.sqrt(2*dff)*np.random.normal(size=(nt,sts0.shape[1],2))
        return fullTraj(sts0, sts1, sigParams0, sigParams1, r0, noises, nt, dt, tau, npts)

