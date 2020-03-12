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
    a0, a1, a2, a3: characterization of gaussian signal for bmp: a0*e^(-(t-a1)^2/a2^2)*{cos(a3), sin(a3)}
    b0, b1, b2, b3: same as "a" but for tgf-beta
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
    def __init__(self, set_param_dict = {}, unset_param_prior_type_dict = {}, unset_param_prior_scale_dict = {}, 
            default_value_params = [], seed = None):
        
        '''
            model_params       = values for all the parameters
            theta_idxs         = indexes of model_params for parameters that go into theta
            theta_prior_types  = types of prior for each theta: 0 = uniform, 1 = gaussian, 2 = exponential
            theta_prior_scales = parameters for the priors of each theta:
                                     type: distribution, params
                                        0: uniform,     [lower, upper]
                                        1: gaussian,    [mu, sigma]
                                        2: exponential, [tau]
                                        3: integer,     [lower,upper]
        '''
        
        #param_list = ['nt','dt','tau','diff','xpos','ypos','a0','a1','a2','a3','b0','b1','b2','b3','nper']
        self.param_default_info = { 
                # name:   [ index   , value        , prior_type, prior_params ] 
                'nt'    : [ 0       , 100          , 3         , [10,1000]  ],
                'dt'    : [ 1       , 1            , 2         , [1]        ],
                'tau'   : [ 2       , 10           , 2         , [10]       ],
                'diff'  : [ 3       , 0.05         , 2         , [0.05]     ],
                'xpos'  : [ 4       , 0            , 1         , [0,1]      ],
                'ypos'  : [ 5       , 0            , 1         , [0,1]      ],
                'a0'    : [ 6       , 1            , 1         , [0,1]      ],
                'a3'    : [ 7       , 11*np.pi/6   , 0         , [0,2*np.pi]],
                'b0'    : [ 8       , 1            , 1         , [0,1]      ],
                'b1'    : [ 9       , 50           , 0         , [0,100]    ],
                'b2'    : [ 10      , 10           , 0         , [0,100]    ],
                'b3'    : [ 11      , np.pi/2      , 0         , [0,2*np.pi]],
                'c0'    : [ 12      , 1            , 1         , [0,1]      ],
                'c1'    : [ 13      , 50           , 0         , [0,100]    ],
                'c2'    : [ 14      , 10           , 0         , [0,100]    ],
                'c3'    : [ 15      , 7*np.pi/6    , 0         , [0,2*np.pi]],
                'nper'  : [ 16      , 100          , 3         , [10,200]   ],
                'yerr'  : [ 17      , 0.0005       , 2         , [0.0005]   ],
                'lag'   : [ 18      , 0            , 3         , [0,20]     ]
                } 
        

        nparams = len(self.param_default_info)
        np.random.seed(seed)

        self.model_params       = np.zeros(nparams) + np.NaN
        self.theta_idxs         = []
        self.theta_prior_types  = []
        self.theta_prior_scales = []
        theta_idx_dict          = {}
        for k,v in self.param_default_info.items():
            if k in set_param_dict:
                # param is fixed by user
                self.model_params[v[0]] = set_param_dict[k]
            elif k in default_value_params:
                # param is fixed to default
                self.model_params[v[0]] = v[1]
            else:
                # param is unfixed
                self.theta_idxs.append(v[0])

                prior_type  = unset_param_prior_type_dict.get(  k, v[2])
                prior_scale = unset_param_prior_scale_dict.get( k, v[3])

                self.theta_prior_types.append( prior_type )
                self.theta_prior_scales.append(prior_scale)
        
        self.ntheta = len(self.theta_idxs)

        # some logic specific to uniformly distributed parameters with units of time
        # unless otherwise specified, these priors should be *defaulted* to span the trajectory timescale
        time_params =['b1','b2','c1','c2']
        if 'nt' in set_param_dict or 'nt' in default_value_params:
            nt = self.model_params[self.param_default_info['nt'][0]]
            for param in time_params:
                idxs = np.where(np.array(self.theta_idxs)==self.param_default_info[param][0])[0]
                if len(idxs) > 0:
                    idx = idxs[0]
                    if param not in unset_param_prior_scale_dict and self.theta_prior_types[idx] in [0,3]:
                        self.theta_prior_scales[idx][1] = nt 


        # param_inits = self.random_parameter_set()
        # for i in range(self.ntheta):
        #     self.model_params[self.theta_idxs[i]] = param_inits[i]

    def set_seed(self, seed):
        np.random.seed(seed)
    
    # end initialization
   
    ##################################################################################################   
    ############### Functions / lists for sampling parameters ########################################
    log_prior_uniform = lambda x, lower, upper: 0 if lower < x < upper else -np.inf
    log_prior_gauss   = lambda x, mu, sig: -(x-mu)**2/sig**2
    log_prior_exp     = lambda x, sc: -x/sc if x > 0 else -np.inf


    prior_func_names  = ['uniform'        , 'gaussian'      , 'exponential'        , 'integer'        ]
    log_prior_funcs   = [log_prior_uniform, log_prior_gauss , log_prior_exp        , log_prior_uniform]
    sampling_funcs    = [np.random.uniform, np.random.normal, np.random.exponential, np.random.randint]
    ###################################################################################################    

    def get_fixed_params(self):
        fixed_params = {}
        for k, v in self.param_default_info.items():
            val = self.model_params[v[0]]
            if not np.isnan(val):
                fixed_params[k]=val
        return fixed_params
    
    def get_theta_labels(self):
        theta_labels = []
        for k,v in self.param_default_info.items():
            if v[0] in self.theta_idxs:
                theta_labels.append(k)
        return theta_labels
    
    def get_prior_info(self):
        theta_info = {}
        thidxs = np.array(self.theta_idxs)
        for k,v in self.param_default_info.items():
            z = np.where(thidxs==v[0])[0]
            if len(z) > 0:
                theta_info[k] = self.prior_func_names[self.theta_prior_types[z[0]]],self.theta_prior_scales[z[0]]
        return theta_info
    
    def log_prior(self, theta):
        prior_tot = 0

        for i in range(self.ntheta):
            prior_tot += self.log_prior_funcs[self.theta_prior_types[i]](theta[i], *self.theta_prior_scales[i])

        return prior_tot

    def random_parameter_set(self):
        
        return [self.sampling_funcs[self.theta_prior_types[i]](*self.theta_prior_scales[i]) for i in range(self.ntheta)]

    def get_params(self, theta): 
        params = self.model_params
        
        for i in range(self.ntheta):
            params[self.theta_idxs[i]] = theta[i]

        return params

    def log_likelihood(self, theta, x, y):
        
        params = self.get_params(theta)
        model  = getTrajBasinProbabilities(x, params, y.shape[1])[:,:,:2]
    
        return -0.5*np.sum((y - model) ** 2 / params[15] )

    def log_probability(self, theta, x, y):
        
        lp = self.log_prior(theta)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_likelihood(theta, x, y)

    def nll(self, theta, x, y):
        return -self.log_likelihood(theta, x, y)

sqrt3over3 = np.sqrt(3)/3
# model running functions
def getTrajBasinProbabilities(x, params, nstg):

    '''
    param indexes:
    nt, dt, tau, diff: 0,1,2,3
    x,y: 4,5
    a0,a3: 6,7
    b0,b1,b2,b3: 8,9,10,11
    c0,c1,c2,c3: 12,13,14,15
    nper: 16
    yerr: 17        
    lag : 18
    '''
    
    trajBasins  = fullTrajD(np.repeat(x[0:6],   int(params[16]), axis=1), 
                            np.repeat(x[6:12],  int(params[16]), axis=1), 
                            params[6:8], params[8:12], params[12:16], params[4:6], 
                            params[3], int(params[0]), params[1], params[2], int(params[18]), nstg)
    
    trajBasinsS = np.array(np.split(trajBasins, range(int(params[16]), trajBasins.shape[1], int(params[16])), axis=1))
    return np.mean(trajBasinsS, axis=2)


def f(r):
    return 2*r + np.vstack([-2*r[:,0]*r[:,1] , r[:,1]**2 - r[:,0]**2]).T

def sigma1(f):
    nrm = np.linalg.norm(f,axis=1)
    return (np.tanh(nrm)*np.divide(f.T, nrm, out=np.zeros_like(f.T), where=nrm!=0)).T

def getBasins(rs):
    basins = np.zeros((rs.shape[0],3))

    inb0   = rs[:,1] > sqrt3over3*np.abs(rs[:,0])
    inb1   = ~inb0 & (rs[:,0]>0)
    inb2   = ~(inb0 | inb1)

    basins[inb0,0] = 1
    basins[inb1,1] = 1
    basins[inb2,2] = 1
    return basins

def rdot(r, tau, l0, l1, l2, v0, v1, v2):
    return 1/tau * (sigma1(f(r) + np.outer(l0,v0) + np.outer(l1,v1) + np.outer(l2,v2)) - r)

def rdot0(r, tau):
    return 1/tau * (sigma1(f(r)) - r)

def getSigSeriesG(sts, nt, a, mu, sig):

    # sts has shape T x M
    # the rest are numbers
    gaus        = a*np.exp(-(np.arange(nt)-mu)**2/sig**2)
    nper        = np.int(nt/sts.shape[0])
    stsRepeated = np.vstack([np.repeat(sts,nper,axis=0),np.zeros((nt-nper*6,sts.shape[1]))])
    return (stsRepeated.T*gaus).T

def fullTrajPos(sts1, sts2, m0, m1, m2, r0, noises, nt, dt, tau, lag, npts=6):
    
    # sts = on/off-ness of bmp at each of the T stages -- should be T x M -- currently T = 6
    # sigParams = parameters for getSigSeries function
    # r0 = initial position on fate landscape 1x2
    # noises = noise at each timestep for each data point --> nt x M x 2
    # nt = number of timesteps (integer)
    # dt = length of timesteps (float)
    # tau = timescale (float)
    
    l0s = np.zeros((nt, sts1.shape[1])) + m0[0]
    l1s = getSigSeriesG(sts1, nt, *m1[0:3]) # nt x M
    l2s = getSigSeriesG(sts2, nt, *m2[0:3]) # nt x M
    v0  = np.array([np.cos(m0[1]), np.sin(m0[1])])
    v1  = np.array([np.cos(m1[3]), np.sin(m1[3])])
    v2  = np.array([np.cos(m2[3]), np.sin(m2[3])])

    # should be able to evaluate m(t) = l0v0+l1v1+l2v2 and feed that into rdot to save some computation time...

    rs      = np.zeros((sts1.shape[1], nt, r0.shape[0])) # M x nt x 2
    rs[:,0] = r0

    for t in range(0, lag):
        rs[:,t+1] = rs[:,t] + dt*noises[t]

    for t in range(lag, nt-1):
        rs[:,t+1] = rs[:,t] + dt*(rdot(rs[:,t], tau, l0s[t], l1s[t], l2s[t], v0, v1, v2) + noises[t])

    return rs

def fullTraj(sts1, sts2, m0, m1, m2, r0, noises, nt, dt, tau, lag, npts=6):
    
    rs      = fullTrajPos(sts1, sts2, m0, m1, m2, r0, noises, nt, dt, tau, lag, npts) 
    tidxs   = np.array(np.around(np.linspace(0,nt-1,npts+1)), dtype='int')[1:]

    return np.array([getBasins(rs[:,t]) for t in tidxs]) # should vectorize getBasins...

def fullTrajD(sts1, sts2, m0, m1, m2, r0, dff, nt, dt, tau, lag, npts=6):
    noises = np.sqrt(2*dff)*np.random.normal(size=(nt,sts1.shape[1],2))
    return fullTraj(sts1, sts2, m0, m1, m2, r0, noises, nt, dt, tau, lag, npts)

def fullTrajPosD(sts1, sts2, m0, m1, m2, r0, dff, nt, dt, tau, lag):
    noises = np.sqrt(2*dff)*np.random.normal(size=(nt,sts1.shape[1],2))
    return fullTrajPos(sts1, sts2, m0, m1, m2, r0, noises, nt, dt, tau, lag)
