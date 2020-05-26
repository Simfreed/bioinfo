#from numba import jit
#import numpy as np
import autograd.numpy as np
import scipy as sc

from autograd.numpy import linalg as linalg
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
            default_value_params = [], log_param_list = ['tau', 'diff', 'b1', 'b2', 'c1', 'c2'], 
            mode = '2d3w_S_h', rdot_depth = None, seed = None):
        
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
            mode = NdMwK or NdMw_sK where 
                    N is dimensionality of landscape, 
                    M is number of wells
                    K is "h" for hard (Heavyside) boundaries and "s" for soft (sigmoid) boundaries
                    the addition of "_s" after the w implies using the Siggia model (only implemented for 2d3w)
                    e.g., 1d2wh: 1D - 2 well,  hard boundaries (Heavyside)
        '''
        
        # need to set: rdotf, basinf, and basin_probsf based on "mode" variable
        self.rdotf          = eval('rdot_{0}'.format(mode[0:-2]))
        self.basinf         = eval('basins_{0}'.format(mode.replace('_S','')))
        self.basin_probsf   = eval('basin_probs_{0}'.format(mode[0:2])) 
        
        # so theoretically i'd want to be able to set a parameter for the well depth,
        # but that'd require using a lambda function? and that's not working so well lately
        # e.g., something like
        # if rdot_depth:
        #   self.rdotf = lambda r: self.rdotf(r, b=rdot_depth)

        np.random.seed(seed)
        
        #param_list = ['nt','dt','tau','diff','xpos','ypos','a0','a1','a2','a3','b0','b1','b2','b3','nper']

        self.param_default_info = { 
                # name:   [ index   , value        , prior_type, prior_params ] 
                'nt'    : [ 0       , 100          , 3         , [10,1000]  ],
                'dt'    : [ 1       , 1            , 2         , [1]        ],
                'tau'   : [ 2       , 50           , 0         , [10,200]   ],
                'diff'  : [ 3       , 0.001        , 2         , [0.001]    ],
                'xpos'  : [ 4       , 0            , 1         , [0,1]      ],
                'ypos'  : [ 5       , 0            , 1         , [0,1]      ],
                'a0'    : [ 6       , 1            , 1         , [0,2]      ],
                'a3'    : [ 7       , 11*np.pi/6   , 0         , [0,2*np.pi]],
                'b0'    : [ 8       , 1            , 1         , [0,100]    ],
                'b1'    : [ 9       , 50           , 0         , [0,100]    ],
                'b2'    : [ 10      , 10           , 0         , [0,100]    ],
                'b3'    : [ 11      , np.pi/2      , 0         , [0,2*np.pi]],
                'c0'    : [ 12      , 1            , 1         , [0,100]    ],
                'c1'    : [ 13      , 50           , 0         , [0,100]    ],
                'c2'    : [ 14      , 10           , 0         , [0,100]    ],
                'c3'    : [ 15      , 7*np.pi/6    , 0         , [0,2*np.pi]],
                'nper'  : [ 16      , 100          , 3         , [10,200]   ],
                'yerr'  : [ 17      , 0.0005       , 2         , [0.0005]   ],
                'lag'   : [ 18      , 0            , 3         , [0,20]     ]
                }

        self.log_param_default_info = {
                'tau'    : [ 2      , np.log10(50) , 0          , [0,4]    ],
                'diff'   : [ 3      , -3           , 0          , [-5,-1]    ],
                'b1'     : [ 9      , np.log10(50) , 0          , [0,4]    ],
                'b2'     : [ 10     , 1            , 0          , [0,4]    ],
                'c1'     : [ 13     , np.log10(50) , 0          , [0,4]    ],
                'c2'     : [ 14     , 1            , 0          , [0,4]    ]
                }
      
        self.log_param_list = log_param_list
        self.log_param_idxs = []
        for k in log_param_list:
            self.param_default_info[k] = self.log_param_default_info[k]
            self.log_param_idxs.append(self.param_default_info[k][0])

        nparams = len(self.param_default_info)

        self.model_params       = np.zeros(nparams) + np.NaN
        self.theta_idxs         = []
        self.theta_prior_types  = []
        self.theta_prior_scales = []
        
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
        # unless otherwise specified, these priors should be *defaulted* to be on order of magnitude of the trajectory timescale
        time_params =['tau','b1','b2','c1','c2']
        if 'nt' in set_param_dict or 'nt' in default_value_params:
            nt = self.model_params[self.param_default_info['nt'][0]]
            for param in time_params:
                idxs = np.where(np.array(self.theta_idxs)==self.param_default_info[param][0])[0]
                if len(idxs) > 0:
                    idx = idxs[0]
                    if param not in unset_param_prior_scale_dict and self.theta_prior_types[idx] in [0,3]:
                        if param in self.log_param_list:
                            self.theta_prior_scales[idx][1] = np.log10(5*nt)
                        else:
                            self.theta_prior_scales[idx][1] = 5*nt 


        # param_inits = self.random_parameter_set()
        # for i in range(self.ntheta):
        #     self.model_params[self.theta_idxs[i]] = param_inits[i]
        
        #def my_basin_probs(x, params, nstg):
        #    return probsf(x, params, nstg, log_param_idxs, self.rdotf, self.basinf)
        
        #self.basin_probsf = my_basin_probs

    def set_seed(self, seed):
        np.random.seed(seed)
    
    # end initialization
   
    ##################################################################################################   
    ############### Functions / lists for sampling parameters ########################################
    log_prior_uniform = lambda x, lower, upper: 0 if lower < x < upper else -np.inf
    log_prior_gauss   = lambda x, mu, sig: -(x-mu)**2/sig**2
    log_prior_exp     = lambda x, sc: -x/sc if x > 0 else -np.inf

#    @jit(nopython=True)
#    def log_prior_uniform(x, lower, upper): 
#        return 0 if lower < x < upper else -np.inf
#    @jit(nopython=True)
#    def log_prior_gauss(x, mu, sig): 
#        return -(x-mu)**2/sig**2
#    @jit(nopython=True)
#    def log_prior_exp(x, sc): 
#        return -x/sc if x > 0 else -np.inf

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
    
    # @jit(nopython=True)
    def log_prior(self, theta):
        prior_tot = 0

        for i in range(self.ntheta):
            #print(theta[i])
            prior_tot += self.log_prior_funcs[self.theta_prior_types[i]](theta[i], *self.theta_prior_scales[i])

        return prior_tot

    def random_parameter_set(self):
        
        return [self.sampling_funcs[self.theta_prior_types[i]](*self.theta_prior_scales[i]) for i in range(self.ntheta)]
    
    def resample_pos(self, pos):
        
        # checks an input position for prior satisfaction
        # if not satisfied, resamples offending parameters
        newpos = pos
        for i in range(self.ntheta):
            pscale = self.theta_prior_scales[i]
            ptype  = self.theta_prior_types[i]
            if ((ptype == 0 or ptype == 3)  and (pos[i] > pscale[1] or pos[i] < pscale[0])) or (ptype == 2 and pos[i] < 0):
                newpos[i] = self.sampling_funcs[ptype](*pscale)
        return newpos

    def make_theta(self, params):
        
        thidxs = np.array(self.theta_idxs)
        th     = np.zeros(self.ntheta)
        
        for k,v in params.items():
            z = np.where(thidxs==self.param_default_info[k][0])[0]
            if len(z) > 0:
                th[z[0]] = v
        return th
       
    # @jit(nopython=True)
    def get_params(self, theta): 
        params = np.copy(self.model_params)
        
        for i in range(self.ntheta):
            params[self.theta_idxs[i]] = theta[i]

        return params
    
    # @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
    def log_likelihood(self, theta, x, y):
        
        params       = self.get_params(theta)
        basin_probs  = self.basin_probsf(x, params, y.shape[2], self.log_param_idxs, self.rdotf, self.basinf)[:,:,:-1]
        errs         = np.array([[y[i,j]-basin_probs[i] for i in range(y.shape[0])] for j in range(y.shape[1])])
    
        return -0.5*np.sum(errs ** 2) / params[17]

    # @jit(nopython=True)
    def log_probability(self, theta, x, y):
        
        lp = self.log_prior(theta)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_likelihood(theta, x, y)

    def nll(self, theta, x, y):
        return -self.log_likelihood(theta, x, y)

sqrt3over3 = np.sqrt(3)/3
# model running functions

# @jit(nopython=True)
def f(r):
    return 2*r + np.vstack([-2*r[:,0]*r[:,1] , r[:,1]**2 - r[:,0]**2]).T

# @jit(nopython=True)
def sigma1(f):
    nrm = np.linalg.norm(f,axis=1)
    return (np.tanh(nrm)*f.T/nrm).T
    #return (np.tanh(nrm)*np.divide(f.T, nrm, out=np.zeros_like(f.T), where=nrm!=0)).T

# @jit(nopython=True)
def basins_2d3w_h(rs):
    # for use with rdot and rdot_2d3w
    basins = np.zeros(list(rs.shape[0:-1])+[3])

    inb0   = rs[...,1] > sqrt3over3*np.abs(rs[...,0])
    inb1   = ~inb0 & (rs[...,0]>0)
    inb2   = ~(inb0 | inb1)

    basins[inb0,0] = 1
    basins[inb1,1] = 1
    basins[inb2,2] = 1
    return basins

# @jit(nopython=True)
def basins_2d4w_h(rs, b=2):
    # for use with rdot_2d4w
    basins = np.zeros(list(rs.shape[0:-1])+[4])

    inb3   = (rs[...,0]*rs[...,0] + rs[...,1]*rs[...,1]) < (b-np.sqrt(b*b-3)) / 3.
    inb0   = ~inb3 & (rs[...,1] > sqrt3over3*np.abs(rs[...,0]))
    inb1   = ~(inb3 | inb0) & (rs[...,0]>0)
    inb2   = ~(inb3 | inb0 | inb1)

    basins[inb0,0] = 1
    basins[inb1,1] = 1
    basins[inb2,2] = 1
    basins[inb3,3] = 1
    return basins

# @jit(nopython=True)
def basins_1d2w_h(rs):
    # for use with rdot_1d2w
    basins = np.zeros(list(rs.shape)+[2])

    inb0   = rs > 0
    inb1   = ~inb0

    basins[inb0,0] = 1
    basins[inb1,1] = 1
    
    return basins

def basins_1d2w_s(rs):
    # for use with rdot_1d2w

    right_basin_p  = sigmoid2(rs) 
    return np.stack([right_basin_p, 1 - right_basin_p], axis=2)

sigmoid2 = lambda x: 1/(1+np.exp(-100*x))
def basins_2d3w_s(rs):
    
    upperBasinP    = sigmoid2(rs[...,1] - sqrt3over3*np.abs(rs[...,0]))
    lowerBasinP    = 1 - upperBasinP
    posXp          = sigmoid2(rs[...,0])
    return np.stack([upperBasinP, lowerBasinP*posXp, lowerBasinP*(1-posXp)], axis=2)

def basins_2d4w_s(rs, b=2):
    
    midBasinP     = sigmoid2((b-np.sqrt(b*b-3)) / 3. - (rs[...,0]*rs[...,0] + rs[...,1]*rs[...,1])) 
    notMidBasinP  = 1 - midBasinP

    upperBasinP    = sigmoid2(rs[...,1] - sqrt3over3*np.abs(rs[...,0]))
    lowerBasinP    = 1 - upperBasinP
    
    posXp          = sigmoid2(rs[...,0])
    
    return np.stack([upperBasinP*notMidBasinP, lowerBasinP*posXp*notMidBasinP, lowerBasinP*(1-posXp)*notMidBasinP, midBasinP], axis=2)

# @jit(nopython=True)
def rdot_2d3w_S(r, tau, tilt):
    return (sigma1(f(r) + tilt) - r) / tau

# gradient of U(r) = -r^4cos(3(phi-pi/2))+b*r^6, 
# with b=2/3, has mimumums at ((0,1), (+/-sqrt(3)/2,-1/2))
# @jit(nopython=True)
def rdot_2d3w(r, tau, tilt, b = 2/3):
    x = r[:,0]
    y = r[:,1]

    
    #rmaginv  = 1. / np.sqrt(x*x+y*y)
    #gradx = 4*x**5                 + x*y**3*( 4*y + 5*rmaginv ) + y*x**3*(  8*y + 9*rmaginv )
    #grady = 4*y**4*( y - rmaginv ) + x**4*(   4*y + 3*rmaginv ) + x*x*y*y*( 8*y + 3*rmaginv )
    
    xsq       = x*x
    ysq       = y*y
    rmagsq    = xsq + ysq
    rmaginv   = 1. / np.sqrt(rmagsq)
    threeXsq  = 3*xsq
    sixBrmag4 = 6*b*rmagsq*rmagsq

    gradx = sixBrmag4*x + (3*threeXsq*x*y + ysq*5*x*y              ) * rmaginv
    grady = sixBrmag4*y + (  threeXsq*xsq + ysq*(threeXsq - 4*ysq) ) * rmaginv
    
    return (-np.array([gradx, grady]).T + tilt) / tau

# gradient of U(r) = r^2-b*r^4cos(3(phi-pi/2))+r^6, 
# with b = 2, has mimumums at ((0,1), (+/-sqrt(3)/2,-1/2))
# @jit(nopython=True)
def rdot_2d4w(r, tau, tilt, b=2):
    
    x = r[:,0]
    y = r[:,1]
    
    xsq      = x*x
    ysq      = y*y
    rmagsq   = xsq + ysq
    bRmaginv = b / np.sqrt(x*x+y*y)
    threeXsq = 3*xsq
    term     = 2+6*rmagsq*rmagsq

    gradx  = x*y*bRmaginv*( 3*threeXsq +        5*ysq             ) + x*term
    grady  = bRmaginv*(   xsq*threeXsq + threeXsq*ysq - 4*ysq*ysq ) + y*term

    #gradx = 2*x*( 1 + 3*x**4   + 6*x*x*y*y   + 3*y**4 - 4*y**3/rmag + 9*y*rmag )
    #grady = 2*(   y + 3*y*x**4 + 6*x*x*y*y*y + 3*y**5 - 4*y*y*rmag  + (3*x**4 + 7*x*x*y*y)/rmag)
    
    return (-np.array([gradx, grady]).T + tilt) / tau

def rdot_1d2w(x, tau, tilt):
    
    return (-2*x*x*x + x/2 + tilt) / tau

# @jit(nopython=True)
def getSigSeriesG(sts, nt, a, mu, sig):

    # sts has shape T x M
    # the rest are numbers
    gaus        = a*np.exp(-(np.arange(nt)-mu)**2/sig**2)
    nper        = np.int(nt/sts.shape[0])
    stsRepeated = np.vstack([np.repeat(sts,nper,axis=0),np.zeros((nt-nper*6,sts.shape[1]))])
    return (stsRepeated.T*gaus).T

# v.shape = 1x2
# l.shape = nt x M

#getTilt = lambda l,v: np.array([l]).transpose((1,2,0)).dot(v)
getTilt = lambda l,v: np.matmul(np.array([l]).transpose((1,2,0)),v)

def pos_traj(sts1, sts2, m0, m1, m2, r0, noises, nt, dt, tau, lag, rdotf = rdot_2d3w_S):
    
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
    v0  = np.array([[np.cos(m0[1]), np.sin(m0[1])]])
    v1  = np.array([[np.cos(m1[3]), np.sin(m1[3])]])
    v2  = np.array([[np.cos(m2[3]), np.sin(m2[3])]])
    
    tilt = getTilt(l0s, v0) + getTilt(l1s, v1) + getTilt(l2s, v2)
    # should be able to evaluate m(t) = l0v0+l1v1+l2v2 and feed that into rdot to save some computation time...

    #rs      = np.zeros((sts1.shape[1], nt, r0.shape[0])) # M x nt x 2
    #rs[:,0] = r0
    #rs      = np.hstack([r0*np.ones((sts1.shape[1],1,r0.shape[0])),  np.zeros((sts1.shape[1], nt-1, r0.shape[0]))])
    rs      = [r0*np.ones((sts1.shape[1],r0.shape[0]))]
    for t in range(0, lag):
        #rs[:,t+1] = rs[:,t] + dt*noises[t]
        rs.append(rs[t] + dt*noises[t])

    for t in range(lag, nt-1):
        #rs[:,t+1] = rs[:,t] + dt*(rdotf(rs[:,t], tau, tilt[t]) + noises[t])
        rs.append(rs[t] + dt*(rdotf(rs[t], tau, tilt[t]) + noises[t]))

    return np.stack(rs,axis=1)

# @jit(nopython=True)
def basin_traj(sts1, sts2, m0, m1, m2, r0, noises, nt, dt, tau, lag, npts = 6, rdotf = rdot_2d3w_S, basinf = basins_2d3w_h):
    
    rs      = pos_traj(sts1, sts2, m0, m1, m2, r0, noises, nt, dt, tau, lag, rdotf) 
    tidxs   = np.array(np.around(np.linspace(0,nt-1,npts+1)), dtype='int')[1:]

    return basinf(rs[:,tidxs]).transpose((1,0,2))

# @jit(nopython=True)
def basin_traj_diff(sts1, sts2, m0, m1, m2, r0, dff, nt, dt, tau, lag, npts=6, rdotf = rdot_2d3w_S, basinf = basins_2d3w_h):
    noises = np.sqrt(2*dff)*np.random.normal(size=(nt,sts1.shape[1],2))
    return basin_traj(sts1, sts2, m0, m1, m2, r0, noises, nt, dt, tau, lag, npts, rdotf, basinf)

def pos_traj_diff(sts1, sts2, m0, m1, m2, r0, dff, nt, dt, tau, lag, rdotf = rdot_2d3w_S):
    noises = np.sqrt(2*dff)*np.random.normal(size=(nt,sts1.shape[1],2))
    return pos_traj(sts1, sts2, m0, m1, m2, r0, noises, nt, dt, tau, lag, rdotf)


# @jit(nopython=True)
def basin_probs_2d(x, params, nstg, log_param_idxs = [], rdotf = rdot_2d3w_S, basinf = basins_2d3w_h, ncond = 3):

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
    
    for i in log_param_idxs:
        params[i] = 10.**params[i]

    trajBasins  = basin_traj_diff(np.repeat(x[0:6],   int(params[16]), axis=1), 
                            np.repeat(x[6:12],  int(params[16]), axis=1), 
                            params[6:8], params[8:12], params[12:16], params[4:6], 
                            params[3], int(params[0]), params[1], params[2], int(params[18]), nstg, rdotf, basinf)

    return np.mean(np.array(np.split(trajBasins, ncond)), axis=1)
    
    #trajBasinsS = np.array(np.split(trajBasins, range(int(params[16]), trajBasins.shape[1], int(params[16])), axis=1))
    #return np.mean(trajBasinsS, axis=2)

def basin_probs_1d(x, params, nstg, log_param_idxs = [], rdotf = rdot_1d2w, basinf = basins_1d2w_h, ncond = 2):

    '''
    param indexes:
    nt, dt, tau, diff: 0,1,2,3
    x0: 4
    a0: 6 
    b0,b1,b2: 8,9,10
    nper: 16 
    yerr: 17        
    lag : 18
    '''
    
    for i in log_param_idxs:
        params[i] = 10.**params[i]

    trajBasins  = basin_traj_diff_1d(np.repeat(x[0:6],  int(params[16]), axis=1), 
            params[6], params[8:11], params[4], 
            params[3], int(params[0]), params[1], params[2], int(params[18]), nstg, rdotf, basinf)
    
    return np.mean(np.array(np.split(trajBasins, ncond)), axis=1)
    
    #trajBasinsS = np.array(np.split(trajBasins, range(int(params[16]), trajBasins.shape[1], int(params[16])), axis=1))
    #return np.mean(trajBasinsS, axis=2)

def pos_traj_1d(sts, m0, m1, x0, noises, nt, dt, tau, lag, rdotf = rdot_1d2w):
    
    # sts = on/off-ness of bmp at each of the T stages -- should be T x M -- currently T = 6
    # m0 = tilt toward development
    # m1 = array of gaussian params for tilt toward BMP
    # x0 = initial position on fate landscape 1x1
    # noises = noise at each timestep for each data point --> nt x M
    # nt = number of timesteps (integer)
    # dt = length of timesteps (float)
    # tau = timescale (float)
    
    l0s = np.zeros((nt, sts.shape[1])) + m0         # if positive, pushes x negative toward neural
    l1s = getSigSeriesG(sts, nt, *m1[0:3]) # nt x M # if positive, pushes x positive toward epidermal
    
    tilt = l1s - l0s

    xs      = [x0*np.ones((sts.shape[1]))]
    for t in range(0, lag):
        xs.append(xs[t] + dt*noises[t])

    for t in range(lag, nt-1):
        xs.append(xs[t] + dt*(rdotf(xs[t], tau, tilt[t]) + noises[t]))

    return np.stack(xs,axis=1)

def basin_traj_1d(sts, m0, m1, x0, noises, nt, dt, tau, lag, npts = 6, rdotf = rdot_1d2w, basinf = basins_1d2w_h):
    
    rs      = pos_traj_1d(sts, m0, m1, x0, noises, nt, dt, tau, lag, rdotf) 
    tidxs   = np.array(np.around(np.linspace(0,nt-1,npts+1)), dtype='int')[1:]
#    return basinf(rs[:,tidxs]).transpose((1,0,2))
    return basinf(rs[:,tidxs])

# @jit(nopython=True)
def basin_traj_diff_1d(sts, m0, m1, x0, dff, nt, dt, tau, lag, npts=6, rdotf = rdot_1d2w, basinf = basins_1d2w_h):
    
    noises = np.sqrt(dff)*np.random.normal(size=(nt,sts.shape[1]))
    
    return basin_traj_1d(sts, m0, m1, x0, noises, nt, dt, tau, lag, npts, rdotf, basinf)

def pos_traj_diff_1d(sts, m0, m1, x0, dff, nt, dt, tau, lag, rdotf = rdot_1d2w):

    noises = np.sqrt(dff)*np.random.normal(size=(nt,sts.shape[1]))
    
    return pos_traj_1d(sts, m0, m1, x0, noises, nt, dt, tau, lag, rdotf)
