import numpy as np

outdir = '/projects/p30129/simonf/out/xenopus/data/siggia_mcmc'
rdot_guess = {
    'nt':100,
    'dt':1,
    'tau':200,
    'nper':100,
    'lag':16,
    'diff':0.0001,
    'xpos':0,
    'ypos':0,
    'a0':1,
    'a3':11*np.pi/6,
    'b0':20,
    'b1':50,
    'b2':20,
    'b3':np.pi/2,
    'c0':30,
    'c1':50,
    'c2':20,
    'c3':7*np.pi/6,
    'yerr':0.0005
}
np.save('{0}/rdot_guess.npy'.format(outdir), rdot_guess) 

# guess for rdot3
nt=2000
rdot3guess = {
    'nt':nt,
    'dt':0.1,
    'tau':50,
    'nper':100,
    'lag':int(0.16*nt),
    'diff':0.01,
    'xpos':0,
    'ypos':0,
    'a0':0.75,
    'a3':11*np.pi/6,
    'b0':3,
    'b1':0.6*nt,
    'b2':0.2*nt,
    'b3':np.pi/2,
    'c0':6,
    'c1':0.55*nt,
    'c2':0.15*nt,
    'c3':7*np.pi/6,
    'yerr':0.01
}
np.save('{0}/rdot3_guess.npy'.format(outdir), rdot3guess) 

# guess for rdot4
b=2
#rdot = lambda r,tau,tilt: w3.rdot4(r,tau,tilt, b)
#rdot = lambda r,tau,tilt: w3.rdot(r,tau,tilt)
# for rdot3, still doesn't work 
nt=4000
paramDictG = {
    'nt':nt,
    'dt':0.1,
    'tau':200,
    'nper':100,
    'lag':int(0.4*nt),
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
np.save('{0}/rdot4_guess.npy'.format(outdir), rdot3guess) 
