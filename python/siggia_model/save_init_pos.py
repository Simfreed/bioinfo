import numpy as np

outdir = '/projects/p31095/simonf/out/xenopus/data/siggia_mcmc'
guess = {
    'nt':100,
    'dt':1,
    'tau':200,
    'nper':100,
    'lag':16,
    'diff':0.0001001,
    'x0':0,
    'y0':0,
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
    'yerr':0.01
}

log_param_list = ['tau','diff','b1','b2','c1','c2']
for k in log_param_list:
    guess[k] = np.log10(guess[k])

np.save('{0}/rdot_guess_log.npy'.format(outdir), guess) 

# guess for rdot3
nt=2000
guess = {
    'nt':nt,
    'dt':0.1,
    'tau':50,
    'nper':100,
    'lag':int(0.16*nt),
    'diff':0.01,
    'x0':0,
    'y0':0,
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
for k in log_param_list:
    guess[k] = np.log10(guess[k])
np.save('{0}/rdot3_guess_log.npy'.format(outdir), guess) 

# guess for rdot4
b=2
#rdot = lambda r,tau,tilt: w3.rdot4(r,tau,tilt, b)
#rdot = lambda r,tau,tilt: w3.rdot(r,tau,tilt)
# for rdot3, still doesn't work 
nt=4000
guess = {
    'nt':nt,
    'dt':0.1,
    'tau':200,
    'nper':100,
    'lag':int(0.01*nt),
    'diff':0.02,
    'x0':0,
    'y0':0,
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
for k in log_param_list:
    guess[k] = np.log10(guess[k])
np.save('{0}/rdot4_guess_log.npy'.format(outdir), guess) 

# guess for rdot2
ninf = -100000
nt=1001
guess = {
    'nt':nt,
    'dt':1,
    'tau':500,
    'nper':100,
    'lag':int(0.16*nt),
    'diff':0.001,
    'x0':0.3,
    'y0':ninf,
    'a0':1.8,
    'a1':400,
    'a3':ninf,
    'b0':3,
    'b1':0.8*nt,
    'b2':0.25*nt,
    'b3':ninf,
    'c0':ninf,
    'c1':ninf,
    'c2':ninf,
    'c3':ninf,
    'yerr':0.01
}

log_param_list = ['tau','diff','a1','b1','b2']
for k in log_param_list:
    guess[k] = np.log10(guess[k])
np.save('{0}/rdot2_guess_log2.npy'.format(outdir), guess) 


