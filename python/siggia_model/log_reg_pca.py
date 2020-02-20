import numpy as np
import scipy as sc
import myfun as mf

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.linear_model import LogisticRegression as logreg

from numpy import linalg as linalg

# directories
datdir  = '/Users/simonfreedman/cqub/xenopus/data'
plotdir = '/Users/simonfreedman/cqub/xenopus/plots'

stagestrs = ['9','10','10.5','11','12','13']

gcts    = np.load('{0}/deGeneCounts.npy'.format(datdir)) 
gtpm     = (gcts.transpose((3,0,1,2)) * 10**6 / np.sum(gcts,axis=3)).transpose((1,2,3,0))
gtpmF    = gtpm.reshape(-1,gtpm.shape[-1]) # arranged as c0_rep0_t0, c0_rep0_t1, c0_rep0_t2...c0_rep1_t0, c0_rep1_t1,...,c0_rep2_t5,c1_rep0_t0,...,c2_rep2_t5
tpmNZmin = np.sort(np.unique(gtpm))[0:5][1]

pcsGrpd,_,_ = mf.logCenPca(gtpmF,tpmNZmin/2)

# compute training data
fulldat  = pcsGrpd
traindat = fulldat[5::6]

X   = traindat
y   = np.array([0,0,0,1,1,1,2,2,2])+1
clf = logreg(random_state = 0)#, multi_class='ovr') #, multi_class = 'auto')

clf.fit(X,y)

preds  = clf.predict_proba(fulldat)
# arranged as c0_rep0_t0, c0_rep0_t1, c0_rep0_t2...c0_rep1_t0, c0_rep1_t1,...,c0_rep2_t5,c1_rep0_t0,...,c2_rep2_t5

predss = np.array(np.split(
    np.array(np.split(preds,9)) # split into each group of 6 time stages
             ,3) # split into each group of 3 replicates, such that
                  # 0th==> trajectory, 1th==> rep, 2th==>stage, 3th==>probability of condition
).transpose((3,0,1,2))

np.save('{0}/log_reg_pca_preds.npy'.format(datdir), preds)
np.save('{0}/log_reg_pca_predss.npy'.format(datdir), predss)
np.savetxt('{0}/log_reg_pca_preds.tsv'.format(datdir), preds)
# now its 0th => p(cond), 1th=>trajectory, 2th=>rep, 3th=>stage

########################
### make 2d plot #######
########################
fig,axs=plt.subplots(1,3,figsize=(20,5))

cols=['r','b','goldenrod']
rcols = list(reversed(cols))
marks = ['s','o','^']
markersizes = [15,12,8]
fillstyles = ['none','none','full']

stagestrs = [9,10,10.5,11,12,13]
nstages = len(stagestrs)

titles = ['epidermal data','neural data','endodermal data']
ylabels = ['epidermal', 'neural', 'endodermal']

for i in range(3): # loop through conditional probabilities
    ax = axs[i]
    plts = []
    for j in range(3): # loop through trajectories
        for k in range(3): # loop through replicates

            lineplt = ax.errorbar(range(nstages),predss[i,j,k], color=cols[j],
                        marker=marks[k],markersize=markersizes[k],fillstyle = fillstyles[k])#,
                    #label='{0} rep {1}'.format(ylabels[k],j))
            plts.append(lineplt)


    ax.set_xlabel('stage')
    ax.set_xticks(range(nstages))
    ax.set_xticklabels(stagestrs)

   # ax.set_title(titles[i],color=cols[i])
    ax.set_ylabel('P({0})'.format(ylabels[i]),color=cols[i])
    ax.set_yticks(np.arange(0,1.2,0.2))
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_color(cols[i])


    mf.set_axs_fontsize(ax, 20)
    ax.set_ylim(-0.05,1.05)

    #ax.set_yscale('symlog')
#axs[0].set_ylabel("probability of condition")
#axs[2].legend()

legend1 = axs[0].legend(plts[0:3],ylabels, loc=(0.05,0.6), title="condition",
                     fontsize=16,frameon=False)
legend1.get_title().set_fontsize('16')
axs[0].add_artist(legend1)

legend2 = axs[1].legend(plts[1:9:3],np.arange(1,4), loc=(0.05,0.6), title="replicate",
                     fontsize=16,frameon=False)
legend2.get_title().set_fontsize('16')
axs[1].add_artist(legend2)
#plt.show()
plt.savefig('{0}/pca_logreg_2d_b.jpg'.format(plotdir),bbox_inches="tight")

###############################
####MAKE 3D PLOT ##############
###############################
fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
fillstyles = ['none','none','none']
marks = ['s','o','^']
markersizes = [15,12,10]

for j in range(3): # loop through trajectories
    for k in range(3): # loop through replicates
        ax.plot(predss[0,j,k],predss[2,j,k], predss[1,j,k],color=cols[j],
                    marker=marks[k],markersize=markersizes[k],fillstyle = fillstyles[k],
                label='{0} rep {1}'.format(ylabels[j],k),zorder=2)
        ax.plot(predss[0,j,k,[0]],predss[2,j,k,[0]], predss[1,j,k,[0]], color=cols[j],
                    marker=marks[k],markersize=markersizes[k],fillstyle = 'full',alpha=0.5,zorder=1)

fs = 18
lp = 15
ax.set_xlabel('P(epidermal)',color='red',fontsize=fs,labelpad=lp)
ax.set_zlabel('P(neural)',color='blue',fontsize=fs,labelpad=lp)
ax.set_ylabel('P(endodermal)',color=cols[2],fontsize=fs,labelpad=lp)

ax.set_zticks(np.arange(0,1.2,0.2))
for tick in ax.zaxis.get_major_ticks():
    tick.label.set_fontsize(fs)
    tick.label.set_color(cols[1])

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(fs)
    tick.label.set_color(cols[0])

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(fs)
    tick.label.set_color(cols[2])



# # create x,y
xx, yy = np.meshgrid(np.arange(0,1,0.005), np.arange(0,1,0.005))
zz = 1-xx-yy
ax.plot(xx[zz>0].reshape(-1),yy[zz>0].reshape(-1), zz[zz>0].reshape(-1)
        ,color='black', marker='.',markersize=1,fillstyle = 'full',zorder=0,alpha=0.1)

# ax.set_zlim(-0.05,1.05)
# ax.set_zlim(-0.05,1.05)
ax.view_init(10,50)

#plt.show()
plt.savefig('{0}/pca_logreg_3d_b.jpg'.format(plotdir),bbox_inches="tight")
