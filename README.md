**experimental repo**

Scripts for modeling / fitting a coarse grained "Waddington's Landscape" model of a developmental decisions to data.

The basic steps are:
- Reduce the dimensionality of the bulk RNA-seq data via PCA
- Classify transcriptomic profiles to probabilistic cell fates via logistic regression
- Fit the probabilistic cell fates trajectories to trajectories on the low dimensional Waddington Landscape via Bayesian inference / MCMC to obtain the parameter distributions for the landscape model.
- Predict new trajectories using the low dimensional model.
- Test using perturbation experiments.
