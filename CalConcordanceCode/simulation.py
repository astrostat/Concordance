import numpy as np
import sys

## To simulate observations for different models.
## N: number of instruments.
## M: number of objects.
## B: logarithm of effective areas.
## G: logarithm of fluxes.
## model: different types of models.
## sigma: the standard deviations to generate data, unnecessary for the 'poisson' and 'both' models.
## xi: the standard deviation for the 'known' constants in the 'constant' model.
def simulateData(N, M, B, G, sigma, model = 'normal', xi = 0.1, tau = 0.05):
    b = np.random.normal(loc = B, scale = tau, size = N)
    
    ## The log-normal model.
    if model == 'normal':
        mu = -sigma**2*0.5 + B[:,np.newaxis] + G
        Y = np.random.normal(loc = mu, scale = sigma, size = (N, M))
    
    ## The poisson model.
    if model == 'poisson':
        mu = B[:,np.newaxis] + G
        C = np.random.poisson(lam = np.exp(mu), size = (N, M))
        ind = np.where(C==0)
        C[C==0] = 0.1
        Y = np.log(C)
        Y[ind[0], ind[1]] = np.log(sys.float_info.min)
        
    ## The log-normal model with 'known' constants.
    if model == 'constant':
        lam = np.random.normal(loc = 0, scale = xi, size = (N, M))
        mu = -sigma**2*0.5 + B[:,np.newaxis] + G 
        Y = np.random.normal(loc = mu, scale = sigma, size = (N, M)) + lam
        
    ## The model with misspecification with poisson model and 'known' constants.
    if model == 'both':
        lam = np.random.uniform(low = 0.8, high = 1.2, size = (N, M))
        mu = B[:,np.newaxis] + G
        C = np.random.poisson(lam = lam*np.exp(mu), size = (N, M))
        ind = np.where(C==0)
        C[C==0] = 0.1
        Y = np.log(C)
        Y[ind[0], ind[1]] = np.log(sys.float_info.min)
    if model == 'poiconst':
        lam = np.random.uniform(low = 0.8, high = 1.2, size = (N, M))
        mu = B[:,np.newaxis] + G
        C = np.random.poisson(lam = np.exp(mu), size = (N, M))
        ind = np.where(C==0)
        C[C==0] = 0.1
        C = C * lam
        Y = np.log(C)
        Y[ind[0], ind[1]] = np.log(sys.float_info.min)
    if model == 'poiconst2':
        lam = np.random.uniform(low = 0.4, high = 1.6, size = (N, M))
        mu = B[:,np.newaxis] + G
        C = np.random.poisson(lam = np.exp(mu), size = (N, M))
        ind = np.where(C==0)
        C[C==0] = 0.1
        C = C * lam
        Y = np.log(C)
        Y[ind[0], ind[1]] = np.log(sys.float_info.min)
    return Y, b
    