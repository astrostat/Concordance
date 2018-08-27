import numpy as np
import sys


def simulateData(N, M, B, G, sigma, model='normal', xi=0.1, tau=0.05):
    """
    The function is to simulate observations for different models.

    Keyword arguments:
    N -- number of instruments
    M -- number of objects
    B -- logarithm of effective areas
    G -- logarithm of fluxes
    sigma -- standard deviations to generate data, unnecessary for the 'poisson' and 'both' models
    model -- different types of models (default 'model')
             Values can be 'normal', 'poisson', 'constant', 'both', 'poiconst' and 'poiconst2'
    xi -- standard deviation for the 'known' constants in the 'constant' model (default 0.1)
    tau -- standard deviation of logarithm of effective areas (default 0.05)
    :return:
    """
    b = np.random.normal(loc=B, scale=tau, size=N)
    
    # The log-normal model.
    if model == 'normal':
        mu = -sigma**2*0.5 + B[:, np.newaxis] + G
        Y = np.random.normal(loc=mu, scale=sigma, size=(N, M))
    
    # The poisson model.
    elif model == 'poisson':
        mu = B[:, np.newaxis] + G
        C = np.random.poisson(lam=np.exp(mu), size=(N, M))
        ind = np.where(C == 0)
        C[C == 0] = 0.1
        Y = np.log(C)
        Y[ind[0], ind[1]] = np.log(sys.float_info.min)
        
    # The misspecified model: log-normal model where T_{ij} is not precisely known, see Section 3.1 of the JASA paper.
    elif model == 'constant':
        lam = np.random.normal(loc=0, scale=xi, size=(N, M))
        mu = -sigma**2*0.5 + B[:, np.newaxis] + G
        Y = np.random.normal(loc=mu, scale=sigma, size=(N, M)) + lam
        
    # The misspecified model: data generating model is Poisson instead of log-Normal and the constants T_{ij} are not precisely known in the sense that there is a multiplicative unknown factor for the Poisson rate.
    elif model == 'both':
        lam = np.random.uniform(low=0.8, high=1.2, size=(N, M))
        mu = B[:, np.newaxis] + G
        C = np.random.poisson(lam=lam*np.exp(mu), size=(N, M))
        ind = np.where(C == 0)
        C[C == 0] = 0.1
        Y = np.log(C)
        Y[ind[0], ind[1]] = np.log(sys.float_info.min)

    # The misspecified model: data generating model is Poisson instead of log-Normal and the constants T_{ij} are not precisely known in the sense that there is a multiplicative unknown factor for the resulting observation/count. See Appendix F of the JASA paper.
    elif model == 'poiconst':
        lam = np.random.uniform(low=0.8, high=1.2, size=(N, M))
        mu = B[:, np.newaxis] + G
        C = np.random.poisson(lam=np.exp(mu), size=(N, M))
        ind = np.where(C == 0)
        C[C == 0] = 0.1
        C = C * lam
        Y = np.log(C)
        Y[ind[0], ind[1]] = np.log(sys.float_info.min)

    # The misspecified model: same as the previous one except that the multiplicative unknow factor is more volatile.
    elif model == 'poiconst2':
        lam = np.random.uniform(low=0.4, high=1.6, size=(N, M))
        mu = B[:, np.newaxis] + G
        C = np.random.poisson(lam=np.exp(mu), size=(N, M))
        ind = np.where(C == 0)
        C[C == 0] = 0.1
        C = C * lam
        Y = np.log(C)
        Y[ind[0], ind[1]] = np.log(sys.float_info.min)

    else:
        return None

    return Y, b
