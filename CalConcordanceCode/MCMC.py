import numpy as np
import pystan
import pickle
from pystan import StanModel
import pandas as pd
import os


def stanTopkl():
    """
    The function complies 'stan' models first and avoids re-complie of the model.
    """
    if os.path.isfile('log_normal.pkl'):
        os.remove('log_normal.pkl')
    sm = StanModel(file = 'log_normal.stan')
    with open('log_normal.pkl', 'wb') as f:
        pickle.dump(sm, f)
            
    if os.path.isfile('log_t.pkl'):
        os.remove('log_t.pkl')
    sm = StanModel(file = 'log_t.stan')
    with open('log_t.pkl', 'wb') as f:
        pickle.dump(sm, f)
            
    
def MCMCFit(Y, b, tau, model='log_normal', df=2, beta=0.01, sigma=1, nu=1, iter=10000, chains=1):
    """
    The function to compute the MCMC posterior of 'B' and 'G' given the counts.
    We use pystan to fit the posterior.

    Keyword arguments:
    Y -- logarithm of the counts
         Each row is observations from a same instrument; each column is observations of a same object.
    b -- prior mean for 'B'
    tau -- prior standard deviation for 'B'
    model -- model used to fit the data (default 'log_normal')
             If 'model' is 'log_normal', we fit the log-normal model with unknown variance.
             If 'model' is 'log_t', we fit the log-t model.
    df -- used when 'model' is 'log_normal', prior parameter for the variance (default 2.0)
    beta -- used when 'model' is 'log_normal', prior parameter for the variance (default 0.01)
    sigma -- used when 'model' is 'log_t', prior parameter for the variance (default 1.0)
    nu -- used when 'model' is 'log_t', prior parameter for the variance (default 1.0)
    iter -- number of iterations in MCMC (default 10000)
    chains -- number of chains in MCMC (default 1)
    """
    N, M = Y.shape
    # Record the index for the observations.
    I = []
    J = []
    y = []
    for i in np.arange(N):
        for j in np.arange(M):
            if not np.isinf(Y[i, j]):
                I.append(i+1)
                J.append(j+1)
                y.append(Y[i, j])
    
    I = np.array(I, dtype=int)
    J = np.array(J, dtype=int)
    y = np.array(y, dtype=float)
    n = len(y)

    if model == 'log_normal':
        dat = {'n': n, 'N': N, 'M': M, 'I': I, 'J': J, 'y': y,
               'df': df, 'beta': beta, 'tau': tau, 'b': b}
        sm = pickle.load(open('log_normal.pkl', 'rb'))
        fit = sm.sampling(data=dat, iter=iter, chains=chains)

    elif model == 'log_t':
        dat = {'n': n, 'N': N, 'M': M, 'I': I, 'J': J, 'y': y,
               'sigma': sigma, 'nu': nu, 'tau': tau, 'b': b}
        sm = pickle.load(open('log_t.pkl', 'rb'))
        fit = sm.sampling(data=dat, iter=iter, chains=chains)

    else:
        return None
        
    return summary_result(fit, model, N, M, n, I, J)


def summary_result(fit, model, N, M, n, I, J):
    """
    The function summarizes the result from pystan.

    Keyword arguments:
    fit -- the MCMC result
    model -- the model used to fit the data
             If 'model' is 'log_normal', we fit the log-normal model with unknown variance.
             If 'model' is 'log_t', we fit the log-t model.
    N -- number of instruments
    M -- number of objects
    n -- total number of observations
    I -- instrument index for each observation
    J -- object index for each observation
    """
    # Summary statistics for 'B', 'G' and 'sigma^2' (log-normal) or 'xi' (log-t).
    resMCMC = fit.summary()
    index = []
    if model == 'log_normal':
        # The MCMC chains for 'B', 'G' and 'sigma^22'.
        chain = np.hstack((fit['B'], fit['G'], fit['sigma2']))
        data = (resMCMC['summary'])[:N+M+N, :]
        for i in np.arange(N):
            index.append('B['+str(i+1)+']')
        for j in np.arange(M):
            index.append('G['+str(j+1)+']')
        for i in np.arange(N):
            index.append('sigma^2['+str(i+1)+']')
    else:
        # The MCMC chains for 'B', 'G' and 'xi'.
        chain = np.hstack((fit['B'], fit['G'], fit['xi']))
        data = (resMCMC['summary'])[:N+M+n, :]
        for i in np.arange(N):
            index.append('B['+str(i+1)+']')
        for j in np.arange(M):
            index.append('G['+str(j+1)+']')
        for k in np.arange(n):
            index.append('xi['+str(I[k])+','+str(J[k])+']')
            
    chain = pd.DataFrame(chain, columns=index)
    res = pd.DataFrame(data, index=index, columns=resMCMC['summary_colnames'])
    
    return chain, res

