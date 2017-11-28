import numpy as np
import pystan
import pickle
from pystan import StanModel
import pandas as pd
import os

## The function to compute the MCMC posterior of 'B' and 'G' given the counts.
## We use pystan to fit the posterior.
## 'Y' is the logarithm of the counts. 
## Each row is observations from a same instrument; each column is observations of a same object.
## When the observation is missing, the corresponding value is -inf.
## 'b' and 'tau' are the priors for 'B' and are all vectors. 
## 'model' indicates which model to fit. 
## If 'model' is 'log_normal', we fit the log-normal model with unknown variance.
## 'df' and 'beta' are the parameters for the prior distribution.
## If 'model' is 'log_t', we fit the log-t model.
## 'sigma' and 'nu' are the parameters for the prior distribution.

def stanTopkl():
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
            
    
def MCMCFit(Y, b, tau, model = 'log_normal', df = 2, beta = 0.01, sigma = 1, nu = 1, iter = 10000, chains = 1):
    N, M = Y.shape
    ## Record the index for the observed values.
    I = []
    J = []
    y = []
    for i in np.arange(N):
        for j in np.arange(M):
            if not np.isinf(Y[i,j]):
                I.append(i+1)
                J.append(j+1)
                y.append(Y[i,j])
    
    I = np.array(I, dtype = int)
    J = np.array(J, dtype = int)
    y = np.array(y, dtype = float)
    n = len(y)
    
    if model == 'log_normal':
        codeFile = 'log_normal.stan'
        dat = {'n':n, 'N':N, 'M':M, 'I':I, 'J':J, 'y':y, 'df':df, 'beta':beta, 'tau':tau, 'b':b}
        sm = pickle.load(open('log_normal.pkl', 'rb'))
        fit = sm.sampling(data = dat, iter = iter, chains = chains)
        #sm = StanModel(file = codeFile)
        #fit = pystan.stan(file=codeFile, data=dat, iter = iter, chains = chains)
        #with open('log_normal.pkl', 'wb') as f:
            #pickle.dump(sm, f)
            
    if model == 'log_t':
        codeFile = 'log_t.stan'
        dat = {'n':n, 'N':N, 'M':M, 'I':I, 'J':J, 'y':y, 'sigma': sigma, 'nu':nu, 'tau':tau, 'b':b}
        sm = pickle.load(open('log_t.pkl', 'rb'))
        fit = sm.sampling(data = dat, iter = iter, chains = chains)
        #sm = StanModel(file = codeFile)
        #fit = pystan.stan(file=codeFile, data=dat, iter = iter, chains = chains)
        #with open('log_t.pkl', 'wb') as f:
            #pickle.dump(sm, f)
        
    return summary_result(fit, model, N, M, n, I, J)
   
## Summarize the result from pystan.  
def summary_result(fit, model, N, M, n, I, J):
    ## Summary statistics for 'B', 'G' and 'sigma^2' (log-normal) or 'xi' (log-t).
    resMCMC = fit.summary()
    index = []
    if model == 'log_normal':
        ## The MCMC chains for 'B', 'G' and 'sigma^22'.
        chain = np.hstack((fit['B'], fit['G'], fit['sigma2']))
        data = (resMCMC['summary'])[:N+M+N,:]
        for i in np.arange(N):
            index.append('B['+str(i+1)+']')
        for j in np.arange(M):
            index.append('G['+str(j+1)+']')
        for i in np.arange(N):
            index.append('sigma^2['+str(i+1)+']')
    else:
        ## The MCMC chains for 'B', 'G' and 'xi'.
        chain = np.hstack((fit['B'], fit['G'], fit['xi']))
        data = (resMCMC['summary'])[:N+M+n,:]
        for i in np.arange(N):
            index.append('B['+str(i+1)+']')
        for j in np.arange(M):
            index.append('G['+str(j+1)+']')
        for k in np.arange(n):
            index.append('xi['+str(I[k])+','+str(J[k])+']')
            
    chain = pd.DataFrame(chain, columns = index)
    res = pd.DataFrame(data, index = index, columns = resMCMC['summary_colnames'])
    
    return chain, res
    
