# Concordance
IACHEC Calibration Concordance

1. Functions for model fitting
    
    a. mapFit(Y, sigma, b, tau)
       This function calculates the posterior mean and variance of 'B' and 'G' under the log_normal model with known variance. 'Y' is the logarithm of the counts. Each row is observations from a same instrument; each column is observations of a same object. When the observation is missing, the value is -inf (-float('Inf) in python). 'sigma' are the known standard deviations. 'b' and 'tau' are the priors for 'B'. 
     
    b. MCMCFit(Y, b, tau, model = 'log_normal', df = 2, beta = 0.01, sigma = 1, nu = 1, iter = 10000, chains = 1)
       This function computes the posterior of 'B', 'G', and 'sigma' given the observed counts using HMC implement via pystan. 'Y' is the logarithm of the counts. Each row is observations from a same instrument; each column is observations of a same object. When the observation is missing, the corresponding value is -inf. 'b' and 'tau' are the priors for 'B' and are all vectors.  'model' indicates which model to fit. If 'model' is 'log_normal', we fit the log-normal model with unknown variance. 'df' and 'beta' are the parameters for the prior distribution of \sigma_i^2. If 'model' is 'log_t', we fit the log-t model. 'sigma' and 'nu' are the parameters for the prior distribution of \xi_{ij}.
       
    c. summary_result(fit, model, N, M, n, I, J)
       This function summarizes the result from pystan. The output are "chain" and "res", the former contains posterior samples of unknown parameters and the latter contains summary statistics. 
       
2. Functions for simulation

   simulateData(N, M, B, G, sigma, model = 'normal', xi = 0.1, tau = 0.05)
  
   This function simulates observations for different models. N is the number of instruments. M is the number of objects. B is the logarithm of effective areas. G is the logarithm of fluxes. "model" denotes different types of models, available choices are "normal", "poisson", "constant", "both", "poiconst", "poiconst2". sigma is the standard deviations to generate data, unnecessary for the 'poisson' and 'both' models. xi is the standard deviation for the 'known' constants in the 'constant' model.
   
3. Functions for making figures

   a. generatehistogramfunc(mcmcChainBG, mapResultBG, N, M, B, G, sigma, figsize1 = 20, figsize2 = 6, ncolfig = 5, sigmatheory = float('nan'), savefigname = 'fig.pdf', divisorxaxt0 = 100, divisorxaxt = 100, plotwhich = [1,0,1], xtickinputB = [0.85, 1, 1.15], xtickinputS = [0.1, 0.3, 0.5], xtickinputG = [1, 2, 3], fontsizelabel = 18, fontsizetick = 14, ticklength = 6, legend = True)
   
      This function generates histograms of the posteriors of B, G, sigma.
   
   b. makeresidueplot (N, M, Y, mcmcRes, mcmcChain, tRes, tChain, savefigname = 'logtlogncompareSIM.pdf')
   
      This function compares the residues of fitted log-Normal and log-t model.
      
    c. comparelognormallogt (N, M, mcmcChain, tChain, B, figsize1 = 20, figsize2 = 6, ncolfig = 5, savefigname = 'logtlogncompareN10M40B5.pdf', legend = True, fontsizelabel = 18, fontsizetick = 14, ticklength = 6, xtickinputB = [4.95, 5, 5.05])
        This function compares histograms of the posterios of B.
 
