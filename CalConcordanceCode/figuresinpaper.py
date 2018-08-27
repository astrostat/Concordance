import numpy as np
import pandas as pd
from simulation import simulateData
from MAP import mapFit
from generatehistogram import generatehistogramfunc, makeresidueplot
from realdata import analyze_real_data, makefig_real_data, makefig_real_data_residue
from MCMC import stanTopkl, MCMCFit
import pickle
import math as math
import matplotlib.pyplot as plt
#stanTopkl()


######### SIMULATIONS ##################

def SimulateFitPlot (N, M, B, G, sigma, generatingmodel = 'poisson', fitmodel = 'log_normal', tau = 0.05, df = 2, beta = 0.01, savefigname = 'fittedBG', sigmatheory = 0.1, outfilemap = 'map', outfilemcmc = 'mcmc', outfilemcmcsummary = 'summary', makefig = 1, divisorxaxt0 = 100, divisorxaxt = 100, plotwhich = [1,0,1], figsize1 = 20, figsize2 = 6, ncolfig = 5):
    Y, b = simulateData(N, M, B, G, sigma, model = generatingmodel, tau = tau)
    mapResultBG = mapFit(Y, sigma, b, tau)
    mapResultBG = np.vstack((np.concatenate((mapResultBG['B'], mapResultBG['G'])), mapResultBG['Sigma']))
    np.save(outfilemap, mapResultBG)
    mcmcChainBG, mcmcResBG = MCMCFit(Y, b, tau, model = fitmodel, df = df, beta = beta)
    np.save(outfilemcmc, mcmcChainBG.values)
    np.save(outfilemcmcsummary, mcmcResBG)
    if makefig == 1:
        generatehistogramfunc(mcmcChainBG.values, mapResultBG, N, M, B, G, sigma, figsize1 = figsize1, figsize2 = figsize2, ncolfig = ncolfig, sigmatheory = sigmatheory, savefigname = savefigname, divisorxaxt = divisorxaxt, divisorxaxt0 = divisorxaxt0, plotwhich = plotwhich)


def SimulateFitPlotCompare (N, M, B, G, sigma, generatingmodel = 'poisson', tau = 0.05, df = 2, 
                            beta = 0.01, nu = 4, s2 = 0.141, savefigname = 'logtlogncompareSIM.pdf', 
                            outfilemap = 'map', outfilemcmc = 'mcmc', outfilemcmcsummary = 'summary', 
                            makefig = 1):
    Y, b = simulateData(N, M, B, G, sigma, model = generatingmodel, tau = tau)
    np.save('fig201710/outlierY', Y)
    mapResultBG = mapFit(Y, sigma, b, tau)
    mapResultBG = np.vstack((np.concatenate((mapResultBG['B'], mapResultBG['G'])), mapResultBG['Sigma']))
    np.save(outfilemap+'lognormal', mapResultBG)
    mcmcChainBG, mcmcResBG = MCMCFit(Y, b, tau, model = 'log_normal', df = df, beta = beta)
    np.save(outfilemcmc+'lognormal', mcmcChainBG.values)
    np.save(outfilemcmcsummary+'lognormal', mcmcResBG)
    tChain, tRes = MCMCFit(Y, b, tau, model = 'log_t', nu = nu, sigma = s2)
    np.save(outfilemcmc+'logt', tChain.values)
    np.save(outfilemcmcsummary+'logt', tRes)
    if makefig == 1:
        makeresidueplot (N, M, Y, mcmcResBG, mcmcChainBG.values, tRes, tChain.values, savefigname = savefigname)


# simulate from Poisson (Simulations 1 and 2)
N = 10
M = 40
sigma = np.repeat(0.1, N*M).reshape((N, M))
tau = np.repeat(0.05, N)
B = np.repeat(1, N)
G = np.repeat(1, M)

SimulateFitPlot (N, M, B, G, sigma, 
                 generatingmodel = 'poisson', 
                 fitmodel = 'log_normal', 
                 tau = tau, df = 2, beta = 0.01, 
                 savefigname = 'fig201710/simPoissonN10M40B1G1.pdf', 
                 sigmatheory = 0.421, 
                 outfilemap = 'fig201710/N10M40poissonsimlognormalfitB1G1map', 
                 outfilemcmc = 'fig201710/N10M40poissonsimlognormalfitB1G1mcmc', 
                 outfilemcmcsummary = 'fig201710/N10M40poissonsimlognormalfitB1G1summary', 
                 makefig = 1, divisorxaxt = 200, divisorxaxt0 = 100, plotwhich = [1,0,1])


## revise figures
N = 10
M = 40
sigma = np.repeat(0.1, N*M).reshape((N, M))
tau = np.repeat(0.05, N)
B = np.repeat(1, N)
G = np.repeat(1, M)
mapest = np.load('fig201710/N10M40poissonsimlognormalfitB1G1map.npy')
mcmc = np.load('fig201710/N10M40poissonsimlognormalfitB1G1mcmc.npy')
generatehistogramfunc(mcmc, mapest, N, M, B, G, sigma, figsize1 = 20, figsize2 = 6, 
                      ncolfig = 5, sigmatheory = 0.421, 
                      savefigname = 'fig201710/simPoissonN10M40B1G1.pdf', 
                      divisorxaxt = 100, divisorxaxt0 = 100, plotwhich = [1,0,1])


N = 10
M = 40
sigma = np.repeat(0.1, N*M).reshape((N, M))
tau = np.repeat(0.05, N)
B = np.repeat(5, N)
G = np.repeat(3, M)


SimulateFitPlot (N, M, B, G, sigma, 
                 generatingmodel = 'poisson', 
                 fitmodel = 'log_normal', 
                 tau = tau, df = 2, beta = 0.01, 
                 savefigname = 'fig201710/simPoissonN10M40B5G3beta001.pdf', 
                 sigmatheory = 0.018, 
                 outfilemap = 'fig201710/N10M40poissonsimlognormalfitB5G3beta001map', 
                 outfilemcmc = 'fig201710/N10M40poissonsimlognormalfitB5G3beta001mcmc',
                 outfilemcmcsummary = 'fig201710/N10M40poissonsimlognormalfitB5G3beta001summary',
                 makefig = 1, divisorxaxt = 85, divisorxaxt0 = 100, plotwhich = [1,0,1])


SimulateFitPlot (N, M, B, G, sigma, 
                 generatingmodel = 'poisson', 
                 fitmodel = 'log_normal', 
                 tau = tau, df = 2, beta = 0.001, 
                 savefigname = 'fig201710/simPoissonN10M40B5G3beta005square.pdf', 
                 sigmatheory = 0.018, 
                 outfilemap = 'fig201710/N10M40poissonsimlognormalfitB5G3beta005squaremap', 
                 outfilemcmc = 'fig201710/N10M40poissonsimlognormalfitB5G3beta005squaremcmc',
                 outfilemcmcsummary = 'fig201710/N10M40poissonsimlognormalfitB5G3beta005squaresummary',
                 makefig = 1, divisorxaxt = 85, divisorxaxt0 = 80, plotwhich = [1,0,1])


## revise figures

N = 10
M = 40
sigma = np.repeat(0.1, N*M).reshape((N, M))
tau = np.repeat(0.05, N)
B = np.repeat(5, N)
G = np.repeat(3, M)
mapest = np.load('fig201710/N10M40poissonsimlognormalfitB5G3beta001map.npy')
mcmc = np.load('fig201710/N10M40poissonsimlognormalfitB5G3beta001mcmc.npy')
generatehistogramfunc(mcmc, mapest, N, M, B, G, sigma, figsize1 = 20, figsize2 = 6, 
                      ncolfig = 5, sigmatheory = 0.018, 
                      savefigname = 'fig201710/simPoissonN10M40B5G3beta001.pdf', 
                      divisorxaxt = 85, divisorxaxt0 = 100, plotwhich = [1,0,1])

mapest = np.load('fig201710/N10M40poissonsimlognormalfitB5G3beta005squaremap.npy')
mcmc = np.load('fig201710/N10M40poissonsimlognormalfitB5G3beta005squaremcmc.npy')
generatehistogramfunc(mcmc, mapest, N, M, B, G, sigma, figsize1 = 20, figsize2 = 6, 
                      ncolfig = 5, sigmatheory = 0.018, 
                      savefigname = 'fig201710/simPoissonN10M40B5G3beta005square.pdf', 
                      divisorxaxt = 85, divisorxaxt0 = 80, plotwhich = [1,0,1])


# simulation 3
N = 10
M = 40
sigma = np.repeat(0.1, N*M).reshape((N, M))
tau = np.repeat(0.05, N)
B = np.repeat(5, N)
G = np.repeat(3, M)


SimulateFitPlot (N, M, B, G, sigma,
                 generatingmodel = 'constant',
                 fitmodel = 'log_normal',
                 tau = tau, df = 2, beta = 0.01,
                 savefigname = 'fig201710/simmis_specified_lognormalN10M40B5G3.pdf',
                 sigmatheory = 0.14,
                 outfilemap = 'fig201710/N10M40simmis_specified_lognormallognormalfitB5G3map',
                 outfilemcmc = 'fig201710/N10M40simmis_specified_lognormallognormalfitB5G3mcmc',
                 outfilemcmcsummary = 'fig201710/N10M40simmis_specified_lognormallognormalfitB5G3summary',
                 makefig = 1, divisorxaxt = 85, divisorxaxt0 = 80, plotwhich = [1,1,1])


## revise figuresN = 10
M = 40
sigma = np.repeat(0.1, N*M).reshape((N, M))
tau = np.repeat(0.05, N)
B = np.repeat(5, N)
G = np.repeat(3, M)
mapest = np.load('fig201710/N10M40simmis_specified_lognormallognormalfitB5G3map.npy')
mcmc = np.load('fig201710/N10M40simmis_specified_lognormallognormalfitB5G3mcmc.npy')
generatehistogramfunc(mcmc, mapest, N, M, B, G, sigma, figsize1 = 20, figsize2 = 9, 
                      ncolfig = 5, sigmatheory = 0.14, 
                      savefigname = 'fig201710/simmis_specified_lognormalN10M40B5G3.pdf', 
                      divisorxaxt = 85, divisorxaxt0 = 80, plotwhich = [1,1,1])


# simulation 6 and 7
N = 10
M = 40
sigma = np.repeat(0.1, N*M).reshape((N, M))
tau = np.repeat(0.05, N)
B = np.repeat(1, N)
G = np.repeat(3, M)

SimulateFitPlot (N, M, B, G, sigma,
                 generatingmodel = 'both',
                 fitmodel = 'log_normal',
                 tau = tau, df = 2, beta = 0.01,
                 savefigname = 'fig201710/simmis_specified_poissonN10M40B1G3.pdf',
                 outfilemap = 'fig201710/N10M40simmis_specified_poissonlognormalfitB1G3map',
                 outfilemcmc = 'fig201710/N10M40simmis_specified_poissonlognormalfitB1G3mcmc',
                 outfilemcmcsummary = 'fig201710/N10M40simmis_specified_poissonlognormalfitB1G3summary',
                 makefig = 1, divisorxaxt = 85, divisorxaxt0 = 80, plotwhich = [1,1,0])



N = 10
M = 40
sigma = np.repeat(0.1, N*M).reshape((N, M))
tau = np.repeat(0.05, N)
B = np.repeat(5, N)
G = np.repeat(3, M)


SimulateFitPlot (N, M, B, G, sigma,
                 generatingmodel = 'both',
                 fitmodel = 'log_normal',
                 tau = tau, df = 2, beta = 0.01,
                 savefigname = 'fig201710/simmis_specified_poissonN10M40B5G3.pdf',
                 outfilemap = 'fig201710/N10M40simmis_specified_poissonlognormalfitB5G3map',
                 outfilemcmc = 'fig201710/N10M40simmis_specified_poissonlognormalfitB5G3mcmc',
                 outfilemcmcsummary = 'fig201710/N10M40simmis_specified_poissonlognormalfitB5G3summary',
                 makefig = 1, divisorxaxt = 85, divisorxaxt0 = 80, plotwhich = [1,1,0])

# simulation 4 and 5


N = 10
M = 40
sigma = np.repeat(0.1, N*M).reshape((N, M))
tau = np.repeat(0.05, N)
B = np.repeat(5, N)
G = np.repeat(3, M)


SimulateFitPlot (N, M, B, G, sigma,
                 generatingmodel = 'poiconst',
                 fitmodel = 'log_normal',
                 tau = tau, df = 2, beta = 0.01,
                 savefigname = 'fig201710/simmis_specified_conspoissonN10M40B5G3.pdf',
                 outfilemap = 'fig201710/N10M40simmis_specified_conspoissonlognormalfitB5G3map',
                 outfilemcmc = 'fig201710/N10M40simmis_specified_conspoissonlognormalfitB5G3mcmc',
                 outfilemcmcsummary = 'fig201710/N10M40simmis_specified_conspoissonlognormalfitB5G3summary',
                 makefig = 1, divisorxaxt = 85, divisorxaxt0 = 80, plotwhich = [1,1,0])


SimulateFitPlot (N, M, B, G, sigma,
                 generatingmodel = 'poiconst2',
                 fitmodel = 'log_normal',
                 tau = tau, df = 2, beta = 0.01,
                 savefigname = 'fig201710/simmis_specified_cons2poissonN10M40B5G3.pdf',
                 outfilemap = 'fig201710/N10M40simmis_specified_cons2poissonlognormalfitB5G3map',
                 outfilemcmc = 'fig201710/N10M40simmis_specified_cons2poissonlognormalfitB5G3mcmc',
                 outfilemcmcsummary = 'fig201710/N10M40simmis_specified_cons2poissonlognormalfitB5G3summary',
                 makefig = 1, divisorxaxt = 85, divisorxaxt0 = 80, plotwhich = [1,1,0])



# with outliers

N = 10
M = 40
sigma = np.repeat(0.1, N*M).reshape((N, M))
tau = np.repeat(0.05, N)
B = np.repeat(5, N)
G = np.repeat(3, M)
G[0] = -2
SimulateFitPlotCompare (N, M, B, G, sigma, generatingmodel = 'poisson', tau = tau, df = 2, 
                            beta = 0.01, nu = 4, s2 = 0.141, savefigname = 'fig201710/logtlogncompareSIM.pdf', 
                            outfilemap = 'fig201710/outliermap', outfilemcmc = 'fig201710/outliermcmc', 
                            outfilemcmcsummary = 'fig201710/outliersummary', makefig = 1)

mcmc = np.load('fig201710/outliermcmclognormal.npy')
mcmcres = np.load('fig201710/outliersummarylognormal.npy')
mcmct = np.load('fig201710/outliermcmclogt.npy')
mcmcrest = np.load('fig201710/outliersummarylogt.npy')
Y = np.load('fig201710/outlierY.npy')
makeresidueplot (N, M, Y, mcmcres, mcmc, mcmcrest, mcmct, savefigname = 'fig201710/logtlogncompareSIM.pdf')











############### Real Data ##########################


analyze_real_data (outputfolder = 'fig201710/realdata/')

makefig_real_data (outputfolder = 'fig201710/realdata/')

makefig_real_data_residue (outputfolder = 'fig201710/realdata/')

