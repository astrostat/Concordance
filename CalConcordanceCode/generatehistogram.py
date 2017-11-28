
import numpy as np
import math as math
import matplotlib.pyplot as plt

def generatehistogramfunc(mcmcChainBG, mapResultBG, N, M, B, G, sigma, figsize1 = 20, figsize2 = 6, ncolfig = 5, sigmatheory = float('nan'), savefigname = 'fig.pdf', divisorxaxt0 = 100, divisorxaxt = 100, plotwhich = [1,0,1]):
    
    plt.figure(figsize = (figsize1, figsize2))
    nrowfig = sum(plotwhich)
    temprow = 0
    if plotwhich[0] == 1:
        xminval = mcmcChainBG[:, range(ncolfig)].min()
        xmaxval = mcmcChainBG[:, range(ncolfig)].max()
        ymaxval = 0.0
        for i in range(ncolfig):
            Bsample = mcmcChainBG[:,i]
            temp = plt.hist(Bsample, color = 'grey', alpha = 0.2, normed = True, edgecolor = "none", bins = 30)
            x = np.linspace(start = Bsample.min(), stop = Bsample.max(), num = 100)
            Bmean = mapResultBG[0,i]
            Bsigma2 = mapResultBG[i+1,i]
            f = 1.0/np.sqrt(2*np.pi*Bsigma2)*np.exp(-0.5*(x-Bmean)**2/Bsigma2)
            ymaxval = max(ymaxval, f.max())
            ymaxval = max(ymaxval, max(temp[0]))    
            ymaxval = ymaxval * 1.05
            xlabels = ['$B_1$', '$B_2$', '$B_3$', '$B_4$', '$B_5$']
        
        for i in np.arange(ncolfig):
            ax = plt.subplot(nrowfig, ncolfig, i+ncolfig*temprow+1)
            Bsample = mcmcChainBG[:,i]
            plt.hist(Bsample, color = 'grey', alpha = 0.2, normed = True, edgecolor = "none", bins = 30)
            plt.axvline(B[i], color = 'k')
            x = np.linspace(start = Bsample.min(), stop = Bsample.max(), num = 100)
            Bmean = mapResultBG[0,i]
            Bsigma2 = mapResultBG[i+1,i]
            f = 1.0/np.sqrt(2*np.pi*Bsigma2)*np.exp(-0.5*(x-Bmean)**2/Bsigma2)
            plt.plot(x, f, 'r--', color = 'k')
            plt.axis([xminval, xmaxval, 0, ymaxval])
            spacetemp = round(math.floor(math.floor(xmaxval * 100 - xminval * 100) / 3) / divisorxaxt0, 2)
            xtickstemp = np.arange(math.floor(xminval * 100) * 0.01, math.floor(xmaxval * 100) * 0.01, spacetemp)
            np.delete(xtickstemp, 0)
            plt.xticks(xtickstemp)
            plt.xlabel(xlabels[i])
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(18)
        temprow = temprow + 1
    
    if plotwhich[1] == 1:
        xminval = mcmcChainBG[:, [x+N for x in range(ncolfig)]].min()
        xmaxval = mcmcChainBG[:, [x+N for x in range(ncolfig)]].max()
        ymaxval = 0.0
        for i in range(ncolfig):
            Gsample = mcmcChainBG[:,N+i]
            temp = plt.hist(Gsample, color = 'grey', alpha = 0.2, normed = True, edgecolor = "none", bins = 30)
            x = np.linspace(start = Gsample.min(), stop = Gsample.max(), num = 100)
            Gmean = mapResultBG[0,i+N]
            Gsigma2 = mapResultBG[1+i+N,i+N]
            f = 1.0/np.sqrt(2*np.pi*Gsigma2)*np.exp(-0.5*(x-Gmean)**2/Gsigma2)
            ymaxval = max(ymaxval, f.max())
            ymaxval = max(ymaxval, max(temp[0]))
        ymaxval = ymaxval * 1.05
        xlabels = ['$G_1$', '$G_2$', '$G_3$', '$G_4$', '$G_5$']
        
        for i in np.arange(ncolfig):
            ax = plt.subplot(nrowfig, ncolfig, i+ncolfig*temprow+1)
            Gsample = mcmcChainBG[:,N+i]
            plt.hist(Gsample, color = 'grey', alpha = 0.2, normed = True, edgecolor = "none", bins = 30)
            plt.axvline(G[i], color = 'k')
            x = np.linspace(start = Gsample.min(), stop = Gsample.max(), num = 100)
            Gmean = mapResultBG[0,i+N]
            Gsigma2 = mapResultBG[1+i+N,i+N]
            f = 1.0/np.sqrt(2*np.pi*Gsigma2)*np.exp(-0.5*(x-Gmean)**2/Gsigma2)
            plt.plot(x, f, 'r--', color = 'k')
            plt.axis([xminval, xmaxval, 0, ymaxval])
            spacetemp = math.floor(math.floor(xmaxval * 100 - xminval * 100) / 3) / 100
            xtickstemp = np.arange(math.floor(xminval * 100) * 0.01, math.floor(xmaxval * 100) * 0.01, spacetemp)
            np.delete(xtickstemp, 0)
            plt.xticks(xtickstemp)
            plt.xlabel(xlabels[i])
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(18)
        temprow = temprow + 1
    
    if plotwhich[2] == 1:
        xminval = np.sqrt(mcmcChainBG[:, [xx+N+M for xx in range(ncolfig)]].min())
        xminval = min(xminval, 0.0)
        xmaxval = np.sqrt(mcmcChainBG[:, [xp+N+M for xp in range(ncolfig)]].max())
        xmaxval = max(xmaxval, 1.1*max(sigma.flatten()))
        xmaxval = max(xmaxval, sigmatheory*1.1)
        ymaxval = 0.0
        for i in np.arange(5):
            ax = plt.subplot(nrowfig, ncolfig, i+ncolfig*temprow+1)
            Sigmasample = np.sqrt(mcmcChainBG[:,N+M+i])
            temp = plt.hist(Sigmasample, color = 'grey', alpha = 0.2, normed = True, edgecolor = "none", bins = 30)
            ymaxval = max(ymaxval, max(temp[0]))
        ymaxval = ymaxval * 1.05
        xlabels = ['$\sigma_1$', '$\sigma_2$', '$\sigma_3$', '$\sigma_4$', '$\sigma_5$']
        
        for i in np.arange(ncolfig):
            ax = plt.subplot(nrowfig, ncolfig, i+ncolfig*temprow+1)
            Sigmasample = np.sqrt(mcmcChainBG[:,N+M+i])
            plt.hist(Sigmasample, color = 'grey', alpha = 0.2, normed = True, edgecolor = "none", bins = 30)
            plt.xlabel(xlabels[i])
            plt.axis([xminval, xmaxval, 0, ymaxval])
            spacetemp = round(math.floor(math.floor(xmaxval * 100 - xminval * 100) / 3) / divisorxaxt, 2)
            xtickstemp = np.arange(math.floor(xminval * 100) * 0.01, math.floor(xmaxval * 100) * 0.01, spacetemp)
            np.delete(xtickstemp, 0)
            plt.xticks(xtickstemp)
            plt.axvline(sigma[i, 0], linestyle='dashed', color = 'k')
            if not math.isnan(sigmatheory):
                plt.axvline(sigmatheory, color = 'k')
        
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(18)
                
    
    plt.tight_layout()
    plt.savefig(savefigname)
    
 

def makeresidueplot (N, M, Y, mcmcRes, mcmcChain, tRes, tChain, savefigname = 'logtlogncompareSIM.pdf'):
    plt.figure(figsize = (15, 6))
    markers = ['o', 's', 'd']
    colors = ['black', 'gray', 'silver']
    #colors = ['#006DDB', '#FFFF6D', '#000000']
    sizes = [80, 40, 40]
    ylim = (-6.0, 6.0)
    
    ax = plt.subplot(2, 1, 1)
    d = mcmcRes
    
    #sigmaFit = d[(N+M):(2*N+M),0]
    sigmaFit = mcmcChain[:,(N+M):(2*N+M)].mean(axis=0)
    B = d[0:N, 0]
    G = d[N:(N+M), 0]
    fit = (B - sigmaFit*0.5)[:,np.newaxis] + G
    residual = Y - fit
    residual = residual/np.sqrt(sigmaFit[:,np.newaxis])
    index = np.arange(N)
    for j in np.arange(3):
        plt.scatter(index, residual[:,j], marker = markers[j], edgecolors='face', color = colors[j], s = sizes[j])
    plt.xlim((-1, N))
    plt.axhline(2, ls = '--', color='k')
    plt.axhline(-2, ls = '--', color='k')
    plt.xticks([])
    
    plt.title('Standardized Residuals (Log-Normal Model)')
    plt.ylim(ylim)
    plt.xlabel('Instruments')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    
    ax = plt.subplot(2, 1, 2)
    d = tRes
    #sigmaFit = 0.141**2/(d[(N+M):(N+M+N*M),0].reshape((N,M)))
    sigmaFit = 0.141**2/tChain[:,(N+M):(N+M+N*M)].mean(axis = 0).reshape((N, M))
    B = d[0:N, 0]
    G = d[N:(N+M), 0]
    
    fit = B[:,np.newaxis]  + G - sigmaFit*0.5
    residual = Y - fit
    residual = residual/np.sqrt(sigmaFit)
    index = np.arange(N)
    for j in np.arange(3):
        plt.scatter(index, residual[:,j], marker = markers[j], edgecolors='face', color = colors[j], s = sizes[j])
    plt.xlim((-1, N))
    plt.axhline(2, ls = '--', color='k')
    plt.axhline(-2, ls = '--', color='k')
    plt.xticks([])
    
    plt.title('Standardized Residuals (Log-t Model)')
    plt.ylim(ylim)
    plt.xlabel('Instruments')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    
    plt.tight_layout()
    plt.savefig(savefigname)
    
    

    
    
    
    
    
    
    
    
    
    
    
    