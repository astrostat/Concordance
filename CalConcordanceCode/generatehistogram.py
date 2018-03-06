
import numpy as np
import math as math
import matplotlib.pyplot as plt

def generatehistogramfunc(mcmcChainBG, mapResultBG, N, M, B, G, sigma, figsize1 = 20, figsize2 = 6, ncolfig = 5, sigmatheory = float('nan'), savefigname = 'fig.pdf', plotwhich = [1,0,1], xtickinputB = [0.85, 1, 1.15], xtickinputS = [0.1, 0.3, 0.5], xtickinputG = [1, 2, 3], fontsizelabel = 18, fontsizetick = 14, ticklength = 6, legend = True):
    nrowfig = sum(plotwhich)
    fig, axarray = plt.subplots(nrowfig, ncolfig, sharey='row', sharex = 'none', figsize = [figsize1, figsize2])
    temprow = 0
    if plotwhich[0] == 1:
        xminval = mcmcChainBG[:, range(ncolfig)].min()
        xmaxval = mcmcChainBG[:, range(ncolfig)].max()
        ymaxval = 0.0
        for i in range(ncolfig):
            Bsample = mcmcChainBG[:,i]
            temp = plt.hist(Bsample, color = 'grey', alpha = 0, normed = True, edgecolor = "none", bins = 30)
            x = np.linspace(start = Bsample.min(), stop = Bsample.max(), num = 100)
            Bmean = mapResultBG[0,i]
            Bsigma2 = mapResultBG[i+1,i]
            f = 1.0/np.sqrt(2*np.pi*Bsigma2)*np.exp(-0.5*(x-Bmean)**2/Bsigma2)
            ymaxval = max(ymaxval, f.max())
            ymaxval = max(ymaxval, max(temp[0]))    
            ymaxval = ymaxval * 1.05
            xlabels = ['$B_1$', '$B_2$', '$B_3$', '$B_4$', '$B_5$']
        for i in np.arange(ncolfig):
            ax = axarray[temprow, i]
            Bsample = mcmcChainBG[:,i]
            ax.hist(Bsample, color = 'grey', alpha = 0.2, normed = True, edgecolor = "none", bins = 30, label = 'posterior samples')
            ax.axvline(B[i], color = 'k', label = 'true value')
            x = np.linspace(start = Bsample.min(), stop = Bsample.max(), num = 100)
            Bmean = mapResultBG[0,i]
            Bsigma2 = mapResultBG[i+1,i]
            f = 1.0/np.sqrt(2*np.pi*Bsigma2)*np.exp(-0.5*(x-Bmean)**2/Bsigma2)
            ax.plot(x, f, 'r--', color = 'k', label = '"known" variance')
            ax.axis([xminval, xmaxval, 0, ymaxval])
            ax.set_xticks(xtickinputB)
            ax.tick_params('both', length = ticklength)
            ax.set_xlabel(xlabels[i])
            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsizetick)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(fontsizelabel)
                    #if i == 0:
                    #handles, labels = ax.get_legend_handles_labels()
                    #ax.set_frame_on(True)
                    #ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox = True, framealpha = 1)
        temprow = temprow + 1
    
    if plotwhich[1] == 1:
        xminval = mcmcChainBG[:, [x+N for x in range(ncolfig)]].min()
        xmaxval = mcmcChainBG[:, [x+N for x in range(ncolfig)]].max()
        ymaxval = 0.0
        for i in range(ncolfig):
            Gsample = mcmcChainBG[:,N+i]
            temp = plt.hist(Gsample, color = 'grey', alpha = 0, normed = True, edgecolor = "none", bins = 30)
            x = np.linspace(start = Gsample.min(), stop = Gsample.max(), num = 100)
            Gmean = mapResultBG[0,i+N]
            Gsigma2 = mapResultBG[1+i+N,i+N]
            f = 1.0/np.sqrt(2*np.pi*Gsigma2)*np.exp(-0.5*(x-Gmean)**2/Gsigma2)
            ymaxval = max(ymaxval, f.max())
            ymaxval = max(ymaxval, max(temp[0]))
        ymaxval = ymaxval * 1.05
        xlabels = ['$G_1$', '$G_2$', '$G_3$', '$G_4$', '$G_5$']
        
        for i in np.arange(ncolfig):
            ax = axarray[temprow, i]
            Gsample = mcmcChainBG[:,N+i]
            ax.hist(Gsample, color = 'grey', alpha = 0.2, normed = True, edgecolor = "none", bins = 30, label = 'posterior samples')
            ax.axvline(G[i], color = 'k', label = 'true value')
            x = np.linspace(start = Gsample.min(), stop = Gsample.max(), num = 100)
            Gmean = mapResultBG[0,i+N]
            Gsigma2 = mapResultBG[1+i+N,i+N]
            f = 1.0/np.sqrt(2*np.pi*Gsigma2)*np.exp(-0.5*(x-Gmean)**2/Gsigma2)
            ax.plot(x, f, 'r--', color = 'k', label = '"known" variance')
            ax.axis([xminval, xmaxval, 0, ymaxval])
            ax.set_xticks(xtickinputG)
            ax.tick_params('both', length = ticklength)
            ax.set_xlabel(xlabels[i])
            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsizetick)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(fontsizelabel)
        temprow = temprow + 1
    
    if plotwhich[2] == 1:
        xminval = np.sqrt(mcmcChainBG[:, [xx+N+M for xx in range(ncolfig)]].min())
        xminval = min(xminval, 0.0)
        xmaxval = np.sqrt(mcmcChainBG[:, [xp+N+M for xp in range(ncolfig)]].max())
        xmaxval = max(xmaxval, 1.1*max(sigma.flatten()))
        xmaxval = max(xmaxval, sigmatheory*1.1)
        ymaxval = 0.0
        for i in np.arange(ncolfig):
            Sigmasample = np.sqrt(mcmcChainBG[:,N+M+i])
            temp = plt.hist(Sigmasample, color = 'grey', alpha = 0, normed = True, edgecolor = "none", bins = 30)
            ymaxval = max(ymaxval, max(temp[0]))
        ymaxval = ymaxval * 1.05
        xlabels = ['$\sigma_1$', '$\sigma_2$', '$\sigma_3$', '$\sigma_4$', '$\sigma_5$']
        
        for i in np.arange(ncolfig):
            ax = axarray[temprow, i]
            Sigmasample = np.sqrt(mcmcChainBG[:,N+M+i])
            ax.hist(Sigmasample, color = 'grey', alpha = 0.2, normed = True, edgecolor = "none", bins = 30, label = 'posterior samples')
            ax.set_xlabel(xlabels[i])
            ax.axis([xminval, xmaxval, 0, ymaxval])
            ax.set_xticks(xtickinputS)
            ax.tick_params('both', length = ticklength)
            ax.axvline(sigma[i, 0], linestyle='dashed', color = 'k', label = '"known" variance')
            if not math.isnan(sigmatheory):
                ax.axvline(sigmatheory, color = 'k', label = 'true value')
            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsizetick)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(fontsizelabel)
                
                #fig.subplots_adjust(wspace=0.13, hspace=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if legend:
       fig.legend(handles, labels, ncol = 3, bbox_to_anchor=(0., 0.95, 1.5, .25), loc=3, fontsize = 14)#loc = 'upper center', fontsize = 'large', fancybox = True, bbox_to_anchor=(0.5, 1.1))#, bbox_to_anchor=(0.5, 1.2), loc=3, mode="expand", borderaxespad=0., fontsize = 'medium')
    fig.tight_layout()
#fig.subplots_adjust(right=0.85)
    fig.savefig(savefigname, bbox_inches="tight" )
    
 

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
    
    

    
    
    

def comparelognormallogt (N, M, mcmcChain, tChain, B, figsize1 = 20, figsize2 = 6, ncolfig = 5, savefigname = 'logtlogncompareN10M40B5.pdf', legend = True, fontsizelabel = 18, fontsizetick = 14, ticklength = 6, xtickinputB = [4.95, 5, 5.05]):
    
    fig, axarray = plt.subplots(2, ncolfig, sharey='row', sharex = 'col', figsize = [figsize1, figsize2])
    
    for indextemp in range(2):
        
        if indextemp == 0:
           mcmcChainBG = mcmcChain
        if indextemp == 1:
           mcmcChainBG = tChain
        xminval = 4.93
        xmaxval = 5.09
        ymaxval = 0.0
        for i in range(ncolfig):
            Bsample = mcmcChainBG[:,i]
            temp = plt.hist(Bsample, color = 'grey', alpha = 0, normed = True, edgecolor = "none", bins = 30)
            x = np.linspace(start = Bsample.min(), stop = Bsample.max(), num = 100)
            Bmean = np.mean(Bsample)
            Bsigma2 = np.var(Bsample)
            f = 1.0/np.sqrt(2*np.pi*Bsigma2)*np.exp(-0.5*(x-Bmean)**2/Bsigma2)
            #ymaxval = max(ymaxval, f.max())
            #ymaxval = max(ymaxval, max(temp[0]))
            #ymaxval = ymaxval * 1.05
            ymaxval = 25
            xlabels = ['$B_1$', '$B_2$', '$B_3$', '$B_4$', '$B_5$']
        for i in np.arange(ncolfig):
            ax = axarray[indextemp, i]
            Bsample = mcmcChainBG[:,i]
            if indextemp == 0:
                ax.hist(Bsample, color = 'grey', alpha = 0.2, normed = True, edgecolor = "none", bins = 30, label = 'posterior samples (log-Normal)')
            if indextemp == 1:
                ax.hist(Bsample, color = 'white', alpha = 0.2, normed = True, edgecolor = "black", bins = 30, label = 'posterior samples (log-t)')
            ax.axvline(B[i], color = 'k', label = 'true value')
            ax.axis([xminval, xmaxval, 0, ymaxval])
            ax.tick_params('both', length = ticklength)
            ax.set_xticks(xtickinputB)
            if indextemp == 1:
                ax.set_xlabel(xlabels[i])
            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsizetick)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(fontsizelabel)
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            if indextemp == 0:
               locationtemp = -0.762
            if indextemp == 1:
               locationtemp = -0.563
            ax.legend(handles, labels, ncol = 2, bbox_to_anchor=(locationtemp, 0.95, 1.5, .25), loc=3, fontsize = 15.5)
    fig.tight_layout()
    fig.savefig(savefigname, bbox_inches="tight" )




    
    
    
    
    
    
    
    
