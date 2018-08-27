from MCMC import MCMCFit
from MAP import mapFit


def analyze_real_data(outputfolder='fig201710/realdata/'):
    """
    The function is to analyze real data.
    """
    # E0102 data
    O = pickle.load(open('realData/O.p', 'rb'))
    Ne = pickle.load(open('realData/Ne.p', 'rb'))
    N = 13
    b = np.zeros(N)
    tau1 = np.repeat(0.025, N)
    tau2 = np.repeat(0.05, N)
    
    _, O1Res = MCMCFit(O, b, tau1, model='log_normal', df=1.5, beta=0.014**2)
    _, O2Res = MCMCFit(O, b, tau2, model='log_normal', df=1.5, beta=0.014**2)
    _, Ne1Res = MCMCFit(Ne, b, tau1, model='log_normal', df=1.5, beta=0.009**2)
    _, Ne2Res = MCMCFit(Ne, b, tau2, model='log_normal', df=1.5, beta=0.009**2)
    O1Res.to_csv(outputfolder+'O1Res.csv')
    O2Res.to_csv(outputfolder+'O2Res.csv')
    Ne1Res.to_csv(outputfolder+'Ne1Res.csv')
    Ne2Res.to_csv(outputfolder+'Ne2Res.csv')

    # 2XMM data
    hard2XMM = pickle.load(open('realData/hard2XMM.p', 'rb'))
    
    N, M = hard2XMM.shape
    b = np.zeros(N)
    tau1 = np.repeat(0.025, N)
    tau2 = np.repeat(0.05, N)
    _, hard2XMM1Res = MCMCFit(hard2XMM, b, tau1, model='log_normal', df=1.5, beta=0.116**2)
    _, hard2XMM2Res = MCMCFit(hard2XMM, b, tau2, model='log_normal', df=1.5, beta=0.116**2)
    medium2XMM = pickle.load(open('realData/medium2XMM.p', 'rb'))
    
    N, M = medium2XMM.shape
    b = np.zeros(N)
    tau1 = np.repeat(0.025, N)
    tau2 = np.repeat(0.05, N)
    _, medium2XMM1Res = MCMCFit(medium2XMM, b, tau1, model='log_normal', df=15, beta=0.288**2)
    _, medium2XMM2Res = MCMCFit(medium2XMM, b, tau2, model='log_normal', df=1.5, beta=0.288**2)
    soft2XMM = pickle.load(open('realData/soft2XMM.p', 'rb'))
    
    N, M = soft2XMM.shape
    b = np.zeros(N)
    tau1 = np.repeat(0.025, N)
    tau2 = np.repeat(0.05, N)
    _, soft2XMM1Res = MCMCFit(soft2XMM, b, tau1, model='log_normal', df=1.5, beta=0.148**2)
    _, soft2XMM2Res = MCMCFit(soft2XMM, b, tau2, model='log_normal', df=1.5, beta=0.148**2)
    hard2XMM1Res.to_csv(outputfolder+'hard2XMM1Res.csv')
    hard2XMM2Res.to_csv(outputfolder+'hard2XMM2Res.csv')
    medium2XMM1Res.to_csv(outputfolder+'medium2XMM1Res.csv')
    medium2XMM2Res.to_csv(outputfolder+'medium2XMM2Res.csv')
    soft2XMM1Res.to_csv(outputfolder+'soft2XMM1Res.csv')
    soft2XMM2Res.to_csv(outputfolder+'soft2XMM2Res.csv')

    # XCAL data
    hard = pickle.load(open('realData/hard.p', 'rb'))    
    N, M = hard.shape
    b = np.zeros(N)
    tau1 = np.repeat(0.025, N)
    tau2 = np.repeat(0.05, N)
    _, hard1Res = MCMCFit(hard, b, tau1, model='log_normal', df=1.5, beta=0.028**2)
    hardChain, hard2Res = MCMCFit(hard, b, tau2, model='log_normal', df=1.5, beta=0.028**2)
    hardtChain, hardtRes = MCMCFit(hard, b, tau2, model='log_t', nu=3, sigma=0.028*1.41)
    
    medium = pickle.load(open('realData/medium.p', 'rb'))    
    N, M = medium.shape
    b = np.zeros(N)
    tau1 = np.repeat(0.025, N)
    tau2 = np.repeat(0.05, N)
    _, medium1Res = MCMCFit(medium, b, tau1, model='log_normal', df=1.5, beta=0.093**2)
    mediumChain, medium2Res = MCMCFit(medium, b, tau2, model='log_normal', df=1.5, beta=0.093**2)
    mediumtChain, mediumtRes = MCMCFit(medium, b, tau2, model='log_t', nu=3, sigma=0.093*1.41)
    
    soft = pickle.load(open('realData/soft.p', 'rb'))    
    N, M = soft.shape
    b = np.zeros(N)
    tau1 = np.repeat(0.025, N)
    tau2 = np.repeat(0.05, N)
    _, soft1Res = MCMCFit(soft, b, tau1, model='log_normal', df=1.5, beta=0.026**2)
    softChain, soft2Res = MCMCFit(soft, b, tau2, model='log_normal', df=1.5, beta=0.026**2)
    softtChain, softtRes = MCMCFit(soft, b, tau2, model='log_t', nu=3, sigma=0.026*1.41)
    
    hard1Res.to_csv(outputfolder+'hard1Res.csv')
    hard2Res.to_csv(outputfolder+'hard2Res.csv')
    hardtRes.to_csv(outputfolder+'hard2tRes.csv')
    medium1Res.to_csv(outputfolder+'medium1Res.csv')
    medium2Res.to_csv(outputfolder+'medium2Res.csv')
    mediumtRes.to_csv(outputfolder+'mediumtRes.csv')
    soft1Res.to_csv(outputfolder+'soft1Res.csv')
    soft2Res.to_csv(outputfolder+'soft2Res.csv')
    softtRes.to_csv(outputfolder+'softtRes.csv')
    pd.DataFrame(hardChain.values).to_csv(outputfolder+'hardChain.csv')
    pd.DataFrame(hardtChain.values).to_csv(outputfolder+'hardtChain.csv')
    pd.DataFrame(mediumChain.values).to_csv(outputfolder+'mediumChain.csv')
    pd.DataFrame(mediumtChain.values).to_csv(outputfolder+'mediumtChain.csv')
    pd.DataFrame(softChain.values).to_csv(outputfolder+'softChain.csv')
    pd.DataFrame(softtChain.values).to_csv(outputfolder+'softtChain.csv')


def calculate_prior_infor(outputfolder='fig201710/realdata/'):
    """
    The function is to compare the prior influence for real data analysis.
    """
    # E0102 data
    tau1 = 0.025
    tau2 = 0.05
    O = pickle.load(open('realData/O.p', 'rb'))
    N, M = O.shape
    O1Res = pd.read_csv(outputfolder+'O1Res.csv', index_col=0)
    O2Res = pd.read_csv(outputfolder+'O2Res.csv', index_col=0)
    Ne1Res = pd.read_csv(outputfolder+'Ne1Res.csv', index_col=0)
    Ne2Res = pd.read_csv(outputfolder+'Ne2Res.csv', index_col=0)
    OW1 = 1/(1 + M*tau1**2/O1Res.values[(N+M):(N*M+N+M), 0])
    OW2 = 1/(1 + M*tau2**2/O2Res.values[(N+M):(N*M+N+M), 0])
    NeW1 = 1/(1 + M*tau1**2/Ne1Res.values[(N+M):(N*M+N+M), 0])
    NeW2 = 1/(1 + M*tau2**2/Ne2Res.values[(N+M):(N*M+N+M), 0])
    weight = pd.DataFrame(np.vstack((OW1, OW2, NeW1, NeW2)).T,
                          columns=['O, tau=0.025', 'O, tau=0.05', 'Ne, tau=0.025', 'Ne, tau=0.05'],
                          index=['RGS1', 'MOS1', 'MOS2', 'pn', 'ACIS-S3', 'ACIS-I3',
                                   'HETG', 'XIS0', 'XIS1', 'XIS2', 'XIS3', 'XRT-WT', 'XRT-PC'])

    weight.to_csv(outputfolder+'priorinfluenceO2Ne.csv')

    # 2XMM data
    hard2XMM1Res = pd.read_csv(outputfolder+'hard2XMM1Res.csv', index_col=0)
    hard2XMM2Res = pd.read_csv(outputfolder+'hard2XMM2Res.csv', index_col=0)
    medium2XMM1Res = pd.read_csv(outputfolder+'medium2XMM1Res.csv', index_col=0)
    medium2XMM2Res = pd.read_csv(outputfolder+'medium2XMM2Res.csv', index_col=0)
    soft2XMM1Res = pd.read_csv(outputfolder+'soft2XMM1Res.csv', index_col=0)
    soft2XMM2Res = pd.read_csv(outputfolder+'soft2XMM2Res.csv', index_col=0)

    hard2XMM = pickle.load(open('realData/hard2XMM.p', 'rb'))
    N, M = hard2XMM.shape
    hard2XMMW1 = 1 / (1 + M * tau1 ** 2 / hard2XMM1Res.values[(N + M):(N * M + N + M), 0])
    hard2XMMW2 = 1 / (1 + M * tau2 ** 2 / hard2XMM2Res.values[(N + M):(N * M + N + M), 0])

    medium2XMM = pickle.load(open('realData/medium2XMM.p', 'rb'))
    N, M = medium2XMM.shape
    medium2XMMW1 = 1 / (1 + M * tau1 ** 2 / medium2XMM1Res.values[(N + M):(N * M + N + M), 0])
    medium2XMMW2 = 1 / (1 + M * tau2 ** 2 / medium2XMM2Res.values[(N + M):(N * M + N + M), 0])

    soft2XMM = pickle.load(open('realData/soft2XMM.p', 'rb'))
    N, M = soft2XMM.shape
    soft2XMMW1 = 1 / (1 + M * tau1 ** 2 / soft2XMM1Res.values[(N + M):(N * M + N + M), 0])
    soft2XMMW2 = 1 / (1 + M * tau2 ** 2 / soft2XMM2Res.values[(N + M):(N * M + N + M), 0])

    weight = np.vstack((np.hstack((hard2XMMW1, hard2XMMW2)),
                        np.hstack((medium2XMMW1, medium2XMMW2)),
                        np.hstack((soft2XMMW1, soft2XMMW2))))

    weight.to_csv(outputfolder + 'priorinfluence2XMM.csv')

    # XCal data
    hard1Res = pd.read_csv(outputfolder+'hard1Res.csv', index_col=0)
    hard2Res = pd.read_csv(outputfolder+'hard2Res.csv', index_col=0)
    medium1Res = pd.read_csv(outputfolder+'medium1Res.csv', index_col=0)
    medium2Res = pd.read_csv(outputfolder+'medium2Res.csv', index_col=0)
    soft1Res = pd.read_csv(outputfolder+'soft1Res.csv', index_col=0)
    soft2Res = pd.read_csv(outputfolder+'soft2Res.csv', index_col=0)
    
    hard = pickle.load(open('realData/hard.p', 'rb'))
    N, M = hard.shape
    hardW1 = 1/(1 + M*tau1**2/hard1Res.values[(N+M):(N*M+N+M), 0])
    hardW2 = 1/(1 + M*tau2**2/hard2Res.values[(N+M):(N*M+N+M), 0])
    
    medium = pickle.load(open('realData/medium.p', 'rb'))
    N, M = medium.shape
    mediumW1 = 1/(1 + M*tau1**2/medium1Res.values[(N+M):(N*M+N+M), 0])
    mediumW2 = 1/(1 + M*tau2**2/medium2Res.values[(N+M):(N*M+N+M), 0])
    
    soft = pickle.load(open('realData/soft.p', 'rb'))
    N, M = soft.shape
    softW1 = 1/(1 + M*tau1**2/soft1Res.values[(N+M):(N*M+N+M), 0])
    softW2 = 1/(1 + M*tau2**2/soft2Res.values[(N+M):(N*M+N+M), 0])
    
    weight = np.vstack((np.hstack((hardW1, hardW2)),
                        np.hstack((mediumW1, mediumW2)),
                        np.hstack((softW1, softW2))))
    
    weight = pd.DataFrame(weight,
                          columns=['pn, tau=0.025', 'mos1, tau=0.025', 'mos2, tau=0.025',
                                   'pn, tau=0.05', 'mos1, tau=0.05', 'mos2, tau=0.05'],
                          index=['hard band 2XMM', 'medium band 2XMM', 'soft band 2XMM',
                                 'hard band', 'medium band', 'soft band'])

    weight.to_csv(outputfolder+'priorinfluenceXCAL.csv')


def makefig_real_data(outputfolder='fig201710/realdata/'):
    """
    The function is to make figures for the real data.
    """
    # E0102 data
    N = 13
    titles = ['O', 'Ne']
    labels = ['RGS1', 'MOS1', 'MOS2', 'pn', 'ACIS-S3', 'ACIS-I3',
              'HETG', 'XIS0', 'XIS1', 'XIS2', 'XIS3', 'XRT-WT', 'XRT-PC']
    O1Res = pd.read_csv(outputfolder+'O1Res.csv', index_col=0)
    O2Res = pd.read_csv(outputfolder+'O2Res.csv', index_col=0)
    Ne1Res = pd.read_csv(outputfolder+'Ne1Res.csv', index_col=0)
    Ne2Res = pd.read_csv(outputfolder+'Ne2Res.csv', index_col=0)
    
    data = np.zeros((2, N, 2, 3))
    d1 = O1Res.values
    d2 = O2Res.values
    data[0, :, 0, :] = d1[0:N, np.array([0, 3, 7])]
    data[0, :, 1, :] = d2[0:N, np.array([0, 3, 7])]
    data[0, :, 0, 1] = data[0, :, 0, 0] - data[0, :, 0, 1]
    data[0, :, 0, 2] = data[0, :, 0, 2] - data[0, :, 0, 0]
    data[0, :, 1, 1] = data[0, :, 1, 0] - data[0, :, 1, 1]
    data[0, :, 1, 2] = data[0, :, 1, 2] - data[0, :, 1, 0]
    
    d1 = Ne1Res.values
    d2 = Ne2Res.values
    data[1, :, 0, :] = d1[0:N, np.array([0, 3, 7])]
    data[1, :, 1, :] = d2[0:N, np.array([0, 3, 7])]
    data[1, :, 0, 1] = data[1, :, 0, 0] - data[1, :, 0, 1]
    data[1, :, 0, 2] = data[1, :, 0, 2] - data[1, :, 0, 0]
    data[1, :, 1, 1] = data[1, :, 1, 0] - data[1, :, 1, 1]
    data[1, :, 1, 2] = data[1, :, 1, 2] - data[1, :, 1, 0]
    
    index = np.arange(N)
    plt.figure(figsize=(16, 10))
    for i in np.arange(2):
        ax = plt.subplot(2, 1, i+1)
        plt.axhline(0, ls='--', color='#000000')
        plt.scatter(index - 0.125, data[i, :, 0, 0], color='#000000')
        plt.scatter(index + 0.125, data[i, :, 1, 0], color='#000000')
        plt.errorbar(index - 0.125, data[i, :, 0, 0], yerr=data[i, :, 0, 1:3].T,
                     fmt=None, ecolor='#000000')
        eb2 = plt.errorbar(index + 0.125, data[i, :, 1, 0], yerr=data[i, :, 1, 1:3].T,
                           fmt=None, ecolor='gray')
        eb2[-1][0].set_linestyle('-.')
        plt.xticks(index, labels)
        plt.xlim((-1, N))
        plt.title(titles[i])
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
    plt.tight_layout()
    plt.savefig(outputfolder+'example1.pdf')
    
    # 2XMM data
    hard2XMM1Res = pd.read_csv(outputfolder+'hard2XMM1Res.csv', index_col=0)
    hard2XMM2Res = pd.read_csv(outputfolder+'hard2XMM2Res.csv', index_col=0)
    medium2XMM1Res = pd.read_csv(outputfolder+'medium2XMM1Res.csv', index_col=0)
    medium2XMM2Res = pd.read_csv(outputfolder+'medium2XMM2Res.csv', index_col=0)
    soft2XMM1Res = pd.read_csv(outputfolder+'soft2XMM1Res.csv', index_col=0)
    soft2XMM2Res = pd.read_csv(outputfolder+'soft2XMM2Res.csv', index_col=0)
    
    N = 3
    titles = ['hard band', 'medium band', 'soft band']
    labels = ['pn', 'MOS1', 'MOS2']
    data = np.zeros((3, N, 2, 3))
    d1 = hard2XMM1Res.values
    d2 = hard2XMM2Res.values
    data[0, :, 0, :] = d1[0:N, np.array([0, 3, 7])]
    data[0, :, 1, :] = d2[0:N, np.array([0, 3, 7])]
    data[0, :, 0, 1] = data[0, :, 0, 0] - data[0, :, 0, 1]
    data[0, :, 0, 2] = data[0, :, 0, 2] - data[0, :, 0, 0]
    data[0, :, 1, 1] = data[0, :, 1, 0] - data[0, :, 1, 1]
    data[0, :, 1, 2] = data[0, :, 1, 2] - data[0, :, 1, 0]
    
    d1 = medium2XMM1Res.values
    d2 = medium2XMM2Res.values
    data[1, :, 0, :] = d1[0:N, np.array([0, 3, 7])]
    data[1, :, 1, :] = d2[0:N, np.array([0, 3, 7])]
    data[1, :, 0, 1] = data[1, :, 0, 0] - data[1, :, 0, 1]
    data[1, :, 0, 2] = data[1, :, 0, 2] - data[1, :, 0, 0]
    data[1, :, 1, 1] = data[1, :, 1, 0] - data[1, :, 1, 1]
    data[1, :, 1, 2] = data[1, :, 1, 2] - data[1, :, 1, 0]
    
    d1 = soft2XMM1Res.values
    d2 = soft2XMM2Res.values
    data[2, :, 0, :] = d1[0:N, np.array([0, 3, 7])]
    data[2, :, 1, :] = d2[0:N, np.array([0, 3, 7])]
    data[2, :, 0, 1] = data[2, :, 0, 0] - data[2, :, 0, 1]
    data[2, :, 0, 2] = data[2, :, 0, 2] - data[2, :, 0, 0]
    data[2, :, 1, 1] = data[2, :, 1, 0] - data[2, :, 1, 1]
    data[2, :, 1, 2] = data[2, :, 1, 2] - data[2, :, 1, 0]

    index = np.arange(N)
    plt.figure(figsize=(15, 4))
    for i in np.arange(3):
        ax = plt.subplot(1, 3, i+1)
        plt.axhline(0, ls='--', color='#000000')
        plt.scatter(index - 0.125, data[i, :, 0, 0], color='#000000')
        plt.scatter(index + 0.125, data[i, :, 1, 0], color='#000000')
        plt.errorbar(index - 0.125, data[i, :, 0, 0], yerr=data[i, :, 0, 1:3].T,
                     fmt=None, ecolor='#000000')
        eb2 = plt.errorbar(index + 0.125, data[i, :, 1, 0], yerr=data[i, :, 1, 1:3].T,
                         fmt=None, ecolor='gray')
        eb2[-1][0].set_linestyle('-.')
        plt.xticks(index, labels)
        plt.xlim((-1, N))
        plt.ylim((-0.10, 0.10))
        plt.title(titles[i])
                     
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
    
    plt.tight_layout()
    plt.savefig(outputfolder+'example2.pdf')
    
    # XCAL data
    hard1Res = pd.read_csv(outputfolder+'hard1Res.csv', index_col=0)
    hard2Res = pd.read_csv(outputfolder+'hard2Res.csv', index_col=0)
    medium1Res = pd.read_csv(outputfolder+'medium1Res.csv', index_col=0)
    medium2Res = pd.read_csv(outputfolder+'medium2Res.csv', index_col=0)
    soft1Res = pd.read_csv(outputfolder+'soft1Res.csv', index_col=0)
    soft2Res = pd.read_csv(outputfolder+'soft2Res.csv', index_col=0)

    ind = [1, 4, 17, 21]
    paras = pickle.load(open('realData/medium_paras.p', 'rb'))
    titles = ['PKS2155-304', '3C120', 'MS0737.9+7441', 'PKS2155-304']
    plt.figure(figsize=(15, 8))
    index = np.arange(5)
    colors = ['#000000', '#000000', '#000000', '#000000', '#000000']
    medium = pickle.load(open('realData/medium.p', 'rb'))
    data = medium
    d1 = medium1Res.values
    d2 = medium2Res.values
    for i in np.arange(4):
        plt.subplot(2, 2, i+1)
        y = np.array([data[0, ind[i]], data[1, ind[i]],
                      data[2, ind[i]], d1[3+ind[i], 0], d2[3+ind[i], 0]])
        yerr = np.zeros((2, 5))
        yerr[:, 0] = (paras['sigma0'])[0, ind[i]]*1.96
        yerr[:, 1] = (paras['sigma0'])[1, ind[i]]*1.96
        yerr[:, 2] = (paras['sigma0'])[2, ind[i]]*1.96
        yerr[0, 3] = d1[3+ind[i], 0] - d1[3+ind[i], 3]
        yerr[1, 3] = d1[3+ind[i], 7] - d1[3+ind[i], 0]
        yerr[0, 4] = d2[3+ind[i], 0] - d2[3+ind[i], 3]
        yerr[1, 4] = d2[3+ind[i], 7] - d2[3+ind[i], 0]
        
        plt.scatter(index, y, color=colors)
        for j in np.arange(5):
            plt.errorbar(index[j], y[j], yerr=yerr[:, j].reshape((2, 1)), fmt='none', ecolor=colors[j])
        plt.xticks(index, ['pn', 'MOS1', 'MOS2', r'$\tau_i=0.025$', r'$\tau_i=0.05$'])
        plt.title(titles[i])
    
    plt.show()
    
    N = 3
    titles = ['hard band', 'medium band', 'soft band']
    labels = ['pn', 'MOS1', 'MOS2']
    data = np.zeros((3, N, 2, 3))
    ylims = [(-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15)]
    d1 = hard1Res.values
    d2 = hard2Res.values
    data[0, :, 0, :] = d1[0:N, np.array([0, 3, 7])]
    data[0, :, 1, :] = d2[0:N, np.array([0, 3, 7])]
    data[0, :, 0, 1] = data[0, :, 0, 0] - data[0, :, 0, 1]
    data[0, :, 0, 2] = data[0, :, 0, 2] - data[0, :, 0, 0]
    data[0, :, 1, 1] = data[0, :, 1, 0] - data[0, :, 1, 1]
    data[0, :, 1, 2] = data[0, :, 1, 2] - data[0, :, 1, 0]
    
    d1 = medium1Res.values
    d2 = medium2Res.values
    data[1, :, 0, :] = d1[0:N, np.array([0, 3, 7])]
    data[1, :, 1, :] = d2[0:N, np.array([0, 3, 7])]
    data[1, :, 0, 1] = data[1, :, 0, 0] - data[1, :, 0, 1]
    data[1, :, 0, 2] = data[1, :, 0, 2] - data[1, :, 0, 0]
    data[1, :, 1, 1] = data[1, :, 1, 0] - data[1, :, 1, 1]
    data[1, :, 1, 2] = data[1, :, 1, 2] - data[1, :, 1, 0]
    
    d1 = soft1Res.values
    d2 = soft2Res.values
    data[2, :, 0, :] = d1[0:N, np.array([0, 3, 7])]
    data[2, :, 1, :] = d2[0:N, np.array([0, 3, 7])]
    data[2, :, 0, 1] = data[2, :, 0, 0] - data[2, :, 0, 1]
    data[2, :, 0, 2] = data[2, :, 0, 2] - data[2, :, 0, 0]
    data[2, :, 1, 1] = data[2, :, 1, 0] - data[2, :, 1, 1]
    data[2, :, 1, 2] = data[2, :, 1, 2] - data[2, :, 1, 0]

    index = np.arange(N)
    plt.figure(figsize=(15, 4))
    for i in np.arange(3):
        ax = plt.subplot(1, 3, i+1)
        plt.axhline(0, ls='--', color='#000000')
        plt.scatter(index - 0.125, data[i, :, 0, 0], color='#000000')
        plt.scatter(index + 0.125, data[i, :, 1, 0], color='#000000')
        plt.errorbar(index - 0.125, data[i, :, 0, 0], yerr=data[i, :, 0, 1:3].T,
                     fmt=None, ecolor='#000000')
        eb2=plt.errorbar(index + 0.125, data[i, :, 1, 0], yerr=data[i, :, 1, 1:3].T,
                         fmt=None, color='gray')
        eb2[-1][0].set_linestyle('-.')
        plt.xticks(index, labels)
        plt.xlim((-1, N))
        plt.ylim(ylims[i])
                     
        plt.title(titles[i])
                     
        for item in ([ax.title, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)

    plt.tight_layout()
    plt.savefig(outputfolder+'example3.pdf')


def makefig_real_data_residue(outputfolder='fig201710/realdata/'):
    """
    The function is to compare the residuals of the XCAL data for log_normal and log_t models.
    """
    hardChain=pd.read_csv(outputfolder+'hardChain.csv', index_col=0)
    hardtChain=pd.read_csv(outputfolder+'hardtChain.csv', index_col=0)
    mediumChain=pd.read_csv(outputfolder+'mediumChain.csv', index_col=0)
    mediumtChain=pd.read_csv(outputfolder+'mediumtChain.csv', index_col=0)
    softChain=pd.read_csv(outputfolder+'softChain.csv', index_col=0)
    softtChain=pd.read_csv(outputfolder+'softtChain.csv', index_col=0)
    hard2Res = pd.read_csv(outputfolder+'hard2Res.csv', index_col=0)
    hardtRes = pd.read_csv(outputfolder+'hard2tRes.csv', index_col=0)
    medium2Res = pd.read_csv(outputfolder+'medium2Res.csv', index_col=0)
    mediumtRes = pd.read_csv(outputfolder+'mediumtRes.csv', index_col=0)
    soft2Res = pd.read_csv(outputfolder+'soft2Res.csv', index_col=0)
    softtRes = pd.read_csv(outputfolder+'softtRes.csv', index_col=0)
    
    labelstemp = ['pn', 'MOS1', 'MOS2']
    markers = ['o', 's', 'd']
    colors = ['black', 'gray', 'silver']

    files = ['hard', 'medium', 'soft']
    titles = ['Hard band (log-Normal)', 'Medium band (log-Normal)', 'Soft band (log-Normal)']
    plt.figure(figsize=(20, 12))
    for i in np.arange(3):
        ax = plt.subplot(3, 2, 2*i+1)
        f = 'realData/' + '/' + files[i] + '.p'
        Y = pickle.load(open(f, 'rb'))
        N, M = Y.shape
        
        if i == 0:
            d = hard2Res.values
            sigmaFit = (np.sqrt(hardChain.values[:, (N+M):(2*N+M)]).mean(axis=0))**2
        
        elif i == 1:
            d = medium2Res.values
            sigmaFit = (np.sqrt(mediumChain.values[:, (N+M):(2*N+M)]).mean(axis=0))**2
        
        elif i == 2:
            d = soft2Res.values
            sigmaFit = (np.sqrt(softChain.values[:, (N+M):(2*N+M)]).mean(axis=0))**2

        else:
            return

        B = d[0:N, 0]
        G = d[N:(N+M), 0]
        fit = (B - sigmaFit*0.5)[:, np.newaxis] + G
        residual = Y - fit
        residual = residual/np.sqrt(sigmaFit[:, np.newaxis])
        
        index = np.arange(M)
        for j in np.arange(N):
            plt.scatter(index, residual[j, :], marker=markers[j],
                        edgecolors='face', color=colors[j])
            plt.xlim((-1, M))
            plt.axhline(3, ls='--', color='k')
            plt.axhline(-3, ls='--', color='k')
            plt.axhline(2, ls=':', color='k')
            plt.axhline(-2, ls=':', color='k')
            plt.yticks([-4, -2, 0, 2, 4])
            plt.xticks([])
            plt.ylim((-4.25, 4.25))
            
        plt.title(titles[i])
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(22)
    
    markers = ['o', 's', 'd']
    colors = ['black', 'gray', 'silver']
    files = ['hard', 'medium', 'soft']
    titles = ['Hard band (log-t)', 'Medium band (log-t)', 'Soft band (log-t)']
    for i in np.arange(3):
        ax = plt.subplot(3, 2, 2*i+2)
        f = 'realData/' + '/' + files[i] + '.p'
        Y = pickle.load(open(f, 'rb'))
        N, M = Y.shape
        
        if i == 0:
            d = hardtRes.values
            sigmaFit = \
                (np.sqrt(0.028**2*2/hardtChain.values[:, (N+M):(N+M+N*M)]).mean(axis=0).reshape((N, M)))**2
        
        elif i == 1:
            d = mediumtRes.values
            sigmaFit = \
                (np.sqrt(0.093**2*2/mediumtChain.values[:, (N+M):(N+M+N*M)]).mean(axis = 0).reshape((N, M)))**2
        
        elif i == 2:
            d = softtRes.values
            sigmaFit = \
                (np.sqrt(0.026**2*2/softtChain.values[:, (N+M):(N+M+N*M)]).mean(axis = 0).reshape((N, M)))**2

        else:
            return
        
        B = d[0:N, 0]
        G = d[N:(N+M), 0]
        fit = B[:, np.newaxis] + G - sigmaFit*0.5
        residual = Y - fit
        residual = residual/np.sqrt(sigmaFit)
        
        index = np.arange(M)
        for j in np.arange(N):
            plt.scatter(index, residual[j, :], marker=markers[j],
                        edgecolors='face', color=colors[j], label=labelstemp[j])
            plt.xlim((-1, M))
            plt.axhline(3, ls='--', color='k')
            plt.axhline(-3, ls='--', color='k')
            plt.axhline(2, ls=':', color='k')
            plt.axhline(-2, ls=':', color='k')
            plt.yticks([])
            plt.xticks([])
            plt.ylim((-4.25, 4.25))
        plt.title(titles[i])
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(22)
    plt.legend(ncol=3, bbox_to_anchor=(-0.28, 3.3, 3, .25), loc=3, fontsize=18)
    plt.tight_layout()
    plt.savefig(outputfolder+'residuelogNormalXCALlogtXCAL.pdf', bbox_inches="tight")

