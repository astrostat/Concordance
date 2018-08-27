import numpy as np


def mapFit(Y, sigma, b, tau):
    """
    The function to get posterior mean and variance of 'B' and 'G' under the log_normal model with known variance.

    Keyword arguments:
    Y -- logarithm of the counts.
         Each row is observations from a same instrument; each column is observations of a same object.
    sigma -- known standard deviation
    b -- prior mean for 'B'
    tau -- prior standard deviation for 'B'
    """
    N, M = Y.shape
    Omega = np.zeros((N+M, N+M))
    Gamma = np.zeros(shape=N+M)
    
    for i in np.arange(N):
        Omega[i, i] = tau[i]**(-2)
        Gamma[i] = b[i]*tau[i]**(-2)
        for j in np.arange(M):
            if not np.isinf(Y[i, j]):
                inv_sigma = sigma[i, j]**(-2)
                Omega[i, i] = Omega[i, i] + inv_sigma
                Omega[N+j, N+j] = Omega[N+j, N+j] + inv_sigma
                Omega[i, j+N] = inv_sigma
                Omega[j+N, i] = inv_sigma
                
                y_inv_sigma = Y[i, j]*inv_sigma + 0.5
                Gamma[i] = Gamma[i] + y_inv_sigma
                Gamma[j+N] = Gamma[j+N] + y_inv_sigma
                
    Mu = np.linalg.solve(Omega, Gamma)
    Sigma = np.linalg.inv(Omega)
        
    # Return the posterior mean of 'B', 'G' and the posterior covariance matrix.
    result = {'B': Mu[:N], 'G': Mu[N:], 'Sigma': Sigma}
    return result
