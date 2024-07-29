import numpy
import scipy.optimize
import utils

#gmm = [[weight, mean, cov] ...]
def logpdf_GMM(X, gmm):
    S = numpy.zeros((len(gmm), X.shape[1]))
    
    for i in range(len(gmm)):
        (w, mu, C) = gmm[i]
        S[i, :] = utils.logpdf_GAU_ND(X, mu, C) + numpy.log(w)
        
    logdens = scipy.special.logsumexp(S, axis=0)
    return (logdens, S)

def Estep(logdens, S):
    return numpy.exp(S-logdens.reshape(1, logdens.size))

def Mstep(X, S, posterior, G):
    psi = 0.01
    N = X.shape[1]
    
    gmmNew = []
    for g in range(G):
        gamma = posterior[g, :]
        
        Z = gamma.sum()
        F = (utils.mrow(gamma)*X).sum(1)
        S = numpy.dot(X, (utils.mrow(gamma)*X).T)
        w = Z/N
        mu = utils.mcol(F/Z)
        Sigma = S/Z - numpy.dot(mu, mu.T)
        U, s, _ = numpy.linalg.svd(Sigma)
        s[s<psi] = psi
        Sigma = numpy.dot(U, utils.mcol(s)*U.T)
        gmmNew.append((w, mu, Sigma))
    return gmmNew


def EMalgorithm(trainM, gmm, typeGMM):
    # flag false if diff becomes smaller than 10^(-6)
    flag = True
    
    (logdens, S) = logpdf_GMM(trainM, gmm)
    oldloglikelihood = numpy.sum(logdens)/trainM.shape[1]
    
    while(flag):
        posterior = Estep(logdens, S)
        
        #different Mstep depending on the type
        if (typeGMM == 'Full'):
            gmm = Mstep(trainM, S, posterior, len(gmm))
        elif (typeGMM == 'Diag'):
            gmm = DiagMstep(trainM, S, posterior, len(gmm))
        elif (typeGMM == 'Tied'):
            gmm = TiedMstep(trainM, S, posterior, len(gmm))
        
        
        (logdens, S) = logpdf_GMM(trainM, gmm)
        loglikelihood = numpy.sum(logdens)/trainM.shape[1]
        
        if (loglikelihood - oldloglikelihood < 10**(-6)): #finished
            flag = False
        elif (loglikelihood-oldloglikelihood < 0):
            print("problem: log likelihood is not increasing")    
        else:
            #update likelihood
            oldloglikelihood = loglikelihood
        
    return gmm

#implementation of the two variations of the Mstep
#with diagonal and tied covariance matrix
def TiedMstep(X, S, posterior, G):
    psi = 0.01
    N = X.shape[1]
    
    gmmNew = []
    sigmaTied = numpy.zeros((X.shape[0],X.shape[0]))
    for g in range(G):
        gamma = posterior[g, :]
        
        Z = gamma.sum()
        F = (utils.mrow(gamma)*X).sum(1)
        S = numpy.dot(X, (utils.mrow(gamma)*X).T)
        
        w = Z/N
        mu = utils.mcol(F/Z)
        #tied
        Sigma = S/Z - numpy.dot(mu, mu.T)
        sigmaTied += Z * Sigma
        
        gmmNew.append((w, mu))
    
    #add the tied
    sigmaTied /= N
    U, s, _ = numpy.linalg.svd(sigmaTied)
    s[s<psi] = psi
    sigmaTied = numpy.dot(U, utils.mcol(s)*U.T)
    
    for g in range(G):
        (w,mu)=gmmNew[g]
        gmmNew[g] = ((w,mu,sigmaTied))
    
    return gmmNew


def DiagMstep(X, S, posterior, G):
    psi = 0.01
    N = X.shape[1]
    
    gmmNew = []
    for g in range(G):
        gamma = posterior[g, :]
        
        Z = gamma.sum()
        F = (utils.mrow(gamma)*X).sum(1)
        S = numpy.dot(X, (utils.mrow(gamma)*X).T)
        w = Z/N
        mu = utils.mcol(F/Z)
        Sigma = S/Z - numpy.dot(mu, mu.T)
        Sigma = Sigma * numpy.eye(Sigma.shape[0])
        U, s, _ = numpy.linalg.svd(Sigma)
        s[s<psi] = psi
        Sigma = numpy.dot(U, utils.mcol(s)*U.T)
        gmmNew.append((w, mu, Sigma))
    return gmmNew

#compute a new set of Gaussians by doubling a given set
def split(GMM):
    alpha = 0.1
    splittedGMM = []
    for i in range(len(GMM)):
        U, s, Vh = numpy.linalg.svd(GMM[i][2])
        # compute displacement vector
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]+d, GMM[i][2]))
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]-d, GMM[i][2]))
    
    return splittedGMM

def LBGalgorithm(GMM, trainM, iterations, typeGMM):
    GMM = EMalgorithm(trainM, GMM, typeGMM)
    for i in range(iterations):
        GMM = split(GMM)
        GMM = EMalgorithm(trainM, GMM, typeGMM)
    return GMM

def compute_posterior(X, gmm):
    S = numpy.zeros((len(gmm), X.shape[1]))
    
    for g in range(len(gmm)):
        (w, mu, C) = gmm[g]
        S[g, :] = utils.logpdf_GAU_ND(X, mu, C) + numpy.log(w)
        
    logdens = scipy.special.logsumexp(S, axis=0)
    
    return logdens

#compT and compNT are the number of iterations for target and not-target gmms
def GMM_train (matrixTrain, vectorTrain, compT, compNT, typeT='Full', typeNT='Full'):
    m0 = matrixTrain[:, vectorTrain==0]
    m1 = matrixTrain[:, vectorTrain==1]
   
    GMM0_init = [(1.0, m0.mean(axis=1).reshape((m0.shape[0], 1)), utils.constrainSigma(numpy.cov(m0).reshape((m0.shape[0], m0.shape[0]))))]
    GMM1_init = [(1.0, m1.mean(axis=1).reshape((m1.shape[0], 1)), utils.constrainSigma(numpy.cov(m1).reshape((m1.shape[0], m1.shape[0]))))]
   

    GMM0 = LBGalgorithm(GMM0_init, m0, compNT, typeNT)
    GMM1 = LBGalgorithm(GMM1_init, m1, compT, typeT)
    
    return [GMM0, GMM1]
        
def GMM_test(GMMs, matrixTest):
    post0 = compute_posterior(matrixTest, GMMs[0])
    post1 = compute_posterior(matrixTest, GMMs[1]) 
    
    res = numpy.vstack((post0,post1))
    return numpy.argmax(res, axis=0)

def GMM_LLR(GMMs, matrixTest):
    post0 = compute_posterior(matrixTest, GMMs[0])
    post1 = compute_posterior(matrixTest, GMMs[1])
    
    return post1 - post0