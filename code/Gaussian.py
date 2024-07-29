import scipy.linalg
import numpy
import utils

def vrow(v):
    return v.reshape(1, v.shape[0])


def vcol(v):
    return v.reshape(v.shape[0], 1)

def maximum_likehood_estimate(x):
    N = x.shape[1]

    muML = vcol(numpy.sum(x, axis=1)/N)

    xCentered = x - vcol(muML)
    CML = numpy.dot(xCentered, xCentered.T)/N

    return (muML, CML)

def Multivariate_Gaussian_Classifier_Train(matrix, vectorLabel):
    muclasses = []
    Sigmaclasses = []
    
    for i in range(vectorLabel.max()+1):
        elemLabel = matrix[:, vectorLabel == i]
        (mu, var) = maximum_likehood_estimate(elemLabel)
        
        muclasses.append(mu)
        Sigmaclasses.append(var)

    return (muclasses, Sigmaclasses)

def Multivariate_Gaussian_Classifier_Test(matrixTest, vectorTest, modelParameters, priorProb):
    (listMu, listVar) = modelParameters
    numLabel = len(listMu)
    
    scoreMatrix = []
    scoreMatrixJoint = numpy.zeros((numLabel, matrixTest.shape[1]), dtype=float)
    
    for j in range(numLabel):
        scoreMatrix.append(utils.logpdf_GAU_ND(matrixTest, listMu[j], listVar[j]))
    
    #sum of log of prior
    priorMatrix = numpy.log(numpy.vstack(priorProb))
    scoreMatrixJoint = scoreMatrix + priorMatrix
    
    MarginalDensity = vrow(scipy.special.logsumexp(scoreMatrixJoint, axis = 0))
    
    posteriorProbabilityLog = scoreMatrixJoint - MarginalDensity
    
    posteriorProbability = numpy.exp(posteriorProbabilityLog)
    
    finalLabelElement = numpy.argmax(posteriorProbability, axis=0) 
    
    return finalLabelElement == vectorTest

def Naive_Bayes_Gaussian_Classifier_Train(matrix, vectorLabel):
    muclasses = []
    Sigmaclasses = []
    
    for i in range(vectorLabel.max()+1):
        elemLabel = matrix[:, vectorLabel == i]
        (mu, var) = maximum_likehood_estimate(elemLabel)
        
        muclasses.append(mu)
        
        varD = numpy.diag(numpy.diag(var)) #naive hypotesis

        Sigmaclasses.append(varD)

    return (muclasses, Sigmaclasses)

def Naive_Bayes_Gaussian_Classifier_Test(matrixTest, vectorTest, modelParameters, priorProb):
    #same structure, we can use again the mvg test
    return Multivariate_Gaussian_Classifier_Test(matrixTest, vectorTest, modelParameters, priorProb)

def Tied_Covariance_Gaussian_Classifier_Train(matrix, vectorLabel):
    muclasses = []
    Sigma = numpy.zeros(matrix.shape[0])
    
    for i in range(vectorLabel.max()+1):
        elemLabel = matrix[:, vectorLabel == i]
        (mu, var) = maximum_likehood_estimate(elemLabel)
        
        muclasses.append(mu)

        Sigma = Sigma + elemLabel.shape[1] * var
    
    Sigma = Sigma/matrix.shape[1]
    return (muclasses, Sigma)

def Tied_Covariance_Gaussian_Classifier_Test(matrixTest, vectorTest, modelParameters, priorProbability):
    #same structure, we can use again the mvg test, we just use the Sigma for all classes
    (listMu, covarianceM) = modelParameters
    
    listVar = [covarianceM] * len(listMu)
    
    newModelPar = (listMu, listVar)
    return Multivariate_Gaussian_Classifier_Test(matrixTest, vectorTest, newModelPar, priorProbability)

def Tied_Naive_Gaussian_Classifier_Train(matrix, vectorLabel):
    muclasses = []
    Sigma = numpy.zeros(matrix.shape[0])
    
    for i in range(vectorLabel.max()+1):
        elemLabel = matrix[:, vectorLabel == i]
        (mu, var) = maximum_likehood_estimate(elemLabel)
        
        muclasses.append(mu)

        Sigma = Sigma + elemLabel.shape[1] * var
    
    Sigma = numpy.diag(numpy.diag(Sigma/matrix.shape[1]))
    
    return (muclasses, Sigma)

def Tied_Naive_Gaussian_Classifier_Test(matrixTest, vectorTest, modelParameters, priorProbability):
    (listMu, covarianceM) = modelParameters
    
    listVar = [covarianceM] * len(listMu)
    
    newModelPar = (listMu, listVar)
    return Multivariate_Gaussian_Classifier_Test(matrixTest, vectorTest, newModelPar, priorProbability)

## for bayes plot we create functions that return LLR
def Multivariate_Gaussian_Classifier_LLR(matrixTest, modelParameters, priorProb):
    (listMu, listVar) = modelParameters
    numLabel = len(listMu)
    
    scoreMatrix = []
    
    for j in range(numLabel):
        scoreMatrix.append(utils.logpdf_GAU_ND(matrixTest, listMu[j], listVar[j]))
    
    return scoreMatrix[1] - scoreMatrix[0]

def Naive_Bayes_Gaussian_Classifier_LLR(matrixTest, modelParameters, priorProb):
    #same structure, we can use again the mvg test
    return Multivariate_Gaussian_Classifier_LLR(matrixTest, modelParameters, priorProb)

def Tied_Covariance_Gaussian_Classifier_LLR(matrixTest, modelParameters, priorProbability):
    #same structure, we can use again the mvg test, we just use the Sigma for all classes
    (listMu, covarianceM) = modelParameters
    
    listVar = [covarianceM] * len(listMu)
    
    newModelPar = (listMu, listVar)
    return Multivariate_Gaussian_Classifier_LLR(matrixTest, newModelPar, priorProbability)

def Tied_Naive_Gaussian_Classifier_LLR(matrixTest, modelParameters, priorProbability):
    (listMu, covarianceM) = modelParameters
    
    listVar = [covarianceM] * len(listMu)
    
    newModelPar = (listMu, listVar)
    return Multivariate_Gaussian_Classifier_LLR(matrixTest, newModelPar, priorProbability)
