import scipy.linalg
import numpy
import utils

def weighted_logreg_obj_wrap(mTrain, vTrain, Lam, prior):
    Z = 2 * vTrain - 1
    n0 = (mTrain[:, vTrain == 0]).shape[1]
    n1 = mTrain.shape[1] - n0
    
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        firstTerm = Lam/2 * numpy.linalg.norm(w)**2
        
        ''''
        Slow version with for loop, replace using logaddexp on matrix
        risk_0 = 0
        risk_1 = 0
        for i in range(mTrain.shape[1]):
            x = vcol(mTrain[:, i])
            
            if vTrain[i] == 0:
                risk_0 += numpy.logaddexp(0, -Z[i]*(numpy.dot(w.T, x) + b))
            else:
                risk_1 += numpy.logaddexp(0, -Z[i]*(numpy.dot(w.T, x) + b))
        '''
        score = (numpy.dot(w.T, mTrain) + b).ravel() #flat the matrix into an array
        
        #take score of samples which true class is zero and multiply it by Z
        risk_0 = (numpy.logaddexp(0, -score[vTrain == 0] * Z[vTrain == 0])).sum()
        risk_1 = (numpy.logaddexp(0, -score[vTrain == 1] * Z[vTrain == 1])).sum()
        
        return firstTerm + (prior/n1)*risk_1 + (1-prior)/n0*risk_0
        
    return logreg_obj

def logistic_regression_train(mTrain, vTrain, Lam,prior):
    logreg_obj = weighted_logreg_obj_wrap(mTrain, vTrain, Lam, prior)
    x0 = numpy.zeros(mTrain.shape[0] + 1)
    (v, _, _) = scipy.optimize.fmin_l_bfgs_b(
        func=logreg_obj, x0=x0, approx_grad=True)

    return (v[0: -1], v[-1])

def logistic_regression_calibration(mTrain, vTrain, mTest, lam, prior):
    logreg_obj = weighted_logreg_obj_wrap(mTrain, vTrain, lam, prior)
    v, _, _ = scipy.optimize.fmin_l_bfgs_b(logreg_obj, 
                                              x0=numpy.zeros(mTrain.shape[0] + 1), approx_grad=True)
    w = v[0:-1]
    b = v[-1]
    calibration = numpy.log(prior / (1 - prior))
    cal_m = numpy.dot(w.T, mTest) + b - calibration
    return cal_m, w, b


def logistic_regression_test(w, b, matrixTest, vectorTest):
    # compute for each element the posterior and use the vectorTest
    S = []
    label = []

    for i in range(matrixTest.shape[1]):
        x = utils.vcol(matrixTest[:, i])
        S.append(numpy.dot(w.T, x) + b)

    for i in range(len(S)):
        label.append(1 if S[i] > 0 else 0)

    return vectorTest == label


def expanded_feature_space(samples):
    result = []
    '''
    Original slow implementation
    for i in range(samples.shape[1]):
        x = vcol(samples[:, i])
        prod = vcol(numpy.dot(x, x.T).flatten())
        
        if i == 0:
            result = numpy.concatenate((prod, x))
        else:
            result = numpy.hstack((result, (numpy.concatenate((prod, x)))))
    '''
    
    #new using apply_along_axis
    def vecxxT(x):
            x = x[:, None]
            prod = numpy.dot(x, x.T).reshape(x.size ** 2, order='F')
            return prod
    
    result = numpy.apply_along_axis(vecxxT, 0, samples)
    #add the original matrix as last row
    phi = numpy.vstack((result, samples))
    
    return phi

def quad_logistic_regression_train(mTrain, vTrain, Lam=1, prior=0.5):
    logreg_obj = weighted_logreg_obj_wrap(mTrain, vTrain, Lam, prior)
    x0 = numpy.zeros(mTrain.shape[0] + 1)
    (v, f, _) = scipy.optimize.fmin_l_bfgs_b(func = logreg_obj, x0 = x0, approx_grad = True, )
    
    return (v[0 : -1], v[-1])

def quad_logistic_regression_test(w, b, matrixTest, vectorTest):
    # compute for each element the posterior and use the vectorTest
    S = []
    label = []
    
    for i in range(matrixTest.shape[1]):
        x = utils.vcol(matrixTest[:, i])
        S.append(numpy.dot(w.T, x) + b)
    
    for i in range(len(S)):
        label.append(1 if S[i] > 0 else 0)
    
    return vectorTest == label

#LLR
def logistic_regression_LLR(w, b, matrixTest):
    # compute for each element the posterior
    S = []

    for i in range(matrixTest.shape[1]):
        x = utils.vcol(matrixTest[:, i])
        score = numpy.dot(w.T, x) + b
        S.append(score.item())

    return S


def quad_logistic_regression_LLR(w, b, matrixTest):
    # compute for each element the posterior
    S = []
    
    for i in range(matrixTest.shape[1]):
        x = utils.vcol(matrixTest[:, i])
        score = numpy.dot(w.T, x) + b
        S.append(score.item())
    
    return S