import numpy
import scipy.optimize
import math
import utils

def svm_wrap(D, zTrain):
    G = numpy.dot(D.T, D)
    Z = numpy.zeros((len(zTrain), len(zTrain)))

    for i in range(len(zTrain)):
        Z[i] = numpy.dot(zTrain[i], zTrain)

    H = Z * G

    def svm_obj(a):
        return 0.5 * numpy.dot(a.T, numpy.dot(H, a)) - numpy.dot(a.T, [1] * len(a))

    return svm_obj


def new_data_space(mTrain, K):
    ks = numpy.full((1, mTrain.shape[1]), K)

    return numpy.concatenate((mTrain, ks))


def gradient_svm_wrap(D, zTrain, C):
    G = numpy.dot(D.T, D)
    Z = numpy.zeros((len(zTrain), len(zTrain)))

    for i in range(len(zTrain)):
        Z[i] = numpy.dot(zTrain[i], zTrain)

    H = Z * G

    def gradient(a):
        return utils.vrow(numpy.dot(H, a) - [1] * len(a))

    return gradient


def SVM_train(trainM, trainV, C, K):
    zTrain = 2*trainV - 1
    D = new_data_space(trainM, K)

    svm_obj = svm_wrap(D, zTrain)
    x0 = numpy.zeros(trainM.shape[1])
    svm_gradient = gradient_svm_wrap(D, zTrain, C)
    bounds = [(0, C)] * trainM.shape[1]

    (a, f, _) = scipy.optimize.fmin_l_bfgs_b(func=svm_obj,
                                             fprime=svm_gradient, x0=x0, bounds=bounds)

    '''
    Slow method
    w_hat = numpy.zeros((D.shape[0], 1))

    for i in range(D.shape[1]):
        w_hat += a[i] * zTrain[i] * vcol(D[:, i])
    '''

    w_hat = utils.vcol(numpy.dot(D, (a * zTrain).T))
    
    
    '''
    primal_SVM = primal_SVM_wrap(D, zTrain, C)
    print('Primal Loss ' + str(primal_SVM(w_hat)))
    print('Dual Loss ' + str(-svm_obj(a)))
    print('Duality gap ' + str(primal_SVM(w_hat) + svm_obj(a)))
    '''
    w = w_hat[0:-1, :]
    b = w_hat[-1, :] * K
    return (w, b)


def primal_SVM_wrap(mTrain, vTrain, C):
    def primal_SVM(w):
        Sum = 0

        for i in range(mTrain.shape[1]):
            x = utils.vcol(mTrain[:, i])
            Sum += max(0, 1 - vTrain[i]*numpy.dot(w.T, x))

        return 0.5 * numpy.linalg.norm(w)**2 + C * Sum

    return primal_SVM


def SupportVectorMachine_val(matrix, vector, iterations, seed=0):
    numpy.random.seed(seed)
    idx = numpy.random.permutation(matrix.shape[1])
    matrixShuffle = matrix[:, idx]
    vectorShuffle = vector[idx]

    C = 0.1
    K = 1

    numElem = int(matrix.shape[1]/iterations)
    accuracy = []
    for i in range(iterations):
        ids = range(i*numElem, (i+1)*numElem)
        testM = matrixShuffle[:, ids]
        testV = vectorShuffle[ids]
        trainM = numpy.delete(matrixShuffle, ids, axis=1)
        trainV = numpy.delete(vectorShuffle, ids)

        alpha = SVM_train(trainM, trainV, C, K)
        res = SVM_test(alpha, testM, testV)
        accuracy.extend(res)

    print('Error rates ' + str((1 - sum(accuracy)/len(accuracy))*100))

    return


def SVM_test(w, b, matrixTest, vectorTest):
    S = []
    label = []

    for i in range(matrixTest.shape[1]):
        x = utils.vcol(matrixTest[:, i])
        S.append(numpy.dot(w.T, x) + b)

    for i in range(len(S)):
        label.append(1 if S[i] > 0 else 0)

    return vectorTest == label


def poly_kernel(x1, x2, c, d, eps):
    return (numpy.dot(x1.T, x2) + c)**d + eps


def radial_basis_kernel(x1, x2, gamma, eps):
    return math.e**(-gamma * numpy.linalg.norm(x1 - x2)**2) + eps


def svm_kernel_wrap_poly(D, zTrain, c, d, eps):
    '''
    Slow version
    G = numpy.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i][j] = poly_kernel(vcol(D[:, i]), vcol(D[:, j]), c, d, eps)
    '''
    G = poly_kernel(D, D, c, d, eps)
    
    '''
    Slow version
    Z = numpy.zeros((len(zTrain), len(zTrain)))
    
    for i in range(len(zTrain)):
        Z[i] = numpy.dot(zTrain[i], zTrain)
    '''
    Z = numpy.dot(zTrain.reshape(zTrain.size, 1), zTrain.reshape(1, zTrain.size))
    
    H = Z * G

    def svm_obj(a):
        return 0.5 * numpy.dot(a.T, numpy.dot(H, a)) - numpy.dot(a.T, [1] * len(a))

    return svm_obj


def svm_gradient_kernel_poly(D, zTrain, c, d, eps):
    G = poly_kernel(D, D, c, d, eps)

    Z = numpy.dot(zTrain.reshape(zTrain.size, 1), zTrain.reshape(1, zTrain.size))

    H = Z * G

    def gradient(a):
        return utils.vrow(numpy.dot(H, a) - [1] * len(a))

    return gradient


def svm_gradient_kernel_radial(G, zTrain):
    Z = numpy.dot(zTrain.reshape(zTrain.size, 1), zTrain.reshape(1, zTrain.size))

    H = Z * G

    def gradient(a):
        return utils.vrow(numpy.dot(H, a) - [1] * len(a))

    return gradient


def svm_kernel_wrap_radial(G, zTrain):
    Z = numpy.dot(zTrain.reshape(zTrain.size, 1), zTrain.reshape(1, zTrain.size))

    H = Z * G

    def svm_obj(a):
        return 0.5 * numpy.dot(a.T, numpy.dot(H, a)) - numpy.dot(a.T, [1] * len(a))

    return svm_obj


def SVM_kernel_train_poly(trainM, trainV, C, K, d, c):
    zTrain = 2*trainV - 1

    svm_obj = svm_kernel_wrap_poly(trainM, zTrain, c, d, K**2)
    x0 = numpy.zeros(trainM.shape[1])
    bounds = [(0, C)] * trainM.shape[1]

    svm_grad = svm_gradient_kernel_poly(trainM, zTrain, c, d, K**2)

    (a, f, _) = scipy.optimize.fmin_l_bfgs_b(func=svm_obj,
                                             x0=x0, bounds=bounds, fprime=svm_grad)

    return a

def SVM_compute_radial_kernel_train(trainM, K, gamma):
    n = trainM.shape[1]
    G = numpy.zeros((n, n))

    for i in range(n):
        for j in range(n):
            G[i][j] = radial_basis_kernel(
                utils.vcol(trainM[:, i]), utils.vcol(trainM[:, j]), gamma, K**2)
    
    return G

def SVM_compute_radial_kernel_test(matrixTrain, matrixTest, K, gamma):
    kern = numpy.zeros((matrixTrain.shape[1], matrixTest.shape[1]))
    
    for i in range(matrixTrain.shape[1]):
        for j in range(matrixTest.shape[1]):
            kern[i, j] = radial_basis_kernel(matrixTrain[:, i], matrixTest[:, j], gamma, K**2)
    
    return kern


def SVM_kernel_train_radial(trainM, zTrain, C, K, gamma, G=[]):
    if G == []: #parameters not passed (used for optimization)
        G = SVM_compute_radial_kernel_train(trainM, K, gamma)
    
    svm_obj = svm_kernel_wrap_radial(G, zTrain)
    x0 = numpy.zeros(trainM.shape[1])
    bounds = [(0, C)] * trainM.shape[1]

    svm_grad = svm_gradient_kernel_radial(G, zTrain)

    (a, _, _) = scipy.optimize.fmin_l_bfgs_b(func=svm_obj,
                                             x0=x0, bounds=bounds, fprime=svm_grad )

    return a

def SVM_kernel_test_poly(matrixTrain, vectorTrain, a, matrixTest, vectorTest, K, d, c):
    S = []
    label = []
    zTrain = 2*vectorTrain - 1

    for i in range(matrixTest.shape[1]):
        x = utils.vcol(matrixTest[:, i])
        
        '''
        Slow version
        score = 0

        for j in range(matrixTrain.shape[1]):
            xj = vcol(matrixTrain[:, j])
            score += a[j] * zTrain[j] * poly_kernel(xj, x, c, d, K**2)
            
        '''
        score = numpy.sum(a * zTrain * poly_kernel(matrixTrain, x, c, d, K**2))
        
        S.append(score)
        
    '''
    Slow versions
    for i in range(len(S)):
        label.append(1 if S[i] > 0 else 0)
    '''
    
    label = 1 * (S > 0)
    
    return vectorTest == label


def SVM_kernel_test_radial(matrixTrain, vectorTrain, a, matrixTest, vectorTest, K, gamma):
    S = []
    label = []
    zTrain = 2*vectorTrain - 1

    for i in range(matrixTest.shape[1]):
        x = utils.vcol(matrixTest[:, i])

        score = 0

        for j in range(matrixTrain.shape[1]):
            xj = utils.vcol(matrixTrain[:, j])
            score += a[j] * zTrain[j] * radial_basis_kernel(xj, x, gamma, K**2)

        S.append(score)
        
    label = 1 * (S > 0)

    return vectorTest == label

#we need three LLR: one linear and two kernel
#note that the name LLR is improper since with SVM there is no probabilistic interpretation
def linearSVM_LLR(w, b, matrixTest):
    S = []

    for i in range(matrixTest.shape[1]):
        x = utils.vcol(matrixTest[:, i])
        score = numpy.dot(w.T, x) + b
        S.append(score.item())
    return S

def SVM_kernel_poly_LLR(matrixTrain, vectorTrain, a, matrixTest, K, d, c):
    zTrain = 2*vectorTrain - 1

    '''
    S = []
    #Slow version
    for i in range(matrixTest.shape[1]):
        x = vcol(matrixTest[:, i])
        
        score = 0

        for j in range(matrixTrain.shape[1]):
            xj = vcol(matrixTrain[:, j])
            score += a[j] * zTrain[j] * poly_kernel(xj, x, c, d, K**2)
        
        
        S.append(score)
    '''
    
    S = numpy.sum(
            numpy.dot(
                    (a * zTrain).reshape(1, matrixTrain.shape[1]),
                    poly_kernel(matrixTrain, matrixTest, c, d, K**2)
            ), axis=0)
    
    return S


def SVM_kernel_radial_LLR(M_train, M_val, K, gamma, zTrain, a, kern=[]):
    '''
    # Slower version
    S = []
    for i in range(matrixTest.shape[1]):
        x = vcol(matrixTest[:, i])

        score = 0

        for j in range(matrixTrain.shape[1]):
            xj = vcol(matrixTrain[:, j])
            score += a[j] * zTrain[j] * radial_basis_kernel(xj, x, gamma, K**2)

        S.append(score)
    '''
    
    if kern == []:
        kern = SVM_compute_radial_kernel_test(M_train, M_val, K, gamma)
    
    S = numpy.sum(numpy.dot(a * utils.vrow(zTrain), kern), axis=0)
    
    return S