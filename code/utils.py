import numpy
import math
import matplotlib.pyplot as plt
import LogReg
import seaborn as sns #import seaborn to plot heatmap

def vrow(v):
    return v.reshape(1, v.shape[0])

def vcol(v):
    return v.reshape(v.shape[0], 1)

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

def load(f):
    v = []
    m = []

    for line in f:
        l = line.rstrip().split(',')
        conversion = [float(i) for i in l[0:10]] #10 dimension
        m.append(conversion)
        v.append(int(l[10]))

    vector = numpy.array(v)
    matrix = numpy.array(m)

    matrix = matrix.T

    return matrix, vector

def plot_scatter(m, v):
    m0 = m[:, v==0]
    m1 = m[:, v==1]
    
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            if(i == j):
                continue
            else:
                plt.figure()
                plt.xlabel(str(i))
                plt.ylabel(str(j))
                plt.scatter(m0[i, :], m0[j, :], color='r')
                plt.scatter(m1[i, :], m1[j, :], color='b')

                plt.legend()
                plt.tight_layout()
            plt.show()


def print_feature(m, v):
    for i in range(m.shape[0]):
        plt.figure()
        plt.xlabel('Feature ' + str(i + 1))
        plt.ylabel('Frequency')
        m0 = m[:, v == 0]
        m1 = m[:, v == 1]
        plt.hist(m0[i, :], bins='auto', density=True, edgecolor="black", alpha = 0.5, label="spoofed fingerprint")
        plt.hist(m1[i, :], bins='auto', density=True, edgecolor="black", alpha = 0.5, label="authentic fingerprint")

        plt.legend()
        # Use with non-default font size to keep axis label inside the figure
        plt.tight_layout()
    plt.show()
    return


def dimension_reduction_PCA(matrix, m):
    mu = matrix.mean(1)
    matrixCentered = matrix - mu.reshape(matrix.shape[0], 1)

    C = numpy.dot(matrixCentered, matrixCentered.T)/float(matrix.shape[1])
    U, _, _ = numpy.linalg.svd(C)
    P = U[:, 0:m]

    return P


def maximum_likehood_estimate(x):
    N = x.shape[1]

    muML = vcol(numpy.sum(x, axis=1)/N)

    xCentered = x - vcol(muML)
    CML = numpy.dot(xCentered, xCentered.T)/N

    return (muML, CML)

def print_heat_map(m, v):
    #we can use the package seaborn to plot our heat map
    plt.figure()
    sns.heatmap(numpy.corrcoef(m), linewidth=0.2, cmap="Greys", square=True, cbar=False)
    plt.figure()
    sns.heatmap(numpy.corrcoef(m[:, v==0]), linewidth=0.2, cmap="Reds", square=True,cbar=False)
    plt.figure()
    sns.heatmap(numpy.corrcoef(m[:, v==1]), linewidth=0.2, cmap="Blues", square=True, cbar=False)
    return

def plot_conf_M(predict, actualC, numC):
    Conf = numpy.zeros((numC, numC))

    for i in range(len(predict)):
        Conf[predict[i]][actualC[i]] += 1

    return Conf

def logpdf_GAU_ND(x, mu, C):
    costant = mu.shape[0]*(-0.5)*math.log(2*math.pi)
    inverseC = numpy.linalg.inv(C)
    _, logDetC = numpy.linalg.slogdet(C)
    costantResult = costant - 0.5*logDetC
    final = numpy.zeros(shape=(x.shape[1]))

    for i in range(x.shape[1]):
        element = x[:, i]
        elemCenter = (vcol(element) - mu)
        result = costantResult - 0.5 * \
            numpy.dot(elemCenter.T, numpy.dot(inverseC, elemCenter))
        final[i] = result

    return final

def k_split(matrix, vector, k, seed=0):
    numpy.random.seed(seed)
    idx = numpy.random.permutation(matrix.shape[1])
    matrixShuffle = matrix[:, idx]
    vectorShuffle = vector[idx]
    
    numElem = int(matrix.shape[1]/k)
    
    folds = []
    labels = []
    
    for i in range(k):
        folds.append(matrixShuffle[:,idx[(i*numElem): ((i+1)*(numElem))]])
        labels.append(vectorShuffle[idx[(i*numElem): ((i+1)*(numElem))]])
        
    return (folds, labels)

def z_norm(mT, mV):
    mean = mT.mean(axis=1)
    standardDeviation = mT.std(axis=1)
    zmT = (mT-mcol(mean))/mcol(standardDeviation)
    zmV = (mV-mcol(mean))/mcol(standardDeviation)
    return (zmT, zmV)

def constrainSigma(sigma):
    psi = 0.01
    U, s, Vh = numpy.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = numpy.dot(U, mcol(s)*U.T)
    return sigma

def calibrate_scores(scoresTrain, labelTrain, scoresTest, Lam, prior):
    (w, b) = LogReg.logistic_regression_train(scoresTrain, labelTrain, Lam, prior)
    calibration = numpy.log(prior / (1 - prior))
    
    cal_scores = numpy.dot(w.T, scoresTest) + b - calibration
    return (cal_scores, w, b)