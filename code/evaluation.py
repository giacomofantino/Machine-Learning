import math
import numpy
import matplotlib.pyplot as plt
import utils
import Gaussian
import LogReg
import SVM
import GMM
import metrics

#where calibration is used we split the training set
#80% is used for training, the remaining 20% for calibration

def MVG_eval(mTrain, Ltrain, mTest, vTest):
    prior_prob=[0.09, 0.91]
    dimension=[10, 9, 8, 7]
    
    scores_MVG=[[] for _ in range(len(dimension))]
    scores_tied=[[] for _ in range(len(dimension))]
    scores_naive=[[] for _ in range(len(dimension))]
    scores_tiednaive=[[] for _ in range(len(dimension))]
    
    #for each dimension
    for j in range(len(dimension)):
        if j != 0:
            P = utils.dimension_reduction_PCA(mTrain, dimension[j])
            M_train = numpy.dot(P.T, mTrain)
            M_eval = numpy.dot(P.T, mTest)
        else:
            M_train = mTrain
            M_eval = mTest
        
        parameters_mvg = Gaussian.Multivariate_Gaussian_Classifier_Train(M_train, Ltrain)
        scores_MVG[j].append(Gaussian.Multivariate_Gaussian_Classifier_LLR(M_eval, parameters_mvg, prior_prob))
        
        parameters_naive = Gaussian.Naive_Bayes_Gaussian_Classifier_Train(M_train, Ltrain)
        scores_naive[j].append(Gaussian.Naive_Bayes_Gaussian_Classifier_LLR(M_eval, parameters_naive, prior_prob))
        
        parameters_tied = Gaussian.Tied_Covariance_Gaussian_Classifier_Train(M_train, Ltrain)
        scores_tied[j].append(Gaussian.Tied_Covariance_Gaussian_Classifier_LLR(M_eval, parameters_tied, prior_prob))
        
        parameters_tiednaive = Gaussian.Tied_Naive_Gaussian_Classifier_Train(M_train, Ltrain)
        scores_tiednaive[j].append(Gaussian.Tied_Naive_Gaussian_Classifier_LLR(M_eval, parameters_tiednaive, prior_prob))
        
    
    #compute minDCF for each classifier and for each dimension
    for j in range(len(dimension)):
        scoresMVG = numpy.hstack(scores_MVG[j])
        minD = metrics.minimum_detection_costs(scoresMVG, vTest, prior_prob[0], 1, 1)
        print('MVG dimension: ' + str(dimension[j]) + ' minDCF = ' + str(minD))
        
        scoresNAIVE = numpy.hstack(scores_naive[j])
        minD = metrics.minimum_detection_costs(scoresNAIVE, vTest, prior_prob[0], 1, 1)
        print('Naive dimension: ' + str(dimension[j]) + ' minDCF = ' + str(minD))
        
        scoresTIED = numpy.hstack(scores_tied[j])
        minD = metrics.minimum_detection_costs(scoresTIED, vTest, prior_prob[0], 1, 1)
        print('Tied dimension: ' + str(dimension[j]) + ' minDCF = ' + str(minD))
        
        scoresTiedNaive = numpy.hstack(scores_tiednaive[j])
        minD = metrics.minimum_detection_costs(scoresTiedNaive, vTest, prior_prob[0], 1, 1)
        print('Tied Naive dimension: ' + str(dimension[j]) + ' minDCF = ' + str(minD))
    return

def QuadLog_reg_computecosts(mTrain, Ltrain, mTest, vTest):
    '''
    #chosen
    Lambda = 0.001623776739188721
    prior_train=(mTrain[:, Ltrain == 1]).shape[1] / mTrain.shape[1] #emp prior
    dimension = 6
    prior_cal = 0.09
    '''
    #optmial
    Lambda = 0.00379269019073225
    prior_train = (mTrain[:, Ltrain == 1]).shape[1] / mTrain.shape[1]
    dimension = 6
    prior_cal = 0.09
    
    
    #shuffling
    numpy.random.seed(0)
    idxTrain = numpy.random.permutation(mTrain.shape[1])
    mTrain = mTrain[:, idxTrain]
    Ltrain = Ltrain[idxTrain]
    
    if dimension != 10:
        P = utils.dimension_reduction_PCA(mTrain, dimension)
        M_train = numpy.dot(P.T, mTrain)
        M_eval = numpy.dot(P.T, mTest)
    else:
        M_train = mTrain
        M_eval = mTest
        
    
    M_train = LogReg.expanded_feature_space(M_train)
    M_eval = LogReg.expanded_feature_space(M_eval)
    
    matrixModel = M_train[:, range(0, int(M_train.shape[1]*0.8))]
    labelModel = Ltrain[range(0, int(M_train.shape[1]*0.8))]
    matrixCal = M_train[:, range(int(M_train.shape[1]*0.8), M_train.shape[1])]
    labelCal = Ltrain[range(int(M_train.shape[1]*0.8), M_train.shape[1])]
    
    scores_calibration = [] #we will use this to train the LR
    scores = [] #these are the scores that will be calibrated
    
    (w, b) = LogReg.quad_logistic_regression_train(matrixModel, labelModel, Lambda, prior_train)
    scores.append(LogReg.quad_logistic_regression_LLR(w, b, M_eval))
    scores = numpy.hstack(scores)
    scores = scores.reshape((1, len(scores)))
    
    scores_calibration.append(LogReg.quad_logistic_regression_LLR(w, b, matrixCal))
    scores_calibration = numpy.hstack(scores_calibration)
    scores_calibration = scores_calibration.reshape(1, len(scores_calibration))
    
    (cal_scores, _, _) = utils.calibrate_scores(scores_calibration, labelCal, scores, 0, prior_cal)
    
    cal_scores = numpy.hstack(cal_scores)
    vTest = numpy.hstack(vTest)
    
    metrics.bayes_error_plot_comparison([cal_scores], vTest, ['QuadLogReg'])
    return

def QuadLog_reg_evaluation(mTrain, Ltrain, mTest, vTest, z_norm=False):
    #prior_train = 0.09
    #Lambdas = numpy.logspace(-4, 2, num=20).tolist()
    Lambdas = [0.00379269019073225]
    prior_train=(mTrain[:, Ltrain == 1]).shape[1] / mTrain.shape[1] #emp prior
    prior_prob=0.09
    dimensions = [6]
    
    
    #shuffling
    numpy.random.seed(0)
    idxTrain = numpy.random.permutation(mTrain.shape[1])
    mTrain = mTrain[:, idxTrain]
    Ltrain = Ltrain[idxTrain]
    
    
    scores = [[[] for _ in range(len(Lambdas))] for _ in range(len(dimensions))]
    
    if z_norm:
        (mTrain, mTest) = utils.z_norm(mTrain, mTest)
    
    for j in range(len(dimensions)):
        if dimensions[j] != 10:
            P = utils.dimension_reduction_PCA(mTrain, dimensions[j])
            M_train = numpy.dot(P.T, mTrain)
            M_eval = numpy.dot(P.T, mTest)
        else:
            M_train = mTrain
            M_eval = mTest
        
        
        #Expansion of features AFTER PCA
        M_train = LogReg.expanded_feature_space(M_train)
        M_eval = LogReg.expanded_feature_space(M_eval)
        
        #Lambda terms
        for i in range(len(Lambdas)):
            Lambda = Lambdas[i]
            
            (w, b) = LogReg.quad_logistic_regression_train(M_train, Ltrain, Lambda, prior_train)
            scores[j][i].append(LogReg.quad_logistic_regression_LLR(w, b, M_eval))
        
    #using the plot to visualize the results
    #plot the minDCF for different values of Lambdas
    plt.figure()
    plt.ylim(0.25, 0.4)
    plt.xlim(xmin = min(Lambdas), xmax = max(Lambdas))
    plt.xscale("log")
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    
    
    colors = ['r', 'g', 'b', 'k', 'm', 'c']
    
    #compute minDCF for each dimension and for each lambda
    for j in range(len(dimensions)):
        scoresDIM = scores[j]
        minDCFV = []
        for i in range(len(scoresDIM)):
            #for each lambda in the dimenions
            scoresLambdaDim = numpy.hstack(scoresDIM[i])
            minD = metrics.minimum_detection_costs(scoresLambdaDim, vTest, prior_prob, 1, 1)
            minDCFV.append(minD)
        
        best_l = str(Lambdas[minDCFV.index(min(minDCFV))])
        print('Best Lambda for dimensionality ' + str(dimensions[j]) + ' is ' + best_l + ' with minDCF ' + str(min(minDCFV)))
    
        plt.plot(Lambdas, minDCFV, color=colors[j], label='LogReg PCA=' + str(dimensions[j]))
        
    plt.grid()
    plt.legend()
    plt.show()
    return

def SVMRBF_computecosts(mTrain, Ltrain, mTest, vTest):
    '''
    #chosen
    C = 4
    gamma = math.e**-7
    dimension = 7
    K = 1
    '''
    #optmial
    C = 10
    gamma = math.e**-6
    dimension = 7
    K = 1
    
    prior_cal = 0.09
    
    #shuffling
    numpy.random.seed(0)
    idxTrain = numpy.random.permutation(mTrain.shape[1])
    mTrain = mTrain[:, idxTrain]
    Ltrain = Ltrain[idxTrain]
    
    if dimension != 10:
        P = utils.dimension_reduction_PCA(mTrain, dimension)
        M_train = numpy.dot(P.T, mTrain)
        M_eval = numpy.dot(P.T, mTest)
    else:
        M_train = mTrain
        M_eval = mTest
    
    matrixModel = M_train[:, range(0, int(M_train.shape[1]*0.8))]
    labelModel = Ltrain[range(0, int(M_train.shape[1]*0.8))]
    matrixCal = M_train[:, range(int(M_train.shape[1]*0.8), M_train.shape[1])]
    labelCal = Ltrain[range(int(M_train.shape[1]*0.8), M_train.shape[1])]
    
    
    scores_calibration = [] #we will use this to train the LR
    scores = []
    
    a = SVM.SVM_kernel_train_radial(matrixModel, 2*labelModel-1, C, K, gamma)
    scores.append(SVM.SVM_kernel_radial_LLR(matrixModel, M_eval, K, gamma, 2*labelModel-1, a))
    scores = numpy.hstack(scores)
    scores = scores.reshape((1, len(scores)))

    scores_calibration.append(SVM.SVM_kernel_radial_LLR(matrixModel, matrixCal, K, gamma, 2*labelModel-1, a))
    scores_calibration = numpy.hstack(scores_calibration)
    scores_calibration = scores_calibration.reshape(1, len(scores_calibration))
    
    (cal_scores, _, _) = utils.calibrate_scores(scores_calibration, labelCal, scores, 0, prior_cal)
    
    cal_scores = numpy.hstack(cal_scores)
    vTest = numpy.hstack(vTest)
    
    metrics.bayes_error_plot_comparison([cal_scores], vTest, ['SVM RBF'])
    return

def SVMRBF_evaluation(mTrain, Ltrain, mTest, vTest, numFolds=5, z_norm=False):
    #Cs = numpy.logspace(-1, 1, num=10).tolist()
    Cs = [10]
    gammas = [math.e**-6]
    #dimensions = [10, 9, 8, 7, 6]
    dimensions = [7]
    
    K=1
    prior_prob=0.09
    
    #shuffling
    numpy.random.seed(0)
    idxTrain = numpy.random.permutation(mTrain.shape[1])
    mTrain = mTrain[:, idxTrain]
    Ltrain = Ltrain[idxTrain]
    
    
    scores = [
        [
            [
                [] for _ in range(len(Cs))
            ] for _ in range(len(gammas))
        ] for _ in range(len(dimensions))
    ]
    
    if z_norm:
        (mTrain, mTest) = utils.z_norm(mTrain, mTest)
    
    for j in range(len(dimensions)):
        if j != 10:
            P = utils.dimension_reduction_PCA(mTrain, dimensions[j])
            M_train = numpy.dot(P.T, mTrain)
            M_eval = numpy.dot(P.T, mTest)
        else:
            M_train = mTrain
            M_eval = mTest
        
        for g in range(len(gammas)): 
            minDCFV = []
            gamma = gammas[g]
            
            #we compute G and kern here since they don't depend on the C value
            #it's an optinal argument for both train and test
            G_train = SVM.SVM_compute_radial_kernel_train(M_train, K, gamma)
            kern = SVM.SVM_compute_radial_kernel_test(M_train, M_eval, K, gamma)
        
            #Lambda terms
            for i in range(len(Cs)):
                C = Cs[i]
                
                a = SVM.SVM_kernel_train_radial(M_train, 2*Ltrain-1, C, K, gamma, G=G_train)
                scores[j][g][i].append(SVM.SVM_kernel_radial_LLR(M_train, M_eval, K, gamma, 2*Ltrain-1, a, kern=kern))
        
    #using the plot to visualize the results
    #plot the minDCF for different values of Lambdas
    plt.figure()
    plt.ylim(0.25, 0.4)
    plt.xlim(xmin = min(Cs), xmax = max(Cs))
    plt.xscale("log")
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    
    
    colors = ['r', 'g', 'b', 'k', 'm', 'c']
    color = 0
    
    #compute minDCF for each dimension and for each lambda
    for j in range(len(dimensions)):
        scoresDIM = scores[j]
        
        for g in range(len(gammas)):
            scoresDimGamma = scoresDIM[g]
        
            minDCFV = []
            for i in range(len(scoresDimGamma)):
                #for each C in the dimenions
                scoresDimGammaC = numpy.hstack(scoresDimGamma[i])
                minD = metrics.minimum_detection_costs(scoresDimGammaC, vTest, prior_prob, 1, 1)
                minDCFV.append(minD)
            
            best_c = str(Cs[minDCFV.index(min(minDCFV))])
            
            print('Best C for SVM RBF (log γ =' + str(math.log(gammas[g])) + ') dimensionality ' + str(dimensions[j]) + ' is ' + best_c + ' with minDCF ' + str(min(minDCFV)))
        
           
            if z_norm:
                plt.plot(Cs, minDCFV, 
                         label='SVM RBF (log γ =' + str(math.log(gammas[g])) +') z_norm dimension: ' + str(dimensions[j]), color=colors[color])
            else:
                plt.plot(Cs, minDCFV, 
                         label='SVM RBF (log γ =' + str(math.log(gammas[g])) +') dimension: ' + str(dimensions[j]), color=colors[color])
            color += 1
    plt.grid()
    plt.legend()
    plt.show()
        
    return

def GMM_nocal_computecosts(mTrain, Ltrain, mTest, vTest):
    '''
    iT = 0
    iNT = 2
    typeT = 'Full'
    typeNT = 'Diag'
    dimension = 8
    '''
    
    #optimal
    iT = 0
    iNT = 2
    typeT = 'Full'
    typeNT = 'Full'
    dimension = 7
    
    scores = []
    
    if dimension != 10:
        P = utils.dimension_reduction_PCA(mTrain, dimension)
        M_train = numpy.dot(P.T, mTrain)
        M_eval = numpy.dot(P.T, mTest)
    else:
        M_train = mTrain
        M_eval = mTest
    
    GMMs = GMM.GMM_train(M_train, Ltrain, iT, iNT, typeT = typeT, typeNT = typeNT)
    scores.append(GMM.GMM_LLR(GMMs, M_eval))
    
    scores = numpy.hstack(scores)
    
    metrics.bayes_error_plot_comparison([scores], vTest, ['GMM'])
    return

def GMM_cal_computecosts(mTrain, Ltrain, mTest, vTest):
    
    #chosen
    iT = 0
    iNT = 2
    typeT = 'Full'
    typeNT = 'Diag'
    dimension = 8
    
    prior_cal = 0.09
    
    numpy.random.seed(0)
    idxTrain = numpy.random.permutation(mTrain.shape[1])
    mTrain = mTrain[:, idxTrain]
    Ltrain = Ltrain[idxTrain]
    
    idxTest = numpy.random.permutation(mTest.shape[1])
    mTest = mTest[:, idxTest]
    vTest = vTest[idxTest]
    
    if dimension != 10:
        P = utils.dimension_reduction_PCA(mTrain, dimension)
        M_train = numpy.dot(P.T, mTrain)
        M_eval = numpy.dot(P.T, mTest)
    else:
        M_train = mTrain
        M_eval = mTest
        
    matrixModel = M_train[:, range(0, int(mTrain.shape[1]*0.8))]
    labelModel = Ltrain[range(0, int(mTrain.shape[1]*0.8))]
    matrixCal = M_train[:, range(int(mTrain.shape[1]*0.8), mTrain.shape[1])]
    labelCal = Ltrain[range(int(mTrain.shape[1]*0.8), mTrain.shape[1])]
    
    
    scores_calibration = [] #we will use this to train the LR
    scores = []
    
    GMMs = GMM.GMM_train(matrixModel, labelModel, iT, iNT, typeT = typeT, typeNT = typeNT)
    scores.append(GMM.GMM_LLR(GMMs, M_eval))
    scores = numpy.hstack(scores)
    scores = scores.reshape((1, len(scores)))
    
    scores_calibration.append(GMM.GMM_LLR(GMMs, matrixCal))
    scores_calibration = numpy.hstack(scores_calibration)
    scores_calibration = scores_calibration.reshape(1, len(scores_calibration))
    
    (cal_scores, _, _) = utils.calibrate_scores(scores_calibration, labelCal, scores, 0, prior_cal)
    cal_scores = numpy.hstack(cal_scores)
    vTest = numpy.hstack(vTest)
    
    metrics.bayes_error_plot_comparison([cal_scores], vTest, ['GMM'])
    return

def GMM_evaluation(Mtrain, Ltrain, Mval, vTest):
    #first iteration
    #dimensions = [10]
    #iterationsT = range(3)
    #iterationsNT = range(6) #da 1 a 32 non target
    
    #second
    dimensions = [7]
    iterationsT = [0]
    iterationsNT = [2]
    z_norm = False
    
    typeT = 'Full'
    typeNT = 'Full'
    
    prior_prob=0.9
    
    scores = [
        [
            [
                [] for _ in range(len(iterationsNT))
            ] for _ in range(len(iterationsT))
        ] for _ in range(len(dimensions))
    ]
    
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    color = 0
    
    if z_norm:
        #normalize our dataset before PCA
        (Mtrain, Mval) = utils.z_norm(Mtrain, Mval)
    
    for j in range(len(dimensions)):
        if dimensions[j] != 10:
            P = utils.dimension_reduction_PCA(Mtrain, dimensions[j])
            M_train = numpy.dot(P.T, Mtrain)
            M_val = numpy.dot(P.T, Mval)
        else:
            M_train = Mtrain
            M_val = Mval
            
        for t in range(len(iterationsT)):
            iT = iterationsT[t]
            
            for nt in range(len(iterationsNT)):
                iNT = iterationsNT[nt]
                
                GMMs = GMM.GMM_train(M_train, Ltrain, iT, iNT, typeT = typeT, typeNT = typeNT)
                scores[j][t][nt].append(GMM.GMM_LLR(GMMs, M_val))
    
    plt.figure()
    plt.ylim(0.2, 1)
    plt.xlim(xmin = 0, xmax = 35)
    plt.xlabel('Non Target')
    plt.ylabel('minDCF')
    
    colors = ['r', 'g', 'b', 'k', 'm', 'c']
    color=0
    
    #compute minDCF for each dimension and for each lambda
    for j in range(len(dimensions)):
        scoresDIM = scores[j]
        
        for t in range(len(iterationsT)):
            scoresDimTarget = scoresDIM[t]
        
            minDCFV = []
            for nt in range(len(iterationsNT)):
                #for each C in the dimenions
                scoresDimTargetNonTarget = numpy.hstack(scoresDimTarget[nt])
                minD = metrics.minimum_detection_costs(scoresDimTargetNonTarget, vTest, prior_prob, 1, 1)
                minDCFV.append(minD)
    
            if z_norm:
                plt.plot([2**numnt for numnt in iterationsNT], minDCFV, 
                         label='GMM z_norm dimension: ' + str(dimensions[j]) + ' Target: ' + str(2**iterationsT[t]), color=colors[color])
            else:
                plt.plot([2**numnt for numnt in iterationsNT], minDCFV, 
                         label='GMM dimension: ' + str(dimensions[j]) + ' Target: ' + str(2**iterationsT[t]), color=colors[color])
            color += 1
            
            best_NT = str(iterationsNT[minDCFV.index(min(minDCFV))])
            
            print('Best nt for GMM target ' + str(2**iterationsT[t]) +' dimensionality ' + str(dimensions[j]) + ' is ' + best_NT + ' with minDCF ' + str(min(minDCFV)))
    
    plt.grid()
    plt.legend()
    plt.show()
    return

def fusionSVMGMM(mTrain, Ltrain, mTest, vTest):
    '''
    #chosen
    iT = 0
    iNT = 2
    typeT = 'Full'
    typeNT = 'Diag'
    dimensionGMM = 8
    
    C = 4
    gamma = math.e**-7
    dimensionSVM = 7
    K = 1
    '''
    #optmial
    iT = 0
    iNT = 2
    typeT = 'Full'
    typeNT = 'Full'
    dimensionGMM = 7
    
    C = 10
    gamma = math.e**-6
    dimensionSVM = 7
    K = 1
    
    prior_cal = 0.09
    
    #shuffling
    numpy.random.seed(0)
    idxTrain = numpy.random.permutation(mTrain.shape[1])
    mTrain = mTrain[:, idxTrain]
    Ltrain = Ltrain[idxTrain]
    
    #since we have different dimensions we have to compute two times the PCA
    if dimensionSVM != 10:
        P = utils.dimension_reduction_PCA(mTrain, dimensionSVM)
        M_trainSVM = numpy.dot(P.T, mTrain)
        M_evalSVM = numpy.dot(P.T, mTest)
    else:
        M_trainSVM = mTrain
        M_evalSVM = mTest
        
    if dimensionGMM != 10:
        P = utils.dimension_reduction_PCA(mTrain, dimensionGMM)
        M_trainGMM = numpy.dot(P.T, mTrain)
        M_evalGMM = numpy.dot(P.T, mTest)
    else:
        M_trainGMM = mTrain
        M_evalGMM = mTest
    
    mSVM = M_trainSVM[:, range(0, int(mTrain.shape[1]*0.8))]
    calSVM = M_trainSVM[:, range(int(mTrain.shape[1]*0.8), mTrain.shape[1])]
    
    mGMM = M_trainGMM[:, range(0, int(mTrain.shape[1]*0.8))]
    calGMM = M_trainGMM[:, range(int(mTrain.shape[1]*0.8), mTrain.shape[1])]
    
    
    labelsTrain = Ltrain[range(0, int(mTrain.shape[1]*0.8))]
    labelsCal =  Ltrain[range(int(mTrain.shape[1]*0.8), mTrain.shape[1])]
    
    #we need the scores of both SVM and GMM
    scores_cal_SVM = []
    scores_SVM = []
    scores_cal_GMM = []
    scores_GMM = []
    
    a = SVM.SVM_kernel_train_radial(mSVM, 2*labelsTrain-1, C, K, gamma)
    scores_SVM.append(SVM.SVM_kernel_radial_LLR(mSVM, M_evalSVM, K, gamma, 2*labelsTrain-1, a))
    scores_cal_SVM.append(SVM.SVM_kernel_radial_LLR(mSVM, calSVM, K, gamma, 2*labelsTrain-1, a))
    
    GMMs = GMM.GMM_train(mGMM, labelsTrain, iT, iNT, typeT = typeT, typeNT = typeNT)
    scores_GMM.append(GMM.GMM_LLR(GMMs, M_evalGMM))
    scores_cal_GMM.append(GMM.GMM_LLR(GMMs, calGMM))
    
    scores_cal_SVM = numpy.hstack(scores_cal_SVM)
    scores_cal_GMM = numpy.hstack(scores_cal_GMM)
    scores_cal = numpy.vstack((scores_cal_SVM, scores_cal_GMM))
    
    
    scores_SVM = numpy.hstack(scores_SVM)
    scores_GMM = numpy.hstack(scores_GMM)
    scores = numpy.vstack((scores_SVM, scores_GMM))
    
    (final_scores, _, _) = utils.calibrate_scores(scores_cal, labelsCal, scores, 0, prior_cal)
    
    final_scores = numpy.hstack(final_scores)
    vTest = numpy.hstack(vTest)
    
    metrics.bayes_error_plot_comparison([final_scores], vTest, ['Fusion SVM GMM'])
    return

def fusionLogRegGMM(mTrain, Ltrain, mTest, vTest):
    '''
    #chosen
    iT = 0
    iNT = 2
    typeT = 'Full'
    typeNT = 'Diag'
    dimensionGMM = 8
    
    Lambda = 0.001623776739188721
    prior_train=(mTrain[:, Ltrain == 1]).shape[1] / mTrain.shape[1] #emp prior
    dimensionLogReg = 6
    '''
    #optmial
    iT = 0
    iNT = 2
    typeT = 'Full'
    typeNT = 'Full'
    dimensionGMM = 7
    
    Lambda = 0.00379269019073225
    prior_train=(mTrain[:, Ltrain == 1]).shape[1] / mTrain.shape[1] #emp prior
    dimensionLogReg = 6
    
    prior_cal = 0.09
    
    #shuffling
    numpy.random.seed(0)
    idxTrain = numpy.random.permutation(mTrain.shape[1])
    mTrain = mTrain[:, idxTrain]
    Ltrain = Ltrain[idxTrain]
    
    #since we have different dimensions we have to compute two times the PCA
    
    if dimensionLogReg != 10:
        P = utils.dimension_reduction_PCA(mTrain, dimensionLogReg)
        M_trainLogReg = numpy.dot(P.T, mTrain)
        M_evalLogReg = numpy.dot(P.T, mTest)
    else:
        M_trainLogReg = mTrain
        M_evalLogReg = mTest
        
    if dimensionGMM != 10:
        P = utils.dimension_reduction_PCA(mTrain, dimensionGMM)
        M_trainGMM = numpy.dot(P.T, mTrain)
        M_evalGMM = numpy.dot(P.T, mTest)
    else:
        M_trainGMM = mTrain
        M_evalGMM = mTest
    
    mLogReg = M_trainLogReg[:, range(0, int(mTrain.shape[1]*0.8))]
    calLogReg = M_trainLogReg[:, range(int(mTrain.shape[1]*0.8), mTrain.shape[1])]
    
    mGMM = M_trainGMM[:, range(0, int(mTrain.shape[1]*0.8))]
    calGMM = M_trainGMM[:, range(int(mTrain.shape[1]*0.8), mTrain.shape[1])]
    
    labelsTrain = Ltrain[range(0, int(mTrain.shape[1]*0.8))]
    labelsCal =  Ltrain[range(int(mTrain.shape[1]*0.8), mTrain.shape[1])]
    
    #we need the scores of both LogReg and GMM
    scores_cal_LogReg = []
    scores_cal_GMM = []
    scores_LogReg = []
    scores_GMM = []
    
    (w, b) = LogReg.quad_logistic_regression_train(mLogReg, labelsTrain, Lambda, prior_train)
    scores_LogReg.append(LogReg.quad_logistic_regression_LLR(w, b, M_evalLogReg))
    scores_cal_LogReg.append(LogReg.quad_logistic_regression_LLR(w, b, calLogReg))
    
    GMMs = GMM.GMM_train(mGMM, labelsTrain, iT, iNT, typeT = typeT, typeNT = typeNT)
    scores_GMM.append(GMM.GMM_LLR(GMMs, M_evalGMM))
    scores_cal_GMM.append(GMM.GMM_LLR(GMMs, calGMM))
    
    scores_cal_LogReg = numpy.hstack(scores_cal_LogReg)
    scores_cal_GMM = numpy.hstack(scores_cal_GMM)
    scores_cal = numpy.vstack((scores_cal_LogReg, scores_cal_GMM))
    
    
    scores_LogReg = numpy.hstack(scores_LogReg)
    scores_GMM = numpy.hstack(scores_GMM)
    scores = numpy.vstack((scores_LogReg, scores_GMM))
    
    (final_scores, _, _) = utils.calibrate_scores(scores_cal, labelsCal, scores, 0, prior_cal)
    
    final_scores = numpy.hstack(final_scores)
    vTest = numpy.hstack(vTest)
    
    metrics.bayes_error_plot_comparison([final_scores], vTest, ['Fusion LogReg GMM'])
    return

def fusionAll(mTrain, Ltrain, mTest, vTest):
    '''
    #chosen 
    C = 4
    gamma = math.e**-7
    dimensionSVM = 7
    K = 1
    
    iT = 0
    iNT = 2
    typeT = 'Full'
    typeNT = 'Diag'
    dimensionGMM = 8
    
    Lambda = 0.001623776739188721
    prior_train=(mTrain[:, Ltrain == 1]).shape[1] / mTrain.shape[1] #emp prior
    dimensionLogReg = 6
    '''
    #optmial
    iT = 0
    iNT = 2
    typeT = 'Full'
    typeNT = 'Full'
    dimensionGMM = 7
    
    Lambda = 0.00379269019073225
    prior_train=(mTrain[:, Ltrain == 1]).shape[1] / mTrain.shape[1] #emp prior
    dimensionLogReg = 6
    
    C = 10
    gamma = math.e**-6
    dimensionSVM = 7
    K = 1
    
    prior_cal = 0.09
    
    #shuffling
    numpy.random.seed(0)
    idxTrain = numpy.random.permutation(mTrain.shape[1])
    mTrain = mTrain[:, idxTrain]
    Ltrain = Ltrain[idxTrain]
    
    #since we have different dimensions we have to compute three times the PCA
    
    if dimensionLogReg != 10:
        P = utils.dimension_reduction_PCA(mTrain, dimensionLogReg)
        M_trainLogReg = numpy.dot(P.T, mTrain)
        M_evalLogReg = numpy.dot(P.T, mTest)
    else:
        M_trainLogReg = mTrain
        M_evalLogReg = mTest
        
    if dimensionGMM != 10:
        P = utils.dimension_reduction_PCA(mTrain, dimensionGMM)
        M_trainGMM = numpy.dot(P.T, mTrain)
        M_evalGMM = numpy.dot(P.T, mTest)
    else:
        M_trainGMM = mTrain
        M_evalGMM = mTest
    
    if dimensionSVM != 10:
        P = utils.dimension_reduction_PCA(mTrain, dimensionSVM)
        M_trainSVM = numpy.dot(P.T, mTrain)
        M_evalSVM = numpy.dot(P.T, mTest)
    else:
        M_trainSVM = mTrain
        M_evalSVM = mTest
    
    
    mSVM = M_trainSVM[:, range(0, int(mTrain.shape[1]*0.8))]
    calSVM = M_trainSVM[:, range(int(mTrain.shape[1]*0.8), mTrain.shape[1])]
    
    mLogReg = M_trainLogReg[:, range(0, int(mTrain.shape[1]*0.8))]
    calLogReg = M_trainLogReg[:, range(int(mTrain.shape[1]*0.8), mTrain.shape[1])]
    
    mGMM = M_trainGMM[:, range(0, int(mTrain.shape[1]*0.8))]
    calGMM = M_trainGMM[:, range(int(mTrain.shape[1]*0.8), mTrain.shape[1])]
    
    labelsTrain = Ltrain[range(0, int(mTrain.shape[1]*0.8))]
    labelsCal =  Ltrain[range(int(mTrain.shape[1]*0.8), mTrain.shape[1])]
    
    scores_cal_LogReg = []
    scores_cal_GMM = []
    scores_LogReg = []
    scores_GMM = []
    scores_cal_SVM = []
    scores_SVM = []
    
    #compute for each classifier the two scores
    (w, b) = LogReg.quad_logistic_regression_train(mLogReg, labelsTrain, Lambda, prior_train)
    scores_LogReg.append(LogReg.quad_logistic_regression_LLR(w, b, M_evalLogReg))
    scores_cal_LogReg.append(LogReg.quad_logistic_regression_LLR(w, b, calLogReg))
    
    a = SVM.SVM_kernel_train_radial(mSVM, 2*labelsTrain-1, C, K, gamma)
    scores_SVM.append(SVM.SVM_kernel_radial_LLR(mSVM, M_evalSVM, K, gamma, 2*labelsTrain-1, a))
    scores_cal_SVM.append(SVM.SVM_kernel_radial_LLR(mSVM, calSVM, K, gamma, 2*labelsTrain-1, a))
    
    GMMs = GMM.GMM_train(mGMM, labelsTrain, iT, iNT, typeT = typeT, typeNT = typeNT)
    scores_GMM.append(GMM.GMM_LLR(GMMs, M_evalGMM))
    scores_cal_GMM.append(GMM.GMM_LLR(GMMs, calGMM))
    
    #stack the scores together to get the data for the calibration step
    scores_cal_LogReg = numpy.hstack(scores_cal_LogReg)
    scores_cal_GMM = numpy.hstack(scores_cal_GMM)
    scores_cal_SVM = numpy.hstack(scores_cal_SVM)
    scores_cal = numpy.vstack((scores_cal_LogReg, scores_cal_GMM))
    scores_cal = numpy.vstack((scores_cal, scores_cal_SVM))
    
    
    scores_LogReg = numpy.hstack(scores_LogReg)
    scores_GMM = numpy.hstack(scores_GMM)
    scores_SVM = numpy.hstack(scores_SVM)
    scores = numpy.vstack((scores_LogReg, scores_GMM))
    scores = numpy.vstack((scores, scores_SVM))
    
    (final_scores, _, _) = utils.calibrate_scores(scores_cal, labelsCal, scores, 0, prior_cal)
    
    final_scores = numpy.hstack(final_scores)
    vTest = numpy.hstack(vTest)
    
    metrics.bayes_error_plot_comparison([final_scores], vTest, ['Fusion LogReg SVM GMM'])
    return