import SVM
import math
import numpy
import utils
import metrics
import GMM
import LogReg


def QuadLogRegCalibrated_validation(m, v, numFolds=5, z_norm=False):
    Lambda = 0.001623776739188721
    prior_train=(m[:, v==1]).shape[1]/m.shape[1]
    prior_prob=0.09
    
    trueLabels = []
    scores = []
    (folds, labels) = utils.k_split(m, v, numFolds)
    
    for k in range(numFolds):
        Mtrain = []
        Ltrain = []
        
        #compute train matrix
        for fold in range(numFolds):
            if fold!=k:
                Mtrain.append(folds[fold])
                Ltrain.append(labels[fold])
        
        Mtrain = numpy.hstack(Mtrain)
        Ltrain = numpy.hstack(Ltrain)
        
        Mval = folds[k]
        Lval = labels[k]
        
        P = utils.dimension_reduction_PCA(Mtrain, 6)
        Mtrain = numpy.dot(P.T, Mtrain)
        Mval = numpy.dot(P.T, Mval)
        
        Mtrain = LogReg.expanded_feature_space(Mtrain)
        Mval = LogReg.expanded_feature_space(Mval)
        
        (w, b) = LogReg.quad_logistic_regression_train(Mtrain, Ltrain, Lambda, prior_train)
        scores.append(LogReg.quad_logistic_regression_LLR(w, b, Mval))
        
        trueLabels.append(Lval)
    
    #calibrate the scores
    scores = numpy.hstack(scores)
    scores = scores.reshape(1, len(scores))
    trueLabels = numpy.hstack(trueLabels)
    
    (cal_scores, trueLabels) = compute_calibrated_scores(scores, trueLabels, numFolds, prior_prob)

    metrics.bayes_error_plot_comparison([cal_scores], trueLabels, ['Calibrated QuadLogReg'])
    return

def RadialSVMCalibrated_validation(m, v, numFolds=5):
    C = 4
    gamma = math.e**-7
    prior = 0.09
    K = 1
    
    trueLabels = []
    scores = []
    (folds, labels) = utils.k_split(m, v, numFolds)
    
    for k in range(numFolds):
        Mtrain = []
        Ltrain = []
        
        #compute train matrix
        for fold in range(numFolds):
            if fold!=k:
                Mtrain.append(folds[fold])
                Ltrain.append(labels[fold])
        
        Mtrain = numpy.hstack(Mtrain)
        Ltrain = numpy.hstack(Ltrain)
        zTrain = 2*Ltrain - 1
        
        Mval = folds[k]
        Lval = labels[k]
        
        P = utils.dimension_reduction_PCA(Mtrain, 7)
        M_train = numpy.dot(P.T, Mtrain)
        M_val = numpy.dot(P.T, Mval)
        
        a = SVM.SVM_kernel_train_radial(M_train, zTrain, C, K, gamma)
        scores.append(SVM.SVM_kernel_radial_LLR(M_train, M_val, K, gamma, zTrain, a))
    
        trueLabels.append(Lval)
    
    #calibrate the scores
    scores = numpy.hstack(scores)
    scores = scores.reshape(1, len(scores))
    trueLabels = numpy.hstack(trueLabels)
    
    (cal_scores, trueLabels) = compute_calibrated_scores(scores, trueLabels, numFolds, prior)

    metrics.bayes_error_plot_comparison([cal_scores], trueLabels, ['Calibrated SVM'])
    return

def GMMCalibrated_validation(m, v, numFolds=5):
    iT = 0
    iNT = 2
    typeT = 'Full'
    typeNT = 'Diag'
    prior = 0.09
    
    trueLabels = []
    scores = []
    (folds, labels) = utils.k_split(m, v, numFolds)
    
    for k in range(numFolds):
        Mtrain = []
        Ltrain = []
        
        #compute train matrix
        for fold in range(numFolds):
            if fold!=k:
                Mtrain.append(folds[fold])
                Ltrain.append(labels[fold])
        
        Mtrain = numpy.hstack(Mtrain)
        Ltrain = numpy.hstack(Ltrain)
        
        Mval = folds[k]
        Lval = labels[k]
        
        P = utils.dimension_reduction_PCA(Mtrain, 8)
        M_train = numpy.dot(P.T, Mtrain)
        M_val = numpy.dot(P.T, Mval)
        
        GMMs = GMM.GMM_train(M_train, Ltrain, iT, iNT, typeT = typeT, typeNT = typeNT)
        scores.append(GMM.GMM_LLR(GMMs, M_val))
        
        trueLabels.append(Lval)
    
    #calibrate the scores
    scores = numpy.hstack(scores)
    scores = scores.reshape(1, len(scores))
    trueLabels = numpy.hstack(trueLabels)
    
    (cal_scores, trueLabels) = compute_calibrated_scores(scores, trueLabels, numFolds, prior)

    metrics.bayes_error_plot_comparison([cal_scores], trueLabels, ['Calibrated GMM'])
    return


def fusionSVMGMM_calibrated(m, v, numFolds=5):
    iT = 0
    iNT = 2
    typeT = 'Full'
    typeNT = 'Diag'
    prior = 0.09
    
    C = 4
    gamma = math.e**-7
    prior = 0.09
    K = 1
    
    trueLabels = []
    scores_1 = []
    scores_2 = []
    (folds, labels) = utils.k_split(m, v, numFolds)
    
    for k in range(numFolds):
        Mtrain = []
        Ltrain = []
        
        #compute train matrix
        for fold in range(numFolds):
            if fold!=k:
                Mtrain.append(folds[fold])
                Ltrain.append(labels[fold])
        
        Mtrain = numpy.hstack(Mtrain)
        Ltrain = numpy.hstack(Ltrain)
        zTrain = 2*Ltrain - 1
        
        Mval = folds[k]
        Lval = labels[k]
        
        P8 = utils.dimension_reduction_PCA(Mtrain, 8)
        M_train8 = numpy.dot(P8.T, Mtrain)
        M_val8 = numpy.dot(P8.T, Mval)
        
        P7 = utils.dimension_reduction_PCA(Mtrain, 7)
        M_train7 = numpy.dot(P7.T, Mtrain)
        M_val7 = numpy.dot(P7.T, Mval)
        
        a = SVM.SVM_kernel_train_radial(M_train7, zTrain, C, K, gamma)
        scores_1.append(SVM.SVM_kernel_radial_LLR(M_train7, M_val7, K, gamma, zTrain, a))
        
        GMMs = GMM.GMM_train(M_train8, Ltrain, iT, iNT, typeT = typeT, typeNT = typeNT)
        scores_2.append(GMM.GMM_LLR(GMMs, M_val8))
        
        trueLabels.append(Lval)
    
    #calibrate the scores
    scores_1 = numpy.hstack(scores_1)
    scores_2 = numpy.hstack(scores_2)
    scores = numpy.vstack((scores_1, scores_2))
    trueLabels = numpy.hstack(trueLabels)
    
    (cal_scores, trueLabels) = compute_calibrated_scores(scores, trueLabels, numFolds, prior)

    metrics.bayes_error_plot_comparison([cal_scores], trueLabels, ['Fusion GMM SVM'])
    return

def fusionQuadLogRegGMM_calibrated(m, v, numFolds=5):
    Lambda = 0.001623776739188721
    prior_train=(m[:, v==1]).shape[1]/m.shape[1]
    
    iT = 0
    iNT = 2
    typeT = 'Full'
    typeNT = 'Diag'
    prior = 0.09
    
    trueLabels = []
    scores_1 = []
    scores_2 = []
    (folds, labels) = utils.k_split(m, v, numFolds)
    
    for k in range(numFolds):
        Mtrain = []
        Ltrain = []
        
        #compute train matrix
        for fold in range(numFolds):
            if fold!=k:
                Mtrain.append(folds[fold])
                Ltrain.append(labels[fold])
        
        Mtrain = numpy.hstack(Mtrain)
        Ltrain = numpy.hstack(Ltrain)
        
        Mval = folds[k]
        Lval = labels[k]
        
        P8 = utils.dimension_reduction_PCA(Mtrain, 8)
        M_train8 = numpy.dot(P8.T, Mtrain)
        M_val8 = numpy.dot(P8.T, Mval)
        
        
        P6 = utils.dimension_reduction_PCA(Mtrain, 6)
        Mtrain6 = numpy.dot(P6.T, Mtrain)
        Mval6 = numpy.dot(P6.T, Mval)
        
        Mtrain6 = LogReg.expanded_feature_space(Mtrain6)
        Mval6 = LogReg.expanded_feature_space(Mval6)
        
        (w, b) = LogReg.quad_logistic_regression_train(Mtrain6, Ltrain, Lambda, prior_train)
        scores_1.append(LogReg.quad_logistic_regression_LLR(w, b, Mval6))
        
        GMMs = GMM.GMM_train(M_train8, Ltrain, iT, iNT, typeT = typeT, typeNT = typeNT)
        scores_2.append(GMM.GMM_LLR(GMMs, M_val8))
        
        trueLabels.append(Lval)
    
    #calibrate the scores
    scores_1 = numpy.hstack(scores_1)
    scores_2 = numpy.hstack(scores_2)
    scores = numpy.vstack((scores_1, scores_2))
    trueLabels = numpy.hstack(trueLabels)
    
    (cal_scores, trueLabels) = compute_calibrated_scores(scores, trueLabels, numFolds, prior)

    metrics.bayes_error_plot_comparison([cal_scores], trueLabels, ['Fusion GMM LogReg'])
    return

def fusionAllModels_calibrated(m, v, numFolds=5):
    Lambda = 0.001623776739188721
    prior_train=(m[:, v==1]).shape[1]/m.shape[1]
    
    iT = 0
    iNT = 2
    typeT = 'Full'
    typeNT = 'Diag'
    
    C = 4
    gamma = math.e**-7
    prior = 0.09
    K = 1
    
    trueLabels = []
    scores_1 = []
    scores_2 = []
    scores_3 = []
    (folds, labels) = utils.k_split(m, v, numFolds)
    
    for k in range(numFolds):
        Mtrain = []
        Ltrain = []
        
        #compute train matrix
        for fold in range(numFolds):
            if fold!=k:
                Mtrain.append(folds[fold])
                Ltrain.append(labels[fold])
        
        Mtrain = numpy.hstack(Mtrain)
        Ltrain = numpy.hstack(Ltrain)
        zTrain = 2*Ltrain - 1
        
        Mval = folds[k]
        Lval = labels[k]
        
        P8 = utils.dimension_reduction_PCA(Mtrain, 8)
        M_train8 = numpy.dot(P8.T, Mtrain)
        M_val8 = numpy.dot(P8.T, Mval)
        
        P7 = utils.dimension_reduction_PCA(Mtrain, 7)
        M_train7 = numpy.dot(P7.T, Mtrain)
        M_val7 = numpy.dot(P7.T, Mval)
        
        P6 = utils.dimension_reduction_PCA(Mtrain, 6)
        Mtrain6 = numpy.dot(P6.T, Mtrain)
        Mval6 = numpy.dot(P6.T, Mval)
        
        Mtrain6 = LogReg.expanded_feature_space(Mtrain6)
        Mval6 = LogReg.expanded_feature_space(Mval6)
        
        (w, b) = LogReg.quad_logistic_regression_train(Mtrain6, Ltrain, Lambda, prior_train)
        scores_1.append(LogReg.quad_logistic_regression_LLR(w, b, Mval6))
        
        a = SVM.SVM_kernel_train_radial(M_train7, zTrain, C, K, gamma)
        scores_2.append(SVM.SVM_kernel_radial_LLR(M_train7, M_val7, K, gamma, zTrain, a))
        
        GMMs = GMM.GMM_train(M_train8, Ltrain, iT, iNT, typeT = typeT, typeNT = typeNT)
        scores_3.append(GMM.GMM_LLR(GMMs, M_val8))
        
        trueLabels.append(Lval)
    
    #calibrate the scores
    scores_1 = numpy.hstack(scores_1)
    scores_2 = numpy.hstack(scores_2)
    scores_3 = numpy.hstack(scores_3)
    scores = numpy.vstack((scores_1, scores_2))
    scores = numpy.vstack((scores, scores_3))
    trueLabels = numpy.hstack(trueLabels)
    
    (cal_scores, trueLabels) = compute_calibrated_scores(scores, trueLabels, numFolds, prior)

    metrics.bayes_error_plot_comparison([cal_scores], trueLabels, ['Fusion model'])
    return

#k-fold approach
def compute_calibrated_scores(scores, trueLabels, numFolds, prior):
    (uncal_scores, labels_scores) = utils.k_split(scores, trueLabels, numFolds)
    
    cal_scores = []
    trueLabels = []
    
    for k in range(numFolds):
        Mtrain = []
        Ltrain = []
        
        #compute train matrix
        for fold in range(numFolds):
            if fold!=k:
                Mtrain.append(uncal_scores[fold])
                Ltrain.append(labels_scores[fold])
        
        Mtrain = numpy.hstack(Mtrain)
        Ltrain = numpy.hstack(Ltrain)
        
        Mval = uncal_scores[k]
        Lval = labels_scores[k]
    
        (scores, _, _) = utils.calibrate_scores(Mtrain, Ltrain, Mval, 0, prior)
        cal_scores.append(scores)
        trueLabels.append(Lval)
    
    cal_scores = numpy.hstack(cal_scores)
    trueLabels = numpy.hstack(trueLabels)
    return (cal_scores, trueLabels)