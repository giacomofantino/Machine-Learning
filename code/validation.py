import math
import numpy
import matplotlib.pyplot as plt
import utils
import Gaussian
import LogReg
import SVM
import GMM
import metrics

#all these validations functions have been programmed using
#the k fold validation approach

def MVG_validation(m, v, numFolds=5):
    prior_prob=[0.09, 0.91]
    dimension=[10, 9, 8, 7]
    
    scores_MVG=[[] for _ in range(len(dimension))]
    scores_tied=[[] for _ in range(len(dimension))]
    scores_naive=[[] for _ in range(len(dimension))]
    scores_tiednaive=[[] for _ in range(len(dimension))]
    
    trueLabels = []
    
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
    
        trueLabels.append(Lval)
        #for each dimension
        for j in range(len(dimension)):
            if j != 0:
                P = utils.dimension_reduction_PCA(Mtrain, dimension[j])
                M_train = numpy.dot(P.T, Mtrain)
                M_val = numpy.dot(P.T, Mval)
            else:
                M_train = Mtrain
                M_val = Mval
            
            parameters_mvg = Gaussian.Multivariate_Gaussian_Classifier_Train(M_train, Ltrain)
            scores_MVG[j].append(Gaussian.Multivariate_Gaussian_Classifier_LLR(M_val, parameters_mvg, prior_prob))
            
            parameters_naive = Gaussian.Naive_Bayes_Gaussian_Classifier_Train(M_train, Ltrain)
            scores_naive[j].append(Gaussian.Naive_Bayes_Gaussian_Classifier_LLR(M_val, parameters_naive, prior_prob))
            
            parameters_tied = Gaussian.Tied_Covariance_Gaussian_Classifier_Train(M_train, Ltrain)
            scores_tied[j].append(Gaussian.Tied_Covariance_Gaussian_Classifier_LLR(M_val, parameters_tied, prior_prob))
            
            parameters_tiednaive = Gaussian.Tied_Naive_Gaussian_Classifier_Train(M_train, Ltrain)
            scores_tiednaive[j].append(Gaussian.Tied_Naive_Gaussian_Classifier_LLR(M_val, parameters_tiednaive, prior_prob))
            
    
    trueLabels = numpy.hstack(trueLabels)
    #compute minDCF for each classifier and for each dimension
    for j in range(len(dimension)):
        scoresMVG = numpy.hstack(scores_MVG[j])
        minD = metrics.minimum_detection_costs(scoresMVG, trueLabels, prior_prob[0], 1, 1)
        print('MVG dimension: ' + str(dimension[j]) + ' minDCF = ' + str(minD))
        
        scoresNAIVE = numpy.hstack(scores_naive[j])
        minD = metrics.minimum_detection_costs(scoresNAIVE, trueLabels, prior_prob[0], 1, 1)
        print('Naive dimension: ' + str(dimension[j]) + ' minDCF = ' + str(minD))
        
        scoresTIED = numpy.hstack(scores_tied[j])
        minD = metrics.minimum_detection_costs(scoresTIED, trueLabels, prior_prob[0], 1, 1)
        print('Tied dimension: ' + str(dimension[j]) + ' minDCF = ' + str(minD))
        
        scoresTiedNaive = numpy.hstack(scores_tiednaive[j])
        minD = metrics.minimum_detection_costs(scoresTiedNaive, trueLabels, prior_prob[0], 1, 1)
        print('Tied Naive dimension: ' + str(dimension[j]) + ' minDCF = ' + str(minD))
    return

def Log_reg_validation(matrix, vector, numFolds=5):
    Lambdas = numpy.logspace(-5, 1, num=30).tolist()
    prior_train=(matrix[:, vector==1]).shape[1]/matrix.shape[1]
    prior_prob=0.09
    dimensions = [10, 9, 8, 7, 6]
    
    scores = [[[] for _ in range(len(Lambdas))] for _ in range(len(dimensions))]
    trueLabels = []
    
    (folds, labels) = utils.utils.k_split(matrix, vector, numFolds)
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
        
        trueLabels.append(Lval)
        
        for j in range(len(dimensions)):
            if j != 0:
                P = utils.dimension_reduction_PCA(Mtrain, dimensions[j])
                M_train = numpy.dot(P.T, Mtrain)
                M_val = numpy.dot(P.T, Mval)
            else:
                M_train = Mtrain
                M_val = Mval
            
            #Lambda terms
            for i in range(len(Lambdas)):
                Lambda = Lambdas[i]
                
                (w, b) = LogReg.logistic_regression_train(M_train, Ltrain, Lambda, prior_train)
                scores[j][i].append(LogReg.logistic_regression_LLR(w, b, M_val))
                
    trueLabels = numpy.hstack(trueLabels)
    
    #using the plot to visualize the results
    #plot the minDCF for different values of Lambdas
    plt.figure()
    plt.ylim(0.4, 0.55)
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
            minD = metrics.minimum_detection_costs(scoresLambdaDim, trueLabels, prior_prob, 1, 1)
            minDCFV.append(minD)
        
        best_l = str(Lambdas[minDCFV.index(min(minDCFV))])
        print('Best Lambda for dimensionality ' + str(dimensions[j]) + ' is ' + best_l + ' with minDCF ' + str(min(minDCFV)))
    
    
        plt.plot(Lambdas, minDCFV, color=colors[j], label='LogReg PCA=' + str(dimensions[j]))
    plt.grid()
    plt.legend()
    plt.show()
        
    return
    
def QuadLog_reg_validation(matrix, vector, numFolds=5, z_norm=False):
    #first iteration
    #prior_train=0.09
    #Lambdas = numpy.logspace(-5, 1, num=20).tolist()
    #dimensions = [10, 9, 8, 7, 6]
    
    #following tests
    Lambdas = [0.001623776739188721]
    prior_train=(matrix[:, vector==1]).shape[1]/matrix.shape[1]
    dimensions = [6]
    
    prior_prob=0.09
    
    scores = [[[] for _ in range(len(Lambdas))] for _ in range(len(dimensions))]
    trueLabels = []
    
    (folds, labels) = utils.k_split(matrix, vector, numFolds)
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
        
        trueLabels.append(Lval)
        
        if z_norm:
            #normalize our dataset before PCA
            (Mtrain, Mval) = utils.z_norm(Mtrain, Mval)
        
        for j in range(len(dimensions)):
            if j != 10:
                P = utils.dimension_reduction_PCA(Mtrain, dimensions[j])
                M_train = numpy.dot(P.T, Mtrain)
                M_val = numpy.dot(P.T, Mval)
            else:
                M_train = Mtrain
                M_val = Mval
            
            
            #Expansion of features AFTER PCA
            M_train = LogReg.expanded_feature_space(M_train)
            M_val = LogReg.expanded_feature_space(M_val)
            
            #Lambda terms
            for i in range(len(Lambdas)):
                Lambda = Lambdas[i]
                
                (w, b) = LogReg.quad_logistic_regression_train(M_train, Ltrain, Lambda, prior_train)
                scores[j][i].append(LogReg.quad_logistic_regression_LLR(w, b, M_val))
            
            
    trueLabels = numpy.hstack(trueLabels)
    
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
            minD = metrics.minimum_detection_costs(scoresLambdaDim, trueLabels, prior_prob, 1, 1)
            minDCFV.append(minD)
        
        best_l = str(Lambdas[minDCFV.index(min(minDCFV))])
        print('Best Lambda for dimensionality ' + str(dimensions[j]) + ' is ' + best_l + ' with minDCF ' + str(min(minDCFV)))
    
    
        plt.plot(Lambdas, minDCFV, color=colors[j], label='LogReg PCA=' + str(dimensions[j]))
    plt.grid()
    plt.legend()
    plt.show()
        
    return    

def LinearSVM_validation(matrix, v, numFolds=5):
    Cs = numpy.logspace(-5, 1, num=30).tolist()
    K=1
    prior_prob=0.09
    dimensions = [10, 9, 8, 7]
    
    scores = [[[] for _ in range(len(Cs))] for _ in range(len(dimensions))]
    trueLabels = []
    
    (folds, labels) = utils.k_split(matrix, v, numFolds)
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
        
        trueLabels.append(Lval)
        
        for j in range(len(dimensions)):
            if j != 0:
                P = utils.dimension_reduction_PCA(Mtrain, dimensions[j])
                M_train = numpy.dot(P.T, Mtrain)
                M_val = numpy.dot(P.T, Mval)
            else:
                M_train = Mtrain
                M_val = Mval
            
            
            for i in range(len(Cs)):
                C = Cs[i]
                
                (w, b) = SVM.SVM_train(M_train, Ltrain, C, K)
                scores[j][i].append(SVM.linearSVM_LLR(w, b, M_val))
            
            
    trueLabels = numpy.hstack(trueLabels)
    
    #using the plot to visualize the results
    plt.figure()
    plt.ylim(0.45, 0.8)
    plt.xlim(xmin = min(Cs), xmax = max(Cs))
    plt.xscale("log")
    plt.xlabel('C')
    plt.ylabel('minDCF')
    
    
    colors = ['r', 'g', 'b', 'k', 'm', 'c']
    
    for j in range(len(dimensions)):
        scoresDIM = scores[j]
        minDCFV = []
        for i in range(len(scoresDIM)):
            #for each C in the dimenions
            scoresCDim = numpy.hstack(scoresDIM[i])
            minD = metrics.minimum_detection_costs(scoresCDim, trueLabels, prior_prob, 1, 1)
            minDCFV.append(minD)
        
        best_c = str(Cs[minDCFV.index(min(minDCFV))])
       
        print('Best C for dimensionality ' + str(dimensions[j]) + ' is ' + best_c + ' with minDCF ' + str(min(minDCFV)))
    
        plt.plot(Cs, minDCFV, color=colors[j], label='SVM PCA=' + str(dimensions[j]))
    plt.grid()
    plt.legend()
    plt.show()
    return

def PolySVM_validation(matrix, v, numFolds=5, z_norm=False):
    #first iteration
    #Cs = numpy.logspace(-5, 1, num=20).tolist()
    prior_prob=0.5
    #dimensions = [10, 8, 7, 6]
    kernel_dim = 2
    c = 1 #what we add to in our kernel function
    
    #other tests
    Cs = [0.0007847599703514606]
    K=1
    dimensions = [7]
    
    scores = [[[] for _ in range(len(Cs))] for _ in range(len(dimensions))]
    trueLabels = []
    
    (folds, labels) = utils.k_split(matrix, v, numFolds)
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
        
        trueLabels.append(Lval)
        
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
            
            
            for i in range(len(Cs)):
                C = Cs[i]
                
                a = SVM.SVM_kernel_train_poly(M_train, Ltrain, C, K, kernel_dim, c)
                scores[j][i].append(SVM.SVM_kernel_poly_LLR(M_train, Ltrain, a, M_val, K, kernel_dim, c))
                
        
    trueLabels = numpy.hstack(trueLabels)
    
    #using the plot to visualize the results
    plt.figure()
    plt.ylim(0.25, 1)
    plt.xlim(xmin = min(Cs), xmax = max(Cs))
    plt.xscale("log")
    plt.xlabel('C')
    plt.ylabel('minDCF')
    
    
    colors = ['r', 'g', 'b', 'k', 'm', 'c']
    
    for j in range(len(dimensions)):
        scoresDIM = scores[j]
        minDCFV = []
        for i in range(len(scoresDIM)):
            #for each C in the dimenions
            scoresCDim = numpy.hstack(scoresDIM[i])
            minD = metrics.minimum_detection_costs(scoresCDim, trueLabels, prior_prob, 1, 1)
            minDCFV.append(minD)
        
        best_c = str(Cs[minDCFV.index(min(minDCFV))])
        print('Best C for dimensionality ' + str(dimensions[j]) + ' is ' + best_c + ' with minDCF ' + str(min(minDCFV)))
    
       
        if z_norm:
            plt.plot(Cs, minDCFV, 
                     label='SVM Poly (' + str(kernel_dim) +') z_norm dimension: ' + str(dimensions[j]), color=colors[j])
        else:
            plt.plot(Cs, minDCFV,
                     label='SVM Poly (' + str(kernel_dim) +') dimension: ' + str(dimensions[j]), color=colors[j])
    plt.grid()
    plt.legend()
    plt.show()
    return

def RadialSVM_validation(matrix, v, numFolds=5, z_norm=False):
    #first iteration
    #Cs = numpy.logspace(-5, 2, num=10).tolist()
    #gammas = [math.e**-6, math.e**-5, math.e**-4]
    #dimensions = [10]
    
    #second iteraion
    #Cs = numpy.logspace(-3, 2, num=10).tolist()
    #gammas = [math.e**-9, math.e**-8, math.e**-7]
    #dimensions = [10]
    
    #third
    #Cs = numpy.logspace(-1, 2, num=20).tolist()
    #gammas = [math.e**-7]
    #dimensions = [10, 8, 7, 6]
    
    #test different applications
    Cs = numpy.logspace(-3, 2, num=10).tolist()
    gammas = [math.e**-7]
    dimensions = [7]
    
    K=1
    prior_prob=0.09
    
    scores = [
        [
            [
                [] for _ in range(len(Cs))
            ] for _ in range(len(gammas))
        ] for _ in range(len(dimensions))
    ]
    trueLabels = []
    
    (folds, labels) = utils.k_split(matrix, v, numFolds)
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
        
        trueLabels.append(Lval)
        
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
            
            for g in range(len(gammas)): 
                minDCFV = []
                gamma = gammas[g]
                
                #we compute G and kern here since they don't depend on the C value
                #it's an optinal argument for both train and test
                G_train = SVM.SVM_compute_radial_kernel_train(M_train, K, gamma)
                kern = SVM.SVM_compute_radial_kernel_test(M_train, M_val, K, gamma)
                
                for i in range(len(Cs)):
                    C = Cs[i]
                
                    a = SVM.SVM_kernel_train_radial(M_train, zTrain, C, K, gamma, G=G_train)
                    scores[j][g][i].append(SVM.SVM_kernel_radial_LLR(M_train, M_val, K, gamma, zTrain, a, kern=kern))
                
        
    trueLabels = numpy.hstack(trueLabels)
    
    #using the plot to visualize the results
    plt.figure()
    plt.ylim(0.25, 1)
    plt.xlim(xmin = min(Cs), xmax = max(Cs))
    plt.xscale("log")
    plt.xlabel('C')
    plt.ylabel('minDCF')
    
    
    colors = ['r', 'g', 'b', 'k', 'm', 'c']
    color=0
    
    for j in range(len(dimensions)):
        scoresDIM = scores[j]
        
        for g in range(len(gammas)):
            scoresDimGamma = scoresDIM[g]
        
            minDCFV = []
            for i in range(len(scoresDimGamma)):
                #for each C in the dimenions
                scoresDimGammaC = numpy.hstack(scoresDimGamma[i])
                minD = metrics.minimum_detection_costs(scoresDimGammaC, trueLabels, prior_prob, 1, 1)
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

def GMM_validation(matrix, v, numFolds=5, z_norm=False):
    #first iteration
    #dimensions = [10]
    #iterationsT = range(3)
    #iterationsNT = range(6) #da 1 a 32 non target
    
    #second
    dimensions = [10, 9, 8, 7]
    iterationsT = [0]
    iterationsNT = [2]
    
    typeT = 'Full'
    typeNT = 'Diag'
    
    prior_prob=0.9
    
    scores = [
        [
            [
                [] for _ in range(len(iterationsNT))
            ] for _ in range(len(iterationsT))
        ] for _ in range(len(dimensions))
    ]
    trueLabels = []
    
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    color = 0
    
    (folds, labels) = utils.k_split(matrix, v, numFolds)
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
        
        trueLabels.append(Lval)
        
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
                
    
    trueLabels = numpy.hstack(trueLabels)
    
    plt.figure()
    plt.ylim(0.2, 1)
    plt.xlim(xmin = 0, xmax = 35)
    plt.xlabel('Non Target')
    plt.ylabel('minDCF')
    
    
    colors = ['r', 'g', 'b', 'k', 'm', 'c']
    color=0
    
    for j in range(len(dimensions)):
        scoresDIM = scores[j]
        
        for t in range(len(iterationsT)):
            scoresDimTarget = scoresDIM[t]
        
            minDCFV = []
            for nt in range(len(iterationsNT)):
                #for each C in the dimenions
                scoresDimTargetNonTarget = numpy.hstack(scoresDimTarget[nt])
                minD = metrics.minimum_detection_costs(scoresDimTargetNonTarget, trueLabels, prior_prob, 1, 1)
                minDCFV.append(minD)
    
            if z_norm:
                plt.plot([2**numnt for numnt in iterationsNT], minDCFV, 
                         label='GMM z_norm dimension: ' + str(dimensions[j]) + ' Target: ' + str(2**iterationsT[t]), color=colors[color])
            else:
                plt.plot([2**numnt for numnt in iterationsNT], minDCFV, 
                         label='GMM dimension: ' + str(dimensions[j]) + ' Target: ' + str(2**iterationsT[t]), color=colors[color])
            color += 1
            
            #best non target
            best_NT = str(iterationsNT[minDCFV.index(min(minDCFV))])
            
            print('Best nt for GMM target ' + str(2**iterationsT[t]) +' dimensionality ' + str(dimensions[j]) + ' is ' + best_NT + ' with minDCF ' + str(min(minDCFV)))
    
    plt.grid()
    plt.legend()
    plt.show()
    return


def compare_models(mTrain, v, numFolds=5):
    #for Quad log, svm rbf and GMM
    scores_1 = []
    scores_2 = []
    scores_3 = []
    trueLabels = []
    
    prior_train=(mTrain[:, v==1]).shape[1]/mTrain.shape[1]
    
    (folds, labels) = utils.k_split(mTrain, v, numFolds)
    
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
        
        #we need three dimensionalities: 8, 7 and 6
        P8 = utils.dimension_reduction_PCA(Mtrain, 8)
        M_train8 = numpy.dot(P8.T, Mtrain)
        M_val8 = numpy.dot(P8.T, Mval)
        
        P7 = utils.dimension_reduction_PCA(Mtrain, 7)
        M_train7 = numpy.dot(P7.T, Mtrain)
        M_val7 = numpy.dot(P7.T, Mval)
        
        P6 = utils.dimension_reduction_PCA(Mtrain, 6)
        M_train6 = numpy.dot(P6.T, Mtrain)
        M_val6 = numpy.dot(P6.T, Mval)
        
        M_train6 = LogReg.expanded_feature_space(M_train6)
        M_val6 = LogReg.expanded_feature_space(M_val6)
        
        
        #now we can compute the scores
        (w, b) = LogReg.quad_logistic_regression_train(M_train6, Ltrain, 0.001623776739188721, prior_train)
        scores_1.append(LogReg.quad_logistic_regression_LLR(w, b, M_val6))
        
        a = SVM.SVM_kernel_train_radial(M_train7, 2*Ltrain-1, 4, 1, math.e**-7)
        scores_2.append(SVM.SVM_kernel_radial_LLR(M_train7, M_val7, 1, math.e**-7, 2*Ltrain-1, a))
        
        GMMs = GMM.GMM_train(M_train8, Ltrain, 0, 2, typeT = 'Full', typeNT = 'Diag')
        scores_3.append(GMM.GMM_LLR(GMMs, M_val8))
        
        trueLabels.append(Lval)
        
    scores_1 = numpy.hstack(scores_1)
    scores_2 = numpy.hstack(scores_2)
    scores_3 = numpy.hstack(scores_3)
    trueLabels = numpy.hstack(trueLabels)
    
    metrics.bayes_error_plot_comparison([scores_1, scores_2, scores_3], trueLabels, ['QuadLogReg', 'SVM RBF', 'GMM'])
    metrics.DET([scores_1, scores_2, scores_3], trueLabels, ['QuadLogReg', 'SVM RBF', 'GMM'])
    return