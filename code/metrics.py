import numpy
import math
import sys
import utils
import matplotlib.pyplot as plt

def minimum_detection_costs (llr, labels, prior, Cfn, Cfp):
    scores = llr.tolist()
    scores.sort()
    scores.insert(0, float('-inf'))
    scores.insert(len(scores), float('inf'))

    dcfMin = sys.maxsize
    lastPredicted = []
    Conf = []
    BDummy = min(prior*Cfn, (1-prior)*Cfp)
    
    for threshold in scores:
        predicted = (llr > threshold).astype(int)
        
        #first threshold is always -inf
        if threshold != float('-inf'):
            #check for new labels
            iMismatch = [i for i in range(len(predicted)) if predicted[i] != lastPredicted[i]]
            
            if len(iMismatch) > 0:
                index = iMismatch[0]
                Conf[lastPredicted[index]][labels[index]] -= 1
                Conf[predicted[index]][labels[index]] += 1
        else:
            #first computation of min
            Conf = utils.plot_conf_M(predicted, labels, 2)
        
        lastPredicted = predicted
        
        FNR = Conf[0][1]/(Conf[0][1] + Conf[1][1])
        FPR = Conf[1][0]/(Conf[1][0] + Conf[0][0])
        dcf = prior*FNR*Cfn + (1-prior)*FPR*Cfn

        if dcf/BDummy < dcfMin:
            dcfMin = dcf/BDummy
    
    return dcfMin

def DET(D, trueLabels, names):
    plt.figure()
    plt.title('DET')
    plt.xlabel('FPR')
    plt.xscale("log")
    plt.ylabel('FNR')
    plt.yscale("log")
    
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    
    for i, scores in enumerate(D):
        thresholds = scores.tolist()
        thresholds.sort()
        thresholds.insert(0, float('-inf'))
        thresholds.insert(len(scores), float('inf'))
        
        FPR = numpy.zeros(len(thresholds))
        FNR = numpy.zeros(len(thresholds))
        lastPredicted = []
        Conf = []
        
        for idx, t in enumerate(thresholds):
            predicted = (scores > t).astype(int)
            
            if t != float('-inf'):
                #check for new labels
                iMismatch = [i for i in range(len(predicted)) if predicted[i] != lastPredicted[i]]
                
                if len(iMismatch) > 0:
                    index = iMismatch[0]
                    Conf[lastPredicted[index]][trueLabels[index]] -= 1
                    Conf[predicted[index]][trueLabels[index]] += 1
            else:
                Conf = utils.plot_conf_M(predicted, trueLabels, 2)
            
            lastPredicted = predicted
            
            FNR[idx] = Conf[0][1]/(Conf[0][1] + Conf[1][1])
            FPR[idx] = Conf[1][0]/(Conf[1][0] + Conf[0][0])
            
        plt.plot(FPR, FNR, label=names[i], color=colors[i])

    plt.legend()
    plt.grid()
    plt.show()
    
def bayes_error_plot_comparison(D, trueLabels, names, ylim=0.6):
    p = numpy.linspace(-3, 3, 21) #possible applications
    
    plt.figure()
    plt.title('Bayes error plot')
    plt.xlabel('log(π/1-π)')
    plt.ylabel('Cost')
    
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    
    #for each model we compute the error plot and we plot it
    for i, scores in enumerate(D):
        
        (dcfMinV, dcfV) = bayes_error_plot(p, scores, trueLabels)
        
        plt.plot(p, dcfMinV, color=colors[i], label=names[i] + '(min cost)')
        plt.plot(p, dcfV, colors[i] + '--', label=names[i] + '(act cost)')
        
        #print of our applications
        print('Application 0.09')
        (minDCF, actDCF) = bayes_error_plot([numpy.log(0.09/(1-0.09))], scores, trueLabels)
        print('For model ' + names[i] + ' minDCF ' + str(minDCF[0]) + ' actDCF ' + str(actDCF[0]))
    
        
        print('Application 0.5')
        (minDCF, actDCF) = bayes_error_plot([numpy.log(0.5/(1-0.5))], scores, trueLabels)
        print('For model ' + names[i] + ' minDCF ' + str(minDCF[0]) + ' actDCF ' + str(actDCF[0]))
        
        #print of our application
        print('Application 0.9')
        (minDCF, actDCF) = bayes_error_plot([numpy.log(0.9/(1-0.9))], scores, trueLabels)
        print('For model ' + names[i] + ' minDCF ' + str(minDCF[0]) + ' actDCF ' + str(actDCF[0]))
    
        
    plt.ylim(0, ylim)
    plt.legend()
    plt.show()

def bayes_error_plot(effPriorLogOdds, ll, labels):
    scores = ll.tolist()
    scores.sort()
    scores.insert(0, float('-inf'))
    scores.insert(len(scores), float('inf'))

    dcfV = []
    dcfMinV = []

    for p in effPriorLogOdds:
        # compute dcf using p as threashold
        prior = 1/(1 + math.exp(-p))

        dcfV.append(actual_dcf(ll, labels, prior, 1, 1))  # normalized
        
        dcfMinV.append(minimum_detection_costs(ll, labels, prior, 1, 1))
    return (dcfMinV, dcfV)

def actual_dcf(llr, labels, prior, Cfn, Cfp):
    threshold = -1 * numpy.log(prior/(1-prior))
    BDummy = min(prior*Cfn, (1-prior)*Cfp)
    
    predictions = (llr > threshold).astype(int)
    
    Conf = utils.plot_conf_M(predictions, labels, 2)
    
    FNR = Conf[0][1]/(Conf[0][1] + Conf[1][1])
    FPR = Conf[1][0]/(Conf[1][0] + Conf[0][0])
    dcf = prior*FNR*Cfn + (1-prior)*FPR*Cfn
    return dcf/BDummy