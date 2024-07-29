import validation
import utils
import Calibration
import evaluation

if(__name__ == '__main__'):
    f = open('train.txt', 'r')
    f2 = open('test.txt', 'r')
 
    matrixTrain, vectorTrain = utils.load(f)
    #utils.plot_scatte r(matrixTrain, vectorTrain)
    #utils.print_feature(matrixTrain,  vectorTrain)
    #utils.print_heat_map(matrixTrain, vectorTrain)
     
    #VALIDATION
    
    #validation.MVG_validation(matrixTrain, vectorTrain)
    #validation.Log_reg_validation(matrixTrain, vectorTrain)
    #validation.QuadLog_reg_validation(matrixTrain, vectorTrain)
    #validation.QuadLog_reg_validation(matrixTrain, vectorTrain, z_norm=True)
    #validation.LinearSVM_validation(matrixTrain, vectorTrain)
    #validation.PolySVM_validation(matrixTrain, vectorTrain )
    #validation.PolySVM_validation(matrixTrain, vectorTrain, z_norm=True )
    #validation.RadialSVM_validation(matrixTrain, vectorTrain)
    #validation.RadialSVM_validation(matrixTrain, vectorTrain, z_norm = True)
    #validation.GMM_validation(matrixTrain, vectorTrain)
    #validation.compare_models(matrixTrain, vectorTrain)
    
    # CALIBRATION
    #Calibration.QuadLogRegCalibrated_validation(matrixTrain, vectorTrain)
    #Calibration.RadialSVMCalibrated_validation(matrixTrain, vectorTrain)
    #Calibration.GMMCalibrated_validation(matrixTrain, vectorTrain)
    #Calibration.fusionSVMGMM_calibrated(matrixTrain, vectorTrain)
    #Calibration.fusionQuadLogRegGMM_calibrated(matrixTrain, vectorTrain)
    #Calibration.fusionAllModels_calibrated(matrixTrain, vectorTrain)
    
    #EVALUATION
    matrixTest, vectorTest = utils.load(f2)
    #evaluation.MVG_eval(matrixTrain, vectorTrain, matrixTest, vectorTest)
    #evaluation.QuadLog_reg_computecosts(matrixTrain, vectorTrain, matrixTest, vectorTest)
    #evaluation.QuadLog_reg_evaluation(matrixTrain, vectorTrain, matrixTest, vectorTest)
    #evaluation.SVMRBF_computecosts(matrixTrain, vectorTrain, matrixTest, vectorTest)
    #evaluation.SVMRBF_evaluation(matrixTrain, vectorTrain, matrixTest, vectorTest)
    #evaluation.GMM_nocal_computecosts(matrixTrain, vectorTrain, matrixTest, vectorTest)
    #evaluation.GMM_cal_computecosts(matrixTrain, vectorTrain, matrixTest, vectorTest)
    #evaluation.GMM_evaluation(matrixTrain, vectorTrain, matrixTest, vectorTest)
    
    #we recall all the computecosts using the optmial hyper parameters
    #evaluation.QuadLog_reg_computecosts(matrixTrain, vectorTrain, matrixTest, vectorTest)
    #evaluation.SVMRBF_computecosts(matrixTrain, vectorTrain, matrixTest, vectorTest)
    #evaluation.GMM_nocal_computecosts(matrixTrain, vectorTrain, matrixTest, vectorTest)
    
    # we do the same process for the fusion models
    #evaluation.fusionSVMGMM(matrixTrain, vectorTrain, matrixTest, vectorTest)
    #evaluation.fusionLogRegGMM(matrixTrain, vectorTrain, matrixTest, vectorTest)
    #evaluation.fusionAll(matrixTrain, vectorTrain, matrixTest, vectorTest)
    
    f.close()
    f2.close()