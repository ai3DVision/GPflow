import GPflow
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import cProfile
import csv
from munkres import Munkres
from IPython import embed

optimizationLoops = 50
tol = 1e-10

def readCsvFile( fileName ):
    reader = csv.reader(open(fileName,'r') )
    dataList = []
    for row in reader:
        dataList.append( [float(elem) for elem in row ] )
    
    return np.array( dataList )
    
def getTrainingTestData():
    overallX = readCsvFile( 'train_inputs' )
    overallY = readCsvFile( 'train_outputs' )
    
    trainIndeces = []
    testIndeces = []
    
    nPoints = overallX.shape[0]
    
    for index in range(nPoints):
        if ( (index%4) == 0):
            trainIndeces.append( index )
        else:
            testIndeces.append( index )
            
    xtrain = overallX[ trainIndeces,: ]
    xtest = overallX[ testIndeces, : ]
    ytrain = overallY[ trainIndeces, : ]
    ytest  = overallY[ testIndeces, : ]
    
    return xtrain,ytrain,xtest,ytest
    
def getPredictPoints():
    predpoints = readCsvFile( 'test_inputs' )
    return predpoints

def getKernel():
    return GPflow.kernels.RBF(1)

def getRegressionModel(X,Y):
    m = GPflow.gpr.GPR(X, Y, kern=getKernel() )
    m.likelihood.variance = 1.
    m.kern.lengthscales = 1.
    m.kern.variance = 1.
    return m
    
def getSparseModel(X,Y,isFITC=False):
    if not(isFITC):
        m = GPflow.sgpr.SGPR(X, Y, kern=getKernel(),  Z=X.copy() )
    else:
        m = GPflow.sgpr.GPRFITC(X, Y, kern=getKernel(),  Z=X.copy() )
    return m

def printModelParameters( model ):
    print "Likelihood variance ", model.likelihood.variance, "\n"
    print "Kernel variance ", model.kern.variance, "\n"
    print "Kernel lengthscale ", model.kern.lengthscales, "\n"

def plotPredictions( ax, model, color, label ):
    xtest = np.sort( readCsvFile( 'test_inputs' ) )
    predMean, predVar = model.predict_y(xtest)
    ax.plot( xtest, predMean, color, label=label )
    ax.plot( xtest, predMean + 2.*np.sqrt(predVar),color )
    ax.plot( xtest, predMean - 2.*np.sqrt(predVar), color )

def trainSparseModel(xtrain,ytrain,exact_model,isFITC):
    sparse_model = getSparseModel(xtrain,ytrain,isFITC)
    sparse_model.likelihood.variance._array = exact_model.likelihood.variance._array.copy()
    sparse_model.kern.lengthscales._array = exact_model.kern.lengthscales._array.copy()
    sparse_model.kern.variance._array = exact_model.kern.variance._array.copy()
    for x in range(optimizationLoops):
       print x
       sparse_model.optimize( display=False, max_iters = 1000 , tol=1e-10 )
       print sparse_model.compute_log_likelihood()
    return sparse_model    

def plotComparisonFigure(xtrain, sparse_model,exact_model, ax_predictions, ax_inducing_points, title, reassignInducing=False ):
    plotPredictions( ax_predictions, exact_model, 'g', label='Exact model' )
    plotPredictions( ax_predictions, sparse_model, 'b', label=title )
    ax_predictions.legend()
    #plt.plot( xtrain, np.ones( xtrain.shape ), 'ro' )
    ax_predictions.plot( sparse_model.Z._array , -1.*np.ones( xtrain.shape ), 'ko' )
    if not(reassignInducing):
        ax_inducing_points.plot( xtrain, sparse_model.Z._array, 'bo' )
    else:
        ax_inducing_points.plot( xtrain, reassignInducingPoints( xtrain, sparse_model.Z._array) , 'bo' )
    xs= np.linspace( ax_inducing_points.get_xlim()[0], ax_inducing_points.get_xlim()[1], 200 )
    ax_inducing_points.plot( xs, xs, 'g' )
    ax_inducing_points.set_xlabel('Optimal inducing point position')
    ax_inducing_points.set_ylabel('Learnt inducing point position')
    
def trainVFEwithFITC(xtrain,ytrain,FITCmodel):
    vfeModel = GPflow.sgpr.SGPR( xtrain, ytrain, kern=getKernel(),  Z=FITCmodel.Z._array.copy() )
    vfeModel.likelihood.variance._array = FITCmodel.likelihood.variance._array.copy()
    vfeModel.kern.lengthscales._array = FITCmodel.kern.lengthscales._array.copy()
    vfeModel.kern.variance._array = FITCmodel.kern.variance._array.copy()
    for x in range(optimizationLoops):
       print x
       vfeModel.optimize( display=False, max_iters = 1000 , tol=1e-10 )    
       print vfeModel.compute_log_likelihood()

    return vfeModel

def reassignInducingPoints( xtrain, inducingPoints ):
    f_xtrain = xtrain.flatten()
    f_inducingPoints = inducingPoints.flatten()
    
    differenceMatrix = (f_xtrain[None,:] - f_inducingPoints[:,None])
    absDifferenceMatrix = np.abs( differenceMatrix ) 
    
    #Find optimal assignment using Hungarian algorithm.
    inds = Munkres().compute( absDifferenceMatrix )
    #embed()
    
    reassignedInducingPoints = f_inducingPoints[inds]
    return reassignedInducingPoints
    

def snelsonDemo():
    xtrain,ytrain,xtest,ytest = getTrainingTestData()
    fig, axes = plt.subplots(3,2)
    
    
    #run exact inference on training data.
    exact_model = getRegressionModel(xtrain,ytrain)
    exact_model.optimize(max_iters = 2000000, tol=tol )
   
    #run sparse model on training data intialized from exact optimal solution.
    VFEmodel = trainSparseModel(xtrain,ytrain,exact_model,False)
    FITCmodel = trainSparseModel(xtrain,ytrain,exact_model,True)
    VFEmodelAdverserial = trainVFEwithFITC(xtrain,ytrain,FITCmodel)

    print "Exact model parameters \n"
    printModelParameters( exact_model )
    print "Sparse model parameters for VFE optimization \n"
    printModelParameters( VFEmodel )
    print "Sparse model parameters for FITC optimization \n"
    printModelParameters( FITCmodel )
    print "Sparse model parameters for VFE adverserial optimization \n"
    printModelParameters( VFEmodelAdverserial )
    
    plotComparisonFigure( xtrain, FITCmodel, exact_model, axes[0,0], axes[0,1], "FITC" )
    plotComparisonFigure( xtrain, VFEmodel, exact_model, axes[1,0], axes[1,1], "VFE" )
    plotComparisonFigure( xtrain, VFEmodelAdverserial, exact_model, axes[2,0], axes[2,1], "VFE adverserial ", True ) 

    axes[0,0].set_ylabel('FITC')
    axes[1,0].set_ylabel('VFE')
    axes[2,0].set_ylabel('VFE adverserial')
    
    fig, axes = plt.subplots(1,1)
    inds = np.argsort( xtrain.flatten() )
    axes.plot( xtrain[inds,:], ytrain[inds,:], 'ro' )
    plotPredictions( axes, exact_model, 'g', None )
    


    
    embed()
    #plt.show()
    
if __name__ == '__main__':
    snelsonDemo()
    
