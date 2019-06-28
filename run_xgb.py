import os
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use('Agg') #this stops matplotlib trying to use Xwindows backend when running remotely
import matplotlib.pyplot as plt
import pandas as pd

from keras import callbacks

from MlClasses.MlData import MlData
from MlClasses.Dnn import Dnn

from MlClasses.Bdt import *
from MlClasses.asimov_obj import *
from MlClasses.xgboost_update import XGBClassifier

from MlFunctions.DnnFunctions import significanceLoss,significanceLossInvert,significanceLoss2Invert,significanceLossInvertSqrt,significanceFull,asimovSignificanceLoss,asimovSignificanceLossInvert,asimovSignificanceFull,truePositive,falsePositive


#make or save dfs
makeDfs = False
saveDfs = True
#appendInputName = 'leonid'
appendInputName = ''

prepareInputs = False #if false, inputs will be read in

#ML options
plotFeatureImportances = False
doCrossVal=False
makeLearningCurve=False

limitSize = 100000 #None #Make this an integer N_events if you want to limit output

makeHistograms=False

#DNN Options
normalLoss=True
sigLossInvert=True
asimovSigLossBothInvert=True
asimovSigLossSysts=[0.01,0.05,0.1,0.2,0.3,0.4,0.5]

#===== Define some useful variables =====

parser = argparse.ArgumentParser()
parser.add_argument("-plot","--makePlots", help = "Make a couple of plots",
                    action="store_true")
parser.add_argument("-class","--doClassification", help = "Make a simple network to carry out classification",
                    action="store_true")
parser.add_argument("-reg","--doRegression", help = "Make a simple network to carry out regressions",
                    action="store_true")
parser.add_argument("-bdt","--doBdtClassification", help = "Make a simple network to carry out classification by sklearn.tree",
                    action="store_true")
parser.add_argument("-xgb","--doXGBClassification", help = "Make a simple network to carry out classification by XGBoost",
                    action="store_true")
parser.add_argument("-hp","--doHyperOpt", help = "Make hyper-parametr optimization for XGBoost classification",
                    action="store_true")
parser.add_argument("-gs","--doGridSearch", help = "Carry out grid search", #if no further argument after classification type, default configs will be used
                    action="store_true")
args = parser.parse_args()

makePlots=args.makePlots
doClassification=args.doClassification
doRegression=args.doRegression
doBdtClassification=args.doBdtClassification
doXGBClassification=args.doXGBClassification
doHyperOpt=args.doHyperOpt
doGridSearch=args.doGridSearch

print "makePlots           ==>", makePlots
print "doClassification    ==>", doClassification
print "doRegression        ==>", doRegression
print "doBdtClassification ==>", doBdtClassification
print "doXGBClassification ==>", doXGBClassification
print "doHyperOpt          ==>", doHyperOpt
print "doGridSearch        ==>", doGridSearch


sigma = 0.1 #set flat systematic error

output='run_xgb_out' # an output directory (then make it if it doesn't exist)
if not os.path.exists(output): os.makedirs(output)

#class to stop training of dnns early
earlyStopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2)

#Use these to calculate the significance when its used for training
#Taken from https://twiki.cern.ch/twiki/bin/view/CMS/SummerStudent2017#SUSY
lumi=30. #luminosity in /fb
expectedBkgd=844000.*8.2e-4*lumi #cross section of ttbar sample in fb times efficiency measured by Marco
#expectedSignal = 17.6*0.059*lumi #for (900,100) #uncompressed model
expectedSignal = 228.195*0.14*lumi #for (600,400) #compressed model

#======== If not doing grid search ========#
#default dnn configurations
dnnConfigs={
     'dnn_batch4096':{'epochs':200,'batch_size':4096,'dropOut':None,'l2Regularization':None,'hiddenLayers':[1.0]},
        }	

#======== If carrying out grid search ========#
#If doing the grid search
def hiddenLayerGrid(nLayers,nNodes):
    hlg=[]
    for nn in nNodes:
        for nl in nLayers:
            hlg.append([nn for x in range(nl)])
        pass
    return hlg

dnnGridParams = dict(
        mlp__epochs=[10,20,50],
        mlp__batch_size=[32,64],
        mlp__hiddenLayers=hiddenLayerGrid([1,2,3,4,5],[2.0,1.0,0.5]),
        mlp__dropOut=[None,0.25,0.5],
        # mlp__activation=['relu','sigmoid','tanh'],
        # mlp__optimizer=['adam','sgd','rmsprop'],
        ## NOT IMPLEMENTED YET:
        # mlp__learningRate=[0.5,1.0],
        # mlp__weightConstraint=[1.0,3.0,5.0]
        )

bdtGridParams = dict(
        base_estimator__max_depth=[3,5],
        base_estimator__min_samples_leaf=[0.05,0.2],
        n_estimators=[400,800]
        )



 #======== Making or reading dataframes ========#
 #Either make the dataframes fresh from the trees or just read them in
if makeDfs:
    print "Making DataFrames"

    signalFile = []#'/nfs/dust/cms/group/susy-desy/marco/training_sample_new/stop_sample_0.root'
    bkgdFile = []#'/nfs/dust/cms/group/susy-desy/marco/training_sample_new/top_sample_0.root'

    for i in range(nInputFiles):
        signalFile.append(' /nfs/dust/cms/group/susy-desy/marco/leonid/stop_600_400/stop_samples_'+str(i)+'.root')
        bkgdFile.append('/nfs/dust/cms/group/susy-desy/marco/training_sample_new/top_sample_'+str(i)+'.root')

    signal = convertTree(signalFile,signal=True,passFilePath=True,tlVectors = ['selJet','sel_lep'])
    bkgd = convertTree(bkgdFile,signal=False,passFilePath=True,tlVectors = ['selJet','sel_lep'])

    # #Expand the variables to 1D
    signal = expandArrays(signal)
    bkgd = expandArrays(bkgd)

    if saveDfs:
        print 'Saving the dataframes'
        # Save the dfs?
        if not os.path.exists('dfs'): os.makedirs('dfs')
        print 'signal size:',len(signal)
        signal.to_pickle('dfs/signal'+appendInputName+'.pkl')
        print 'bkgd size:',len(bkgd)
        bkgd.to_pickle('dfs/bkgd'+appendInputName+'.pkl')
else:
    print "Loading DataFrames"

    signal = pd.read_pickle('dfs/signalleonid.pkl')
    bkgd = pd.read_pickle('dfs/bkgdleonid.pkl')
 
 #======== Prepare or read inputs ========# 
 #Carry out the organisation of the inputs or read them in if it's already done
    if prepareInputs:
        print 'Preparing inputs'

        # Put the data in a format for the machine learning:
        # combine signal and background with an extra column indicating which it is

        signal['signal'] = 1
        bkgd['signal'] = 0

        combined = pd.concat([signalleonid,bkgdleonid])

    else:
	print 'Reading prepared files'
        combined = pd.read_pickle('dfs/combinedleonid'+appendInputName+'.pkl')

 #======== Choosing variables ========#
#Now carry out machine learning (with some algo specific diagnostics)
    #Choose the variables to train on

    chosenVars = {
            
            # #The 4 vectors only, if don't want HL variables
            # 'fourVector':['signal',
            # 'sel_lep_pt','sel_lep_eta','sel_lep_phi','sel_lep_m',
            # 'selJet_phi','selJet_pt','selJet_eta','selJet_m','MET'],
            #
            # 'fourVectorBL':['signal','lep_type','selJetB',
            # 'sel_lep_pt','sel_lep_eta','sel_lep_phi','sel_lep_m',
            # 'selJet_phi','selJet_pt','selJet_eta','selJet_m','MET'],
            #
            # 'fourVectorMT':['signal',
            # 'sel_lep_pt','sel_lep_eta','sel_lep_phi','sel_lep_m',
            # 'selJet_phi','selJet_pt','selJet_eta','selJet_m','MET','MT'],
            #
            # 'fourVectorMT2W':['signal',
            # 'sel_lep_pt','sel_lep_eta','sel_lep_phi','sel_lep_m',
            # 'selJet_phi','selJet_pt','selJet_eta','selJet_m','MET','MT2W'],
            #
            # 'fourVectorHT':['signal',
            # 'sel_lep_pt','sel_lep_eta','sel_lep_phi','sel_lep_m',
            # 'selJet_phi','selJet_pt','selJet_eta','selJet_m','MET','HT'],
            #
            # #A vanilla analysis with HL variables and lead 3 jets
            'vanilla':['signal','HT','MET','MT','MT2W','n_jet',
            'n_bjet','sel_lep_pt0','sel_lep_eta0','sel_lep_phi0',
            'selJet_phi0','selJet_pt0','selJet_eta0','selJet_m0',
            'selJet_phi1','selJet_pt1','selJet_eta1','selJet_m1',
            'selJet_phi2','selJet_pt2','selJet_eta2','selJet_m2'],

            }

    trainedModels={}

    for varSetName,varSet in chosenVars.iteritems():

        print ''
        print '==========================='
        print 'Analysing var set '+varSetName
        print '==========================='

        #Pick out the expanded arrays
        columnsInDataFrame = []
        for k in combined.keys():
            for v in varSet:
                #Little trick to ensure only the start of the string is checked
                if varSetName is 'vanilla':
                    if ' '+v+' ' in ' '+k+' ': columnsInDataFrame.append(k)
                elif ' '+v in ' '+k: columnsInDataFrame.append(k)


        #Select just the features we're interested in
        #For now setting NaNs to 0 for compatibility
        combinedToRun = combined[columnsInDataFrame].copy()
        combinedToRun.fillna(0,inplace=True)

        # print columnsInDataFrame
        # print exit()




if makePlots:
    #======== Make a couple of plots: ========#

    #Calculate the weights for each event and add them to the dataframe
    signalWeight = expectedSignal/(combinedToRun.signal==1).sum() #divide expected events by number in dataframe
    bkgdWeight   = expectedBkgd/(combinedToRun.signal==0).sum()

    #Add a weights column with the correct weights for background and signal
    combinedToRun['weight'] = combinedToRun['signal']*signalWeight+(1-combinedToRun['signal'])*bkgdWeight

    #Choose some variables to plot and loop over them
    varsToPlot = ['HT']

    for v in varsToPlot:

        print 'Plotting',v
        maxRange=max(combinedToRun[v])
        #Plot the signal and background but stacked on top of each other
        plt.hist([combinedToRun[combinedToRun.signal==0][v],combinedToRun[combinedToRun.signal==1][v]], #Signal and background input
                label=['background','signal'],
                bins=50, range=[0.,maxRange], 
                stacked=True, color = ['g','r'],
                weights=[combinedToRun[combinedToRun.signal==0]['weight'],combinedToRun[combinedToRun.signal==1]['weight']]) #supply the weights
        plt.yscale('log')
        plt.xlabel(v)
        plt.legend()
        plt.savefig(os.path.join(output,'hist_'+v+'.pdf')) #save the histogram
        plt.clf() #Clear it for the next one

    combinedToRun = combinedToRun.drop('weight',axis=1) #drop the weight to stop inference from it as truth variable

#======== Now everything is ready can start the machine learning ========#

if plotFeatureImportances:
    print 'Making feature importances'
    #Find the feature importance with a random forest classifier
    featureImportance(combinedToRun,'signal','testPlots/mlPlots/'+varSetName+'/featureImportance')

print 'Splitting up data'

mlDataC = MlData(combinedToRun,'signal') #call it C for now to avoid editing

#Now split pseudorandomly into training and testing
#Split the development set into training and testing
#(forgetting about evaluation for now)
mlDataC.prepare(evalSize=0.0,testSize=0.33,limitSize=limitSize)
        
print 'Data output before standardise:'
mlDataC.output(number_of_lines=5)
        
mlDataC.standardise()

print 'Data output after standardise:'
mlDataC.output(number_of_lines=5)


      

if doClassification:

    #=============================================================
    #===== Make a simple network to carry out classification =====
    #=============================================================

    print 'Running classification'
    print '-----Timer start-----'
    start_time = time.time()

    #===== Setup and run the network  =====
    
    print 'Setting up network'
    
    if doGridSearch: 
        print 'Running DNN grid search'
        dnnC = Dnn(mlDataC,'testPlots/mlPlots/'+varSetName+'/dnnGridSearch')
        dnnC.setup()
        dnnC.gridSearch(param_grid=dnnGridParams,kfolds=3,epochs=20,batch_size=32,n_jobs=4)
   
    else:
        #deep neural net
        for name,config in dnnConfigs.iteritems():

            if normalLoss:
                print 'Defining and fitting DNN',name
                dnn = Dnn(mlDataC,'testPlots/mlPlots/'+varSetName+'/'+name)
                dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                        extraMetrics=[
                            significanceLoss(expectedSignal,expectedBkgd),significanceFull(expectedSignal,expectedBkgd),
                            asimovSignificanceFull(expectedSignal,expectedBkgd,sigma),truePositive,falsePositive
                            ])
                dnn.fit(epochs=config['epochs'],batch_size=128,callbacks=[earlyStopping])
                dnn.save()
                if doCrossVal:
                    print ' > Carrying out cross validation'
                    dnn.crossValidation(kfolds=5,epochs=config['epochs'],batch_size=config['batch_size'])
                if makeLearningCurve:
                    print ' > Making learning curves'
                    dnn.learningCurve(kfolds=5,n_jobs=1)

                print ' > Producing diagnostics'
                dnn.explainPredictions()
                dnn.diagnostics(batchSize=8192)
                dnn.makeHepPlots(expectedSignal,expectedBkgd,asimovSigLossSysts,makeHistograms=makeHistograms)

                trainedModels[varSetName+'_'+name]=dnn
            
            if sigLossInvert:
                print 'Defining and fitting DNN with significance loss function',name
                dnn = Dnn(mlDataC,'testPlots/mlPlots/sigLossInvert/'+varSetName+'/'+name)
                dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                        loss=significanceLossInvert(expectedSignal,expectedBkgd),
                        extraMetrics=[
                            significanceLoss(expectedSignal,expectedBkgd),significanceFull(expectedSignal,expectedBkgd),
                            asimovSignificanceFull(expectedSignal,expectedBkgd,sigma),truePositive,falsePositive
                        ])
                dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'],callbacks=[earlyStopping])
                dnn.save()
                print ' > Producing diagnostics'
                dnn.diagnostics(batchSize=8192)
                dnn.makeHepPlots(expectedSignal,expectedBkgd,asimovSigLossSysts,makeHistograms=makeHistograms)

                trainedModels[varSetName+'_sigLossInvert_'+name]=dnn

            if asimovSigLossBothInvert:

                for chosenSyst in asimovSigLossSysts:

                    systName = str(chosenSyst).replace('.','p')

                    #First set up a model that trains on the sig loss
                    print 'Defining and fitting DNN with inverted asimov significance loss function',name
                    dnn = Dnn(mlDataC,'testPlots/mlPlots/asimovSigLossBothInvertSyst'+systName+'/'+varSetName+'/'+name)
                    dnn.setup(hiddenLayers=config['hiddenLayers'],dropOut=config['dropOut'],l2Regularization=config['l2Regularization'],
                            loss=significanceLoss2Invert(expectedSignal,expectedBkgd),
                            extraMetrics=[
                                asimovSignificanceLossInvert(expectedSignal,expectedBkgd,chosenSyst),asimovSignificanceFull(expectedSignal,expectedBkgd,chosenSyst),
                                significanceFull(expectedSignal,expectedBkgd),truePositive,falsePositive
                            ])

                    dnn.fit(epochs=5,batch_size=config['batch_size'])
                    dnn.diagnostics(batchSize=8192,subDir='pretraining')
                    dnn.makeHepPlots(expectedSignal,expectedBkgd,[chosenSyst],makeHistograms=makeHistograms,subDir='pretraining')

                    #Now recompile the model with a different loss and train further
                    dnn.recompileModel(asimovSignificanceLossInvert(expectedSignal,expectedBkgd,chosenSyst))
                    dnn.fit(epochs=config['epochs'],batch_size=config['batch_size'],callbacks=[earlyStopping])
                    dnn.save()
                    print ' > Producing diagnostics'
                    dnn.diagnostics(batchSize=8192)
                    dnn.makeHepPlots(expectedSignal,expectedBkgd,[chosenSyst],makeHistograms=makeHistograms)

                    trainedModels[varSetName+'_asimovSigLossBothInvert_'+name+systName]=dnn

  

    print '----Timer stop----'
    print 'General CPU time: ', time.time()-start_time

#====================================#
#======== Bdt Classification ========#
#====================================#
if doBdtClassification:

    if doGridSearch:
        print 'Running BDT grid search'
        bdt = Bdt(mlDataC,'testPlots/mlPlots/'+varSetName+'/bdtGridSearch')
        bdt.setup(AdaBoostClassifier(DecisionTreeClassifier(max_depth=3,min_samples_leaf=0.05),
                          algorithm='SAMME',n_estimators=1000, learning_rate=0.5))
        # ====================DecisionTreeClassifier args=========================
        # min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
        # max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
        # class_weight=None, presort=False
        # ====================AdaBoostClassifier Args=============================
        # base_estimator=None, n_estimators=50, learning_rate=1.0,
        # algorithm=SAMME.R, random_state=None

        bdt.gridSearch(param_grid=bdtGridParams,kfolds=3,n_jobs=4)

    else:
        print 'Running BdtClassification'
        print '-----Timer start-----'
        start_time = time.time()

        print 'Defining BDT'
        bdt = Bdt(mlDataC,output+'/BDT')

        print 'Setup BDT'
        bdt.setup(AdaBoostClassifier(DecisionTreeClassifier(max_depth=3,min_samples_leaf=0.05),
                          algorithm='SAMME',n_estimators=1000, learning_rate=0.5))
        # ====================DecisionTreeClassifier args=========================
        # min_weight_fraction_leaf=0.0, max_features=None, random_state=None, 
        # max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
        # class_weight=None, presort=False
        # ====================AdaBoostClassifier Args=============================
        # base_estimator=None, n_estimators=50, learning_rate=1.0, 
        # algorithm=SAMME.R, random_state=None

        print 'Fitting BDT'
        bdt.fit()

    print 'Diagnostic BDT'  
    bdt.diagnostics()

    print 'Making HEP plots'
    bdt.makeHepPlots(expectedSignal,expectedBkgd,systematics=[0.2],makeHistograms=False)

    print '----Timer stop----'
    print 'General CPU time: ', time.time()-start_time

#====================================#
#======== XGB Classification ========#
#====================================#
if doXGBClassification:

    print 'Running XGBClassification'
    print '------Timer start--------'
    start_time = time.time()

    if doGridSearch:
        print 'Running XGB grid search'
        bdt = Bdt(mlDataC, 'testPlots/mlPlots/'+varSetName+'/xgbGridSearch')
        print 'Setup XGB'
        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Asimov objective function~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        bdt.setup(cls=XGBClassifier,objective=asimov_obj,expected_events=[expectedSignal,expectedBkgd],
                  sigma=0.3,separation_facet=0.98,
                  n_estimators=1000,subsample=0.8,max_depth=17,gamma=0,min_child_weight=2,colsample_bylevel=0.6)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Default objective function~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #bdt.setup(cls=XGBClassifier,objective='binary:logistic',expected_events=[expectedSignal,expectedBkgd],
                  #sigma=sigma,separation_facet=0.5,
                  #subsample=0.9,max_depth=8,gamma=0.1,min_child_weight=15,colsample_bylevel=0.9)

        bdt.setup_metrics(expectedSignal,expectedBkgd,sigma)
        bdt.gridSearch(param_grid=bdtGridParams,kfolds=3,n_jobs=4)

         # ========================XGBClassifier args===============================
        # max_depth=3, learning_rate=0.1, n_estimators=100, silent=True,
        # objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None,
        # gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,colsample_bytree=1,
        # colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,
        # random_state=0, seed=None, missing=None, **kwarg

        # bdt.change_separation_facet(0.5)
        # print bdt.asimov_output()

        # bdt.change_separation_facet(0.95)
        # print bdt.asimov_output()


    else:

        print 'Defining XGB'
        bdt = Bdt(mlDataC,output+'/XGB')


        print 'Setup XGB'
        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Asimov objective function~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        bdt.setup(cls=XGBClassifier,objective=asimov_obj,expected_events=[expectedSignal,expectedBkgd],
                  sigma=0.3,separation_facet=0.98,
                  n_estimators=1000,subsample=0.8,max_depth=17,gamma=0,min_child_weight=2,colsample_bylevel=0.6)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Default objective function~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #bdt.setup(cls=XGBClassifier,objective='binary:logistic',expected_events=[expectedSignal,expectedBkgd],
                  #sigma=sigma,separation_facet=0.5,
                  #subsample=0.9,max_depth=8,gamma=0.1,min_child_weight=15,colsample_bylevel=0.9)

        bdt.setup_metrics(expectedSignal,expectedBkgd,sigma)

        # ========================XGBClassifier args===============================
        # max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, 
        # objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, 
        # gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,colsample_bytree=1,
        # colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
        # random_state=0, seed=None, missing=None, **kwarg

        print 'Fitting XGB'
        bdt.fit()

        # bdt.change_separation_facet(0.5)
        # print bdt.asimov_output()

        # bdt.change_separation_facet(0.95)
        # print bdt.asimov_output()

        print 'Diagnostic XGB'  
        bdt.diagnostics()

        print 'Making HEP plots'
        bdt.makeHepPlots(expectedSignal,expectedBkgd,systematics=[0.1,0.3,0.5],makeHistograms=False)

        #print 'Making Hist plots'
        #bdt.makeHistPlot(expectedSignal,expectedBkgd)

    print '----Timer stop----'
    print 'General CPU time: ', time.time()-start_time

if doRegression:

    #=========================================================
    #===== Make a simple network to carry out regression =====
    #=========================================================

    #now we've seen a classification example, try a similar thing with regression
    #try to predict a higher level variable from the low level inputs

    print 'Running regression'

    print 'Preparing data'

    #Just pick the 4-vectors to train on
    subset = ['HT']
    for k in combined.keys():
        for v in ['selJet','sel_lep']:
            if ' '+v in ' '+k: subset.append(k)

    print 'Using subset',subset
    combinedToRun=combined[subset]

    combinedToRun=combinedToRun.fillna(0) #NaNs in the input cause problems

    #insert the dataframe without the background class and the variable for regression
    mlDataR = MlData(combinedToRun,'HT') 

    mlDataR.split(evalSize=0.0,testSize=0.3) #Split into train and test sets, leave out evaluation set for now

    #Now decide whether we want to standardise the dataset
    #it is worth seeing what happens to training with and without this option
    #(this must be done after the split to avoid information leakage)
    #mlDataR.standardise() #find this causes problems with regression

    print 'Setting up network'

    dnnR=Dnn(mlDataR,os.path.join(output,'regression'),doRegression=True)

    #here sets up with mean squared error and a linear output neuron
    dnnR.setup(hiddenLayers=[20,20],dropOut=None,l2Regularization=None)#,loss='mean_squared_error')

    print 'Fitting'
    dnnR.fit(epochs=100,batch_size=128,callbacks = [earlyStopping])

    print 'Making diagnostics'
    dnnR.diagnostics() #make regression specific diagnostics
