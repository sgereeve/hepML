import os
import numpy as np
from functools import partial, update_wrapper
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials,space_eval

from pandasPlotting.Plotter import Plotter
from asimovErrors import Z,eZ,asimov_scorer_function,asimov_metric
from MlClasses.PerformanceTests import classificationReport,rocCurve,compareTrainTest,learningCurve
from MlClasses.Config import Config
#from MlFunctions.DnnFunctions import *




class Bdt(object):
    '''Take some data split into test and train sets and train a bdt on it'''
    def __init__(self,data,output=None):

        print "Data split result: "
        print "train:",len(data.y_train)
        print "test:",len(data.y_test)
        print "eval:",len(data.y_eval)

        self.data = data 
        self.output = output
        self.config = Config(output=output)

        self.score=None
        self.crossValResults=None

        # should be checked it valid for Bdt
        self.scoreTypes = ['acc']

        self.bestHyperOpr_score = None
        self.bestHyperOpt_param = None
  
        #self.scorer = metrics.make_scorer(asimov_scorer_function,
        #    greater_is_better = True, needs_proba = False)
    
    def setup(self,cls=None,objective=None,expected_events=[None,None],sigma=0.1,**bdtArgs):
        '''This method represents initialization of bosted DT. In case of objective != None - cls has to be initialized with 
        customize objective function'''
        expectedSignal,expectedBack = expected_events[0],expected_events[1]

        if objective == None:
            print "None objective is choosen!"
            self.cls = cls
            self.bdt = self.cls(**bdtArgs)
            self.init_param = dict(objective=objective,**bdtArgs) 

        elif isinstance(objective, basestring):
            print "Default objective is choosen!"
            self.cls = cls
            self.bdt = self.cls(objective=objective,**bdtArgs)
            self.init_param = dict(objective=objective,**bdtArgs) 

        elif objective != None:
            print "Custome objective is choosen!"
            obj_fun = partial(objective,
                            expectedSignal = expectedSignal,
                            expectedBkgd = expectedBack,
                            sigma=sigma)
            self.init_param = dict(objective=obj_fun,**bdtArgs) 
            self.cls = cls 
            self.bdt = self.cls(**self.init_param) 

        print "FACET:",self.bdt.separation_facet
        # self.config.addToConfig('Class used: ',self.bdt)
        self.config.addToConfig('Vars used: ',self.data.X.columns.values)
        self.config.addToConfig('BDT was activated: ',type(self.bdt).__name__)
        self.config.addToConfig('nEvalEvents',len(self.data.y_eval.index))
        self.config.addToConfig('nDevEvents',len(self.data.y_dev.index))
        self.config.addToConfig('nTrainEvents',len(self.data.y_train.index))
        self.config.addToConfig('nTestEvents',len(self.data.y_test.index))
        self.config.addToConfig('DT arguments',bdtArgs)
    
    def setup_metrics(self,expectedSignal,expectedBkgd,sigma):
        '''Subroutine to setup metrics which is used 
        inside hyperopt minimization'''
        self.scorer = partial(asimov_scorer_function,
                           expectedBkgd = expectedBkgd,
                           expectedSignal = expectedSignal,
                           sig = sigma)
        self.metrics = partial(asimov_metric,
                           expectedBkgd = expectedBkgd,
                           expectedSignal = expectedSignal,
                           sig = sigma)
    
        #update_wrapper(metrics_,partial)

        #self.scorer = metrics.make_scorer(metrics_, 
        #                                  greater_is_better = True,
        #                                  needs_proba = False)


    def fit(self):
        # self.history = self.bdt.fit(self.data.X_train, self.data.y_train)

        self.history = self.bdt.fit(self.data.X_train,self.data.y_train,
                                    eval_set=[(self.data.X_test,self.data.y_test)],
                                    early_stopping_rounds=10,
                                    eval_metric=self.metrics,verbose=True)

    def predict_proba(self,X_test):
        return self.bdt.predict_proba(X_test)[:,1]

    def crossValidation(self,kfolds=3,n_jobs=4):
        '''K-means cross validation'''
        self.crossValResults = cross_val_score(self.bdt, self.data.X_dev, self.data.y_dev,scoring='accuracy',n_jobs=n_jobs,cv=kfolds)
        self.config.addLine('CrossValidation')
        self.config.addToConfig('kfolds',kfolds)
        self.config.addLine('')
        
    def hyperopt_train_test(self,param):
        #function calculate accuracy and returns formated output for HyperOpt
        kfolds=3
        n_jobs=6
        cls = self.cls(**dict(self.init_param,**param))
        # cls.separation_facet = self.separation_facet 

        CV_gen = cross_val_score(cls, self.data.X_train, self.data.y_train,scoring=self.scorer,n_jobs=n_jobs,cv=kfolds)
        # print CV_gen
        score = CV_gen.mean()

        if self.bestHyperOpr_score == None:
            self.bestHyperOpr_score = abs(score)
            self.bestHyperOpt_param = param
        
        if abs(score) < self.bestHyperOpr_score:
            self.bestHyperOpr_score = score
            self.bestHyperOpt_param = param

        print 'best result: ',self.bestHyperOpr_score, 1.0/self.bestHyperOpr_score, self.bestHyperOpt_param
        print 'current result: ', score, 1.0/score, param
        return {'loss': abs(score), 'status': STATUS_OK}

    def print_best_hyperopt_results(self):
        print '~~~~~~~~~~~ best values ~~~~~~~~~~~~~~'
        print 'best:',self.bestHyperOpr_score
        print 'parametrs:',self.bestHyperOpt_param

    def asimov_output(self):
        return asimov_scorer_function(self.data.y_test,self.testPrediction())

    def gridSearch(self,param_grid,kfolds=3,n_jobs=4):
        '''Implementation of the sklearn grid search for hyper parameter tuning, 
        making use of kfolds cross validation.
        Pass a dictionary of lists of parameters to test on. Choose number of cores
        to run on with n_jobs, -1 is all of them'''

        grid = GridSearchCV(estimator=self.bdt,param_grid=param_grid,scoring='accuracy',n_jobs=n_jobs,cv=kfolds)
        self.gridResult = grid.fit(self.data.X_dev, self.data.y_dev)

        #Save the results
        if not os.path.exists(self.output): os.makedirs(self.output)
        outFile = open(os.path.join(self.output,'gridSearchResults.txt'),'w')
        outFile.write("Best: %f using %s \n\n" % (self.gridResult.best_score_, self.gridResult.best_params_))
        means = self.gridResult.cv_results_['mean_test_score']
        stds = self.gridResult.cv_results_['std_test_score']
        params = self.gridResult.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            outFile.write("%f (%f) with: %r\n" % (mean, stdev, param))
        outFile.close()

    def saveConfig(self):

        if not os.path.exists(self.output): os.makedirs(self.output)
        self.config.saveConfig()

    def classificationReport(self,doEvalSet=False):

        if doEvalSet: #produce report for dev and eval sets instead
            X_test=self.data.X_eval
            y_test=self.data.y_eval
            X_train=self.data.X_dev
            y_train=self.data.y_dev
            append='_eval'
        else:
            X_test=self.data.X_test
            y_test=self.data.y_test
            X_train=self.data.X_train
            y_train=self.data.y_train
            append=''

        if not os.path.exists(self.output): os.makedirs(self.output)
        f=open(os.path.join(self.output,'classificationReport'+append+'.txt'),'w')
        f.write( 'Performance on test set:')
        classificationReport(self.bdt.predict_class(X_test),self.predict_proba(X_test),y_test,f)

        f.write( '\n' )
        f.write('Performance on training set:')
        classificationReport(self.bdt.predict_class(X_train),self.predict_proba(X_train),y_train,f)

        if self.crossValResults is not None:
            f.write( '\n\nCross Validation\n')
            f.write("Cross val results: %.2f%% (%.2f%%)" % (self.crossValResults.mean()*100, self.crossValResults.std()*100))
        
    def rocCurve(self,doEvalSet=False):

        if doEvalSet: #produce report for dev and eval sets instead
            rocCurve(self.predict_proba(self.data.X_eval),self.data.y_eval,output=self.output,append='_eval')
            rocCurve(self.predict_proba(self.data.X_dev),self.data.y_dev,output=self.output,append='_dev')
        else:
            rocCurve(self.predict_proba(self.data.X_test),self.data.y_test,output=self.output)
            rocCurve(self.predict_proba(self.data.X_train),self.data.y_train,output=self.output,append='_train')

    def compareTrainTest(self,doEvalSet=False):
        if doEvalSet:
            compareTrainTest(self.predict_proba,self.data.X_dev,self.data.y_dev,\
                    self.data.X_eval,self.data.y_eval,self.output,append='_eval')
        else:
            compareTrainTest(self.predict_proba,self.data.X_train,self.data.y_train,\
                    self.data.X_test,self.data.y_test,self.output)

    def learningCurve(self,kfolds=3,n_jobs=1):
        learningCurve(self.bdt,self.data.X_dev,self.data.y_dev,self.output,cv=kfolds,n_jobs=n_jobs)

    def diagnostics(self,doEvalSet=False):

        self.saveConfig()
        self.classificationReport(doEvalSet=doEvalSet)
        self.rocCurve(doEvalSet=doEvalSet)
        self.compareTrainTest(doEvalSet=doEvalSet)

    def plotDiscriminator(self):
        plotDiscriminator(self.bdt,self.data.X_test,self.data.y_test, self.output)

    def testPrediction(self):
        return self.bdt.predict_class(self.data.X_test)

    def getAccuracy(self):
        if not self.score:
            self.score = self.bdt.score(self.data.X_test,self.data.y_test)
        return self.score

    def makeHistPlot(self,expectedSignal,expectedBkgd):
        #===== Make a couple of plots: =====


        signal_ = self.data.y_test
        dataSet_ = self.data.X_test
        
        print "Initial set: ", len(signal_)

        #Calculate the weights for each event and add them to the dataframe
        signalWeight = expectedSignal/(signal_==1).sum() 
        bkgdWeight   = expectedBkgd/(signal_==0).sum()

        weight_ = signal_ *signalWeight+(1-signal_)*bkgdWeight


        predict = self.bdt.predict_class(dataSet_)
        #print predict

        signal = signal_[predict == 1]
        dataSet = dataSet_[predict == 1]

        print "Final set: ", len(signal)

        #Add a weights column with the correct weights for background and signal
        weight = signal *signalWeight+(1-signal)*bkgdWeight

        #Choose some variables to plot and loop over them
        varsToPlot = ['HT']

        # Before Classification:
        for v in varsToPlot:

            print 'Plotting',v
            maxRange=max(dataSet_[v])
            minRange=min(dataSet_[v])
            #Plot the signal and background but stacked on top of each other
            plt.hist([dataSet_[signal_==1][v],dataSet_[signal_==0][v]], #Signal and background input
                    label=['signal','background'],
                    bins=50, range=[minRange,maxRange], 
                    stacked=True, color = ['r','g'],
                    weights=[weight_[signal_==1],weight_[signal_==0]]) #supply the weights
            # plt.yscale('log')
            plt.xlabel(v)
            plt.plot([], [], ' ', label="Events: %d" % len(signal_))
            plt.legend()
            plt.savefig(os.path.join(self.output,'hist_'+v+'_before.pdf')) #save the histogram
            plt.clf()

        # After Classification:
        for v in varsToPlot:

            print 'Plotting',v
            maxRange=max(dataSet[v])
            minRange=min(dataSet[v])
            #Plot the signal and background but stacked on top of each other
            plt.hist([dataSet[signal==1][v],dataSet[signal==0][v]], #Signal and background input
                    label=['signal','background'],
                    bins=50, range=[minRange,maxRange], 
                    stacked=True, color = ['r','g'],
                    weights=[weight[signal==1],weight[signal==0]]) #supply the weights
            # plt.yscale('log')
            plt.xlabel(v)
            plt.plot([], [], ' ', label="Events: %d" % len(signal))
            plt.legend()
            plt.savefig(os.path.join(self.output,'hist_'+v+'_after.pdf')) #save the histogram
            plt.clf() #Clear it for the next one

    def makeHepPlots(self,expectedSignal,expectedBackground,systematics=[0.0001],makeHistograms=True,subDir=None,customPrediction=None):
        '''Plots intended for binary signal/background classification
        
            - Plots significance as a function of discriminator output
            - Plots the variables for signal and background given different classifications 
            - If reference variables are available in the data they will also be plotted 
            (to be implemented, MlData.referenceVars)
        '''

        if subDir:
            oldOutput = self.output
            self.output=os.path.join(self.output,subDir)


        names = self.data.X.columns.values

        #Then predict the results and save them
        if customPrediction is None:
            # predictionsTest = self.testPrediction()
            predictionsTest = self.predict_proba(self.data.X_test)
        else:
            predictionsTest = customPrediction

        dataTest=pd.DataFrame(self.data.scaler.inverse_transform(self.data.X_test),columns=names)

        #Add the predictions and truth to a data frame
        dataTest['truth']=self.data.y_test.as_matrix()
        dataTest['pred']=predictionsTest

        signalSizeTest = len(dataTest[dataTest.truth==1])
        bkgdSizeTest = len(dataTest[dataTest.truth==0])
        signalWeightTest = float(expectedSignal)/signalSizeTest
        bkgdWeightTest = float(expectedBackground)/bkgdSizeTest

        def applyWeight(row,sw,bw):
            if row.truth==1: return sw
            else: return bw

        dataTest['weight'] = dataTest.apply(lambda row: applyWeight(row,signalWeightTest,bkgdWeightTest), axis=1)

        #save it for messing about
        #dataTest.to_pickle('dataTestSigLoss.pkl')

        #Produce a cumulative histogram of signal and background (truth) as a function of score
        #Plot it with a log scTrue

        h1=plt.hist(dataTest[dataTest.truth==0]['pred'],weights=dataTest[dataTest.truth==0]['weight'],bins=5000,color='b',alpha=0.8,label='background',cumulative=-1)
        h2=plt.hist(dataTest[dataTest.truth==1]['pred'],weights=dataTest[dataTest.truth==1]['weight'],bins=5000,color='r',alpha=0.8,label='signal',cumulative=-1)
        # plt.yscale('log')
        plt.ylabel('Cumulative event counts / 0.02')
        plt.xlabel('Classifier output')
        plt.legend()
 
        plt.savefig(os.path.join(self.output,'cumulativeWeightedDiscriminator.pdf'))
        plt.clf()

        #From the cumulative histograms plot s/b and s/sqrt(s+b)

        s=h2[0]
        b=h1[0]

        plt.plot((h1[1][:-1]+h1[1][1:])/2,s/b)
        plt.title('sig/bkgd on test set')
        plt.savefig(os.path.join(self.output,'sigDivBkgdDiscriminator.pdf'))
        plt.clf()

        plt.plot((h1[1][:-1]+h1[1][1:])/2,s/np.sqrt(s+b))
        plt.title('sig/sqrt(sig+bkgd) on test set, best is '+str(max(s/np.sqrt(s+b))))
        plt.savefig(os.path.join(self.output,'sensitivityDiscriminator.pdf'))
        plt.clf()

        for systematic in systematics:
            # sigB=systematic*b
            #
            # toPlot=np.sqrt(2*( (s+b) * np.log( (s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB) ) - b*b*np.log( 1+sigB*sigB*s/(b*(b+sigB*sigB)) ) / (sigB*sigB) ))
            #plt.plot((h1[1][:-1]+h1[1][1:])/2,Z(s,b,systematic))
            toPlot = Z(s,b,systematic)
            plt.plot((h1[1][:-1]+h1[1][1:])/2,toPlot)
            es = signalWeightTest*np.sqrt(s/signalWeightTest)
            eb = bkgdWeightTest*np.sqrt(b/bkgdWeightTest)
            error=eZ(s,es,b,eb,systematic)
            # plt.plot((h1[1][:-1]+h1[1][1:])/2,toPlot-error)
            # plt.plot((h1[1][:-1]+h1[1][1:])/2,toPlot+error)
            plt.fill_between((h1[1][:-1]+h1[1][1:])/2,toPlot-error,toPlot+error,linewidth=0,alpha=0.6)
            maxIndex=np.argmax(toPlot)
            plt.title('Systematic '+str(systematic)+', s: '+str(round(s[maxIndex],1))+', b:'+str(round(b[maxIndex],1))+', best significance is '+str(round(toPlot[maxIndex],2))+' +/- '+str(round(error[maxIndex],2)))
            
            plt.xlabel('Cut on classifier score')
            plt.ylabel('Asimov estimate of significance')
            plt.savefig(os.path.join(self.output,'asimovDiscriminatorSyst'+str(systematic).replace('.','p')+'.pdf'))
            plt.clf()

            print "Best Asimov: ", str(round(toPlot[maxIndex],2))+' +/- '+str(round(error[maxIndex],2))
            print "Probability: ", ((h1[1][:-1]+h1[1][1:])/2)[maxIndex]    

            if makeHistograms: #Do this on the full set


                #Start with all the data and standardise it
                if self.data.standardised:
                    data = pd.DataFrame(self.data.scaler.transform(self.data.X))
                    #data = self.data.X.apply(self.data.scaler.transform)
                else:
                    data = self.data.X

                predictions= self.bdt.predict_class(data.as_matrix())
            
                #Now unstandardise
                if self.data.standardised:
                    data=pd.DataFrame(self.data.scaler.inverse_transform(data),columns=names)

                data['truth']=self.data.y.as_matrix()
                data['pred']=predictions

                signalSize = len(data[data.truth==1])
                bkgdSize = len(data[data.truth==0])
                signalWeight = float(expectedSignal)/signalSize
                bkgdWeight = float(expectedBackground)/bkgdSize

                data['weight'] = data.apply(lambda row: applyWeight(row,signalWeight,bkgdWeight), axis=1)

                #Plot all other interesting variables given classification
                p = Plotter(data,os.path.join(self.output,'allHistsSyst'+str(systematic).replace('.','p')))
                p1 = Plotter(data[data.pred>float(maxIndex)/len(toPlot)],os.path.join(self.output,'signalPredHistsSyst'+str(systematic).replace('.','p')))
                p2 = Plotter(data[data.pred<float(maxIndex)/len(toPlot)],os.path.join(self.output,'bkgdPredHistsSyst'+str(systematic).replace('.','p')))

                p.plotAllStackedHists1D('truth',weights='weight',log=True)
                p1.plotAllStackedHists1D('truth',weights='weight',log=True)
                p2.plotAllStackedHists1D('truth',weights='weight',log=True)


        if subDir:
            self.output=oldOutput 


        pass
