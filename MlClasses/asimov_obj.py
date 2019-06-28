import numpy as np
from  pandas import *
import math
from scipy import optimize
from numpy import log,power,sqrt
from autograd.numpy import log,power,sqrt
from autograd import grad,hessian,hessian_vector_product

def evalAsimov(preds, dtrain):
    labels = dtrain.get_label()
    return 'my-error', float(sum(labels != (preds > 0.0))) / len(labels)

def asimov(X, sig):
    eps=0.000001
    s,b = X[0],X[1]

    result = 1./(sqrt( -2.0/(sig*sig)*log( b/( eps + b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)+\
        2.0*( b+s)*log(( b+s)*( b+(b*b)*(sig*sig))/(eps + (b*b)+( b+s)*(b*b)*(sig*sig))+eps)))
    return result

def grad_asimov(X,sig):
    return grad(asimov)(X,sig)

def asimov_obj(y_true,preds,expectedSignal=None,expectedBkgd=None,sigma=0.1):
    

    # print y_true,preds
    y = y_true
    # weight = dtrain.get_weight()

#dk
#    yhat = 1.0 / (1.0 + np.exp(-preds))
    yhat = preds
#    print len(y_true),y_true,
#    print np.min(preds),np.max(preds),preds    
    
    #print 'expected:',expectedSignal ,expectedBkgd
    #DK OK
    signalWeight=float(expectedSignal)/np.sum(y_true)
    bkgdWeight=float(expectedBkgd)/np.sum(1-y_true) 

    #print 'weights:', signalWeight, bkgdWeight
    # approx ams using soft probability instead of hard predicted class
    #s = np.sum( weight * y * yhat )
    #b = np.sum( weight * (1.-y) * yhat )
    s = np.sum( y * yhat ) * signalWeight 
    b = np.sum( (1.-y) * yhat ) * bkgdWeight 
#    print "signal:",s," back:", b,asimov([s,b],sigma),sigma,expectedSignal,expectedBkgd
    eps = np.sqrt(np.finfo(float).eps)
#dk
#    s_prime,b_prime = optimize.approx_fprime([s,b], asimov, [eps*s,eps*b],sigma) 
#    print 'Num:  ',s_prime,b_prime
    #numerical same
    s_prime,b_prime = grad_asimov([s,b],sigma)  
#    print 'Auto: ',s_prime,b_prime

    # grad = (s_prime * y + b_prime * (1.-y)) * weight * yhat * (1.-yhat)
#dk
    grad = (s_prime*(yhat-y)+b_prime*(1-(yhat-y)))
#    grad = -(s_prime*(yhat-y)+b_prime*(yhat-y))
    # grad = -(s_prime*y+b_prime*(1-y))
    # grad = optimize.approx_fprime(yhat, asimov, eps*yhat, y, weight) 
#proper calc
#    grad =-(s_prime*signalWeight*y+b_prime*bkgdWeight*(1-y))/asimov([s,b],sigma)**2
    grad = (s_prime*signalWeight*y+b_prime*bkgdWeight*(1-y))*10
#dk
    hess = np.ones(yhat.shape)/(10**3) #constant

#    print grad, hess
    return grad, hess