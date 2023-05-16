# -*- coding: utf-8 -*-
"""
Created on Mon May 31 09:57:21 2021

@author: olivi
"""
# %% importing libraries
from time import time
import os
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from numpy import linalg
from itertools import cycle
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix, plot_roc_curve,roc_curve,roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold,cross_val_score, GridSearchCV, RandomizedSearchCV 
from skopt import BayesSearchCV
import pickle
import warnings
warnings.filterwarnings('ignore')

# %% Load Train and Test datasets

def import_fused(timept,train_or_test,N_or_P):
    """
    Parameters
    ----------
    timept : 'str'
        the time point to be chosen
    loc : str
        The sub directory. Default is fused. It will probably always be set to default 
    train_or_test: str
        indicates whether you generate the training or testing data
    N_or_P: str
        indicates whether you use positives or negatives
    Returns
    -------
    fused_array: the array containing all the positive or negative (as chosen) training or testing data (as chosen),
    for the chosen timepoint. 
    Note: the data is ordered

    """
    path=''
    images = [f for f in os.listdir(path) if f.endswith(timept+' min_'+train_or_test+'.csv')]
    fused_array=np.ones((18,1))
    for name in images:
        if N_or_P in name:
            # print(name)
            data=np.genfromtxt(path+name,delimiter=';',skip_header=0, skip_footer=0)
            fused_array=np.concatenate((fused_array,data),axis=1)  
    fused_array=fused_array[:,1:]
    fused_array=np.transpose(fused_array)
    return(fused_array)


def data_indiv(timept,N_or_P):    
    """
    Imports all CSV file of each sample without its headers and first column, and transposes it into an np.array
    
    Parameters
    ----------
    timept: str
    The timepoint to be used (only the number, ie '1','10')
    loc: str (default is fused and that probably won't change)
    Returns
    -------
    a list with the arrays ordered from P1 to P19
    """
    data=[]
    if N_or_P=='P':
        for pat in pat_list:
            path=''
            data_i=np.genfromtxt(path+pat+'_'+timept+' min_test.csv',delimiter=';',\
                             skip_header=0, skip_footer=0)     
            data_i=np.transpose(data_i)
            data.append(data_i) 
    if N_or_P=='N':
        for pat in neg_list:
            path=''
            data_i=np.genfromtxt(path+pat+'_'+timept+' min_test.csv',delimiter=';',\
                             skip_header=0, skip_footer=0)     
            data_i=np.transpose(data_i)
            data.append(data_i)
    return(data)

def prepare_dataset(*args):
    """
    From the classes that are given, generates a shuffled training set
    
    Parameters
    ----------
    the arrays, which are the classes to be concatenated and shuffled
    
    Returns
    -------
    X_full, y_full, respectively the concatenated and shuffled X and y
    """
    
    data_full=np.concatenate(*args, axis=0)
    data_full=shuffle(data_full, random_state=1)
    X_full=data_full[:,1:]
    y_full=data_full[:,0]
    return(X_full,y_full)  
  
# %% related to general SVM   
def add_class_single(array, num_class=0):
    length=np.shape(array)[0]
    class_array=np.full((length,1),num_class)
    final_array=np.concatenate((class_array,array),axis=1)
    return(final_array)

def svm_prediction(X_train, y_train, X_test, C_, gam, deg, kern):
    clf = svm.SVC(C=C_, kernel=kern, gamma=gam, degree=deg, probability=True)
    clf.fit(X_train, y_train)
    # filename = 'finalized_model.sav' #for when saving the weights will be necessary
    # pickle.dump(clf, open(filename, 'wb'))
    predicted= clf.predict(X_test)
    proba=clf.predict_proba(X_test)
    scores_test=clf.decision_function(X_test)
    scores_train=clf.decision_function(X_train)
    return(predicted, scores_test, scores_train, proba)

def majority(predicted):
    confidence=np.sum(predicted)/np.size(predicted) 
    return(confidence)


# %% ROC related functions

def plot_ROC(X_train, X_test, y_train, y_test,C_, gam, deg, kern):
    clf = svm.SVC(C=C_, kernel=kern, gamma=gam, degree=deg)
    clf.fit(X_train, y_train)
    y_score = clf.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
    auc=roc_auc_score(y_test, y_score)
    return(auc,fpr, tpr, thresholds)


#%% import data
if __name__ == "__main__":

    kern='rbf' 
    deg=1
    
    pat_list=['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13',\
          'P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26',\
              'P27','P28','P29','P30','P31','P32','P33','P34']  
    neg_list=['N1','N2','N3','N4','N5','N6','N7','N8','N9','N10','N11','N12','N13','N14','N15']
    
    
# ## 10 min
    positive_train_10=import_fused('10','train','P')
    positive_test_10=import_fused('10','test','P') #for a test on entire test dataset   
    negative_train_10=import_fused('10','train','N')
    negative_test_10=import_fused('10','test','N')
 
if __name__ == "__main__":
    
    class_label=["negative","positive"]  
    
    ##  create the classes for train    
    class_0_train=add_class_single(negative_train_10, num_class=0)
    class_1_train=add_class_single(positive_train_10, num_class=1) # the data is not shuffled yet

    X_train, y_train=prepare_dataset((class_0_train, class_1_train)) #the data is now shuffled
   
    C_,gam,kern=

    
#%% Test aggregated

   
    class_0_test=add_class_single(negative_test_10, num_class=0)
    class_1_test=add_class_single(positive_test_10, num_class=1) #data not shuffled yet
    X_test, y_test=prepare_dataset((class_0_test, class_1_test)) #data now shuffled
    
    predicted_class, scores_test, scores_train, proba=svm_prediction(X_train,\
        y_train, X_test, C_, gam, deg, kern)
    
    conf=confusion_matrix(y_test, predicted_class)
    tn, fp, fn, tp=confusion_matrix(y_test, predicted_class).ravel()
    sens=tp/(tp+fn)
    spe=tn/(tn+fp)
    acc=(tn+tp)/(tn+tp+fn+fp)
    plot_conf(conf, class_label)
    auc,fpr,tpr,threshold=plot_ROC(X_train,X_test,y_train,y_test,C_, gam, deg, kern)

#%% Test individual

#for lists of all the results
    all_P_test=data_indiv(timept='10',N_or_P='P')
    all_N_test=data_indiv(timept='10',N_or_P='N')

    predicted_class_P=[]
    scores_test_P=[]
    proba_P=[]
    confidence_P=[]
    for patient in all_P_test:
        predicted_pos,scores_test_pos,scores_train_pos,proba_pos=svm_prediction(X_train,\
            y_train, patient, C_,gam, deg, kern)
        confidence_pos=majority(predicted_pos)
        predicted_class_P.append(predicted_pos)
        scores_test_P.append(scores_test_pos)
        proba_P.append(proba_pos)
        confidence_P.append(confidence_pos)
    scores_train=scores_train_pos

    predicted_class_N=[]
    scores_test_N=[]
    proba_N=[]
    confidence_N=[]
    for negative in all_N_test:
        predicted_neg,scores_test_neg,scores_train_neg,proba_neg=svm_prediction(X_train,\
            y_train, negative, C_,gam, deg, kern)
        confidence_neg=majority(predicted_neg)
        predicted_class_N.append(predicted_neg)
        scores_test_N.append(scores_test_neg)
        proba_N.append(proba_neg)
        confidence_N.append(confidence_neg)

    arr_ones=np.ones((15,))
    acc_N=arr_ones-confidence_N

    
    