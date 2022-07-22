# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 09:17:49 2022

@author: fizah
"""
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras import Sequential,Input,Model
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np

class EDA:
    def distplot_graph(self, cont_col,df):
        # continuous data
        for i in cont_col:
            plt.figure()
            sns.distplot(df[i])
            plt.show()
            
    def countplot_graph(self, cat_col,df):
        #categorical data
        for i in cat_col:
            plt.figure()
            sns.countplot(df[i])
            plt.show()
    
    def cramers_corrected_stat(self,confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    
        
class ModelDevelopment:
    def  dl_model(self,X_shape,nb_class,nb_node=128,dropout_rate=0.3):
             
        model = Sequential()
        model.add(Input(shape=X_shape)) #Input is the number of column of features, or X
        model.add(Dense(nb_node,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_node,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class,activation='softmax'))
        model.summary()
        
        return model
    
class ModelEvaluation:
    def plot_hist_graph(self,hist):
        plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.legend(['Training loss', 'validation loss'])
        plt.show()

        plt.figure()
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.legend(['Training Acc', 'validation Acc'])
        plt.show()