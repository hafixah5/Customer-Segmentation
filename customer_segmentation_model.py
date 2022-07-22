# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:25:19 2022

@author: fizah
"""
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from ModulesCustomer import EDA,ModelDevelopment,ModelEvaluation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential,Input,Model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import plot_model
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import scipy.stats as ss
import missingno as msno
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import pickle
import os


pd.set_option('display.max_columns', None)
#%% Constants

CSV_PATH = os.path.join(os.getcwd(),'dataset','Train.csv')
LOGS_PATH = os.path.join(os.getcwd(),'logs', 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
OHE_PATH = os.path.join(os.getcwd(),'pickle_files','ohe.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
#%% Functions

def cramers_corrected_stat(matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(matrix)[0]
    n = matrix.sum()
    phi2 = chi2/n
    r,k = matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))



#%% 1) Data loading

df = pd.read_csv(CSV_PATH)

#%% 2) Data Inspection

df.head()
df.info()
df.describe().T
df.isna().sum()
msno.matrix(df)
# dropping id (not important for analysis), 
# and days_since_prev_campaign_contact (too many NaNs (over 25k))
df = df.drop(labels = ['id','days_since_prev_campaign_contact'], axis =1)
columns = list(df.columns)

cat_col = list(df.columns[df.dtypes=='object'])
cont_col = list(df.columns[(df.dtypes =='int64') | (df.dtypes =='float64')])

# Making changes to the lists as 'term_deposit_subscribed' is not continuous
cat_col.append('term_deposit_subscribed')        
cont_col.pop()

# visualizations of the variables
eda = EDA()
eda.distplot_graph(cont_col,df)
eda.countplot_graph(cat_col, df)
#imbalanced data for 'term_deposit_subscribed' column
#%% 3) Data Cleaning

#Change object to numerical values (encoding) 
le = LabelEncoder()
for i in cat_col:
    temp = df[i]
    temp[temp.notnull()] = le.fit_transform(temp[temp.notnull()]) 
    df[i] = pd.to_numeric(temp, errors = 'coerce')
    df[i] = le.fit_transform(df[i])
    ENCODER_PATH = os.path.join(os.getcwd(),'pickle_files',i + '_encoder_pkl') 
    pickle.dump(le, open(ENCODER_PATH,'wb'))

# Impute NaNs
knn_imp = KNNImputer(n_neighbors=5)
df = knn_imp.fit_transform(df)
df = pd.DataFrame(df)
df.columns = columns
#rechecking NaNs
df.isna().sum()
msno.matrix(df)
df.duplicated().sum() # no duplicates

#%% 4) Feature Selection

# cont vs cat
selected_features = []
for i in cont_col:
    lr=LogisticRegression()
    y_reg = np.expand_dims(df['term_deposit_subscribed'], axis = -1)
    x_reg = np.expand_dims(df[i],axis =-1)
    lr.fit(x_reg, y_reg)
    print(i, ':', lr.score(x_reg, y_reg))
    if lr.score(x_reg, y_reg) > 0.5:
        selected_features.append(i)

# cat vs cat
for i in cat_col:
    print(i)  
    matrix = pd.crosstab(df[i], df['term_deposit_subscribed']).to_numpy()
    print(cramers_corrected_stat(matrix))
    if cramers_corrected_stat(matrix) > 0.5:
        selected_features.append(i)

#removing 'term_deposit_subscribed' from selected_features list
selected_features.pop()

# Finalized selected features to be used are:
# customer_age,balance,day_of_month,last_contact_duration,
# num_contacts_in_campaign,num_contacts_prev_campaign.

y = df['term_deposit_subscribed']
y = np.expand_dims(y, axis=-1)
ohe = OneHotEncoder()
y = ohe.fit_transform(np.array(y).reshape(-1,1))
y = y.toarray()

with open(OHE_PATH,'wb') as file:
    pickle.dump(OHE_PATH,file)

X = df[['customer_age','balance','day_of_month','last_contact_duration','num_contacts_in_campaign','num_contacts_prev_campaign']]
#%% 5) Model Development

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state =123)

md = ModelDevelopment()
model = md.dl_model(X_shape=6, nb_class=2,nb_node=128,dropout_rate=0.3)
model.compile(optimizer = 'adam', loss='categorical_crossentropy',
              metrics=['acc'])

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)
early_callback = EarlyStopping(monitor = 'val_loss',patience=5) 

hist = model.fit(X_train, y_train,epochs=50,
                 validation_data=(X_test,y_test),
                 callbacks=[tensorboard_callback, early_callback])

plot_model(model,show_shapes=(True,False),show_layer_names=True)

model.save(MODEL_SAVE_PATH)
#%% Model Evaluation

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

#%% Classification report
y_pred= np.argmax(model.predict(X_test),axis=1)
y_test= np.argmax(y_test,axis=1)
print(classification_report(y_test,y_pred))










