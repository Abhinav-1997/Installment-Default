# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:09:11 2019

@author: ABHINAV TIWARY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import metrics
from sklearn import feature_selection
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.metrics import log_loss
from sklearn.preprocessing import  StandardScaler
pd.set_option("display.max_columns",100)


scaler = StandardScaler()
df=pd.read_csv("f:/dataset/loans.csv")
df.columns
df.info()
df.describe()
df.isnull().sum()
df.isnull().sum()/df.shape[0]
df.dropna(inplace=True)
le=preprocessing.LabelEncoder()
df["pp"]=le.fit_transform(df["purpose"])
df.drop("purpose",axis=1,inplace=True)

a=df["int.rate"].min()
b=df["int.rate"].max()
df["int.rate"]=(df["int.rate"]-a)/(b-a)
a=df["installment"].min()
b=df["installment"].max()
df["installment"]=(df["installment"]-a)/(b-a)
a=df["revol.util"].min()
b=df["revol.util"].max()
df["revol.util"]=(df["revol.util"]-a)/(b-a)
a=df["dti"].min()
b=df["dti"].max()
df["dti"]=(df["dti"]-a)/(b-a)
a=df["fico"].min()
b=df["fico"].max()
df["fico"]=(df["fico"]-a)/(b-a)

##############################Removing Outlier 
m5=df["fico"]
iqr5=m5.quantile(q=.75) - m5.quantile(q=.25)
m5[(m5>m5.quantile(q=.75)+ 1.5*iqr5) | (m5<m5.quantile(q=.25)- 1.5*iqr5)].shape#outlier removed
df.drop(m5[(m5>m5.quantile(q=.75)+ 1.5*iqr5) | (m5<m5.quantile(q=.25)- 1.5*iqr5)].index,axis=0,inplace=True)


m1=df["int.rate"]
iqr2=m1.quantile(q=.75) - m1.quantile(q=.25)
m1[(m1>m1.quantile(q=.75)+ 1.5*iqr2) | (m1<m1.quantile(q=.25)- 1.5*iqr2)].shape[0]#outlier removed
df.drop(m1[(m1>m1.quantile(q=.75)+ 1.5*iqr2) | (m1<m1.quantile(q=.25)- 1.5*iqr2)].index,axis=0,inplace=True)

m2=df["installment"]
iqr2=m2.quantile(q=.75) - m2.quantile(q=.25)
m2[(m2>m2.quantile(q=.75)+ 1.5*iqr2) | (m2<m2.quantile(q=.25)- 1.5*iqr2)].shape[0]
df.drop(m2[(m2>m2.quantile(q=.75)+ 1.5*iqr2) | (m2<m2.quantile(q=.25)- 1.5*iqr2)].index,axis=0,inplace=True)

m8=df["revol.util"]
iqr8=m8.quantile(q=.75) - m8.quantile(q=.25)
m8[(m8>m8.quantile(q=.75)+ 1.5*iqr8) | (m8<m8.quantile(q=.25)- 1.5*iqr8)].shape
df.drop(m8[(m8>m8.quantile(q=.75)+ 1.5*iqr8) | (m8<m8.quantile(q=.25)- 1.5*iqr8)].index,axis=0,inplace=True)
df.info()

m7=df["revol.bal"]#too many outliers
iqr7=m7.quantile(q=.75) - m7.quantile(q=.25)
m7[(m7>m7.quantile(q=.75)+ 1.5*iqr7) | (m7<m7.quantile(q=.25)- 1.5*iqr7)].shape


m8=df["revol.util"]
iqr8=m8.quantile(q=.75) - m8.quantile(q=.25)
m8[(m8>m8.quantile(q=.75)+ 1.5*iqr8) | (m8<m8.quantile(q=.25)- 1.5*iqr8)].shape
df.drop(m8[(m8>m8.quantile(q=.75)+ 1.5*iqr8) | (m8<m8.quantile(q=.25)- 1.5*iqr8)].index,axis=0,inplace=True)
##################################################
df.info()

#########################################################normalising data
scaler.fit_transform(df["int.rate"].values.reshape(-1,1))
sns.distplot(df["int.rate"])#can be furthur normalised with valur replacement
sns.distplot(df["fico"])#slightly right skewed




scaler.fit_transform(df["dti"].values.reshape(-1,1)) 


#####rest continuous
'''
['credit.policy', 'int.rate', 'installment', 'log.annual.inc',
       'fico', 'revol.util', 'inq.last.6mths', 'pub.rec']
'''

##################################################################################################3

#feature selection
df.info()
X=df.drop("not.fully.paid",axis=1)
y=df["not.fully.paid"]
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.2,random_state=42,stratify=y)

selector=feature_selection.RFECV(estimator=neighbors.KNeighborsRegressor(n_neighbors=5),min_features_to_select=5,scoring="recall")
selector.fit(Xtrain,ytrain)
Xtrain.columns.values[selector.get_support()].shape
Xtrain1=Xtrain[Xtrain.columns.values[selector.get_support()]]
Xtest1=Xtest[Xtest.columns.values[selector.get_support()]]
Xtrain1.columns.values



df.info()
df.drop("ndti",axis=1,inplace=True)
