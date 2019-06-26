# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 23:12:33 2019

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

m8=df["revol.util"]
iqr8=m8.quantile(q=.75) - m8.quantile(q=.25)
m8[(m8>m8.quantile(q=.75)+ 1.5*iqr8) | (m8<m8.quantile(q=.25)- 1.5*iqr8)].shape
df.drop(m8[(m8>m8.quantile(q=.75)+ 1.5*iqr8) | (m8<m8.quantile(q=.25)- 1.5*iqr8)].index,axis=0,inplace=True)

scaler.fit_transform(df["int.rate"].values.reshape(-1,1))
scaler.fit_transform(df["dti"].values.reshape(-1,1)) 

c0,c1=0,0
dfn=df.copy(deep=True)
for idx,ser in df.iterrows():
    if ser["not.fully.paid"]==1:
        if c1<750:
            c1=c1+1
        else:
            dfn.drop(idx,inplace=True)
    else:
        if c0<1500:
            c0=c0+1
        else:
            dfn.drop(idx,inplace=True)
print(dfn["not.fully.paid"].value_counts())
dftrain=dfn.copy(deep=True)
dftest=df.copy(deep=True)
dftest.drop(dftrain.index,inplace=True)
print(dftrain.shape,dftest.shape)

def printresult(actual,predicted):
    confmatrix=metrics.confusion_matrix(actual,predicted)
    accscore=metrics.accuracy_score(actual,predicted)
    precscore=metrics.precision_score(actual,predicted)
    recscore=metrics.recall_score(actual,predicted)
    print(confmatrix)
    print("accuracy : {:.4f}".format(accscore))
    print("precision : {:.4f}".format(precscore))
    print("recall : {:.4f}".format(recscore))
    print("f1-score : {:.4f}".format(metrics.f1_score(actual,predicted)))
    print("AUC : {:.4f}".format(metrics.roc_auc_score(actual,predicted)))


dftrain.info()

trmodel1=tree.DecisionTreeClassifier(criterion="entropy")
trmodel1.fit(dftrain.drop("not.fully.paid",axis=1),dftrain["not.fully.paid"])
pred=trmodel1.predict(dftest.drop("not.fully.paid",axis=1))
printresult(dftest.drop("not.fully.paid",axis=1),pred)



rf=ensemble.RandomForestClassifier(criterion="entropy")
rf.fit(dftrain.drop(["not.fully.paid","pub.rec","delinq.2yrs"],axis=1),dftrain["not.fully.paid"])
prediction=rf.predict(dftest.drop(["not.fully.paid","pub.rec","delinq.2yrs"],axis=1))
printresult(dftest.drop("not.fully.paid",axis=1),prediction)






model=naive_bayes.BernoulliNB()
model.fit(dftrain.drop(["not.fully.paid","fico","int.rate","installment","days.with.cr.line","log.annual.inc"],axis=1),dftrain["not.fully.paid"])
prediction=model.predict(dftest.drop(["not.fully.paid","fico","int.rate","installment","days.with.cr.line","log.annual.inc"],axis=1))
printresult(dftest.drop("not.fully.paid",axis=1),prediction)





