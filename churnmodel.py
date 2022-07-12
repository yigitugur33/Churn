#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Veri hazırlık kütüphaneleri"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
get_ipython().system('pip install xgboost')


"""Modelleme Kütüphaneleri"""
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

"""Model Eleme"""
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

"""Diğer"""
import os
import warnings
# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category = ConvergenceWarning)


# Veri setinin Yüklenmesi
dt = pd.read_csv('/Users/uguryigit/Downloads/Churn_Modelling.csv')



# İlk beş satır
dt.head()

dt = dt.drop(columns=["RowNumber","CustomerId","Surname"])

dt.describe()

pd.DataFrame(dt.isnull().sum(),columns=["Count"])


# Exited (churn) -- CreditScore (Kredi skoru)
sns.violinplot( x=dt["Exited"], y=dt["CreditScore"], linewidth=5)
plt.title("Credit Score Distribution of Churn (Exited)")
plt.show()


# Exited (churn) -- Age (Yaş)
sns.violinplot( x=dt["Exited"], y=dt["Age"], linewidth=5)
plt.title("Age of Customers Distribution of Churn (Exited)")
plt.show()


# Exited (Churn) -- Tenure 
sns.violinplot( x=dt["Exited"], y=dt["Tenure"], linewidth=5)
plt.title("Tenure of Customers Distribution of Churn (Exited)")
plt.show()

# Exited (Churn) -- Balance 
sns.violinplot( x=dt["Exited"], y=dt["Balance"], linewidth=5)
plt.title("Balance of Customers Distribution of Churn (Exited)")
plt.show()

# Balance boxplot
dt[["Balance"]].boxplot()

# Exited (Churn) -- NumOfProducts 
sns.violinplot( x=dt["Exited"], y=dt["NumOfProducts"], linewidth=5)
plt.title("Number of Products of Customers Distribution of Churn (Exited)")
plt.show()

# Exited (Churn) -- EstimatedSalary
sns.violinplot( x=dt["Exited"], y=dt["EstimatedSalary"], linewidth=5)
plt.title("Estimated Salary of Customers Distribution of Churn (Exited)")
plt.show()


# Matrix
correlationColumns = dt[["CreditScore","Age","Tenure"
    ,"Balance","NumOfProducts","EstimatedSalary"]]
sns.set()
corr = correlationColumns.corr()
ax = sns.heatmap(corr
                 ,center=0
                 ,annot=True
                 ,linewidths=.2
                 ,cmap="YlGnBu")
plt.show()


# In[2]:


# predictors and target (exited - churn)
predictors = dt.iloc[:,0:10]
target = dt.iloc[:,10:]

# erkek = 1, kadın = 0
predictors['isMale'] = predictors['Gender'].map({'Male':1, 'Female':0})
# Geography one shot encoder
predictors[['France', 'Germany', 'Spain']] = pd.get_dummies(predictors['Geography'])

predictors = predictors.drop(columns=['Gender','Geography','Spain'])

normalization = lambda x:(x-x.min()) / (x.max()-x.min())

transformColumns = predictors[["Balance","EstimatedSalary","CreditScore"]]
predictors[["Balance","EstimatedSalary","CreditScore"]] = normalization(transformColumns)

# Train and test splitting
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size=0.25, random_state=0)
pd.DataFrame({"Train Row Count":[x_train.shape[0],y_train.shape[0]],
              "Test Row Count":[x_test.shape[0],y_test.shape[0]]},
             index=["X (Predictors)","Y (Target)"])


# In[3]:


# Karar Ağacı - Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred_dtc = dtc.predict(x_test)
dtc_acc = accuracy_score(y_test,y_pred_dtc)
# Lojistik Regresyon - Logistic Regression
logr = LogisticRegression()
logr.fit(x_train,y_train)
y_pred_logr = logr.predict(x_test)
logr_acc = accuracy_score(y_test,y_pred_logr)
# Naif Bayes - Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred_gnb = gnb.predict(x_test)
gnb_acc = accuracy_score(y_test,y_pred_gnb)
# K En Yakın Komşu - K Neighbors Classifier
knn = KNeighborsClassifier( metric='minkowski')
knn.fit(x_train,y_train)
y_pred_knn = knn.predict(x_test)
knn_acc = accuracy_score(y_test,y_pred_knn)
# Rassal Ağaçlar - Random Forrest
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)
rfc_acc = accuracy_score(y_test,y_pred_rfc)
# Sinir Ağları - Neural Network
nnc = MLPClassifier()
nnc.fit(x_train,y_train)
y_pred_nnc = nnc.predict(x_test)
nnc_acc = accuracy_score(y_test,y_pred_nnc)


# Sonuçların bir tabloya yazdırılması
pd.DataFrame({"Algorithms":["Decision Tree","Logistic Regression","Naive Bayes","K Neighbors Classifier","Random Ferest","Neural Network"],
              "Scores":[dtc_acc, logr_acc, gnb_acc, knn_acc, rfc_acc,nnc_acc]})


# In[ ]:




