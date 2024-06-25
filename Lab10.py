# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:41:20 2024

@author: Rachana
"""
# import the packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load the dataset
data = pd.read_csv(r'C:\Users\Rachana\Desktop\LU\Business Analytics\Datasets\heart_disease.csv')

# Basics analysis
data.info()
print(data.isna().sum())
data.dropna(axis = 0, inplace = True)
print(data.isna().sum())

# Data exploration
y = data['TenYearCHD']
y.value_counts()
sn.countplot(x='TenYearCHD', data = data)
Count_risk_CHD = len(data[data['TenYearCHD']==1])
Count_no_risk_CHD = len(data[data['TenYearCHD']==0])
pct_risk = Count_risk_CHD/(Count_risk_CHD + Count_no_risk_CHD)
pct_no_risk = Count_no_risk_CHD/(Count_risk_CHD + Count_no_risk_CHD)
print('Percentage of CHD risk is:', str(round(pct_risk*100,2))+'%')
data.groupby('TenYearCHD').mean()
pd.crosstab(data.male,data.TenYearCHD).plot(kind='bar')
plt.title('Frequency of 10 year CHD risk by gender')
plt.xlabel('Job')
plt.ylabel('Frequency')
plt.figure(figsize=(15,15))
sn.heatmap(data.corr(), annot=True)

# Finding useful predictors
X=data.drop(['TenYearCHD'],axis=1)
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())

# Final model
X_new = data.drop(['TenYearCHD', 'currentSmoker', 'BPMeds',
'prevalentStroke','totChol', 'glucose'],axis=1)
logit_model=sm.Logit(y,X_new)
result=logit_model.fit()
print(result.summary())

# Testing accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set:{:.2f}'.format(logreg.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
