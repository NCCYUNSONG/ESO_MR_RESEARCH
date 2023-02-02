import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from collections import Counter
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score

data1 = pd.read_excel('/home/liu/PycharmProjects/pythonProject-MR/image/TTCOM_RFE6C.xlsx', engine='openpyxl')
data1.head()
train_x=data1.iloc[:, 2:]#choose parameters
train_y=data1['pCR']#the endpoint,0-npcr,1-pcr
data2 = pd.read_excel('/home/liu/PycharmProjects/pythonProject-MR/image/EX_AFTTCOM.xlsx', engine='openpyxl')
aa = train_x.columns
bb = list(aa)
test_x=data2.loc[:, bb]#choose parameters
test_y=data2['pCR']
Counter(train_y)
Counter(test_y)

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
model4= RandomForestClassifier(n_estimators=400, max_depth=9, max_features=1,random_state=21) #consistent with hyperparameters from Grid-search
model4.fit(train_x,train_y)
pred4_test=model4.predict(test_x)
pred4_test
metrics.accuracy_score(pred4_test, test_y)
metrics.accuracy_score(model4.predict(train_x), train_y)

from sklearn.model_selection import GridSearchCV
ne = np.arange(100,500,100)
md=np.arange(2,12,1)
mf=np.arange(1,4,1) #can be modified if marginal value is reached
param_grid = {'n_estimators': ne, 'max_depth': md, 'max_features':mf}
GS = GridSearchCV(RandomForestClassifier(random_state=90)
                       , param_grid = param_grid
                       , cv = 5
                       ,scoring='roc_auc'
                       ,n_jobs=12
                      )
GS.fit(train_x,train_y)
print(GS.best_params_)
n_estimators = GS.best_params_['n_estimators']
max_depth  = GS.best_params_['max_depth']
max_feartures = GS.best_params_['max_features']
n_estimators, max_depth, max_feartures



from sklearn.metrics import roc_auc_score
train_yp = model4.predict_proba(train_x)[:,1]
test_yp = model4.predict_proba(test_x)[:,1]
model4_train_auc=roc_auc_score(train_y, train_yp)
model4_test_auc=roc_auc_score(test_y, test_yp)
model4_train_auc, model4_test_auc

data1 = pd.read_excel('/home/liu/PycharmProjects/pythonProject-MR/image/TTCOM_RFE6C.xlsx', engine='openpyxl')
data1.head()
train_x=data1.iloc[:, 2:]#choose parameters
train_y=data1['pCR']#the endpoint,0-npcr,1-pcr
data2 = pd.read_excel('/home/liu/PycharmProjects/pythonProject-MR/image/EX_AFTTCOM.xlsx', engine='openpyxl')
aa = train_x.columns
bb = list(aa)
test_x=data2.loc[:, bb]#choose parameters
test_y=data2['pCR']
Counter(train_y)
Counter(test_y)


import xgboost as xgb
model7 = XGBClassifier(gamma = 0.45, max_depth =2 , min_child_weight =1, subsample = 0.9, colsample_bytree = 0.6,reg_lambda=0.1)
model7.fit(train_x,train_y)
pred7 = model7.predict(test_x)
acc_train_l = model7.score(train_x, train_y)
acc_test_l = model7.score(test_x, test_y)
print(acc_train_l, acc_test_l)

from sklearn.model_selection import GridSearchCV
ga = np.arange(0.05,1,0.05)
md=np.arange(1,8,1)
sb=np.arange(0.5,1.5,0.1)
mc=[1, 3, 5, 7]
cb = np.arange(0.5,0.9,0.1)
lm = np.arange(0.1,0.5,0.1) #can be modified if marginal value is reached
param_grid = {'gamma': ga, 'subsample': sb, 'max_depth':md, 'min_child_weight':mc, 'colsample_bytree':cb, 'reg_lambda':lm }
GS = GridSearchCV(xgb.XGBClassifier(random_state=21)
                       , param_grid = param_grid
                       , cv = 5
                       ,scoring='roc_auc'
                       ,n_jobs=12
                      )
GS.fit(train_x,train_y)
print(GS.best_params_)

train_yp = model7.predict_proba(train_x)[:,1]
test_yp = model7.predict_proba(test_x)[:,1]
model7_train_auc=roc_auc_score(train_y, train_yp)
model7_test_auc=roc_auc_score(test_y, test_yp)
model7_train_auc, model7_test_auc