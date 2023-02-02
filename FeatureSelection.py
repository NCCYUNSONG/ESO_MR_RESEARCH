import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split

data = pd.read_excel('/home/liu/PycharmProjects/pythonProject-MR/image/AFTTCOM.xlsx', engine='openpyxl')
fea = data.iloc[:, 2:]
corr_matrix = fea.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
af_c = data.drop(to_drop, axis=1)

af_c.to_excel('TTCOM_C.xlsx')

data1 = pd.read_excel('/home/liu/PycharmProjects/pythonProject-MR/image/TTCOM_C.xlsx', engine='openpyxl')
data1.head()
train_x=data1.iloc[:, 2:]#choose parameters
train_y=data1['pCR']#the endpoint,0-npcr,1-pcr
RFC = RandomForestClassifier(n_estimators=100,random_state=21)
sele_rfe = RFE(RFC, n_features_to_select=6, step=1).fit(train_x, train_y)
sele_rfe.n_features_
sele_rfe.ranking_
sele_rfe.support_
x_rfe = train_x[train_x.columns[sele_rfe.support_]]
z = data1.iloc[:, 0:2]
x_rfe = pd.concat([z, x_rfe], axis=1)
x_rfe.to_excel('TTCOM_RFE6.xlsx')