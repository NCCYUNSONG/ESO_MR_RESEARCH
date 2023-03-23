import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from collections import Counter
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score

data1 = pd.read_excel('/home/liu/PycharmProjects/pythonProject-MR/image/TTCOM_RFECV2.xlsx', engine='openpyxl')
data1.head()
train_x=data1.iloc[:, 2:]#choose clinical parameters
train_y=data1['pCR']#the endpoint,0-npcr,1-pcr
data2 = pd.read_excel('/home/liu/PycharmProjects/pythonProject-MR/image/EXAFTTCOM.xlsx', engine='openpyxl')
aa = train_x.columns
bb = list(aa)
test_x=data2.loc[:, bb]#choose clinical parameters
test_y=data2['pCR']
Counter(train_y)
Counter(test_y)

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
model4= RandomForestClassifier(n_estimators=50, max_depth=7, max_features='sqrt',min_samples_leaf=1,min_samples_split=8,random_state=90,criterion='gini')
model4.fit(train_x,train_y)
pred4_test=model4.predict(test_x)
pred4_test
metrics.accuracy_score(pred4_test, test_y)
metrics.accuracy_score(model4.predict(train_x), train_y)

from sklearn.model_selection import GridSearchCV

param_grid =          {
        'n_estimators': np.arange(50, 501, 100),
        'max_features': ['sqrt', 'log2'],
        'max_depth': np.arange(3, 8, 2),
        'min_samples_split': np.arange(2, 11, 2),
        'min_samples_leaf': np.arange(1, 6, 1),
        'bootstrap': [True, False]
    },
GS = GridSearchCV(RandomForestClassifier(random_state=90)
                       , param_grid = param_grid
                       , cv = 5
                       ,scoring='roc_auc'
                       ,n_jobs=12
                      )
GS.fit(train_x,train_y)
print(GS.best_params_)


from sklearn.metrics import roc_auc_score
train_yp = model4.predict_proba(train_x)[:,1]
test_yp = model4.predict_proba(test_x)[:,1]
model4_train_auc=roc_auc_score(train_y, train_yp)
model4_test_auc=roc_auc_score(test_y, test_yp)
model4_train_auc, model4_test_auc
