import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from collections import Counter
import pingouin as pg

data = pd.read_excel('/home/liu/PycharmProjects/pythonProject-MR/image/MR_feature.xlsx', engine='openpyxl')
data.head()

data_2 = pd.read_excel('/home/liu/PycharmProjects/pythonProject-MR/image/MRtt_feature.xlsx', engine='openpyxl')
data_2.head()
data.insert(0, 'reader', np.ones(data.shape[0]))
data_2.insert(0, 'reader', np.ones(data_2.shape[0]) * 2)
data_tt = pd.concat([data, data_2])

data_tt.columns
featuresaftt = []
iccd = pd.DataFrame()
features = np.array(data_tt.columns)
for i in features[2:]:
    icc = pg.intraclass_corr(data=data_tt, targets='ID', raters='reader', ratings=i, nan_policy='omit')
    if icc.loc[1, 'ICC'] > 0.75:
        featuresaftt.append(i)

list1 = features.tolist()
df = pd.DataFrame()
for k in list1:
    if k in featuresaftt:
        df_add = data[k]
        df = pd.concat([df, df_add], axis=1)
    elif k == 'ID':
        df_add2 = data['ID']
        df = pd.concat([df, df_add2], axis=1)
df.to_excel('MR_afTTf.xlsx')