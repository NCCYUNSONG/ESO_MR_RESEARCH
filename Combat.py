from combat.pycombat import pycombat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data1 = pd.read_excel('/home/liu/PycharmProjects/pythonProject-MR/image/MR_AFTT2COM.xlsx', engine='openpyxl',index_col=u"ID")
data2 = pd.read_excel('/home/liu/PycharmProjects/pythonProject-MR/image/EX_feature.xlsx', engine='openpyxl',index_col=u"ID")


data1_T = pd.DataFrame(data1.values.T,columns=data1.index,index=data1.columns)
data2_T = pd.DataFrame(data2.values.T,columns=data2.index,index=data2.columns)

df_expression = pd.concat([data1_T,data2_T],join='inner',axis=1)

plt.boxplot(df_expression)
plt.show()

len(df_expression)

batch = []
datasets = [data1_T,data2_T]
for j in range(len(datasets)):
    batch.extend([j for _ in range(len(datasets[j].columns))])

data_corrected = pycombat(df_expression, batch)

plt.boxplot(data_corrected)
plt.show()
data_out = pd.DataFrame(data_corrected.T,columns=data_corrected.index,index=data_corrected.columns)

data_out.to_excel('af_TTcombat.xlsx')