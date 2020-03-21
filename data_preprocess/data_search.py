#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#训练数据初步统计
train_data_path = r"C:\Users\ccs\Documents\dataWhale\used_car_train_20200313\used_car_train_20200313.csv"
train_data_df =  pd.read_csv(train_data_path, sep=' ')
# print("原数据的数量:\n",train_data_df.count())
# print("去重后的数量:\n",train_data_df.drop_duplicates().count())
train_data_df.describe(include='all')


# In[ ]:


#空值统计
train_data_df.isnull().sum()


# In[15]:


#对于字符型指标---分组统计
get_ipython().run_line_magic('matplotlib', 'inline')
category_columns  = ['model', 'brand', 'bodyType', 'fuelType','gearbox','notRepairedDamage','regionCode','seller', 'offerType']
# print(train_data_df['model'])
for i in category_columns:
    print(train_data_df.groupby(i).size())


# In[25]:


#对于连续型指标---正态分布检验
from scipy.stats import stats
numerical_columns  = ['regDate','power', 'kilometer','creatDate', 'price', 'v_0', 'v_1', 'v_2', 'v_3',
           'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',
           'v_13', 'v_14']
# print(scipy.stats.shapiro(train_data_df['regDate']))
# k2, p = stats.normaltest(x)
print('看P-Value是否满足正态分布，小于0.001即满足',list(map(lambda x:scipy.stats.normaltest(train_data_df[x])[1],numerical_columns)))


# In[26]:


#连续型指标的均值、方差、标准差，四分位数，最大最小值观察
train_data_df[numerical_columns].describe(include='all')


# In[29]:


#连续型指标异常值检测--超过80%的指标检验异常则判定为该样本异常，需过滤
import numpy as np
from collections import Counter


# Outlier detection
def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # quartile spacing (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than n outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers

detect_outliers(train_data_df,int(len(numerical_columns)*0.8),numerical_columns)


# In[ ]:


#绘制各变量的散点图
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.pairplot(train_data_df['v_0', 'v_1', 'v_2', 'v_3',
           'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',
           'v_13', 'v_14'])
plt.show()


# In[7]:


#自变量之间是否存在多重共线性关系检验
# 结果显示，若VIF低于10，说明自变量之间并不存在多重共线性的隐患，否则，存在
import patsy.highlevel
import statsmodels.api as sm
import statsmodels.compat.scipy
import statsmodels.stats.outliers_influence
col_list = list(train_data_df.columns)
col_list.remove('price')
y, X = patsy.highlevel.dmatrices('price~{columns}'.format(columns=('+').join(col_list)), data=train_data_df,
                                     return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = ['{:.5f}'.format(statsmodels.stats.outliers_influence.variance_inflation_factor(X.values, i)) for i in
                         range(X.shape[1])]
vif["features"] = X.columns
print(vif)


# In[ ]:






