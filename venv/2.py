import gc

import pandas as pd
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv(r'C:\Users\TEARS\Desktop\train.csv')
submit_example = pd.read_csv(r'C:\Users\TEARS\Desktop\submit_example.csv')
test = pd.read_csv(r'C:\Users\TEARS\Desktop\test.csv')

data['user_id'] = data['user_id'].astype('int32')
data['product_id'] = data['product_id'].astype('int32')
data['category_id'] = data['category_id'].astype('int32')
lbe = LabelEncoder()
data['brand'].fillna('0', inplace=True)
data['brand'] = lbe.fit_transform(data['brand'])
data['brand'] = data['brand'].astype('int32')
data['event_type'].fillna('0', inplace=True)
data['event_type'] = lbe.fit_transform(data['event_type'])
data['event_type'] = data['brand'].astype('int32')
# data['event_time'] = pd.to_datetime(data['event_time'], format='%Y-%m-%d %H:%M:%S')
data.fillna(0, inplace=True)
data=data.drop(columns="user_session")
train_X = data
test_data = test

# 构建特征
groups = train_X.groupby('user_id')
#u1：按user_id分组后，每组的数量
temp = groups.size().reset_index().rename(columns={0: 'u1'})
matrix = temp
temp = groups['product_id'].agg([('u2', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['category_id'].agg([('u3', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['brand'].agg([('u5', 'nunique')]).reset_index()
# TODO 根据用户购买行为去构建特征
temp = groups['event_type'].value_counts().unstack().reset_index().rename(
    columns={0: 'u7', 1: 'u8', 2: 'u9', 3: 'u10'})
matrix = matrix.merge(temp, on='user_id', how='left')

label_list = []
for name, group in groups:
    product_id = int(group.iloc[-1, 2])
    label_list.append([name, product_id])

train_data = matrix.merge(pd.DataFrame(label_list, columns=['user_id', 'label'], dtype=int), on='user_id', how='left')

# 构建特征
groups = test_data.groupby('user_id')
temp = groups.size().reset_index().rename(columns={0: 'u1'})
test_matrix = temp
temp = groups['product_id'].agg([('u2', 'nunique')]).reset_index()
matrix = test_matrix.merge(temp, on='user_id', how='left')
temp = groups['category_id'].agg([('u3', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['brand'].agg([('u5', 'nunique')]).reset_index()
# TODO 根据用户购买行为去构建特征
temp = groups['event_type'].value_counts().unstack().reset_index().rename(
    columns={0: 'u7', 1: 'u8', 2: 'u9', 3: 'u10'})
test_data = matrix.merge(temp, on='user_id', how='left')

test_data = test_data.drop(['user_id'], axis=1)

train_X, train_y = train_data.drop(['label', 'user_id'], axis=1), train_data['label']


import  numpy as np
# 导入分析库
import lightgbm as lgb


model = lgb.LGBMClassifier()
model.fit(
    train_X,
    train_y,
)
prob = model.predict(test_data)
test_y=test['product_id']
submit_example['product_id'] = prob
from sklearn.metrics import accuracy_score
submit_example.to_csv('baseline1.csv', index=False)



