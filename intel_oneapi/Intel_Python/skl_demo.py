# Copyright 2015-2023 Lenovo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 1.导包
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn import linear_model
import time
import numpy as np

# 2.设置随机数
rand_val = 520


def set_rand(rand_val):
    np.random.rand(rand_val)


set_rand(rand_val)

# 3.实例化需要测试的算法
linear_reg = linear_model.LinearRegression()  # 线性回归
logistic_reg_cla = linear_model.LogisticRegression(C=5)  # 逻辑回归，实际上是分类模型
ridge_reg = linear_model.Ridge()  # 岭回归
km1 = KMeans(n_clusters=1000)  # K-Means聚类，聚为1000类
km2 = KMeans(n_clusters=5)  # K-Means聚类，聚为5类
km3 = KMeans(n_clusters=20)  # K-Means聚类，聚为20类
dbscan = DBSCAN()  # DBSCAN聚类
pca = PCA()  # 主成分分析
svc = SVC(C=2)  # 支持向量机
elasnet_reg = linear_model.ElasticNet()  # 弹性网络回归

# 4.常见模型训练
# K-Means 聚类
cla_x = np.random.random((20000, 20))  # 20K*20的数据
st = time.time()
km1.fit(cla_x)  # K=1000
print(f'K-Means 20K*20 usage time: {time.time() - st}')

cla_x = np.random.random((1000000, 50))  # 1M*50的数据
st = time.time()
km2.fit(cla_x)  # K=5
print(f'K-Means 1M*50 usage time: {time.time() - st}')

cla_x = np.random.random((100000, 50))  # 100K*50的数据
st = time.time()
km3.fit(cla_x)  # K=20
print(f'K-Means 100K*50 usage time: {time.time() - st}')

# DBSCAN 聚类
cla_x = np.random.random((100000, 50))  # 100K*50的数据
st = time.time()
dbscan.fit(cla_x)
print(f'DBSCAN 100K*50 usage time: {time.time() - st}')

# PCA
reg_x = np.random.random((9000000, 100))  # 9M*100
reg_y = np.random.random(9000000)
st = time.time()
linear_reg.fit(reg_x, reg_y)
print(f'PCA 9M*100 usage time: {time.time() - st}')

# LinearRegression
reg_x = np.random.random((30000000, 20))  # 30M*20
reg_y = np.random.random(30000000)
st = time.time()
linear_reg.fit(reg_x, reg_y)
print(f'LinearRegression 30M*20 usage time: {time.time() - st}')

reg_x = np.random.random((4000000, 100))  # 4M*100
reg_y = np.random.random(4000000)
st = time.time()
linear_reg.fit(reg_x, reg_y)
print(f'LinearRegression 4M*100 usage time: {time.time() - st}')

# Ridge 回归
reg_x = np.random.random((90000000, 20))  # 90M*20
reg_y = np.random.random(90000000)
st = time.time()
ridge_reg.fit(reg_x, reg_y)
print(f'Ridge 90M*20 usage time: {time.time() - st}')

reg_x = np.random.random((9000000, 100))  # 9M*100
reg_y = np.random.random(9000000)
st = time.time()
ridge_reg.fit(reg_x, reg_y)
print(f'Ridge 9M*100 usage time: {time.time() - st}')

# logistic 回归
cla_x = np.random.random((10000000, 20))  # 10M*20
cla_y = np.random.randint(0, 5, 10000000)
st = time.time()
logistic_reg_cla.fit(cla_x, cla_y)
print(f'logistic 10M*20 usage time: {time.time() - st}')

cla_x = np.random.random((2000000, 100))  # 2M*100
cla_y = np.random.randint(0, 5, 2000000)
st = time.time()
logistic_reg_cla.fit(cla_x, cla_y)
print(f'logistic 2M*100 usage time: {time.time() - st}')

# SVMs
cla_x = np.random.random((15300, 22))  # 15.3K*22
cla_y = np.random.randint(0, 5, 15300)
st = time.time()
svc.fit(cla_x, cla_y)
print(f'SVMs 15.3K*22 usage time: {time.time() - st}')

cla_x = np.random.random((9900, 123))  # 9.9K*123
cla_y = np.random.randint(0, 5, 9900)
st = time.time()
svc.fit(cla_x, cla_y)
print(f'SVMs 9.9K*123 usage time: {time.time() - st}')

# ElasticNet
reg_x = np.random.random((9630000, 90))  # 463K*100
reg_y = np.random.random(9630000)
st = time.time()
elasnet_reg.fit(reg_x, reg_y)
print(f'ElasticNet 463K*100 usage time: {time.time() - st}')

