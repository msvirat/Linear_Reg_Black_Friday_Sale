# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 20:38:32 2021

@author: Sathiya vigraman M
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
os.chdir('D:/Dataset')

pd.pandas.set_option('display.max_columns',None)

#import data
blackfriday_df = pd.read_csv('Black_friday/train.csv')
blackfriday_test = pd.read_csv('Black_friday/test.csv')

#Shape of data
blackfriday_df.shape
blackfriday_test.shape

#colunm name of data
blackfriday_df.columns
blackfriday_test.columns

#removing y from train and combine train and test for EDA
blackfriday_df_train = blackfriday_df.drop('Purchase', axis = 1)
blackfriday = blackfriday_df_train.append(blackfriday_test)

#finding null value percentage
blackfriday.isnull().mean()

#finding howmuch null values affecting Y value
for i in blackfriday:
    data = blackfriday_df.copy()
    data[i] = np.where(data[i].isnull(), 1, 0)
    data.groupby(i)['Purchase'].median().plot.bar()
    plt.title(i)
    plt.show()

del i, data
#removing NA values
blackfriday.Product_Category_2.describe()
blackfriday.Product_Category_2.head()
#finding outlier in Product_Category_2 
upper_lim = blackfriday.Product_Category_2.quantile(0.75)
lower_lim = blackfriday.Product_Category_2.quantile(0.25)
blackfriday.Product_Category_2[(blackfriday.Product_Category_2 > blackfriday.Product_Category_2.quantile(0.90)) & (blackfriday.Product_Category_2 > blackfriday.Product_Category_2.quantile(0.10))]


blackfriday.Product_Category_2.fillna(blackfriday.Product_Category_2.median(), inplace=True)



del upper_lim, lower_lim

#drop unwanted colunms
blackfriday.drop(['User_ID', 'Product_ID', 'Product_Category_3'], axis = 1, inplace=True)

blackfriday.columns

#gether information about each colunm
blackfriday.Gender.info()
sns.bar(blackfriday.Gender.unique(), blackfriday.Gender.value_counts())


blackfriday.Age.value_counts()
sns.bar(blackfriday.Age.unique(), blackfriday.Age.value_counts())

blackfriday.Occupation.value_counts()
sns.bar(blackfriday.Occupation.unique(), blackfriday.Occupation.value_counts())

blackfriday.City_Category.value_counts()
sns.bar(blackfriday.City_Category.unique(), blackfriday.City_Category.value_counts())

blackfriday.Stay_In_Current_City_Years.value_counts()
sns.bar(blackfriday.Stay_In_Current_City_Years.unique(), blackfriday.Stay_In_Current_City_Years.value_counts())

blackfriday.Marital_Status.value_counts()
sns.barplot(blackfriday.Marital_Status.unique(), blackfriday.Marital_Status.value_counts())

blackfriday.Product_Category_1.describe()
sns.barplot(blackfriday.Product_Category_1.unique(), blackfriday.Product_Category_1.value_counts())

blackfriday.Product_Category_2.describe()
sns.barplot(blackfriday.Product_Category_2.unique(), blackfriday.Product_Category_2.value_counts())


blackfriday = pd.get_dummies(blackfriday, columns=['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status'])

'''
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
blackfriday = mms.fit_transform(MinMaxScaler)
'''
#Scling X values
blackfriday.dtypes


def norm_func(i):
	x = (i-i.min())	/(i.max()-i.min())
	return(x)

blackfriday = norm_func(blackfriday)

#split train and test data
modal_data = blackfriday.iloc[:len(blackfriday_df), :]
test_modal = blackfriday.iloc[len(blackfriday_df):,:]
#reset index of each data
modal_data.reset_index(drop=True, inplace=True)
test_modal.reset_index(drop=True, inplace=True)


#split train into train and test
train_x, test_x, train_y, test_y = train_test_split(modal_data, blackfriday_df.Purchase, test_size = 0.30)


#Y-continous---Regression methods--------

#----Linear Regression------------

#from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


#poly = PolynomialFeatures(2)

#train_x_transform = poly.fit_transform(X = train_x)
#test_x_transform = poly.fit_transform(X = test_x)



linear_modal = LinearRegression()

linear_modal.fit(X= train_x, y = train_y)

linear_modal.score(X = train_x, y = train_y)#93.84

linear_test_accurecy = linear_modal.score(X = test_x, y = test_y)#77.80

linear_modal.predict(X = test_x)


#---------Decision tree-------------

from sklearn.tree import DecisionTreeRegressor

decision_tree_modal = DecisionTreeRegressor()

decision_tree_modal.fit(X = train_x, y = train_y)

decision_tree_modal.score(X = train_x, y = train_y)

decision_tree_modal_test_accurecy = decision_tree_modal.score(X = test_x, y = test_y)

#----------Random forest regression----

from sklearn.ensemble import RandomForestRegressor

random_forest_modal = RandomForestRegressor()

random_forest_modal.fit(X = train_x, y = train_y)

random_forest_modal.score(X = train_x, y = train_y)

random_forest_modal_test_accurecy = random_forest_modal.score(X = test_x, y = test_y)


#-----------SVM regression---------
from sklearn.svm import SVR

svr_modal = SVR()

svr_modal.fit(X = train_x, y = train_y)

svr_modal.score(X = train_x, y = train_y)

svr_modal_test_accurecy = svr_modal.score(X = test_x, y = test_y)

#-----------Neural network regression----
from sklearn.neural_network import MLPRegressor

nnet_modal = MLPRegressor()

nnet_modal.fit(X = train_x, y = train_y)
nnet_modal.score(X = train_x, y = train_y)

nnet_modal_test_accurecy = nnet_modal.score(X = test_x, y = test_y)


#---------Lesso regression---------------

from sklearn.linear_model import Lasso

lesso_modal = Lasso()

lesso_modal.fit(X = train_x, y = train_y)
lesso_modal.score(X = train_x, y = train_y)

lesso_modal_test_accurecy = lesso_modal.score(X = test_x, y = test_y)


#---------Ridge regression-----------

from sklearn.linear_model import Ridge

ridge_modal = Ridge()

ridge_modal.fit(X = train_x, y = train_y)
ridge_modal.score(X = train_x, y = train_y)

ridge_modal_test_accurecy = ridge_modal.score(X = test_x, y = test_y)


#------------Ada boost-----------

from sklearn.ensemble import AdaBoostRegressor

ada_boost_modal = AdaBoostRegressor()

ada_boost_modal.fit(X = train_x, y = train_y)
ada_boost_modal.score(X = train_x, y = train_y)

ada_boost_modal_test_accurecy = ada_boost_modal.score(X = test_x, y = test_y)


#---------XG boost---------------

from sklearn.ensemble import GradientBoostingRegressor

XG_boost_modal = GradientBoostingRegressor()

XG_boost_modal.fit(X = train_x, y = train_y)
XG_boost_modal.score(X = train_x, y = train_y)

XG_boost_modal_test_accurecy = XG_boost_modal.score(X = test_x, y = test_y)

#------------Best modal----------


accuracy_all = {'LinearRegression': linear_test_accurecy, 'Decision Tree Regression': decision_tree_modal_test_accurecy, 'Random Forest Regression': random_forest_modal_test_accurecy, 'SVM Regression': svr_modal_test_accurecy, 'Neural Network Regression': nnet_modal_test_accurecy, 'Lesso Regression': lesso_modal_test_accurecy, 'Ridge Regression': ridge_modal_test_accurecy, 'Ada boost regression': ada_boost_modal_test_accurecy, 'XG boost Regression': XG_boost_modal_test_accurecy}

print('Best regression method for Blackfirday is ', max(accuracy_all, key = accuracy_all.get),', and R square is ', max(linear_test_accurecy, decision_tree_modal_test_accurecy, random_forest_modal_test_accurecy, svr_modal_test_accurecy, nnet_modal_test_accurecy, lesso_modal_test_accurecy, ridge_modal_test_accurecy, ada_boost_modal_test_accurecy, XG_boost_modal_test_accurecy))

#For time being---Take Linear Regr. modal as best modal

linear_modal_x = LinearRegression()

linear_modal_x.fit(X = modal_data, y = blackfriday_df.Purchase)

linear_modal_x.score(X = modal_data, y = blackfriday_df.Purchase)

linear_modal_y = linear_modal_x.predict(X = test_modal)


#-----------Convert y as discrete and use classfication methods-------
train_y



from sklearn.decomposition import PCA

#pca_modal = PCA(n_components=20) #n components means no of components

pca_modal = PCA()

pca_modal.fit(X = train_x)

pca_variance = pca_modal.explained_variance_ratio_

cum_variance = np.cumsum(np.round(pca_variance, 4)*100)

plt.plot(cum_variance)

pca_transform = pca_modal.fit_transform(X = train_x)
pca_transform = pd.DataFrame(pca_transform)

pca_transform_new = pca_transform.iloc[:,0:20]

linear_modal_pca = LinearRegression()

linear_modal_pca.fit(X = pca_transform_new, y = train_y)

linear_modal_pca.score(X = pca_transform_new, y = train_y)

linear_modal_y = pca_transform_new.predict(X = test_modal)










