# House Rent Predicting
"""
Created on Sat Sep 22 23:58:00 2018

@author: Vikram Singh
"""


#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


#Importing the dataset and Encoding the categorical variable
dataset = pd.read_csv('Data.csv')
vikram0 = pd.get_dummies(dataset['type'])
dataset = pd.concat([dataset,vikram0],axis = 1)
vikram3 = pd.get_dummies(dataset['lease_type'])
dataset = pd.concat([dataset,vikram3],axis = 1)
vikram8 = pd.get_dummies(dataset['furnishing'])
dataset = pd.concat([dataset,vikram8],axis = 1)
vikram9 = pd.get_dummies(dataset['parking'])
dataset = pd.concat([dataset,vikram9],axis = 1)
vikram10 = pd.get_dummies(dataset['facing'])
dataset = pd.concat([dataset,vikram10],axis = 1)
vikram11 = pd.get_dummies(dataset['water_supply'])
dataset = pd.concat([dataset,vikram11],axis = 1)
vikram12 = pd.get_dummies(dataset['building_type'])
dataset = pd.concat([dataset,vikram12],axis = 1)
       
#Create the input dataset and output dataset
x = dataset.iloc[:,[4,5,7,8,9,10,13,14,15,17,18,19,20,23,26,27,28,29,30,32,33,34,36,37,39,40,41,43,44,45,46,47,48,49,51,52,54,55,56]].values
y = dataset.iloc[:, 24].values

#Encoding the categorial data
for i in range(0,25000):
    s = x[i , 12]
    json_acceptable_string = s.replace("'", "\"")
    d = json.loads(json_acceptable_string) 
    if 'GYM' in d.keys():
        d['GYM'] = 0
    if 'LIFT' in d.keys():
        d['LIFT'] = 0
    x[i , 12] = sum(d.values())


#Spliting the dataset into the training set and test set 
from  sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#Feature Scaling 


#Apply the dataset into the machine learning model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


#Predicting the House rent for dataset
y_pred_back = regressor.predict(x_test)



#Building the optimal model using the Backward Elimination
#Building the optimal model using BackWard Elimination
import statsmodels.formula.api  as sm
x = np.append(arr = np.ones((25000,1)).astype(int),values = x_train ,axis = 1)
x_opt = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
print(regressor_OLS.summary())


x_opt = x[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
print(regressor_OLS.summary())



#Visualising the machine learning model for given dataset


#visualising the machine learning model (in more efficient method and find the efficient curve for given problem)


