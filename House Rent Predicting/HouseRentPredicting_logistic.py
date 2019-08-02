# Home Rent Predicting(Logistic Regression)
"""
Created on Sat Sep 22 22:14:46 2018

@author: Lazy_Booleans
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
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


#Apply the dataset into the Logistic  machine learning model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)

#Predictig the House rent for dataset
y_pred_logistic = classifier.predict(x_test)

#showing the indivisual accuracy of the test set
prince = 0
for i in range(0,5000):
    print("y_test data ",end= " ")
    print(y_test[i])
    print("y_test data ",end= " ")
    print(y_pred_logistic[i])
    vikram = 100 - (abs(y_test[i] - y_pred_logistic[i])/y_test[i])*100
    print("efficiency = ",end = " ")
    print(vikram)
    prince = prince + vikram


#Show your Total efficiency
print("Total efficiency " ,end = " ")
print(prince/5000)

    

    


