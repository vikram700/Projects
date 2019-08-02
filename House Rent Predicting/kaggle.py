# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 03:52:43 2018

@author: user
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
X = dataset.iloc[:,[4,5,7,8,9,10,13,14,15,17,18,19,20,23,26,27,28,29,30,32,33,34,36,37,39,40,41,43,44,45,46,47,48,49,51,52,54,55,56]].values
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
    X[i , 12] = sum(d.values())


#Spliting the dataset into the training set and test set 
from  sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# building the optimal model using backward elimination
# SL = 0.05 and eliminating those features which have p > SL
import statsmodels.formula.api as sm
X_train = np.append(arr = np.ones((891,1)).astype(int), values = X_train, axis = 1)
X_train_opt = X_train[:,[0, 1, 2, 3, 4, 5, 6, 7, 8]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
regressor_OLS.summary()
X_train_opt = X_train[:,[0, 2, 3, 4, 5, 6, 7, 8]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
regressor_OLS.summary()
X_train_opt = X_train[:,[0, 2, 3, 4, 5, 6, 8]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
regressor_OLS.summary()
X_train_opt = X_train[:,[0, 2, 3, 4, 5, 6]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
regressor_OLS.summary()

# Optimized training and test sets
# X_train_opt is optimized training set
y_train_opt = y_train
X_test = np.append(arr = np.ones((418,1)).astype(int), values = X_test, axis = 1)
X_test_opt = X_test[:,[0, 2, 3, 4, 5, 6]]
y_test_opt = y_test


# Making ANN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 3, kernel_initializer='glorot_uniform', activation = 'relu', input_dim = 6))

# Adding second hidden layer
classifier.add(Dense(units = 3, kernel_initializer='glorot_uniform', activation = 'relu'))

# adding output layer
# activation='softmax' if there are mor the two category in output
classifier.add(Dense(units = 1, kernel_initializer='glorot_uniform', activation = 'sigmoid'))

# Compiling ANN
# categorical_crossentropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN to training set
classifier.fit(X_train_opt, y_train_opt, batch_size = 10, epochs = 300)

# predicting the test set result
y_pred_ann = classifier.predict(X_test_opt)
y_pred_ann = (y_pred_ann > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_ann = confusion_matrix(y_test, y_pred_ann)
# accuracy (265+127)/418 = 93.77%

#######################################  Logistic Regression  ################################################

# Fitting Logistic Regression to the opt Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_opt, y_train_opt)

# predicting the test set result
y_pred_logreg = classifier.predict(X_test_opt)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_logreg = confusion_matrix(y_test_opt, y_pred_logreg)
# accuracy (252+141)/418 = 94


##############################  KNN  #######################################################################

# Fitting classifier to the opt Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifier.fit(X_train_opt, y_train_opt)

# predicting the test set result
y_pred_knn = classifier.predict(X_test_opt)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test_opt, y_pred_knn)
# accuracy (261+111)/418 ~ 89%


############################  Support Vector Machine (SVM)  ################################################  

# Fitting Support Vector Machine (SVM) to the opt Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train_opt, y_train_opt)

# predicting the test set result
y_pred_svm = classifier.predict(X_test_opt)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test_opt, y_pred_svm)
# accuracy (262+117)/418 = 90.66%


##########################  Naive Bayes  ####################################################################

# Fitting Naive Bayes classifier to the opt Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train_opt, y_train_opt)

# predicting the test set result
y_pred_NB = classifier.predict(X_test_opt)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_NB = confusion_matrix(y_test_opt, y_pred_NB)
# accuracy (238+147)/418 = 92.1% 


#########################   Decision Tree Classification  #####################################################

# Fitting Decision Tree Classification classifier to the opt Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy' , random_state=42)
classifier.fit(X_train_opt, y_train_opt)

# predicting the test set result
y_pred_Dtree = classifier.predict(X_test_opt)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_Dtree = confusion_matrix(y_test_opt, y_pred_Dtree)
# accuracy (225 + 100)/418 = 77.75%


########################  Random forest classification  #######################################################

# Fitting Random forest classification classifier to the opt Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 42) 
classifier.fit(X_train_opt, y_train_opt) 

# predicting the test set result
y_pred_Rforest = classifier.predict(X_test_opt)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_Rforest = confusion_matrix(y_test, y_pred_Rforest)
# accuracy (226+108)/418= 79.9%