# -*- coding: utf-8 -*-
"""


"""

# Import the Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


#Data Preprocessing
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values #Start from third column until last one get all columns
y = dataset.iloc[:, -1].values # Get last column as independent value

#Lable Encode the String column to numerical values
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2]) #Encode the gender index second column

#OneHot Encoding to Country Column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough') #Encoding index 1 which is country column
X = np.array(ct.fit_transform(X))

#Train the Model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)#Split the dataset by 20 percent for test and rest for train

#Feature Scaling to all our features since all variables are numerical
sc = StandardScaler()
X_train = sc.fit_transform(X_train) #Scaling on all features of X_train dataset
X_test = sc.transform(X_test) #Scaling on all features of X_test dataset

#Buid an Artifiacl Neural Network
model = tf.keras.models.Sequential() #Initilize the ANN
model.add(tf.keras.layers.Dense(units=6, activation='relu')) #Add First Input Dense Layer / All features are nueron
model.add(tf.keras.layers.Dense(units=6, activation='relu')) #Add Secong Hidden Layer
#Since prediction result will be boolen value 0 or 1 and activation must be sigmoid
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #Adding the output layer /
#Since we have one output classification then we choose the activation as sigmoid otherwise we should choose softmax

#Compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#The best one are the optimers that can perform to gas the gradient descent is adam optimezer
#Since we are doing binary classification like 0 and 1 then loss function must always be binary_crossentropy

#TRAINING THE MODEL
model.summary()
history = model.fit(X_train, y_train, batch_size = 32, epochs = 100)
#Batch Size : Instead of compairing a prediction to the result one by one we do that by several prediction result 
#Mumber of learning times

#PREDICT BY ONE EXAMPLE
print(model.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) < 0.5)
#If bigger then .5 then output is 1 otherwise result will be 0

#PREDICTON ON TEST DATASET
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)#If prediction result will be over .5 then result will 1 otherwise result will be 0
#Print prediction dataset with real dataset and compare them. Left one is prediction result right one is real dataset
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

score = model.evaluate(X_test, y_test, verbose=1)

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


#Visulizing

from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

