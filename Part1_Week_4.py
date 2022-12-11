# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 20:56:41 2022

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:13:19 2022

@author: HP
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
import random
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
 
np.seterr(over='ignore')



def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict
datadict1 = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\cifar-10-batches-py\data_batch_1')
#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/test_batch')
datadict2 = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\cifar-10-batches-py\data_batch_2')
datadict3 = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\cifar-10-batches-py\data_batch_3')
datadict4 = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\cifar-10-batches-py\data_batch_4')
datadict5 = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\cifar-10-batches-py\data_batch_5')
print("this is datadict/n")
#print(datadict)

#X_Train and Y_train data for batch_1
X1 = np.array(datadict1["data"]) #X_train_for batch_1
Y1 = np.array(datadict1["labels"]) #Y_Train_for-batch_1

#print("SIZE OF X1 training data set is")
#print(X1)
#print(np.shape(X1))
#print("SIZE OF Y1 Training data set is ")
#print(np.shape(Y1))

#X_Train and Y_train data for batch_2
X2 = np.array(datadict2["data"]) #X_train_for batch_2
Y2 = np.array(datadict2["labels"]) #Y_Train_for-batch_2

#X_Train and Y_train data for batch_3
X3 = np.array(datadict3["data"]) #X_train_for batch_3
Y3 = np.array(datadict3["labels"]) #Y_Train_for-batch_3

#X_Train and Y_train data for batch_4
X4= np.array(datadict4["data"]) #X_train_for batch_4
Y4 = np.array(datadict4["labels"]) #Y_Train_for-batch_4

#X_Train and Y_train data for batch_5
X5 = np.array(datadict5["data"]) #X_train_for batch_5
Y5 = np.array(datadict5["labels"]) #Y_Train_for-batch_5
#now_concatinate_whole_5_batches_into_1_array_of_X_train, Y_train

print("Train Images ")
X=np.concatenate((X1,X2,X3,X4,X5),axis=0)
X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
print("Shape of X_Train Images after reshaping ")
print(X)
print(np.shape(X))

print("===============================")
print("===============================")
print("Normalizing Train Images")
X=X/255.0
Y=np.concatenate((Y1,Y2, Y3,Y4,Y5),axis=0)
#Y = Y.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")\
print(" X_train Normalized first 5 rows")
print(X[:5])
print("===============================")
print("===============================")
#do concatenate X and Y of all 5 batches here
("/n printing Y train data sets named as labels of Train_images /n")
print("now Y_train will print")
print(Y)
print("SIZE OF Y_Train")
print(np.shape(Y))


#Converting Y_Train_classes to_One_Hot_encoder
print("===============================")
print("===============================")
print("Converting Y_train to One Hot Encoder")
print(" Y_train One Hot Encoded first 5 rows")
#print(Y[:5])
Y_train_categorical=keras.utils.to_categorical(
                      Y, num_classes=10, dtype='float32')
print(Y_train_categorical[:5])
print("Y_train One Hot Encoded Complete")
print("===============================")
labeldict = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\cifar-10-batches-py\batches.meta')
label_names = labeldict["label_names"]
print("True Values of labels are "+ str(label_names))


print("calling test batch data now")
datadict_test = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\cifar-10-batches-py\test_batch')
X_test = np.array(datadict_test["data"])
print("Normalizing test_images ")
X_test=X_test/255.0
print("shape of X_test dataset before reshaping is "+ str(np.shape(X_test)))
print(X_test)
print("X_test after reshaping")
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

Y_test = np.array(datadict_test["labels"])

print("printing Y_test ")
print(Y_test[:5])

print("===============================")
print("===============================")
print("Converting Y_test to One Hot Encoder")

print(" Y_test One Hot Encoded first 5 rows")
#Converting Y_Train_classes to_One_Hot_encoder
Y_test_categorical=keras.utils.to_categorical (Y_test, num_classes=10, dtype='float32')
print(Y_test_categorical[:5])

print("Y_test One Hot Encoded Complete")
print("===============================")
 


print("shape of X_test dataset is "+ str(np.shape(X_test)))
print("shape of Y_test dataset is "+str(np.shape(Y_test)))

print("===============================")
print("===============================")
print("Makig 5 neurons layer")


model =   keras.Sequential([
    
        keras.layers.Flatten(input_shape=(32,32,3)), #Input layer with given input is X_train so shape of X_Train
        layers.Dense(5, activation='sigmoid'), #1st hidden layer with 5 neurons  
        layers.Dense(10, activation='sigmoid') #output layers has 10 classes so 10 neurons 
    ])


opt=keras.optimizers.SGD(lr=0.02)
model.compile(opt,
              loss='categorical_crossentropy', #categorical_crossentropy cause Y_train and Y_test is One_Hot_encoded or its a discrete value
              metrics=['accuracy'])

model.fit(X,Y_train_categorical,epochs=5 )
print("===============================")
print("Learning complete")
print("===============================")
print("===============================")
print("Prediction for X_test")
print("Preding 10 Outputs, one for each class")
model.predict(X_test)
np.shape(model.predict(X_test))
















