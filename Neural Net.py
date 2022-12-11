# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:13:19 2022

@author: HP
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras

import matplotlib.pyplot as plt
import pickle
import numpy as np
from random import random
import random
from tqdm import tqdm
 
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

print("SIZE OF X1 training data set is")
print(X1)
print(np.shape(X1))


print("SIZE OF Y1 Training data set is ")
print(np.shape(Y1))

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


print("Normalizing Train Images")

X=X/255.0
Y=np.concatenate((Y1,Y2, Y3,Y4,Y5),axis=0)
#Y = Y.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
print("/n printing X data set named as Train_images /n")
print(X)
#do concatenate X and Y of all 5 batches here
("/n printing Y train data sets named as labels of Train_images /n")
print("now Y will print")
print(Y)

print("SIZE OF Y_Train")
print(np.shape(Y))
a=np.shape(Y)
print(a[0])

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

print("shape of X_test dataset is "+ str(np.shape(X_test)))
print("shape of Y_test dataset is "+str(np.shape(Y_test)))



class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

#print(class_names[1]) out put will be automobile at 1 idex
# Part 2 - data_exploration

def plot_sample(X, y, class_index):

    plt.figure(figsize=(30,2)) #control size of image
    plt.imshow(X[class_index])
    plt.xlabel(class_names[y[class_index]])
    
plot_sample(X, Y, 1)    

#part 3 - Normalization, already done above

#print("Normalizing Train Images")
#X=X/255.0
#print("Normalizing test_images ")
#X_test=X_test/255.0

#part 4- Make a simple Artificial Neural Network

ann =   models.Sequential([
    
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'), #1st deep layers with 3000 neurons
        layers.Dense(1000, activation='relu'), #2nd deep layers with 1000 neurons
        layers.Dense(10, activation='softmax') #third layers with 1000 neurons 
    ])

keras.optimizers.SGD(lr=0.2)
ann.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X, Y, epochs=5, verbose=1)


#now we will make cnn neural network
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='sigmoid')])


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#now efficiency will get print till 10 epox
cnn.fit(X, Y, epochs=10)

#evalute efficiency of X_test and Y_test and compare with cnn epox 10th efficiency
cnn.evaluate(X_test, Y_test)















