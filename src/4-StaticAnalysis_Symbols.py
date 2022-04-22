from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from keras.models import model_from_json
from PIL import Image
from sklearn.utils import shuffle
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import collections
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.decomposition import PCA
f = open("../Pickles/filteredFilesTrainTest","rb")
FilesBenignTrain, FilesMalwareTrain, FilesBenignTest, FilesMalwareTest = pickle.load(f)

pathBenign = "../linuxBenignStaticAnalysisResults/symbols/"
pathMalware = "../malwareStaticAnalysisResults/symbols/"


x_train = []
y_train = []
x_test = []
y_test = []

trainFunctionNames = set()
counter = 0
for filename in FilesBenignTrain:
    counter += 1
    print(counter,len(FilesBenignTrain))
    f = open(pathBenign+filename+".pickle","rb")
    x = pickle.load(f)
    trainFunctionNames = trainFunctionNames.union(set(x.keys()))
    # for key in list(x.keys()):
    #     trainFunctionNames.append(key)

counter = 0
for filename in FilesMalwareTrain:
    counter += 1
    print(counter,len(FilesMalwareTrain))
    f = open(pathMalware+filename+".pickle","rb")
    x = pickle.load(f)
    trainFunctionNames = trainFunctionNames.union(set(x.keys()))
    # for key in list(x.keys()):
    #     trainFunctionNames.append(key)

trainFunctionNames = list(trainFunctionNames)

dicLocations = {}
for i in range(len(trainFunctionNames)):
    dicLocations[trainFunctionNames[i]] = i

f = open("../Pickles/Static/SymbolsF","wb")
pickle.dump([trainFunctionNames,dicLocations],f)


counter = 0
for filename in FilesBenignTrain:
    counter += 1
    print(counter,len(FilesBenignTrain))
    f = open(pathBenign+filename+".pickle","rb")
    x = pickle.load(f)
    x_train.append(([0]*len(trainFunctionNames)))
    for x_part in x.keys():
        x_train[-1][dicLocations[x_part]] = x[x_part][1]
    y_train.append(0)
    # for key in list(x.keys()):
    #     trainFunctionNames.append(key)

counter = 0
for filename in FilesMalwareTrain:
    counter += 1
    print(counter,len(FilesMalwareTrain))
    f = open(pathMalware+filename+".pickle","rb")
    x = pickle.load(f)
    x_train.append(([0]*len(trainFunctionNames)))
    for x_part in x.keys():
        x_train[-1][dicLocations[x_part]] = x[x_part][1]
    y_train.append(1)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
f = open("../Pickles/Static/SymbolsTrainDataP1", "wb")
pickle.dump([x_train[:int(len(x_train)/2)],y_train[:int(len(x_train)/2)]],f)
f = open("../Pickles/Static/SymbolsTrainDataP2", "wb")
pickle.dump([x_train[int(len(x_train)/2):],y_train[int(len(x_train)/2):]],f)

del x_train,y_train

counter = 0
for filename in FilesBenignTest:
    counter += 1
    print(counter,len(FilesBenignTest))
    f = open(pathBenign+filename+".pickle","rb")
    x = pickle.load(f)
    x_test.append(([0]*len(trainFunctionNames)))
    for x_part in x.keys():
        try:
            x_test[-1][dicLocations[x_part]] = x[x_part][1]
        except:
            pass
    y_test.append(0)
    # for key in list(x.keys()):
    #     trainFunctionNames.append(key)

counter = 0
for filename in FilesMalwareTest:
    counter += 1
    print(counter,len(FilesMalwareTest))
    f = open(pathMalware+filename+".pickle","rb")
    x = pickle.load(f)
    x_test.append(([0]*len(trainFunctionNames)))
    for x_part in x.keys():
        try:
            x_test[-1][dicLocations[x_part]] = x[x_part][1]
        except:
            pass
    y_test.append(1)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
f = open("../Pickles/Static/SymbolsTestData", "wb")
pickle.dump([x_test,y_test],f)

f = open("../Pickles/Static/SymbolsTrainDataP1","rb")
x_train_1, y_train_1 = pickle.load(f)
f = open("../Pickles/Static/SymbolsTrainDataP2","rb")
x_train_2, y_train_2 = pickle.load(f)
x_train = np.concatenate((x_train_1,x_train_2))
y_train = np.concatenate((y_train_1,y_train_2))
f = open("../Pickles/Static/SymbolsTestData","rb")
x_test, y_test = pickle.load(f)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# print(x_train[0])
# print(y_train[0])
# print(x_test[0])
# print(y_test[0])
# exit()

print("==========================================================================")
print("LR")
print("==========================================================================")
clf = LogisticRegression(random_state=0).fit(x_train, y_train)
score = clf.score(x_test, y_test)
print("The LR score is:",score)
f = open("../model/Static/SymbolsBaselineLR","wb")
pickle.dump(clf,f)


print("==========================================================================")
print("RF")
print("==========================================================================")
clf = RandomForestClassifier(random_state=0).fit(x_train, y_train)
score = clf.score(x_test, y_test)
print("The RF score is:",score)
f = open("../model/Static/SymbolsBaselineRF","wb")
pickle.dump(clf,f)
print("==========================================================================")



print("==========================================================================")
print("Deep Learning")
print("==========================================================================")
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1]))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))
# create model
model = Sequential()
model.add(keras.layers.Dense(64, activation='relu',input_shape=(x_train.shape[1:])))
# model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.25))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(256, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation="softmax"))

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.fit(x_train,y_train, epochs=20, batch_size=32)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', scores[1])


model_json = model.to_json()
with open("../model/Static/SymbolsBaselineDNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../model/Static/SymbolsBaselineDNN.h5")
print("Saved model to disk")


# print("==========================================================================")
# print("CNN")
# print("==========================================================================")
# x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
# x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
# # create model
# model = Sequential()
#
# filter = 64
# # create model
# model = Sequential()
# model.add(keras.layers.Conv1D(filter,3,padding="same",activation="relu",input_shape=(x_train.shape[1:])))
# # model.add(keras.layers.Conv1D(filter,3,padding="valid",activation="relu"))
# # model.add(keras.layers.MaxPooling1D())
# model.add(keras.layers.Dropout(0.25))
# model.add(keras.layers.Conv1D(filter*2,3,padding="same",activation="relu"))
# # model.add(keras.layers.Conv1D(filter*2,3,padding="valid",activation="relu"))
# # model.add(keras.layers.MaxPooling1D())
# # model.add(keras.layers.Dropout(0.25))
# # model.add(keras.layers.Conv1D(filter*3,3,padding="same",activation="relu"))
# # model.add(keras.layers.Conv1D(filter*3,3,padding="valid",activation="relu"))
# # model.add(keras.layers.MaxPooling1D())
# model.add(keras.layers.Dropout(0.25))
# model.add(keras.layers.Flatten())
# # model.add(keras.layers.Dense(256, activation='relu'))
# # model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(2, activation="softmax"))
#
# # Compile model
# model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
# model.fit(x_train,y_train, epochs=10, batch_size=64)
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test accuracy:', scores[1])
# model_json = model.to_json()
# with open("../model/Static/SymbolsBaselineCNN.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("../model/Static/SymbolsBaselineCNN.h5")
# print("Saved model to disk")
