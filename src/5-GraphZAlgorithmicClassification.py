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

f = open("../Pickles/Graph/DataAlgorithmic","rb")
x_train,y_train,x_test,y_test = pickle.load(f)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

print("==========================================================================")
print("LR")
print("==========================================================================")
clf = LogisticRegression(random_state=0).fit(x_train, y_train)
score = clf.score(x_test, y_test)
print("The LR score is:",score)
f = open("../model/Graph/AlgorithmicBaselineLR","wb")
pickle.dump(clf,f)


print("==========================================================================")
print("RF")
print("==========================================================================")
clf = RandomForestClassifier(random_state=0).fit(x_train, y_train)
score = clf.score(x_test, y_test)
print("The RF score is:",score)
f = open("../model/Graph/AlgorithmicBaselineRF","wb")
pickle.dump(clf,f)
print("==========================================================================")



# print("==========================================================================")
# print("Deep Learning")
# print("==========================================================================")
# # pca = PCA(n_components=0.999)
# # pca.fit(x_train)
# # f = open("../model/Static/SymbolsBaselineDNN_PCA","wb")
# # pickle.dump(pca,f)
#
# # f = open("../model/Static/SymbolsBaselineDNN_PCA","rb")
# # pca = pickle.load(f)
# #
# # x_train_pca = pca.transform(x_train)
# # x_test_pca = pca.transform(x_test)
# x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
# x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
# # print(x_train_pca.shape)
# # create model
# model = Sequential()
# model.add(keras.layers.Dense(128, activation='relu',input_shape=(x_train.shape[1:])))
# # model.add(keras.layers.Dropout(0.25))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(128, activation='relu'))
# # model.add(keras.layers.Dropout(0.25))
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
# with open("../model/Graph/AlgorithmicBaselineDNN.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("../model/Graph/AlgorithmicBaselineDNN.h5")
# print("Saved model to disk")


print("==========================================================================")
print("CNN")
print("==========================================================================")
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
# create model
model = Sequential()

filter = 64
# create model
model = Sequential()
model.add(keras.layers.Conv1D(filter,3,padding="same",activation="relu",input_shape=(x_train.shape[1:])))
model.add(keras.layers.Conv1D(filter,3,padding="valid",activation="relu"))
model.add(keras.layers.MaxPooling1D())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv1D(filter*2,3,padding="same",activation="relu"))
model.add(keras.layers.Conv1D(filter*2,3,padding="valid",activation="relu"))
model.add(keras.layers.MaxPooling1D())
# model.add(keras.layers.Dropout(0.25))
# model.add(keras.layers.Conv1D(filter*3,3,padding="same",activation="relu"))
# model.add(keras.layers.Conv1D(filter*3,3,padding="valid",activation="relu"))
# model.add(keras.layers.MaxPooling1D())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(256, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation="softmax"))

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
model.fit(x_train,y_train, epochs=10, batch_size=64)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', scores[1])
model_json = model.to_json()
with open("../model/Graph/AlgorithmicBaselineCNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../model/Graph/AlgorithmicBaselineCNN.h5")
print("Saved model to disk")
