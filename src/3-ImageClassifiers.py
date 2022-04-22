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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


label = "Baseline"
size = (64,64)
f = open("../Pickles/Images/"+label+str(size),"rb")
x_train, y_train, x_test, y_test = pickle.load(f)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255



## DNN ##
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# print('y_train shape:', y_train.shape)
#
# # Convert class vectors to binary class matrices.
# y_train = keras.utils.to_categorical(y_train, 2)
# y_test = keras.utils.to_categorical(y_test, 2)
# x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
# x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
#
# model = Sequential()
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy',optimizer="adam" , metrics=['accuracy'])
#
# model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test), shuffle=True)
#
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
#
#
#
# model_json = model.to_json()
# with open("../model/Image/BaselineDNN.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("../model/Image/BaselineDNN.h5")
# print("Saved model to disk")


# RF ##
# x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
# x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
#
#
# clf = RandomForestClassifier().fit(x_train, y_train)
# score = clf.score(x_test, y_test)
# print("Accuracy:",score)
# f = open("../model/Image/BaselineRF","wb")
# pickle.dump(clf,f)



# ## LR ##
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))


clf = LogisticRegression(random_state=0).fit(x_train, y_train)
score = clf.score(x_test, y_test)
print("Accuracy:",score)
f = open("../model/Image/BaselineLR","wb")
pickle.dump(clf,f)
