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

def load_data(label,size):
    if os.path.exists("../Pickles/Images/"+label+str(size)):
        f = open("../Pickles/Images/"+label+str(size),"rb")
        xtrain,ytrain,xtest,ytest = pickle.load(f)
        return [xtrain,ytrain],[xtest,ytest]

    pathBenign = "../images/Benign/"
    pathMalware = "../images/Malware/"

    f = open("../Pickles/filteredFilesTrainTest","rb")
    FilesBenignTrain, FilesMalwareTrain, FilesBenginTest, FilesMalwareTest = pickle.load(f)

    x_train = []
    y_train = []
    x_test = []
    y_test = []


    for filename in FilesBenignTrain:
        pic = Image.open(pathBenign+filename+".png")
        pic = pic.resize(size);
        pix = np.array(pic)
        x_train.append(pix)
        y_train.append(0)

    for filename in FilesMalwareTrain:
        pic = Image.open(pathMalware+filename+".png")
        pic = pic.resize(size);
        pix = np.array(pic)
        x_train.append(pix)
        y_train.append(1)



    for filename in FilesBenginTest:
        pic = Image.open(pathBenign+filename+".png")
        pic = pic.resize(size);
        pix = np.array(pic)
        x_test.append(pix)
        y_test.append(0)

    for filename in FilesMalwareTest:
        pic = Image.open(pathMalware+filename+".png")
        pic = pic.resize(size);
        pix = np.array(pic)
        x_test.append(pix)
        y_test.append(1)



    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    f = open("../Pickles/Images/"+label+str(size),"wb")
    pickle.dump([x_train,y_train,x_test,y_test],f)
    return [x_train,y_train],[x_test,y_test]


(x_train, y_train), (x_test, y_test) = load_data(label="Baseline",size=(64,64))

# # Input image dimensions.
# input_shape = x_train.shape[1:]
#
# # Normalize data.
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
#
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# print('y_train shape:', y_train.shape)
#
# # Convert class vectors to binary class matrices.
# y_train = keras.utils.to_categorical(y_train, 2)
# y_test = keras.utils.to_categorical(y_test, 2)
#
# model = Sequential()
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
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
# with open("../Model/Image/BaselineSimple.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("../Model/Image/BaselineSimple.h5")
# print("Saved model to disk")
