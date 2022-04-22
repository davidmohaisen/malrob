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


results = []
for pert in np.arange(0.00,1.01,0.01):
    results.append([])
    noise = np.random.normal(pert, pert, x_test.shape)
    # x_test_new = x_test + x_test*noise
    x_test_new = x_test + (np.amax(x_train)-np.amin(x_train))*noise

    print("==========================================================================")
    print("LR")
    print("==========================================================================")
    x_test_new_ = np.reshape(x_test_new,(x_test_new.shape[0],x_test_new.shape[1]*x_test_new.shape[2]))
    f = open("../model/Image/BaselineLR","rb")
    clf = pickle.load(f)

    score = clf.score(x_test_new_, y_test)
    print("The LR score is:",score)
    results[-1].append(score)

    print("==========================================================================")
    print("RF")
    print("==========================================================================")
    x_test_new_ = np.reshape(x_test_new,(x_test_new.shape[0],x_test_new.shape[1]*x_test_new.shape[2]))
    f = open("../model/Image/BaselineRF","rb")
    clf = pickle.load(f)
    score = clf.score(x_test_new_, y_test)
    print("The RF score is:",score)
    results[-1].append(score)

    print("==========================================================================")


    print("==========================================================================")
    print("Deep Learning")
    print("==========================================================================")

    # Convert class vectors to binary class matrices.
    y_test_n = keras.utils.to_categorical(y_test, 2)
    x_test_new = np.reshape(x_test_new,(x_test_new.shape[0],x_test_new.shape[1],x_test_new.shape[2],1))

    json_file = open('../model/Image/BaselineDNN.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("../model/Image/BaselineDNN.h5")

    model.compile(loss='categorical_crossentropy',optimizer="adam" , metrics=['accuracy'])

    scores = model.evaluate(x_test_new, y_test_n, verbose=1)
    print('Test accuracy:', scores[1])
    results[-1].append(scores[1])


f = open("/media/ahmed/HDD/ICDCS2021/Pickles/Noise/Image","wb")
pickle.dump(results,f)
for i in range(101):
    print(i, results[i])
