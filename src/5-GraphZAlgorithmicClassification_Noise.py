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


results = []
for pert in np.arange(0.00,1.01,0.01):
    results.append([])
    noise = np.random.normal(pert, pert, x_test.shape)
    # x_test_new = x_test + x_test*noise
    x_test_new = x_test + (np.amax(x_train)-np.amin(x_train))*noise

    print("==========================================================================")
    print("LR")
    print("==========================================================================")
    f = open("../model/Graph/AlgorithmicBaselineLR","rb")
    clf = pickle.load(f)

    score = clf.score(x_test_new, y_test)
    print("The LR score is:",score)
    results[-1].append(score)

    print("==========================================================================")
    print("RF")
    print("==========================================================================")
    f = open("../model/Graph/AlgorithmicBaselineRF","rb")
    clf = pickle.load(f)
    score = clf.score(x_test_new, y_test)
    print("The RF score is:",score)
    results[-1].append(score)

    print("==========================================================================")


    print("==========================================================================")
    print("Deep Learning")
    print("==========================================================================")
    x_test_new = np.reshape(x_test_new,(x_test_new.shape[0],x_test_new.shape[1],1))

    json_file = open('../model/Graph/AlgorithmicBaselineCNN.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("../model/Graph/AlgorithmicBaselineCNN.h5")

    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

    scores = model.evaluate(x_test_new, y_test, verbose=1)
    print('Test accuracy:', scores[1])
    results[-1].append(scores[1])


f = open("/media/ahmed/HDD/ICDCS2021/Pickles/Noise/Algorithmic","wb")
pickle.dump(results,f)
for i in range(101):
    print(i, results[i])
