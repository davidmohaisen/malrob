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


f = open("../Pickles/filteredFilesTrainTest","rb")
_, _, FilesBenignTest, FilesMalwareTest = pickle.load(f)

path = "../data/staticAnalysisResults/"
insideFoldersBenign = ["packedBenign/","packedBenign-Best/","strippedBenign/","paddedBenign/","packedMalware/","packedMalware-Best/","strippedMalware/","paddedMalware/"]



results = ""

for fileFolder in insideFoldersBenign:
    print(fileFolder)
    results+="===============================\n"+fileFolder+"\n"

    FilesTest = None
    label = -1

    FeatureToGet = "hexdump/"

    x_test_HexDump = []
    y_test_HexDump = []
    if "Benign" in fileFolder:
        FilesTest = FilesBenignTest
        label = 0
    else:
        FilesTest = FilesMalwareTest
        label = 1
    counter = 0
    for filename in FilesTest:
        counter += 1
        try:
            f = open(path+fileFolder+FeatureToGet+filename+".pickle","rb")
            x = pickle.load(f)
            x_test_HexDump.append(x)
        except:
            x_test_HexDump.append(([0]*len(trainFunctionNames)))
        y_test_HexDump.append(label)

    x_test_HexDump = np.asarray(x_test_HexDump)
    y_test_HexDump = np.asarray(y_test_HexDump)


    f = open("../Pickles/Static/SymbolsF","rb")
    trainFunctionNames,dicLocations = pickle.load(f)

    FeatureToGet = "symbols/"


    x_test_Symbols = []
    y_test_Symbols = []
    if "Benign" in fileFolder:
        FilesTest = FilesBenignTest
        label = 0
    else:
        FilesTest = FilesMalwareTest
        label = 1
    counter = 0
    for filename in FilesTest:
        counter += 1
        try:
            f = open(path+fileFolder+FeatureToGet+filename+".pickle","rb")
            x = pickle.load(f)
            x_test_Symbols.append(([0]*len(trainFunctionNames)))
            for x_part in x.keys():
                try:
                    x_test_Symbols[-1][dicLocations[x_part]] = x[x_part][1]
                except:
                    pass
        except:
            x_test_Symbols.append(([0]*len(trainFunctionNames)))
        y_test_Symbols.append(label)

    x_test_Symbols = np.asarray(x_test_Symbols)
    y_test_Symbols = np.asarray(y_test_Symbols)



    f = open("../Pickles/Static/SectionsF","rb")
    trainFunctionNames,dicLocations = pickle.load(f)

    FeatureToGet = "sections/"


    x_test_Sections = []
    y_test_Sections = []
    if "Benign" in fileFolder:
        FilesTest = FilesBenignTest
        label = 0
    else:
        FilesTest = FilesMalwareTest
        label = 1
    counter = 0
    for filename in FilesTest:
        counter += 1
        try:
            f = open(path+fileFolder+FeatureToGet+filename+".pickle","rb")
            x = pickle.load(f)
            x_test_Sections.append(([0]*len(trainFunctionNames)))
            for x_part in x.keys():
                try:
                    x_test_Sections[-1][dicLocations[x_part]] = x[x_part][1]
                except:
                    pass
        except:
            x_test_Sections.append(([0]*len(trainFunctionNames)))
        y_test_Sections.append(label)

    x_test_Sections = np.asarray(x_test_Sections)
    y_test_Sections = np.asarray(y_test_Sections)


    f = open("../Pickles/Static/SegmentsF","rb")
    trainFunctionNames,dicLocations = pickle.load(f)

    FeatureToGet = "segments/"


    x_test_Segments = []
    y_test_Segments = []
    if "Benign" in fileFolder:
        FilesTest = FilesBenignTest
        label = 0
    else:
        FilesTest = FilesMalwareTest
        label = 1
    counter = 0
    for filename in FilesTest:
        counter += 1
        try:
            f = open(path+fileFolder+FeatureToGet+filename+".pickle","rb")
            x = pickle.load(f)
            x_test_Segments.append(([0]*len(trainFunctionNames)))
            for x_part in x.keys():
                try:
                    x_test_Segments[-1][dicLocations[x_part]] = x[x_part][1]
                except:
                    pass
        except:
            x_test_Segments.append(([0]*len(trainFunctionNames)))
        y_test_Segments.append(label)

    x_test_Segments = np.asarray(x_test_Segments)
    y_test_Segments = np.asarray(y_test_Segments)




    x_test = np.concatenate((x_test_HexDump,x_test_Symbols,x_test_Sections,x_test_Segments), axis = 1)
    y_test = y_test_Symbols



    print("==========================================================================")
    print("LR")
    print("==========================================================================")
    # clf = LogisticRegression(random_state=0).fit(x_train, y_train)
    f = open("../model/Static/CombinedBaselineLR","rb")
    clf = pickle.load(f)

    score = clf.score(x_test, y_test)
    print("The LR score is:",score)
    results += "The LR score is: "+str(score)+"\n"

    print("==========================================================================")
    print("RF")
    print("==========================================================================")
    f = open("../model/Static/CombinedBaselineRF","rb")
    clf = pickle.load(f)
    score = clf.score(x_test, y_test)
    print("The RF score is:",score)
    results += "The RF score is: "+str(score)+"\n"

    print("==========================================================================")



    print("==========================================================================")
    print("Deep Learning")
    print("==========================================================================")
    x_test_new = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))
    # create model

    json_file = open('../model/Static/CombinedBaselineDNN.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("../model/Static/CombinedBaselineDNN.h5")

    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

    scores = model.evaluate(x_test_new, y_test, verbose=1)
    print('Test accuracy:', scores[1])
    results += "The NN score is: "+str(scores[1])+"\n"
print(results)
