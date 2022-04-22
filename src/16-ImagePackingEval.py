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

path = "/media/ahmed/HDD/ICDCS2021/images/Manipulated/"
insideFoldersBenign = ["packedBenign/","packedBenign-Best/","strippedBenign/","paddedBenign/","packedMalware/","packedMalware-Best/","strippedMalware/","paddedMalware/"]

def load_data(label,size,pathToDealWith,isBenign):

    f = open("../Pickles/filteredFilesTrainTest","rb")
    filesTest = None
    if isBenign:
        _, _, filesTest, _ = pickle.load(f)
    else:
        _, _, _, filesTest = pickle.load(f)



    x_test = []
    y_test = []

    for filename in filesTest:
        f = open(pathToDealWith+filename,"rb")
        image = pickle.load(f)
        image = np.resize(image, size)
        x_test.append(image)
        if isBenign:
            y_test.append(0)
        else:
            y_test.append(1)


    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    print(x_test.shape)
    print(y_test.shape)
    return [x_test,y_test]
#
# label = "Baseline"
# size = (64,64)
# f = open("../Pickles/Images/"+label+str(size),"rb")
# _, _, x_test, y_test = pickle.load(f)
# x_test_benign = []
# y_test_benign = []
# x_test_malware = []
# y_test_malware = []
# for i in range(len(x_test)):
#     if y_test[i] == 0:
#         x_test_benign.append(x_test[i])
#         y_test_benign.append(y_test[i])
#     else:
#         x_test_malware.append(x_test[i])
#         y_test_malware.append(y_test[i])
#
# x_test_benign = np.asarray(x_test_benign)
# y_test_benign = np.asarray(y_test_benign)
# x_test_malware = np.asarray(x_test_malware)
# y_test_malware = np.asarray(y_test_malware)
#
#
# # Normalize data.
# x_test_benign = x_test_benign.astype('float32') / 255
#
#
# print("==========================================================================")
# print("LR")
# print("==========================================================================")
# x_test_ = np.reshape(x_test_benign,(x_test_benign.shape[0],x_test_benign.shape[1]*x_test_benign.shape[2]))
# f = open("../model/Image/BaselineLR","rb")
# clf = pickle.load(f)
# score = clf.score(x_test_, y_test_benign)
# print("The LR score is:",score)
# print("==========================================================================")
# print("RF")
# print("==========================================================================")
# x_test_ = np.reshape(x_test_benign,(x_test_benign.shape[0],x_test_benign.shape[1]*x_test_benign.shape[2]))
# f = open("../model/Image/BaselineRF","rb")
# clf = pickle.load(f)
# score = clf.score(x_test_, y_test_benign)
# print("The RF score is:",score)
#
#
# print("==========================================================================")
#
#
# print("==========================================================================")
# print("Deep Learning")
# print("==========================================================================")
#
# # Convert class vectors to binary class matrices.
# y_test_n = keras.utils.to_categorical(y_test_benign, 2)
# x_test_new = np.reshape(x_test_benign,(x_test_benign.shape[0],x_test_benign.shape[1],x_test_benign.shape[2],1))
#
# json_file = open('../model/Image/BaselineDNN.json','r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights("../model/Image/BaselineDNN.h5")
#
# model.compile(loss='categorical_crossentropy',optimizer="adam" , metrics=['accuracy'])
#
# scores = model.evaluate(x_test_new, y_test_n, verbose=1)
# print('Test accuracy:', scores[1])
#
#
#
#
# # Normalize data.
# x_test_malware = x_test_malware.astype('float32') / 255
#
#
# print("==========================================================================")
# print("LR")
# print("==========================================================================")
# x_test_ = np.reshape(x_test_malware,(x_test_malware.shape[0],x_test_malware.shape[1]*x_test_malware.shape[2]))
# f = open("../model/Image/BaselineLR","rb")
# clf = pickle.load(f)
# score = clf.score(x_test_, y_test_malware)
# print("The LR score is:",score)
# print("==========================================================================")
# print("RF")
# print("==========================================================================")
# x_test_ = np.reshape(x_test_malware,(x_test_malware.shape[0],x_test_malware.shape[1]*x_test_malware.shape[2]))
# f = open("../model/Image/BaselineRF","rb")
# clf = pickle.load(f)
# score = clf.score(x_test_, y_test_malware)
# print("The RF score is:",score)
#
#
# print("==========================================================================")
#
#
# print("==========================================================================")
# print("Deep Learning")
# print("==========================================================================")
#
# # Convert class vectors to binary class matrices.
# y_test_n = keras.utils.to_categorical(y_test_malware, 2)
# x_test_new = np.reshape(x_test_malware,(x_test_malware.shape[0],x_test_malware.shape[1],x_test_malware.shape[2],1))
#
# json_file = open('../model/Image/BaselineDNN.json','r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights("../model/Image/BaselineDNN.h5")
#
# model.compile(loss='categorical_crossentropy',optimizer="adam" , metrics=['accuracy'])
#
# scores = model.evaluate(x_test_new, y_test_n, verbose=1)
# print('Test accuracy:', scores[1])
#
#
# exit()

results = ""
for pathEval in insideFoldersBenign:
    print(pathEval)
    results+="===============================\n"+pathEval+"\n"
    isBenign = False
    if "Benign" in pathEval:
        isBenign = True
    (x_test, y_test) = load_data(label="Baseline",size=(64,64),pathToDealWith=path+pathEval,isBenign=isBenign)

    print(x_test.shape)
    print(y_test.shape)

    # Input image dimensions.
    input_shape = x_test.shape[1:]

    # Normalize data.
    x_test = x_test.astype('float32') / 255


    print("==========================================================================")
    print("LR")
    print("==========================================================================")
    x_test_ = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
    f = open("../model/Image/BaselineLR","rb")
    clf = pickle.load(f)

    score = clf.score(x_test_, y_test)
    print("The LR score is:",score)
    results += "The LR score is: "+str(score)+"\n"
    print("==========================================================================")
    print("RF")
    print("==========================================================================")
    x_test_ = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
    f = open("../model/Image/BaselineRF","rb")
    clf = pickle.load(f)
    score = clf.score(x_test_, y_test)
    print("The RF score is:",score)
    results += "The RF score is: "+str(score)+"\n"


    print("==========================================================================")


    print("==========================================================================")
    print("Deep Learning")
    print("==========================================================================")

    # Convert class vectors to binary class matrices.
    y_test_n = keras.utils.to_categorical(y_test, 2)
    x_test_new = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))

    json_file = open('../model/Image/BaselineDNN.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("../model/Image/BaselineDNN.h5")

    model.compile(loss='categorical_crossentropy',optimizer="adam" , metrics=['accuracy'])

    scores = model.evaluate(x_test_new, y_test_n, verbose=1)
    print('Test accuracy:', scores[1])
    results += "The NN score is: "+str(scores[1])+"\n"


print(results)
