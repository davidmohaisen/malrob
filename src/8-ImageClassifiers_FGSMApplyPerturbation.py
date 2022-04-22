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
from PIL import Image
from art.classifiers import KerasClassifier
import sys
import os
import pickle
import os
import scipy
import array
import numpy
import scipy.misc
import imageio
from PIL import Image
from shutil import copyfile
sys.path.append(".")
from WhiteBoxAttackFGSM import FastGradientMethod

f = open("../Pickles/Images/ImageBinaryInjection","rb")
x_pert = pickle.load(f)
x_pert = x_pert.reshape((x_pert.shape[0],x_pert.shape[1],x_pert.shape[2]))

f = open("../Pickles/filteredFilesTrainTest","rb")
_, _, FilesBenginTest, FilesMalwareTest = pickle.load(f)



count = 0
path = "/media/ahmed/HDD/MalwareTemporalRobustness/Data/ELF/filtered/Benign/"
for file in FilesBenginTest:
    Filename = path+file;
    f = open(Filename,'rb');
    ln = os.path.getsize(Filename);
    width = 256;
    rem = ln%width;
    a = array.array("B");
    a.fromfile(f,ln-rem);
    f.close();

    lines = int(len(a)/width)

    pic = Image.fromarray(x_pert[count])
    pic = pic.crop((0, 32, 64, 64))

    pic = pic.resize((256,128));
    pic = pic.resize((256,lines));
    pix = np.array(pic)
    pix = pix * 255
    pix = pix.astype("byte")
    pix = pix.reshape((pix.shape[0]*pix.shape[1]))
    # pix_bytes = []
    # for pixInt in pix:
    #     pix_bytes.append(b''+pixInt)
    #
    # print(pix[:10])
    # print(pix_bytes[:10])
    # exit()

    copyfile(Filename, "/media/ahmed/HDD/ICDCS2021/data/benData/paddedBenign/"+file)

    f = open("/media/ahmed/HDD/ICDCS2021/data/benData/paddedBenign/"+file,"ab")
    f.write(pix)
    f.close()
    count += 1

# count = len(FilesBenginTest)

path = "/media/ahmed/HDD/MalwareTemporalRobustness/Data/ELF/filtered/Malware/"
for file in FilesMalwareTest:
    Filename = path+file;
    f = open(Filename,'rb');
    ln = os.path.getsize(Filename);
    width = 256;
    rem = ln%width;
    a = array.array("B");
    a.fromfile(f,ln-rem);
    f.close();

    lines = int(len(a)/width)

    pic = Image.fromarray(x_pert[count])
    pic = pic.crop((0, 32, 64, 64))

    pic = pic.resize((256,128));
    pic = pic.resize((256,lines));
    pix = np.array(pic)
    pix = pix * 255
    pix = pix.astype("byte")
    pix = pix.reshape((pix.shape[0]*pix.shape[1]))
    # pix_bytes = []
    # for pixInt in pix:
    #     pix_bytes.append(b''+pixInt)
    #
    # print(pix[:10])
    # print(pix_bytes[:10])
    # exit()

    copyfile(Filename, "/media/ahmed/HDD/ICDCS2021/data/malData/paddedMalware/"+file)
    # exit()
    f = open("/media/ahmed/HDD/ICDCS2021/data/malData/paddedMalware/"+file,"ab")
    f.write(pix)
    f.close()
    count += 1
