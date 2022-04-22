import os
import pickle
import os
import scipy
import array
import numpy
import scipy.misc
import imageio
from PIL import Image
from os import listdir
from os.path import isfile, join

benignName = []
malwareName = []

folders = ["functions","hexdump","sections","segments","strings","symbols"]


path = "../linuxBenignStaticAnalysisResults/"
fileNames = []
for folder in folders:
    onlyfiles = [f for f in listdir(path+folder+"/") if isfile(join(path+folder+"/", f))]
    for i in range(len(onlyfiles)):
        onlyfiles[i] = onlyfiles[i].replace(".pickle","")
    fileNames.append(set(onlyfiles))

benignName = fileNames[0]
for fileName in fileNames:
    benignName = benignName.intersection(fileName)

benignName = list(benignName)


path = "../malwareStaticAnalysisResults/"
fileNames = []
for folder in folders:
    onlyfiles = [f for f in listdir(path+folder+"/") if isfile(join(path+folder+"/", f))]
    for i in range(len(onlyfiles)):
        onlyfiles[i] = onlyfiles[i].replace(".pickle","")
    fileNames.append(set(onlyfiles))

malwareName = fileNames[0]
for fileName in fileNames:
    malwareName = malwareName.intersection(fileName)

malwareName = (list(malwareName))[:3000]


f = open("../Pickles/filteredFiles","wb")
pickle.dump([benignName,malwareName],f)


f = open("../Pickles/filteredFilesTrainTest","wb")
pickle.dump([benignName[:int(0.8*len(benignName))],malwareName[:int(0.8*len(malwareName))],benignName[int(0.8*len(benignName)):],malwareName[int(0.8*len(malwareName)):]],f)
