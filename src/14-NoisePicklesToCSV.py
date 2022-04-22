from shutil import copyfile
from os import listdir
from os.path import isfile, join
from os import walk
import pickle

files = ["Adjacency","Algorithmic","Combined","HexDump","Image","Sections","Segments","String","Symbols"]
path = "/media/ahmed/HDD/ICDCS2021/Pickles/Noise/"

for file in files:
    f = open(path+file,"rb")

    data = pickle.load(f)
    toWrite = "Pert,"
    for i in range(len(data)):
        toWrite += str(i)+"%,"
    toWrite += "\nLR,"
    for i in range(len(data)):
        toWrite += str(data[i][0])+","
    toWrite += "\nRF,"
    for i in range(len(data)):
        toWrite += str(data[i][1])+","
    toWrite += "\nNN,"
    for i in range(len(data)):
        toWrite += str(data[i][2])+","

    f = open(path+file+".csv","w")
    f.write(toWrite)
    # print(toWrite)
    # exit()
