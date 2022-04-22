import networkx as nx
# import matplotlib.pyplot as plt
import os
import sys
import pygraphviz
import numpy as np
from shutil import copyfile
import random
import pickle
#IoT malware features

pathAll = "../data/staticAnalysisResults/"
insideFoldersBenign = ["packedBenign/","packedBenign-Best/","paddedBenign/","strippedBenign/"]
insideFoldersMalware = ["packedMalware/","packedMalware-Best/","paddedMalware/","strippedMalware/"]

savePath = "../Pickles/Graph/Manipulated/"

f = open("../Pickles/filteredFilesTrainTest","rb")
_, _, FilesBenignTest, FilesMalwareTest = pickle.load(f)


for insideFolder in insideFoldersBenign:
    x_test = []
    y_test = []

    for file in FilesBenignTest:
        print(file)
        nodes_density = []
        loc = pathAll + insideFolder+"graphs/"+ file+".dot"
        g = ""
        try:
            g = nx.drawing.nx_agraph.read_dot(loc)
        except:
            print("Passed Sample")
            pass
        if g!= "" :
            A = nx.adjacency_matrix(g)
            A = A.todense()
            A = np.array(A)
            # print(A.shape)
            arrToAdd = np.zeros((100,100))
            # print(arrToAdd.shape)
            for j in range(min(100,len(A))):
                for k in range(min(100,len(A[j]))):
                    arrToAdd[j][k] = A[j][k]
            # print(arrToAdd.shape)
            # exit()
            x_test.append(arrToAdd)
            y_test.append(0)

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    f = open(savePath+insideFolder+"DataAdjacency","wb")
    pickle.dump([x_test,y_test],f)


for insideFolder in insideFoldersMalware:
    x_test = []
    y_test = []

    for file in FilesMalwareTest:
        print(file)
        nodes_density = []
        loc = pathAll + insideFolder+"graphs/"+ file+".dot"
        g = ""
        try:
            g = nx.drawing.nx_agraph.read_dot(loc)
        except:
            print("Passed Sample")
            pass
        if g!= "" :
            A = nx.adjacency_matrix(g)
            A = A.todense()
            A = np.array(A)
            # print(A.shape)
            arrToAdd = np.zeros((100,100))
            # print(arrToAdd.shape)
            for j in range(min(100,len(A))):
                for k in range(min(100,len(A[j]))):
                    arrToAdd[j][k] = A[j][k]
            # print(arrToAdd.shape)
            # exit()
            x_test.append(arrToAdd)
            y_test.append(1)

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    f = open(savePath+insideFolder+"DataAdjacency","wb")
    pickle.dump([x_test,y_test],f)
