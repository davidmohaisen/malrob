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
            node_cnt = len(list(nx.nodes(g)))
            edge_cnt = len(list(nx.edges(g)))
            avg_shortest_path = ""
            shortest_path = []
            closeness = []
            diameter = 0
            radius = 0
            current_flow_closeness = ""
            try:
                avg_shortest_path = nx.average_shortest_path_length(g)
                shortest_path = nx.shortest_path(g)
                closeness = nx.algorithms.centrality.closeness_centrality(g)
                # shortest_betweenness = nx.algorithms.centrality.betweenness_centrality(g)
                degree_centrality = nx.algorithms.centrality.degree_centrality(g)
                density = nx.density(g)

            except:
                print("Unexpected error:", loc)
            sp_length = []
            for i in shortest_path:
                sp_length.append(shortest_path[i])
            shortestPathsArray = []
            for i in range(len(sp_length)):
                for x in sp_length[i] :
                    if (len(sp_length[i][x])-1)==0 :
                        continue
                    shortestPathsArray.append((len(sp_length[i][x])-1))

            if (len(shortestPathsArray))== 0 :
                continue

            maxShortestPath = np.max(shortestPathsArray)
            minShortestPath = np.min(shortestPathsArray)
            meanShortestPath = np.mean(shortestPathsArray)
            medianShortestPath = np.median(shortestPathsArray)
            stdShortestPath = np.std(shortestPathsArray)
            closeness_list = list(closeness.values())
            # betweenness_list = list(shortest_betweenness.values())
            degree_list = list(degree_centrality.values())
            out = [np.max(degree_list),np.min(degree_list),np.mean(degree_list),np.median(degree_list),np.std(degree_list),np.max(closeness_list),np.min(closeness_list),np.mean(closeness_list),np.median(closeness_list),np.std(closeness_list),maxShortestPath,minShortestPath,meanShortestPath,medianShortestPath,stdShortestPath,node_cnt,edge_cnt,density]
            # out = [np.max(degree_list),np.min(degree_list),np.mean(degree_list),np.median(degree_list),np.std(degree_list),np.max(betweenness_list),np.min(betweenness_list),np.mean(betweenness_list),np.median(betweenness_list),np.std(betweenness_list),np.max(closeness_list),np.min(closeness_list),np.mean(closeness_list),np.median(closeness_list),np.std(closeness_list),maxShortestPath,minShortestPath,meanShortestPath,medianShortestPath,stdShortestPath,node_cnt,edge_cnt,density]
            x_test.append(out)
            y_test.append(0)

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    f = open(savePath+insideFolder+"DataAlgorithmic","wb")
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
            node_cnt = len(list(nx.nodes(g)))
            edge_cnt = len(list(nx.edges(g)))
            avg_shortest_path = ""
            shortest_path = []
            closeness = []
            diameter = 0
            radius = 0
            current_flow_closeness = ""
            try:
                avg_shortest_path = nx.average_shortest_path_length(g)
                shortest_path = nx.shortest_path(g)
                closeness = nx.algorithms.centrality.closeness_centrality(g)
                # shortest_betweenness = nx.algorithms.centrality.betweenness_centrality(g)
                degree_centrality = nx.algorithms.centrality.degree_centrality(g)
                density = nx.density(g)

            except:
                print("Unexpected error:", loc)
            sp_length = []
            for i in shortest_path:
                sp_length.append(shortest_path[i])
            shortestPathsArray = []
            for i in range(len(sp_length)):
                for x in sp_length[i] :
                    if (len(sp_length[i][x])-1)==0 :
                        continue
                    shortestPathsArray.append((len(sp_length[i][x])-1))

            if (len(shortestPathsArray))== 0 :
                continue

            maxShortestPath = np.max(shortestPathsArray)
            minShortestPath = np.min(shortestPathsArray)
            meanShortestPath = np.mean(shortestPathsArray)
            medianShortestPath = np.median(shortestPathsArray)
            stdShortestPath = np.std(shortestPathsArray)
            closeness_list = list(closeness.values())
            # betweenness_list = list(shortest_betweenness.values())
            degree_list = list(degree_centrality.values())
            out = [np.max(degree_list),np.min(degree_list),np.mean(degree_list),np.median(degree_list),np.std(degree_list),np.max(closeness_list),np.min(closeness_list),np.mean(closeness_list),np.median(closeness_list),np.std(closeness_list),maxShortestPath,minShortestPath,meanShortestPath,medianShortestPath,stdShortestPath,node_cnt,edge_cnt,density]
            # out = [np.max(degree_list),np.min(degree_list),np.mean(degree_list),np.median(degree_list),np.std(degree_list),np.max(betweenness_list),np.min(betweenness_list),np.mean(betweenness_list),np.median(betweenness_list),np.std(betweenness_list),np.max(closeness_list),np.min(closeness_list),np.mean(closeness_list),np.median(closeness_list),np.std(closeness_list),maxShortestPath,minShortestPath,meanShortestPath,medianShortestPath,stdShortestPath,node_cnt,edge_cnt,density]
            x_test.append(out)
            y_test.append(1)

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    f = open(savePath+insideFolder+"DataAlgorithmic","wb")
    pickle.dump([x_test,y_test],f)
