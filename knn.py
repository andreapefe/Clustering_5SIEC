# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:57:12 2023

@author: andrea
"""
import numpy as np
import matplotlib . pyplot as plt
import time
from sklearn import cluster
from scipy . io import arff
#
# Les donnees sont dans datanp (2 dimensions )
# f0 : valeurs sur la premiere dimension


path ='artificial'
databrut = arff.loadarff ( open ( path +'\\xclara.arff' , 'r') )
datanp = [[x[0],x[1]] for x in databrut [0]]
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ -0.499261 , -1.51369 , -1.60321 , ...]
# Ex pour f1 = [ -0. 0612356 , 0.265446 , 0.362039 , ...]
datanp = np.asarray(datanp)
f0 = datanp[:,0] # tous les elements de la premiere colonne
f1 = datanp[:,1] # tous les elements de la deuxieme colonne

print (" Appel KMeans pour une valeur fixee de k ")
tps1 = time . time ()
k=3
model = cluster . KMeans ( n_clusters =k , init ='k-means++')
model . fit ( datanp )
tps2 = time . time ()
labels = model . labels_
iteration = model . n_iter_
plt.scatter ( f0 , f1 , c=labels , s=8 )
plt.title (" Donnees apres clustering Kmeans ")
plt.show    ()
print ("nb clusters =",k ,", nb iter =", iteration , ", ... runtime = ", round (( tps2 - tps1 )*1000 , 2 ) ,"ms")