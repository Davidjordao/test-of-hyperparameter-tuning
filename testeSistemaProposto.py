#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 20:51:34 2021

@author: david jordão m. b. santos
"""
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

dataset_treinamento = '/home/david/Área de Trabalho/Sistema Proposto/Etapa de Treinamento/Features/treinamento_vgg19.mat'
dataset_teste = '/home/david/Área de Trabalho/Sistema Proposto/Etapa de Teste/Features/teste_vgg19.mat'

seed = 42
np.random.seed = seed
        
aux_treinamento = loadmat(dataset_treinamento)
bk_features_treino = aux_treinamento['features']
bk_labels_treino = aux_treinamento['labels']
        
aux_teste = loadmat(dataset_teste)
features_teste = aux_teste['features']
labels_teste = aux_teste['labels']


def __KMEANS__(nClusters, features_treino, labels_treino, seed):

    kmeans_model = KMeans(n_clusters=nClusters, random_state=seed)
    
    aux_treino_zero = features_treino[0:20,:]
    aux_treino_zero = kmeans_model.fit(aux_treino_zero).cluster_centers_
    
    aux_treino_um = features_treino[20:34,:]
    aux_treino_um = kmeans_model.fit(aux_treino_um).cluster_centers_    
                
    aux_treino_dois = features_treino[34:66,:]
    aux_treino_dois = kmeans_model.fit(aux_treino_dois).cluster_centers_            
                                      
    labels_treino = np.zeros((3*nClusters,features_treino.shape[1]))
    labels_treino =np.zeros((3*nClusters))
                
    labels_um = np.ones((1,nClusters))
    labels_dois = 2*labels_um
    labels_treino[nClusters:2*nClusters] = labels_um
    labels_treino[2*nClusters:3*nClusters] = labels_dois
      
    
    features_treino = np.zeros((3*nClusters,features_treino.shape[1]))                      
    features_treino[0:nClusters,:]  = aux_treino_zero 
    features_treino[nClusters:2*nClusters,:] = aux_treino_um
    features_treino[2*nClusters:3*nClusters] = aux_treino_dois
    
    return features_treino, labels_treino


def __SVM__(par_c, par_gamma, features_treino, labels_treino, features_teste, labels_teste):

    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C = par_c, gamma = par_gamma))
    clf.fit(features_treino,np.squeeze(labels_treino))
    resposta = clf.predict(features_teste)
    acc = np.mean(labels_teste == resposta)
    print("SVM>>",acc)
    __PLOT__(labels_teste, resposta,'SVM')

def __KNN__(par_k, features_treino, labels_treino,features_teste, labels_teste):
    
    
    neigh = KNeighborsClassifier(n_neighbors=par_k)
    neigh.fit(features_treino, labels_treino)
    resposta = neigh.predict(features_teste)
    acc = np.mean(labels_teste == resposta)
    print("KNN>>",acc)
    __PLOT__(labels_teste, resposta,'KNN')

def __RANDOMFOREST__(n_features, features_treino, features_teste, seed):
    
    clf = RandomForestClassifier(max_features=n_features,random_state = seed)
    clf.fit(features_treino,np.squeeze(labels_treino))
    resposta = clf.predict(features_teste)
    acc = np.mean(labels_teste == resposta)
    print("RANDOMFOREST>>",acc)
    __PLOT__(labels_teste, resposta,'RandomForest')

    
def __STANDARDSCALER__(features_treino, features_teste):

    scaler = StandardScaler()
    scaler.fit(features_treino)
    features_treino = scaler.transform(features_treino)
    features_teste = scaler.transform(features_teste)
    
    return features_treino, features_teste

def __PLOT__(labels_teste, resposta, titulo):
    
    cm = confusion_matrix(np.squeeze(labels_teste), resposta)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); 
    ax.set_title(titulo); 
    ax.xaxis.set_ticklabels(['Aquecido', 'Pouco Aq.','Muito Aq.']); ax.yaxis.set_ticklabels(['Aquecido', 'Pouco Aq.','Muito Aq.']);
    ax.set_xlabel('Predição do sistema');ax.set_ylabel('Condição real do item');     
    local_save =  '/home/david/Área de Trabalho/Sistema Proposto/Etapa de Teste/Plot/vgg19/'+ titulo+'.png'
    plt.savefig(local_save,dpi=300,transparent = True)

#--KNN--
nClusters = int(input("Entre com valor de médias do K-means(KNN):"))
print("OK...")
features_treino, labels_treino = __KMEANS__(nClusters, bk_features_treino, bk_labels_treino, seed)
par_k = int(input("Entre com o hiperparâmetro do K-NN:"))
print("OK...")
__KNN__(par_k, features_treino, labels_treino,features_teste, labels_teste)
print()
del  features_treino, labels_treino

#--RF--
nClusters = int(input("Entre com valor de médias do K-means(RF):"))
print("OK...")
features_treino, labels_treino = __KMEANS__(nClusters, bk_features_treino, bk_labels_treino, seed)
n_features = float(input("Entre com o hiperparâmetro RandomForest:"))
print("Ok...")
__RANDOMFOREST__(n_features, features_treino, features_teste, seed)
print()
del  features_treino, labels_treino

#--SVM--
nClusters = int(input("Entre com valor de médias do K-means(SVM):"))
print("OK...")
features_treino, labels_treino = __KMEANS__(nClusters, bk_features_treino, bk_labels_treino, seed)
par_c = float(input("Entre com o hiperparâmetro C do SVM:"))
print("OK...")
par_gamma= float(input("Entre com o hiperparâmetro Gamma do SVM:"))
print("OK...")
__SVM__(par_c, par_gamma, features_treino, labels_treino, features_teste, labels_teste)
del  features_treino, labels_treino  