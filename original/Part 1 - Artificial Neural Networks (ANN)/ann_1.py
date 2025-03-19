#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:38:56 2019

@author: juangabriel
"""

# Redes Neuronales Artificales

# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras

# Parte 1 - Pre procesado de datos


# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')
print(dataset.shape)

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Codificar datos Categóricos
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Se usa label ya que es si o no - 1 o 0
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)


