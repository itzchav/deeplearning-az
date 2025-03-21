#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:38:56 2019

@author: juangabriel
"""

#************************************************
#Pre procesado de datos
#************************************************




# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')
print(dataset.shape)

X = dataset.iloc[:, 3:13].values
X_original = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#El OneHotEncoder en las nuevas versiones está OBSOLETO
#onehotencoder = OneHotEncoder(categorical_features=[1])
#X = onehotencoder.fit_transform(X).toarray()

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("Churn_Modelling",        # Un nombre de la transformación
         OneHotEncoder(categories='auto', sparse=False), # La clase a la que transformar
         [1]            # Las columnas a transformar.
         )
    ], remainder='passthrough'
)


X = transformer.fit_transform(X)
X = X[:, 1:]
'''
X = np.array(X)
y = np.array(y)

# Asegurarse de que los valores de X y y sean numéricos
X = X.astype(np.float32)  # Convertir a float32
y = y.astype(np.float32)  # Convertir a float32 (si es clasificación binaria)
'''
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)




# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#************************************************
#Construir RNA
#************************************************

import keras
from keras.models import Sequential
from keras.layers import Dense

#************************************************
#Inicializar Red Neuronal
#clasificador
#************************************************

classifier = Sequential()

#************************************************
#Capas de la RNA
#************************************************

#Añadir capa de entrada, y la primer capa oculta
classifier.add(Dense(units=6, kernel_initializer= "uniform", activation='relu', input_dim=X_train.shape[1]))
#segunda capa oculta
classifier.add(Dense(units=6, kernel_initializer= "uniform", activation='relu'))
#ultima capa oculta
classifier.add(Dense(units=1, kernel_initializer= "uniform", activation='sigmoid'))


#************************************************
#Compilar RNA
#************************************************
epocas = 2
classifier.compile(optimizer='adam', loss="binary_crossentropy", metrics="accuracy" )
classifier



#************************************************
#Gráficar Accuracy y perdida de las epocas
#************************************************
history = classifier.fit(X_train, y_train, batch_size=(10), epochs=epocas)
perdida, precision = classifier.evaluate(X_test)
print("PERDIDA: ", perdida)
print("PRESICIÓN: ", precision)

historial_dict = history.history
historial_dict.keys()


# Variables para las gráficas
acc = historial_dict['accuracy']
val_acc = historial_dict.get('val_accuracy', [])  # Puede no existir en algunos casos
perdida = historial_dict['loss']

# Graficar la pérdida
epocas = range(1, len(acc) + 1)
plt.plot(epocas, perdida, 'go', label='Pérdida de entrenamiento')
plt.title('Pérdida de entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()


plt.plot(epocas, acc, 'bo', label='Precisión de entrenamiento')
plt.title('Precisión del entrenamiento y la validación')
plt.xlabel('Epocas')
plt.ylabel('Precisión')
plt.legend(loc='lower right')

plt.show()

#************************************************
#Predicción
#************************************************

y_pred  = classifier.predict(X_test)

#************************************************
# Predicción de nuevos datos

#************************************************


X_predict = [600, 'Spain', 'Male', 40, 3, 60000, 2, 1, 1, 50000]

# Convertir X_predict a DataFrame
df_predict = pd.DataFrame([X_predict], columns=['CustomerId', 'Geography', 'Gender', 'Age', 'Tenure', 
                   
                                                'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
# Seleccionar las características de X_predict (eliminar la columna 'CustomerId' ya que no es relevante para la predicción)
X_predict = df_predict.iloc[:, :].values  # Esto incluye todas las columnas excepto 'CustomerId'

X_predict[:, 1] = labelencoder_X_1.fit_transform(X_predict[:, 1])
X_predict[:, 2] = labelencoder_X_2.fit_transform(X_predict[:, 2])


# Aplicar la transformación de OneHotEncoder para 'Geography' y 'Gender' (esto asegura que los datos tengan la misma forma que X_train)
X_predict = transformer.transform(X_predict)
X_predict = X_predict[:, 1:]
# No eliminamos ninguna columna adicional ya que 'transformer' se encargará de eso

# Escalar las características numéricas
X_predict = sc_X.transform(X_predict)

# Realizar la predicción
y_pred_new = classifier.predict(X_predict)
print("Predicción:", y_pred_new)

y_pred_new = (y_pred_new > 0.5)
print("Predicción:", y_pred_new)