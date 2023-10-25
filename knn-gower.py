# Se importan las librerias necesarias
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter   #se usa para sacar el most_common al final
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split


class KNN:
    def __init__(self, k=10):
        self.k = k

    def fit(self, atributos, clase):
        self.atributos = atributos
        self.clase = clase

    def is_binary_feature(self, feature):
        unique_values = np.unique(feature)
        return len(unique_values) == 2

    def gower_distance(self, row, dataset):
        n = len(row)
        cont_indices = [i for i in range(n) if not self.is_binary_feature(row[i])]
        cat_indices = [i for i in range(n) if i not in cont_indices]
        cont_x = np.array([row[i] for i in cont_indices])  # Filtrar elementos dentro del rango
        distances = []
        for _, train_row in dataset.iterrows():
            cont_y = np.array([train_row[i] for i in cont_indices if i < len(cont_indices)])
            cont_dist = cdist(cont_x.reshape(1, -1), cont_y.reshape(1, -1), metric='euclidean')[0][0]
            cat_y = np.array([train_row[i] for i in cat_indices if i < len(train_row)])  # Filtrar elementos dentro del rango
            cat_dist = np.sum(row[cat_indices] != cat_y) / len(cat_indices)
            print('cont', cont_dist, cat_dist)
            print ('denominador', cont_indices, cat_indices)
            total_dist = (cont_dist + cat_dist) / (len(cont_indices) + len(cat_indices))
            distances.append(total_dist)

        return distances

    def predecir(self, nuevo_registro):
        print('nuevo registro', nuevo_registro)
        clases = [self._predecir(row) for _, row in nuevo_registro.iterrows()]
        return np.array(clases)

    def _predecir(self, row):
        distances = self.gower_distance(row, self.atributos)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.clase.iloc[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]





    
dataset = pd.read_csv('data_cardiovascular_risk.csv').dropna()
dataset['sex'] = dataset['sex'].replace({'M': 1, 'F': 0})
dataset['is_smoking'] = dataset['is_smoking'].replace({'YES': 1, 'NO':0})
copyDataset = dataset.copy().drop(columns=['id','TenYearCHD']) #Dataset sin las etiquetas
labels = dataset['TenYearCHD'] #Etiquetas dataset
train_dataset, test_dataset, labels_train, labels_test = train_test_split(copyDataset,labels, test_size = 0.25, random_state=42) # en el 3er parametro se ve que se puede jugar con el tamaño de los conjuntos de entrenamiento.
print("Dataset Training: ", len(train_dataset))

knn = KNN(k=1)
knn.fit(train_dataset,labels_train)

predictions = knn.predecir(test_dataset)
print(predictions)

