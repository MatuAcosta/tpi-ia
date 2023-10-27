# Se importan las librerias necesarias
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score
#se usa para sacar el most_common al final

class KNN:
    def __init__(self, k=10):
        self.k = k

    def fit(self, atributos, clase):
        self.atributos = atributos
        self.clase = clase

    def distancia_euclidea(self, punto1, punto2):        # point1 and point2 son numpy arrays representando las coordenadas de dos puntos en el espacio multidimensional.
        #print('punto1', punto1, 'punto2', punto2)
        distancia = np.sqrt(np.sum((punto1 - punto2) ** 2))
        if np.isnan(distancia):    #si es null ignoramos con High Value (por ahora)
            return 9999999
        else:
            return distancia

    def predecir(self, nuevo_registro):
        clase = [self._predecir(punto) for punto in nuevo_registro.values]
        return np.array(clase)
    def predecir_ponderado(self, nuevo_registro):
        clase = [self._predecir_ponderado(punto) for punto in nuevo_registro.values]
        return np.array(clase)

    # Calculamos distancias entre test_point y todos los ejemplos en el training set
    def _predecir(self, test_point):
        distancias = [self.distancia_euclidea(test_point, train_point) for train_point in self.atributos.values]
        k_indices = np.argsort(distancias)[:self.k] # Ordenamos por distancia y devolver índices de los primeros k vecinos.
        k_nearest_labels = [self.clase.iloc[i] for i in k_indices] # Extraemos las clases o etiquetas de las k muestras de entrenamiento del vecino más cercano
        count_yes = 0
        count_no = 0
        for i in range(self.k):
            if(k_nearest_labels[i] == 1 ): 
                count_yes += 1
        for i in range(self.k):
            if(k_nearest_labels[i] == 0 ): 
                count_no += 1
        if (count_yes > count_no):
                return 1
        return 0
    
    def _predecir_ponderado(self, test_point):
        distancias = [self.distancia_euclidea(test_point, train_point) for train_point in self.atributos.values]
        k_indices = np.argsort(distancias)[:self.k]
        k_nearest_labels = [self.clase.iloc[i] for i in k_indices]
        # Calcular los pesos ponderados para las etiquetas (cuadrado del inverso de la distancia)
        weighted_labels = [1 / (distancias[i] ** 2) for i in k_indices]
        count_yes = 0
        count_no = 0
        for i in range(self.k):
            if(k_nearest_labels[i] == 1 ): 
                count_yes += weighted_labels[i]
        for i in range(self.k):
            if(k_nearest_labels[i] == 0 ): 
                count_no += weighted_labels[i]
        if (count_yes > count_no):
            return 1
        return 0
    def evaluar_predicciones(self,predictions, test_labels):
        # Calcula la precisión
        #accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions,pos_label=0)
        return precision
    
    def cross_validation(self,predictions,test_labels):
        # Compara para cada prediccion si coincide con el valor de etiqueta en el dataset test_labels
        accuracy = np.sum(predictions == test_labels) / len(test_labels)
        return accuracy
    
    def save_predictions(self,predictions,test_labels, filename):
        pred = pd.DataFrame({'Valor real': test_labels, 'Predicción': predictions})
        pred.to_csv( filename +'.csv')
        
        
