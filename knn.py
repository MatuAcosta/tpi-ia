# Se importan las librerias necesarias
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score
#se usa para sacar el most_common al final
import math
class KNN:
    def __init__(self, k=10):
        self.k = k

    def fit(self, atributos, clase):
        self.atributos = atributos
        self.clase = clase

    def distancia(self,punto1,punto2):
        sumatoria = 0
        for index,attr in enumerate(punto1):
            sumatoria += (attr - punto2[index]) ** 2
        return math.sqrt(sumatoria)
    
    def predecir(self, nuevo_registro):
        predicciones = [self.calcular_prediccion(punto) for punto in nuevo_registro.values]
        return np.array(predicciones)
    def predecir_ponderado(self, nuevo_registro):
        predicciones = [self.calcular_prediccion_ponderado(punto) for punto in nuevo_registro.values]
        return np.array(predicciones)
    # Calculamos distancias entre test_point y todos los ejemplos en el training set
    def calcular_prediccion(self, test_point):
        distancias = [self.distancia(test_point, train_point) for train_point in self.atributos.values]
        #print('distancias normal',distancias[:self.k])
        index_distancias = np.argsort(distancias)[:self.k] # Ordenamos por distancia y devolver índices de los primeros k vecinos.
        k_nearest_labels = [self.clase.iloc[i] for i in index_distancias] # Extraemos las clases o etiquetas de las k muestras de entrenamiento del vecino más cercano
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
    
    def calcular_prediccion_ponderado(self, test_point):
        distancias = [self.distancia(test_point, train_point) for train_point in self.atributos.values]
        #print('distancias ponderado',distancias[:self.k])
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
    
    ##esta función nos ayuda a calcular las predicciones por etiqueta, sirve para contorll
    def evaluar_predicciones(self,predictions, test_labels):
        # Calcula la precisión
        #accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions,pos_label=0)
        return precision
    
    def cross_validation(self,predictions,test_labels):
        # Compara para cada prediccion si coincide con el valor de etiqueta en el dataset test_labels
        accuracy = np.sum(predictions == test_labels) / len(test_labels)
        return accuracy
    
    def k_optimo(self,accuracy):
        max = 0
        index_max = 0
        for index, acc in enumerate(accuracy):
            if(acc > max):
                max = acc
                index_max = index
        return index_max
    
    def save_predictions(self,predictions,test_labels, filename):
        pred = pd.DataFrame({'Valor real': test_labels, 'Predicción': predictions})
        pred.to_csv( './csv/' + filename +'.csv')
        
        
