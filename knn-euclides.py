# Se importan las librerias necesarias
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter   #se usa para sacar el most_common al final
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


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
    def distancia_ponderada(self, punto1, punto2):
        distancia = np.sqrt(np.sum((punto1 - punto2) ** 2))
        if np.isnan(distancia):    # Si es nulo, ignorar con un valor alto (por ahora)
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
        #print('Mostramos las primeras 10 distancias:')
        #print(distancias[0:9])
        # Ordenamos por distancia y devolver índices de los primeros k vecinos.
        k_indices = np.argsort(distancias)[:self.k]
        # Extraemos las clases o etiquetas de las k muestras de entrenamiento del vecino más cercano
        k_nearest_labels = [self.clase.iloc[i] for i in k_indices]
        #print((k_nearest_labels))
        #Devuelve la etiqueta de clase más común
        most_common = Counter(k_nearest_labels).most_common(1)
        #print('La etiqueta mas común es', most_common[0][0])
        # return most_common[0][0]
        count_yes = 0
        count_no = 0
        for i in range(self.k):
            value_label = k_nearest_labels[i] == most_common[0][0]
            #print('condicion always true?', value_label)
            if(value_label and k_nearest_labels[i] == 1 ): 
                count_yes += 1
            elif (value_label and k_nearest_labels[i] == 0):
                count_no += 1
            #print('COUNTERS',[count_yes, count_no])
            if (count_yes > count_no):
                #print('entre a 1')
                return 1
            return 0
    
    def evaluar_predicciones(self,predictions, test_labels):
        # Calcula la precisión
        accuracy = accuracy_score(test_labels, predictions)
        return accuracy

    def _predecir_ponderado(self, test_point):
        distancias = [self.distancia_ponderada(test_point, train_point) for train_point in self.atributos.values]
        k_indices = np.argsort(distancias)[:self.k]
        #print('DISTANCIAS', k_indices)
        k_nearest_labels = [self.clase.iloc[i] for i in k_indices]
       #print('MAS CERCANO',self.atributos.iloc[k_indices[0]])
        # Calcular los pesos ponderados para las etiquetas (cuadrado del inverso de la distancia)
        weighted_labels = [1 / (distancias[i] ** 2) for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        #print('WEIGHTED', weighted_labels)
        count_yes = 0
        count_no = 0
        for i in range(self.k):
            value_label = k_nearest_labels[i] == most_common[0][0]
            if(value_label and k_nearest_labels[i] == 1 ): 
                count_yes += weighted_labels[i]
            elif (value_label and k_nearest_labels[i] == 0):
                count_no += weighted_labels[i]
        print('COUNTERS',[count_yes, count_no])
        if (count_yes > count_no):
            return 1
        return 0

def to_csv(datasets, names):
    i = 0
    for dataset in datasets:
        dataset.to_csv(names[i])
        i += 1

def div_dataset(dataset, drop_columns):
    copy_dataset = dataset.copy().drop(columns=drop_columns)
    labels = dataset['TenYearCHD']
    train_dataset, test_dataset, labels_train, labels_test = train_test_split(copy_dataset,labels, test_size = 0.2, random_state=42) # en el 3er parametro se ve que se puede jugar con el tamaño de los conjuntos de entrenamiento.
    return train_dataset,test_dataset,labels_train,labels_test

def normalize_dataset(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)
    return pd.DataFrame(dataset_scaled)


dataset = pd.read_csv('data_cardiovascular_risk.csv').dropna()
dataset['sex'] = dataset['sex'].replace({'M': 1, 'F': 0})
dataset['is_smoking'] = dataset['is_smoking'].replace({'YES': 1, 'NO':0})
train_dataset, test_dataset, labels_train, labels_test = div_dataset(dataset, ['id','TenYearCHD'])
to_csv([train_dataset, test_dataset, labels_train, labels_test],['training_dataset.csv','test_dataset.csv','labels_train.csv','labels_test.csv'])

train_dataset = normalize_dataset(train_dataset)
test_dataset = normalize_dataset(test_dataset)

to_csv([train_dataset,test_dataset],['training_dataset_scaled.csv','test_dataset_scaled.csv'])


def ejecutarKNN(train_dataset, labels_train,test_dataset,labels_test):
    accuracy_ponderado=[]
    accuracy = []
    for k in range(1,8):
        knn = KNN(k)
        knn.fit(train_dataset,labels_train)
        predictions_ponderado = knn.predecir_ponderado(test_dataset)
        predictions = knn.predecir(test_dataset)
        print(predictions)
        accuracy.append(knn.evaluar_predicciones(predictions,labels_test))
        accuracy_ponderado.append(knn.evaluar_predicciones(predictions_ponderado, labels_test))
        #print(f"Precisión del modelo: {accuracy:.2f}")
    print('NORMAL',accuracy)
    print('PONDERADO',accuracy_ponderado)

def ejecutarKNN_simple(train_dataset, labels_train,test_dataset,labels_test):
     for k in range(5,10):
        knn = KNN(k)
        knn.fit(train_dataset,labels_train)
        new_data = {
            'age':[70],
            'education':[1],
            'sex':[0],
            'is_smoking':[1],
            'cigsPerDay':[5],
            'BPMeds':[1],
            'prevalentStroke':[1],
            'prevalentHyp':[0],
            'diabetes':[1],
            'totChol': [280.0],
            'sysBP': [102.0],
            'diaBP': [68.0],
            'BMI': [28.15],
            'heartRate': [60.0],
            'glucose': [80.0]
        }
        new_df = pd.DataFrame(new_data)
        prediction_normal = knn.predecir(new_df)
        prediction_ponderado = knn.predecir_ponderado(new_df)
        print('PREDICCION para kponderadoS= ',k, prediction_ponderado)
        print('PREDICCION para k normal= ',k, prediction_normal)

ejecutarKNN(train_dataset=train_dataset,labels_train=labels_train,test_dataset=test_dataset,labels_test=labels_test,)
#ejecutarKNN_simple(train_dataset=train_dataset,labels_train=labels_train,test_dataset=test_dataset,labels_test=labels_test,)



    # def distancia_gower(self, punto1, punto2):
    #   n = len(punto1)
    #   cont_indices = [i for i in range(n) if isinstance(punto1[i], (int, float))]
    #   cat_indices = [i for i in range(n) if i not in cont_indices]
    #   cont_x = np.array([punto1[i] for i in cont_indices])
    #   cont_y = np.array([punto2[i] for i in cont_indices])
    #   cat_x = np.array([punto1[i] for i in cat_indices])
    #   cat_y = np.array([punto2[i] for i in cat_indices])
    #   cont_dist = cdist(cont_x.reshape(1, -1), cont_y.reshape(1, -1), metric='euclidean')[0][0]
    #   cat_dist = np.sum(cat_x != cat_y) / len(cat_indices)
    #   return (cont_dist + cat_dist) / (len(cont_indices) + len(cat_indices))