# Se importan las librerias necesarias
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from knn import KNN
from sklearn.neighbors import KNeighborsClassifier

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

def ejecutarKNN(train_dataset, labels_train,test_dataset,labels_test):
    cross = []
    cross_ponderado = []
    for k in range(1,11):
        knn = KNN(k)
        knn.fit(train_dataset,labels_train)
        predictions_ponderado = knn.predecir_ponderado(test_dataset)
        predictions = knn.predecir(test_dataset)
        print(predictions)
        cross.append(knn.cross_validation(predictions=predictions,test_labels=labels_test))
        cross_ponderado.append(knn.cross_validation(predictions=predictions_ponderado,test_labels=labels_test))
    print('CROSS',cross)
    print('CROSS PONDERADO',cross_ponderado)

def ejecutarKNN_simple(train_dataset, labels_train,test_dataset,labels_test):
     for k in range(5,6):
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
            'prevalentHyp':[1],
            'diabetes':[1],
            'totChol': [295.0],
            'sysBP': [102.0],
            'diaBP': [68.0],
            'BMI': [28.15],
            'heartRate': [60.0],
            'glucose': [120.0]
        }
        new_df = normalize_dataset( pd.DataFrame(new_data))
        new_df = pd.DataFrame(new_data)
        print('NEW_DF',new_df)
        prediction_normal = knn.predecir(new_df)
        prediction_ponderado = knn.predecir_ponderado(new_df)
        print('PREDICCION para kponderadoS= ',k, prediction_ponderado)
        print('PREDICCION para k normal= ',k, prediction_normal)


dataset = pd.read_csv('data_cardiovascular_risk.csv').dropna()
dataset['sex'] = dataset['sex'].replace({'M': 1, 'F': 0})
dataset['is_smoking'] = dataset['is_smoking'].replace({'YES': 1, 'NO':0})
train_dataset, test_dataset, labels_train, labels_test = div_dataset(dataset, ['id','TenYearCHD'])
to_csv([train_dataset, test_dataset, labels_train, labels_test],['training_dataset.csv','test_dataset.csv','labels_train.csv','labels_test.csv'])

# train_dataset = normalize_dataset(train_dataset)
# test_dataset = normalize_dataset(test_dataset)

to_csv([train_dataset,test_dataset],['training_dataset_scaled.csv','test_dataset_scaled.csv'])
ejecutarKNN(train_dataset=train_dataset,labels_train=labels_train,test_dataset=test_dataset,labels_test=labels_test,)

#ejecutarKNN_simple(train_dataset=train_dataset,labels_train=labels_train,test_dataset=test_dataset,labels_test=labels_test,)
# precisiones = []
# for i in range(1,10):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(train_dataset, labels_train)
#     y_pred = knn.predict(test_dataset)
#     accuracy = accuracy_score(labels_test, y_pred)
#     precisiones.append(accuracy)
# print('precisiones', precisiones)

  
 #RESULTADOS SIN ESCALAR   
#precisiones [0.7764505119453925, 0.8361774744027304, 0.8225255972696246, 0.8344709897610921, 0.8225255972696246, 0.8344709897610921, 0.8310580204778157, 0.8430034129692833, 0.841296928327645]
#NORMAL [0.7764505119453925, 0.8361774744027304, 0.8225255972696246, 0.8344709897610921, 0.8225255972696246, 0.8344709897610921, 0.8310580204778157]
#PONDERADO [0.7764505119453925, 0.7764505119453925, 0.8156996587030717, 0.8174061433447098, 0.8225255972696246, 0.8276450511945392, 0.825938566552901]


#results with scaled data
#precisiones [0.7286689419795221, 0.8071672354948806, 0.7832764505119454, 0.8174061433447098, 0.7918088737201365, 0.8310580204778157, 0.8191126279863481, 0.8310580204778157, 0.8242320819112628]
#NORMAL [0.7286689419795221, 0.8071672354948806, 0.7832764505119454, 0.8174061433447098, 0.7918088737201365, 0.8310580204778157, 0.8191126279863481, 0.8310580204778157, 0.8242320819112628, 0.8430034129692833]
#PONDERADO [0.7286689419795221, 0.7286689419795221, 0.7781569965870307, 0.7798634812286689, 0.78839590443686, 0.7969283276450512, 0.8071672354948806, 0.8191126279863481, 0.8174061433447098, 0.8208191126279863]
