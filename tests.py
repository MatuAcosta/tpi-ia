
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from scipy.stats import mode
from scipy.stats import normaltest


from collections import Counter   #se usa para sacar el most_common al final
dataset = pd.read_csv('data_cardiovascular_risk.csv')
dataset_2 = pd.read_csv('data_cardiovascular_risk.csv')
dataset['sex'] = dataset['sex'].replace({'M': 1, 'F': 0})
dataset['is_smoking'] = dataset['is_smoking'].replace({'YES': 1, 'NO':0})

print(dataset.describe())




glucose = dataset['glucose']
mean = glucose.mean()
mediana = glucose.median()
moda = glucose.mode()

print(f"Mediana:  {mediana}, Media: {mean}, Moda: {moda[0]}")


# stat, p = normaltest(glucose)
# print('Estadisticos' + str(stat) + str(p))
# #--PEARSON R


# # -- CORRELACION LINEAL
# correlation = dataset.corr()
# print(correlation)

# #coeficiente_correlacion, valor_p = pearsonr(dataset[['glucose']], dataset[['diabetes']])
# # Calcular la regresión lineal
# slope, intercept, r_value, p_value, std_err = linregress(dataset['glucose'],dataset['diabetes'])

# # Desempaquetar y mostrar los valores
# print(f"Pendiente: {slope}")
# print(f"Intersección: {intercept}")
# print(f"Coeficiente de correlación: {r_value}")
# print(f"Valor p: {p_value}")




# #PORCENTAJE FALTANTES ------

# #porc_faltantes = (dataset.isna().sum() / len(dataset)) * 100
# #print(porc_faltantes)

# # -- GRAFICO GLUCOSA DISTRIBUCION
sns.displot(dataset['education'])
plt.show()

# #--- IMPUTACION DE DATOS EN GLUCOSA --
# missing_vars = ['glucose'] #variables con data faltante
# predictor_vars = ['diabetes'] #variables que ayudan a predecir
# imputer = IterativeImputer(estimator=BayesianRidge())
# imputed_data = imputer.fit_transform(dataset_2[predictor_vars + missing_vars])
# dataset_2[missing_vars] = imputed_data[:, -len(missing_vars):]
# #print(dataset_2)
# dataset_2.to_csv('dataset_imputado.csv')
# porc_faltantes_2 = (dataset_2.isna().sum() / len(dataset)) * 100
# print(porc_faltantes_2)
# sns.displot(dataset_2['glucose'])
# plt.show()



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