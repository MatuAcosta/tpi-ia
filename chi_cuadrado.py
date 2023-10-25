import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from scipy.stats import mode
from scipy.stats import chi2_contingency

categoric_variables_faltantes = ['education', 'BPMeds']
dataset = pd.read_csv('data_cardiovascular_risk.csv')

for category in categoric_variables_faltantes:
    moda = dataset[category].mode()
    print(type(moda[0].astype(int)))
    dataset[category].fillna(moda[0], inplace=True)

porc_faltantes = (dataset.isna().sum() / len(dataset)) * 100
print(porc_faltantes)

dataset.to_csv('dataset_categoricos.csv')
