
# Librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Lectura de datos

ruta = 'C:/Users/Bradalis/Desktop/LenguajesDeProgramacion/Datasets/CSV/Cars/all_bikez_curated.csv'

df = pd.read_csv(ruta) 
print(f'DF: \n{df.head()}')

print(df.columns)

# Division de datos

df_x = df.drop(['Rating'], axis = 1)

x = df_x
y = df['Rating']

print('\n', type(x))
print(type(y))