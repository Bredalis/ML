
# Feature_Engineering
# Librerias

import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer

# Dataset

url = 'C:/Users/Bradalis/Desktop/LenguajesDeProgramacion/Datasets/CSV/all_bikez_curated.csv'
df = pd.read_csv(url)
df = df.iloc[:, [num for num in range(0, 5)]]

print('DF: \n', df)
print(df.columns)

# Cantidad de datos
# faltantes en cada columna

print(df.isnull().sum())

# Separar df en 
# numericas y categoricas

df_numerico = df[['Rating']]

# Aplicar distintas imputaciones
# Imputar con la media

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(df_numerico)
df_numerico['Mean'] = imputer.transform(df_numerico)

# Imputar con la mediana

imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
imputer.fit(df_numerico[['Rating']])
df['Median'] = imputer.transform(df_numerico[['Rating']])

# Imputar con un valor constante

imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', 
	fill_value = 99999)

imputer.fit(df_numerico[['Rating']])
df_numerico['Constante'] = imputer.transform(df_numerico[['Rating']])

# Imputar con un valor mas frecuente

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imputer.fit(df_numerico[['Rating']])
df_numerico['Mas_Frecuente'] = imputer.transform(df_numerico[['Rating']])

print(df_numerico)