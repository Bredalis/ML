
# Librerias

import pandas as pd
import seaborn as sns

# Lectura del dataset

df = sns.load_dataset("tips")
print(df.head())

# Remover datos categoricas

df_numerico = df.drop(["sex", "smoker", "day", "time"], axis = 1)
print(df_numerico)

# Guardar datos categoricos

df_categoricas = df.filter(["sex", "smoker", "day", "time"])
print(df_categoricas)

# Convertir a columnas numericas

categorico_numerico = pd.get_dummies(df_categoricas, 
	drop_first = True, dtype = int)
print(categorico_numerico)

# DF final con datos numericos

df = pd.concat([df_numerico, categorico_numerico], axis = 1)
print(df)