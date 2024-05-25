
# Feature Engineering
# Librerias

import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer

# Dataset

df = pd.read_csv("Advertising.csv")

print("DF: \n", df)
print(df.columns)

# Cantidad de datos
# faltantes en cada columna

print(df.isnull().sum())

# Separar df en 
# numericas y categoricas

df_numerico = df[["Sales"]]

# Aplicar distintas imputaciones
# Imputar con la media

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer.fit(df_numerico)
df_numerico["Mean"] = imputer.transform(df_numerico)

# Imputar con la mediana

imputer = SimpleImputer(missing_values = np.nan, strategy = "median")
imputer.fit(df_numerico[["Sales"]])
df["Median"] = imputer.transform(df_numerico[["Sales"]])

# Imputar con un valor constante

imputer = SimpleImputer(missing_values = np.nan, strategy = "constant", 
	fill_value = 99999)

imputer.fit(df_numerico[["Sales"]])
df_numerico["Constante"] = imputer.transform(df_numerico[["Sales"]])

# Imputar con un valor mas frecuente

imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
imputer.fit(df_numerico[["Sales"]])
df_numerico["Mas_Frecuente"] = imputer.transform(df_numerico[["Sales"]])

print(df_numerico)