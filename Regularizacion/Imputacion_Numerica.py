
# Feature Engineering
# Librerias

import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer

# Dataset

df = pd.read_csv("Advertising.csv")

print("DF: \n", df)
print(df.columns)

# Cantidad de datos faltantes

print(df.isnull().sum())

# Separar df en 
# numericas y categoricas

df_numerico = df[["Sales"]]

# Aplicar distintas imputaciones

def imputaciones(estrategia, datos, columna, valor = None):
	imputer = SimpleImputer(missing_values = np.nan, strategy = estrategia, 
		fill_value = valor)

	imputer.fit(datos)
	df_numerico[columna] = imputer.transform(datos)

imputaciones("mean", df_numerico, "Mean")
imputaciones("median", df_numerico[["Sales"]], "Median")
imputaciones("constant", df_numerico[["Sales"]], "Constante", 99999)
imputaciones("most_frequent", df_numerico[["Sales"]], "Mas_Frecuente")

print(df_numerico)