
# Librerías
import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer

# Cargar el dataset
df = pd.read_csv("../CSV/Advertising.csv")
print("DF:\n", df)

# Cantidad de datos faltantes
print("Datos faltantes:\n", df.isnull().sum())

# Separar columnas numéricas
df_numerico = df[["Sales"]]

# Función para aplicar distintas imputaciones
def imputaciones(estrategia, datos, columna, valor=None):
    imputer = SimpleImputer(missing_values = np.nan, strategy = estrategia, fill_value = valor)
    imputer.fit(datos)
    df_numerico[columna] = imputer.transform(datos)

# Aplicar distintas estrategias de imputación
imputaciones("mean", df_numerico, "Mean")
imputaciones("median", df_numerico[["Sales"]], "Median")
imputaciones("constant", df_numerico[["Sales"]], "Constante", 99999)
imputaciones("most_frequent", df_numerico[["Sales"]], "Mas_Frecuente")

# Mostrar el DataFrame con imputaciones
print(df_numerico)