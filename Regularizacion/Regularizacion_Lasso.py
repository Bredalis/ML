
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

df = pd.read_csv("../CSV/Advertising.csv") 

# Eliminar columnas de tipo "object"

df = df.select_dtypes(exclude = "object")
df = df.fillna(value = 0)

# Escalar las características

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Dividir los datos en entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(df_scaled, df["Sales"], 
	test_size = 0.2, random_state = 42)

# Modelo

lasso = Lasso(alpha = 0.1)

# Entrenamiento

lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

# Metrica

print(mean_squared_error(y_test, y_pred))

# Obtener coeficientes

coefs = lasso.coef_

# Nombres de las caracteristicas y coeficientes

feature_names = df.columns
feature_coefs = list(zip(feature_names, coefs))

# Ordenar las caracteristicas segun el valor absoluto 
# del coeficiente en orden descendente

feature_coefs.sort(key = lambda x: abs(x[1]), reverse = True)

# Mostrar las caracteristicas mas 
# importantes y sus coeficientes

for feature, coef in feature_coefs:
	print(f"{feature}: {coef}")