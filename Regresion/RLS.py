
# Objetivo es aumentar las ventas de un producto.
# Utilizaremos el dataset "Advertising" que incluye 
# datos de ventas en 200 mercados y presupuestos de 
# publicidad en TV, radio y diario. Nuestra tarea es 
# identificar la relación entre la inversión en publicidad 
# y las ventas. Las variables predictoras son los 
# presupuestos para cada canal (TV, radio, diario).

# Librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Traer dataset

df = pd.read_csv("Advertising.csv")

# Mostar principales datos

print(df.head())

# Configurar todas las
# graficas a 20 x 5

plt.rcParams["figure.figsize"] = (20, 5)

# Grafica

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
df.plot.scatter(x = "TV", y = "Sales", ax = ax1)
df.plot.scatter(x = "Radio", y = "Sales", ax = ax2)
df.plot.scatter(x = "Newspaper", y = "Sales", ax = ax3)

plt.show()

# Modelo

modelo = LinearRegression(fit_intercept = True)

# Dividimos los datos en X y Y

X = df.loc[:, ["TV"]]
y = df["Sales"]

print(X)
print(y)

print(X.shape)
print(y.shape)

# Entrenamiento

modelo.fit(X, y)

# Coeficientes

print(modelo.coef_)

# Predecir con 4 
# unidades de inversion

print(modelo.predict([[4]]))

# Grafica

plt.scatter(X, y)
plt.plot(X, modelo.predict(X))

plt.show()