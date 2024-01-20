
# Librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Mostrar los numeros
# float con 2 decimales

pd.options.display.float_format = '{:.2f}'.format

# Lectura de datos

url = 'https://datasets-humai.s3.amazonaws.com/datasets/properati_caba_2021.csv'
df = pd.read_csv(url)

# Definimos la semilla

SEMILLA = 1992

print(df.head())

# Correlacion entre las
# variables numericas

print(df.corr())
print(df.columns)

# Grafica

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 5))

df.plot.scatter(x = 'rooms', y = 'price', ax = ax1)
df.plot.scatter(x = 'bathrooms', y = 'price', ax = ax2)
df.plot.scatter(x = 'surface_total', y = 'price', ax = ax3)
df.plot.scatter(x = 'surface_covered', y = 'price', ax = ax4)

plt.show()

# Division de datos

x = df.drop(['price'], axis = 1)
y = df['price']

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = SEMILLA)

# Definicion de las variables predictorias

variables_exogenas = ['surface_total', 'bathrooms']

# Matriz de X

x_train_modelo_sup_baños = x_train[variables_exogenas]

print(x_train_modelo_sup_baños.head())

# Modelo

modelo = LinearRegression(fit_intercept = True)

# Entrenamiento

modelo.fit(x_train_modelo_sup_baños, y_train)

coeficientes = modelo.coef_
intercepto = modelo.intercept_

beta_1, beta_2 = coeficientes[0], coeficientes[1]

print(beta_1)
print(beta_2)

# Definimos una función para obtener los coeficientes en un dataframe
def obtener_coeficientes(modelo, lista_variables):

  '''Crea un dataframe con los coeficientes estimados de un modelo'''
  # Creo la lista de nombres de variables
  lista_variables = ['intercepto'] + lista_variables

  # Intercepto
  intercepto = modelo.intercept_

  # Lista coeficientes excepto el intercepto
  coeficientes = list(modelo.coef_)

  # Lista completa coeficientes
  lista_coeficientes = [intercepto] + coeficientes

  return pd.DataFrame({'variable': lista_variables, 'coeficiente': lista_coeficientes})

obtener_coeficientes(modelo, variables_exogenas)