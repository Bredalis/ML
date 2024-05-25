
# Librerias

import time
import pandas as pd
import numpy as np
import itertools as it
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Definimos la grilla

parametros = {
  "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
  "l1_ratio": [0, 0.25, 0.5, 0.75, 1]
}

# Modelo

modelo = ElasticNet()

grid_search = GridSearchCV(estimator = modelo, param_grid = parametros, 
                          cv = 20, scoring = "neg_mean_squared_error")

# Cargar el conjunto de datos

df = pd.read_csv("Advertising.csv")
print(df.columns)

df = df.select_dtypes(exclude = "object")
df = df.fillna(value = 0)

# Separar las características (X_train) 
# y la variable objetivo (y_train)

X = df.drop("Sales", axis = 1)
y = df["Sales"]

# Division de datos

X_train, X_test, y_train, y_test = train_test_split(X, y, 
  random_state = 42, test_size = 0.2)

print(X_train.shape)
print(y_train.shape)

# Ajustamiento del modelo

tiempo_inicial = time.time()
tiempo_grid = time.time() - tiempo_inicial
grid_search.fit(X_train, y_train)

print(f"Tiempo de ejecucion: {tiempo_grid}")
print(f"Mejor combinacion de parametros: {grid_search.best_params_}")
print(f"Definicion del modelo: {grid_search.best_estimator_}")

# Entrenamiento

grid_search.fit(X_train, y_train)

# Optimizacion

y_pred_grid = grid_search.best_estimator_.predict(X_test)
mse_grid = mean_squared_error(y_test, y_pred_grid)

# Crear un df con la informacion

df_evaluacion = pd.DataFrame({
  "Estrategia": ["GridSearch"], "Tiempo": [tiempo_grid], 
  "Metrica": ["MSE"], "Valor": [mse_grid], "Folds": [20] 
})

print("\n", df_evaluacion)

y_pred = grid_search.predict(X_test)
mse = mean_squared_error(y_pred, y_test)

print(mse)

tiempo_final = tiempo_inicial - time.time()

print(f"Tiempo de ejecucion: {tiempo_final}")
print(f"Mejor combinacion de parametros: {grid_search.best_params_}")
print(f"Definicion del modelo: {grid_search.best_estimator_}")

feature_coef = list(zip(df.columns, grid_search.best_estimator_.coef_))

for feature, coef in feature_coef:
  print(f"{feature}: {coef}")

df_evaluacion = pd.DataFrame({
  "Estrategia": ["GridSearch"], "Timepo": [tiempo_final],
  "Metrica": ["MSE"], "Valor": [mse], "Folds": [20]
})

print(df_evaluacion)