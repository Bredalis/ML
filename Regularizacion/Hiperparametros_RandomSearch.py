
# Librerias 

import time
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Definimos los parametros

parametros = {
	"l1_ratio": stats.uniform(0, 1),
	"alpha": stats.uniform(0.0001, 0.9999)
}

n_iteraciones = 15
semilla = 2021
tiempo_inicial = time.time()

# Ajustamos el modelo

modelo = ElasticNet()
random_search = RandomizedSearchCV(estimator = modelo, n_iter = n_iteraciones,
	param_distributions = parametros, cv = 20, 
	scoring = "neg_mean_squared_error",	random_state = semilla)

# Cargar el conjunto de datos

df = pd.read_csv("../CSV/Advertising.csv")

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

# Entrenamiento

random_search.fit(X_train, y_train)
tiempo_random = (time.time() - tiempo_inicial)

print(f"Tiempo de ejecucion: {tiempo_random}")
print(f"Mejor combinacion: {random_search.best_params_}")
print(f"Definicion del modelo: {random_search.best_estimator_}")
print(f"Combinaciones evaluadas: {n_iteraciones}")

# Mostrar los coeficientes

feature_coefs = list(zip(df.columns, random_search.best_estimator_.coef_))

for feature, coef in feature_coefs:
	print(f"{feature}: {coef}")

# Optimizacion

y_pred_random = random_search.best_estimator_.predict(X_test)
mse_random = mean_squared_error(y_test, y_pred_random)

# Crear un df con la informacion

df_evaluacion = pd.DataFrame({
	"Estrategia": ["RandomSearch"], "Tiempo": [tiempo_random], 
	"Metrica": ["MSE"], "Valor": [mse_random], "Folds": [20]
})

print("\n", df_evaluacion)