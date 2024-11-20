
# Librerías
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Definir la grilla de parámetros
parametros = {
  "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
  "l1_ratio": [0, 0.25, 0.5, 0.75, 1]
}

# Modelo
modelo = ElasticNet()

# Configuración de GridSearchCV
grid_search = GridSearchCV(estimator = modelo, param_grid = parametros, 
  cv = 20, scoring = "neg_mean_squared_error")

# Cargar y preparar el conjunto de datos
df = pd.read_csv("../CSV/Advertising.csv")
df = df.select_dtypes(exclude = "object").fillna(value = 0)

# Separar características (X) y variable objetivo (y)
X = df.drop("Sales", axis = 1)
y = df["Sales"]

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, 
  test_size = 0.2, random_state = 42)

# Ajuste del modelo
tiempo_inicial = time.time()
grid_search.fit(X_train, y_train)
tiempo_grid = time.time() - tiempo_inicial

# Resultados de la búsqueda
print(f"Tiempo de ejecución: {tiempo_grid}")
print(f"Mejor combinación de parámetros: {grid_search.best_params_}")
print(f"Mejor modelo: {grid_search.best_estimator_}")

# Predicción y evaluación
y_pred_grid = grid_search.best_estimator_.predict(X_test)
mse_grid = mean_squared_error(y_test, y_pred_grid)

# Crear DataFrame con la evaluación
df_evaluacion = pd.DataFrame({
    "Estrategia": ["GridSearch"], "Tiempo": [tiempo_grid], 
    "Metrica": ["MSE"], "Valor": [mse_grid], "Folds": [20]
})
print(df_evaluacion)

# Coeficientes de las características
feature_coef = list(zip(df.columns, grid_search.best_estimator_.coef_))
for feature, coef in feature_coef:
    print(f"{feature}: {coef}")