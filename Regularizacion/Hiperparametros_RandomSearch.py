
# Librerías
import time
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Definir los parámetros para RandomizedSearchCV
parametros = {
	"l1_ratio": stats.uniform(0, 1),
    "alpha": stats.uniform(0.0001, 0.9999)
}

n_iteraciones = 15
semilla = 2021
tiempo_inicial = time.time()

# Ajustar el modelo
modelo = ElasticNet()
random_search = RandomizedSearchCV(estimator = modelo, n_iter = n_iteraciones, 
	param_distributions = parametros, cv = 20, scoring = "neg_mean_squared_error", 
	random_state = semilla)

# Cargar y preparar el conjunto de datos
df = pd.read_csv("../CSV/Advertising.csv")
df = df.select_dtypes(exclude = "object").fillna(0)

# Separar las características y la variable objetivo
X = df.drop("Sales", axis = 1)
y = df["Sales"]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, 
	test_size = 0.2, random_state = 42)

# Entrenamiento del modelo
random_search.fit(X_train, y_train)
tiempo_random = time.time() - tiempo_inicial

# Resultados
print(f"Tiempo de ejecución: {tiempo_random}")
print(f"Mejor combinación de parámetros: {random_search.best_params_}")
print(f"Mejor modelo: {random_search.best_estimator_}")
print(f"Combinaciones evaluadas: {n_iteraciones}")

# Coeficientes del modelo
feature_coefs = list(zip(df.columns, random_search.best_estimator_.coef_))
for feature, coef in feature_coefs:
    print(f"{feature}: {coef}")

# Predicciones y evaluación
y_pred_random = random_search.best_estimator_.predict(X_test)
mse_random = mean_squared_error(y_test, y_pred_random)

# Crear un DataFrame con los resultados
df_evaluacion = pd.DataFrame({
    "Estrategia": ["RandomSearch"], "Tiempo": [tiempo_random], 
    "Métrica": ["MSE"], "Valor": [mse_random], "Folds": [20]
})

print("\n", df_evaluacion)