
# Librerías
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

# Cargar dataset
df = pd.read_csv("../CSV/Advertising.csv")

# Seleccionar columnas numéricas y manejar valores nulos
df = df.select_dtypes(include = ["int64", "float64"]).fillna(0)

# División de los datos
X = df.drop(["Sales"], axis = 1)
y = df[["Sales"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
	test_size = 0.2, random_state = 42)

# Definir el pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()), 
    ("enet", ElasticNet())
])

# Definir los parámetros para la búsqueda en grid
parametros = {
    "enet__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
    "enet__l1_ratio": [0, 0.25, 0.5, 0.75, 1]
}

# Crear el GridSearchCV
grid = GridSearchCV(pipeline, param_grid = parametros, 
	cv = 20, scoring = "neg_mean_squared_error")

# Ajustar el modelo
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

# Resultados
print(f"Mejor score: {grid.score(X_test, y_pred)}")
print(f"Mejores parámetros: {grid.best_params_}")