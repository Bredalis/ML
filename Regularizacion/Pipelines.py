
# Librerias

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

# Dataset

df = pd.read_csv("Advertising.csv")

# Incluir columnas: int64, float64

df = df.select_dtypes(include = ["int64", "float64"])
df = df.fillna(0)

print(df.head())
print(f"DF Columnas: \n {df.columns}")
print(f"DF cantidad de Columnas: \n {len(df.columns)}")

# Division de datos

X = df.drop(["Sales"], axis = 1)
y = df[["Sales"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
	random_state = 42, test_size = 0.2)

print(X_train.shape)
print(y_train.shape)

# Definimos la tupla de pasos
# para ingresarselo al pipeline

pasos = [("scaler", StandardScaler()), ("enet", ElasticNet())]

# Creamos el pipeline

pipeline = Pipeline(pasos)

# Vamos a ealphaejecutar una busqueda de grilla 
# definimos los parametros a evaluar debemos poner el 
# nombre del estimador seguido de __ y luego el parametro

parametros = {
	"enet__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
	"enet__l1_ratio": [0, 0.25, 0.5, 0.75, 1]
}

# Creamos la busqueda de grilla
# con el pipeline

grid = GridSearchCV(pipeline, param_grid = parametros, cv = 20,
	scoring = "neg_mean_squared_error")

# Ajustamos nuestro modelo

grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

print(f"Mejor score: {grid.score(X_test, y_pred)}")
print(f"Mejores parametros: {grid.best_params_}")