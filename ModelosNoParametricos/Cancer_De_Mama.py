
# Librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Obtenemos el dataset

data = load_breast_cancer()
print(data.keys())

# Separar los datos

""" 
Invertir mapeo a:
- Benigno = 0 no es canceroso
- Maligno = 1 es canceroso 
"""

data_clases = [data.target_names[1], data.target_names[0]]
data_target = [1 if x == 0 else 0 for x in list(data.target)]
data_features = list(data.feature_names)

# Crear df con los datos

df = pd.DataFrame(data.data[:, :], columns = data_features)
print(df.info)

# Dividir los datos

X = df
y = pd.Series(data_target)

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
	random_state = 0)

# Evaluar el modelo con distintas 
# distribuciones de los datos

valores_k = list(range(1, 50, 4))
resultados_train_u = []
resultados_test_u = []
resultados_train_w = []
resultados_test_w = []

for k in valores_k:

	# Instanciamos el modelo uniforme
	clf_u = KNeighborsClassifier(n_neighbors = k, weights = "uniform")
	clf_u.fit(X_train, y_train)

	y_train_pred = clf_u.predict(X_train)
	y_pred = clf_u.predict(X_test)

	resultados_train_u.append(accuracy_score(y_train_pred, y_train))
	resultados_test_u.append(accuracy_score(y_pred, y_test))

	# Instanciamos el modelo distance
	clf_w = KNeighborsClassifier(n_neighbors = k, weights = "distance")
	clf_w.fit(X_train, y_train)
	y_train_pred = clf_w.predict(X_train)
	y_pred = clf_w.predict(X_test)

	resultados_train_w.append(accuracy_score(y_train_pred, y_train))
	resultados_test_w.append(accuracy_score(y_pred, y_test))

# Grafica de los datos

f, ax = plt.subplots(1, 2, figsize = (14, 5), sharey = True)
ax[0].plot(valores_k, resultados_train_u, valores_k, resultados_test_u)
ax[0].legend(["Pesos uniformes - train", "Pesos uniformes - test"])
ax[0].set(xlabel = "k", ylabel = "accuracy")

ax[1].plot(valores_k, resultados_train_w, valores_k, resultados_test_w)
ax[1].legend(["Pesos distancia - train", "Pesos distancia - test"])
ax[1].set(xlabel = "k")

plt.show()

# Buscar el mejor modelo basandose en 
# la validacion cruzada y GridSearchCV 

modelo = KNeighborsClassifier()

n_neighbors = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 50])
param_grid = {
	"n_neighbors": n_neighbors, "weights": ["uniform", "distance"],
	"metric": ["euclidean", "chebyshev", "manhattan"]
}

grid = GridSearchCV(estimator = modelo, param_grid = param_grid)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(pd.DataFrame(grid.cv_results_).sample(3))
print(classification_report(y_test, grid.best_estimator_.predict(X_test), target_names = data_clases))