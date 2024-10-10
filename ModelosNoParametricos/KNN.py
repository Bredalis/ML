
# Librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

# Obtener datos del dataset

X, y = make_classification(n_samples = 200, n_clusters_per_class = 2, 
	n_classes = 2, n_features = 2, n_informative = 2, 
	n_redundant = 0, random_state = 1, class_sep = 1.1)

print(X)
print(y)

# Graficar los datos

df = pd.DataFrame(dict(x = X[:, 0], y = X[:, 1], label = y))
print(df)

colores = {0: "red", 1: "blue"}
fig, ax = plt.subplots()

agrupacion_etiquetas = df.groupby("label")

for key, grupo in agrupacion_etiquetas:
	grupo.plot(ax = ax, kind = "scatter", x = "x", y = "y", 
		label = key, color = colores[key])

plt.show()

# Creacion y entrenamiento del modelo

def grafica(cantidad, tipo):

	modelo = KNeighborsClassifier(n_neighbors = cantidad, weights = tipo)
	modelo.fit(X, y)

	cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
	cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

	# Una grilla y setear un step

	h = .02 
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])

	# Usamos un pcolormesh

	Z = Z.reshape(xx.shape)
	plt.figure()
	plt.pcolormesh(xx, yy, Z, cmap = cmap_light)

	# Ploteamos también los puntos de entrenamiento

	plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cmap_bold)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.title("2-Class classification (k = %i, weights = '%s')"
		% (cantidad, tipo))

	plt.show()

# Grafica con k = 10

grafica(10, "uniform")

# k = 200

grafica(200, "uniform")

# k = 200 distance

grafica(200, "distance")