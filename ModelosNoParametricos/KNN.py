
# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

# Generar dataset
X, y = make_classification(
    n_samples = 200, n_clusters_per_class = 2, n_classes = 2, n_features = 2,
    n_informative = 2, n_redundant = 0, random_state = 1, class_sep = 1.1
)

# Crear DataFrame para graficar
df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label": y})

# Graficar los datos originales
colores = {0: "red", 1: "blue"}
fig, ax = plt.subplots()
for key, grupo in df.groupby("label"):
    grupo.plot(ax = ax, kind = "scatter", x = "x", y = "y", label = key, color = colores[key])
plt.title("Datos Originales")
plt.show()

# Función para graficar la clasificación
def graficar_clasificacion(k, tipo_pesos):
    modelo = KNeighborsClassifier(n_neighbors = k, weights = tipo_pesos)
    modelo.fit(X, y)

    # Crear grilla
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Graficar región de decisión
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light, shading = "auto")

    # Graficar puntos de datos
    plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cmap_bold, edgecolor = "k")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"Clasificación con k={k}, pesos='{tipo_pesos}'")
    plt.show()

# Visualización con diferentes valores de k y tipos de pesos
graficar_clasificacion(10, "uniform")
graficar_clasificacion(200, "uniform")
graficar_clasificacion(200, "distance")