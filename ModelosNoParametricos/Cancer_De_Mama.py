
# Librerías
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Cargar el dataset
data = load_breast_cancer()

# Invertir el mapeo de las clases: 0 = Benigno, 1 = Maligno
clases = [data.target_names[1], data.target_names[0]]
target = [1 if x == 0 else 0 for x in data.target]
features = list(data.feature_names)

# Crear DataFrame
df = pd.DataFrame(data.data, columns = features)

# División de características y etiquetas
X = df
y = pd.Series(target)

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, 
	test_size = 0.25, random_state = 0)

# Evaluar el modelo con diferentes valores de k
valores_k = range(1, 50, 4)
resultados_uniformes = {"train": [], "test": []}
resultados_ponderados = {"train": [], "test": []}

def evaluar_modelo(k, tipo_pesos, resultados):
    modelo = KNeighborsClassifier(n_neighbors = k, weights = tipo_pesos)
    modelo.fit(X_train, y_train)

    resultados["train"].append(accuracy_score(y_train, modelo.predict(X_train)))
    resultados["test"].append(accuracy_score(y_test, modelo.predict(X_test)))

for k in valores_k:
    evaluar_modelo(k, "uniform", resultados_uniformes)
    evaluar_modelo(k, "distance", resultados_ponderados)

# Gráfica de los resultados
plt.figure(figsize = (14, 5))
plt.subplot(1, 2, 1)
plt.plot(valores_k, resultados_uniformes["train"], label = "Train")
plt.plot(valores_k, resultados_uniformes["test"], label = "Test")
plt.title("Pesos Uniformes")
plt.xlabel("k")
plt.ylabel("Precisión")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(valores_k, resultados_ponderados["train"], label = "Train")
plt.plot(valores_k, resultados_ponderados["test"], label = "Test")
plt.title("Pesos Ponderados")
plt.xlabel("k")
plt.legend()

plt.tight_layout()
plt.show()

# GridSearchCV para encontrar el mejor modelo
param_grid = {
    "n_neighbors": [1, 2, 3, 5, 8, 10, 15, 20, 30, 50],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "chebyshev", "manhattan"]
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 5)
grid.fit(X_train, y_train)

# Resultados del mejor modelo
print(f"Mejores parámetros: {grid.best_params_}")
print(f"Reporte de clasificación:\n{classification_report(y_test, 
	grid.best_estimator_.predict(X_test), target_names = clases)}")