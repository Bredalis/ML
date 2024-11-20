
# Librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
confusion_matrix)

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

# Instanciar el modelo con balanceo de clases
modelo = DecisionTreeClassifier(
    criterion = "gini", max_depth = 2, min_samples_leaf = 2,
    min_samples_split = 2, ccp_alpha = 0, class_weight = "balanced"
)

# Entrenamiento del modelo
modelo.fit(X_train, y_train)

# Evaluación del modelo
train_accuracy = accuracy_score(y_train, modelo.predict(X_train))
test_report = classification_report(y_test, modelo.predict(X_test))
print(f"Precisión en entrenamiento: {train_accuracy:.2f}")
print(f"Reporte de clasificación:\n{test_report}")

# Matriz de confusión
cf_matrix = confusion_matrix(y_test, modelo.predict(X_test))
sns.heatmap(cf_matrix, annot = True, fmt = "d", cmap = "Blues")
plt.title("Matriz de Confusión")
plt.ylabel("Valores Reales")
plt.xlabel("Predicciones")
plt.show()

# Visualización del árbol
plt.figure(figsize = (12, 8))
plot_tree(modelo, feature_names = features, class_names = clases, 
	filled = True)
plt.title("Visualización del Árbol de Decisión")
plt.show()