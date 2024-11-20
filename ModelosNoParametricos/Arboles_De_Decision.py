
# Librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
confusion_matrix)

# Cargar el dataset
data = load_breast_cancer()
print(f"Características del dataset: {data.keys()}")

# Crear DataFrame con características y etiquetas
df = pd.DataFrame(data.data, columns = data.feature_names)
df["target"] = [1 if x == 0 else 0 for x in data.target]  # 1: Maligno, 0: Benigno

# Separar características (X) y etiquetas (y)
X = df.drop(columns = "target")
y = df["target"]

# División del dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size = 0.25, random_state = 0)

# Instanciar y entrenar modelo de Árbol de Decisión
modelo = DecisionTreeClassifier(
    criterion = "gini", max_depth = 2, min_samples_leaf = 2,
    min_samples_split = 2, ccp_alpha = 0
)
modelo.fit(X_train, y_train)

# Evaluación del modelo
train_accuracy = accuracy_score(y_train, modelo.predict(X_train))
test_accuracy = accuracy_score(y_test, modelo.predict(X_test))
print(f"Precisión en entrenamiento: {train_accuracy:.2f}")
print(f"Reporte de clasificación:\n{classification_report(y_test, modelo.predict(X_test))}")

# Matriz de confusión
cf_matrix = confusion_matrix(y_test, modelo.predict(X_test))
sns.heatmap(cf_matrix, annot = True, fmt = "d", cmap = "Blues")
plt.title("Matriz de Confusión")
plt.ylabel("Valores Reales")
plt.xlabel("Predicciones")
plt.show()

# Importancia de características
importancias = pd.Series(modelo.feature_importances_, index = data.feature_names)
importantes_top5 = importancias.sort_values(ascending = False).head(5)
print(f"Top 5 características más importantes:\n{importantes_top5}")

# Gráfico de características importantes
plt.figure(figsize = (8, 5))
importantes_top5.plot.barh(color = "skyblue")
plt.title("Top 5 Características Más Importantes")
plt.xlabel("Importancia")
plt.gca().invert_yaxis()
plt.show()