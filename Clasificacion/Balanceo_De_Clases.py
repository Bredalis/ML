
# Librerías
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, 
recall_score, f1_score)

# Cargar el dataset
url = "https://datasets-humai.s3.amazonaws.com/datasets/titanic_clase4.csv"
df = pd.read_csv(url)

# Selección de columnas relevantes
columnas = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Fare", "Embarked"]
df = df[columnas]

# Eliminar registros con datos faltantes
df.dropna(inplace = True)

# One-Hot Encoding para variables categóricas
variables_categoricas = ["Sex", "Pclass", "Embarked"]
df = pd.get_dummies(df, columns = variables_categoricas, 
	drop_first = True, dtype = int)

# División de los datos
X = df.drop("Survived", axis = 1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
	test_size = 0.2, random_state = 99)

# Balanceo de datos en el conjunto de entrenamiento
train = X_train.copy()
train["Survived"] = y_train.values

positivos = train[train["Survived"] == 1]
negativos = train[train["Survived"] == 0]

negativos_sub = negativos.sample(n = len(positivos), random_state = 99)
train_sub = pd.concat([positivos, negativos_sub])

# Nueva división de datos balanceados
X_train = train_sub.drop("Survived", axis = 1)
y_train = train_sub["Survived"]

# Modelo de Regresión Logística
modelo = LogisticRegression(max_iter = 10000)
modelo.fit(X_train, y_train)

# Predicción
y_pred = modelo.predict(X_test)

# Métricas de rendimiento
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1: {f1_score(y_test, y_pred):.2f}")