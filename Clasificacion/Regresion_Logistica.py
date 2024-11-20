
# Librerías
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

# División de datos en características y etiqueta
X = df.drop("Survived", axis = 1)
y = df["Survived"]

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, 
	test_size = 0.2, random_state = 99)

# Modelo de Regresión Logística
modelo = LogisticRegression(max_iter = 10000)
modelo.fit(X_train, y_train)

# Predicción y evaluación
y_pred = modelo.predict(X_test)
rendimiento = modelo.score(X_test, y_test)
print(f"Rendimiento del modelo: {rendimiento:.2f}")

# Coeficientes del modelo
coeficientes = pd.DataFrame(modelo.coef_[0], X.columns, 
	columns = ["Coeficiente"])
print(coeficientes)
print(f"Intercepto: {modelo.intercept_[0]:.2f}")