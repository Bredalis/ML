
# Librerias

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Obtener el dataset

url = "https://datasets-humai.s3.amazonaws.com/datasets/titanic_clase4.csv"
df = pd.read_csv(url)

print(df.head())
print(df.info())

# Preparacion del dataset

columnas = [
	"Survived", "Pclass", "Sex", "Age", "SibSp",
	"Fare", "Embarked"
]

df = df[columnas]

# Eliminar registros con
# datos faltantes

df.dropna(inplace = True)
print(df)
print(df.info())

# Hacer One-Hot-Encoder
# a las variables categoricas

variables_categoricas = ["Sex", "Pclass", "Embarked"]

for variable in variables_categoricas:
	lista_categorica = pd.get_dummies(df[variable], 
		prefix = variable, drop_first = True, dtype = int)
	df = df.join(lista_categorica)

variables_datos = df.columns.values.tolist()
variables_mantener = [i for i in variables_datos if i not in variables_categoricas]
df = df[variables_mantener]

print(df)

# Division de los datos

X = df.iloc[:, df.columns != "Survived"]
y = df.iloc[:, df.columns == "Survived"]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
	random_state = 99) 

# Modelo

logistic_reg = LogisticRegression(max_iter = 10000)
logistic_reg.fit(X, y)

# Prediccion

y_pred = logistic_reg.predict(X_test)
print(f"Rendimiento: {logistic_reg.score(X_test, y_test)}")

print(logistic_reg.coef_)
print(logistic_reg.intercept_)

coeficientes = pd.DataFrame(logistic_reg.coef_[0], 
	X.columns, columns = ["Coefs"])

print(coeficientes)