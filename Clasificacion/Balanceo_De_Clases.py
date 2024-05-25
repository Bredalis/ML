
# Librerias

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, 
	recall_score, f1_score)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
	random_state = 99) 

train = X_train
train["Survived"] = y_train.Survived.to_list()

positivos = train[train.Survived == 1]
print(len(positivos))

negativos = train[train.Survived == 0]
print(len(negativos))

# Crear un subset

negativos_sub = negativos.sample(n = len(positivos), random_state = 99)
print(len(negativos_sub))

# Balancear los sets

train_sub = pd.concat([positivos, negativos_sub])
print(len(train_sub))

# Nueva dividion de datos

X_train = train_sub.loc[:, train_sub.columns != "Survived"]
y_train = train_sub.loc[:, train_sub.columns == "Survived"]

# Modelo

modelo_logistico = LogisticRegression(max_iter = 10000)
modelo_logistico.fit(X_train, y_train)

# Prediccion

y_pred = modelo_logistico.predict(X_test)

# Metricas

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1: {f1_score(y_test, y_pred)}")