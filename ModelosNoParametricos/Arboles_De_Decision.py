
# Librerias

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
	confusion_matrix)

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

# Instanciamos el modelo y 
# entremoslo con el dataset de autos

arbol = DecisionTreeClassifier(
	criterion = "gini", max_depth = 2, min_samples_leaf = 2,
	min_samples_split = 2, ccp_alpha = 0
)

arbol.fit(X_train, y_train)
print(accuracy_score(y_train, arbol.predict(X_train)))
print(classification_report(y_true = y_test, y_pred = arbol.predict(X_test)))

# Grafica de los errores del 
# modelo con la matriz de confusion

cf_matrix = confusion_matrix(y_test, arbol.predict(X_test))
sns.heatmap(cf_matrix, annot = True)
plt.show()

# Features importantes

print(arbol.feature_importances_)

# Calcular las 5 feature importantes mas altas

importantes_features = pd.Series(arbol.feature_importances_).sort_values(
	ascending = False)[:5]
print(importantes_features)

# Grafica de las features

f5_nombres = list(pd.Series(data.feature_names)[importantes_features.index.to_list()])
fig, ax = plt.subplots()
importantes_features.plot.barh(ax = ax)
ax.set_yticklabels(f5_nombres)
ax.invert_yaxis()
plt.show()