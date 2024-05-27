
# Librerias

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model

plt.style.use("ggplot")
plt.rc("font", size = 12)

# Limitaciones de la Regresion

lista_1 = [* range(10, 30)]
lista_2 = [
	0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

print(lista_1)
print(lista_2)

# Crear df

df = pd.DataFrame(list(zip(lista_1, lista_2)), columns = ["Edad", "Compras"])
print(df)

# Grafica de los datos
 
sns.scatterplot(data = df, x = "Edad", y = "Compras")
plt.show()

# Division de los datos

X = df["Edad"].values.reshape(-1, 1)
y = df["Compras"].values.reshape(-1, 1)

# Modelo

modelo_reg = linear_model.LinearRegression()
modelo_reg.fit(X, y) 

# Prediccion

df["Compras_pred"] = modelo_reg.predict(X)
print(df)

# Grafica de las predicciones

sns.regplot(
	data = df, x = "Edad", y = "Compras", ci = None)
plt.show()

# Grafica con los 
# datos no lineales

sns.regplot(
	data = df, x = "Edad", y = "Compras",
	logistic = True, ci = None)

plt.show()