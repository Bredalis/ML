
# Librerías
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Estilo de gráficos
plt.style.use("ggplot")
plt.rc("font", size = 12)

# Datos
edad = list(range(10, 30))
compras = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Crear DataFrame
df = pd.DataFrame({"Edad": edad, "Compras": compras})

# Graficar los datos
sns.scatterplot(data = df, x = "Edad", y = "Compras")
plt.show()

# Preparar los datos para el modelo
X = df["Edad"].values.reshape(-1, 1)
y = df["Compras"].values.reshape(-1, 1)

# Crear y entrenar el modelo
modelo_reg = LinearRegression()
modelo_reg.fit(X, y)

# Predicción
df["Compras_pred"] = modelo_reg.predict(X)
print(df)

# Graficar las predicciones
sns.regplot(data = df, x = "Edad", y = "Compras", ci = None)
plt.show()

# Graficar datos no lineales con regresión logística
sns.regplot(data = df, x = "Edad", y = "Compras", logistic = True, ci = None)
plt.show()