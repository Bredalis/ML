
# Librerías
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar dataset
df = pd.read_csv("../CSV/Advertising.csv")

# Mostrar los primeros datos
print(df.head())

# Configurar tamaño de las gráficas
plt.rcParams["figure.figsize"] = (20, 5)

# Graficar las relaciones
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
df.plot.scatter(x = "TV", y = "Sales", ax = ax1)
df.plot.scatter(x = "Radio", y = "Sales", ax = ax2)
df.plot.scatter(x = "Newspaper", y = "Sales", ax = ax3)
plt.show()

# Modelo de regresión lineal
modelo = LinearRegression(fit_intercept=True)

# Dividir los datos en X (predictoras) y Y (variable objetivo)
X = df[["TV"]]
y = df["Sales"]

# Entrenar el modelo
modelo.fit(X, y)

# Coeficientes del modelo
print("Coeficientes:", modelo.coef_)

# Predicción para 4 unidades de inversión en TV
print("Predicción con 4 unidades de inversión:", modelo.predict([[4]]))

# Graficar los resultados
plt.scatter(X, y)
plt.plot(X, modelo.predict(X), color="red")
plt.show()