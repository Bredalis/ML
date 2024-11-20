
# Librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(42)
X = np.random.rand(100, 1) * 10  
y = 3 * X + np.random.randn(100, 1) * 2 

# Crear un DataFrame
df = pd.DataFrame({"X": X.flatten(), "y": y.flatten()})

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df[["X"]], df["y"], 
    test_size = 0.2, random_state = 42)

# Entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Graficar los resultados
plt.scatter(X_test, y_test, color = "black", label = "Datos reales")
plt.plot(X_test, y_pred, color = "blue", linewidth = 3, label = "Predicciones")
plt.xlabel("Variable Independiente")
plt.ylabel("Variable Dependiente")
plt.title("Predicciones del Modelo de Regresión Lineal")
plt.legend()
plt.show()

# Mostrar métricas de evaluación
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")