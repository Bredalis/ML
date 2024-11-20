
# Librerías
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

class RegresionLinealMultiple:
    def __init__(self):
        self.modelo = LinearRegression()
        self.coeficientes = None
        self.intercepto = None

    def entrenar(self, X, y):
        # División de los datos en entrenamiento y prueba
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, 
            test_size = 0.2, random_state = 42)
        self.modelo.fit(X_entrenamiento, y_entrenamiento)

        # Guardar coeficientes e intercepto
        self.coeficientes = self.modelo.coef_
        self.intercepto = self.modelo.intercept_

        # Evaluación del rendimiento del modelo
        y_prediccion = self.modelo.predict(X_prueba)
        mse = mean_squared_error(y_prueba, y_prediccion)
        r2 = r2_score(y_prueba, y_prediccion)

        print(f"MSE: {mse}")
        print(f"R2: {r2}")

    def predecir(self, X_nuevos):
        return self.modelo.predict(X_nuevos)

# Crear instancia del modelo
modelo_rlm = RegresionLinealMultiple()

# Datos de ejemplo
datos = pd.DataFrame({
    "X1": [1, 2, 3, 4, 5],
    "X2": [2, 4, 5, 4, 5],
    "y": [3, 5, 7, 8, 9]
})

# Separación de variables
X = datos[["X1", "X2"]]
y = datos["y"]

# Entrenamiento del modelo
modelo_rlm.entrenar(X, y)

# Hacer predicciones con nuevos datos
X_nuevos = pd.DataFrame({
    "X1": [6, 7],
    "X2": [8, 9]
})

predicciones = modelo_rlm.predecir(X_nuevos)
print(f"Predicciones: {predicciones}")