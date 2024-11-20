
# Librerías
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Cargar y preparar los datos
df = pd.read_csv("../CSV/Advertising.csv")
df = df.select_dtypes(exclude = "object").fillna(0)

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns = ["Sales"]))
y = df["Sales"]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
    test_size = 0.2, random_state = 42)

# Modelo Lasso
lasso = Lasso(alpha = 0.1)

# Entrenamiento del modelo
lasso.fit(X_train, y_train)

# Predicción y métrica
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Coeficientes de las características
feature_names = df.drop(columns =  ["Sales"]).columns
feature_coefs = list(zip(feature_names, lasso.coef_))

# Ordenar características por la magnitud del coeficiente
feature_coefs.sort(key =  lambda x: abs(x[1]), reverse = True)

# Mostrar las características más importantes
print("\nCaracterísticas más importantes:")
for feature, coef in feature_coefs:
    print(f"{feature}: {coef}")