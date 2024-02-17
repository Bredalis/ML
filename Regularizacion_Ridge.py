
# Librerias

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dataset

url = 'C:/Users/Bradalis/Desktop/LenguajesDeProgramacion/Datasets/CSV/all_bikez_curated.csv'

df = pd.read_csv(url)

print(f'DF: \n {df}')
print(f'DF Columnas: \n {df.columns}')
print(f'DF cantidad de Columnas: \n {len(df.columns)}')

# Eliminar columnas con object

df = df.select_dtypes(exclude = 'object')
df = df.fillna(0)

# Escalar las caracteristicas

escalador = StandardScaler()
df_scaled = escalador.fit_transform(df)

print(f'DF escalado:\n {df_scaled}')

# Division de datos

x_train, x_test, y_train, y_test = train_test_split(df_scaled, df['Rating'], 
	random_state = 42, test_size = 0.2)

print(x_train.shape)
print(y_train.shape)

# Modelo

modelo = Ridge(alpha = 0.1)

# Entrenamiento

modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

# Metricas

print(f'MSE: {mean_squared_error(y_test, y_pred)}')

# Obtener las mejores
# caracteristicas para el modelo

coefs = modelo.coef_

# Caracteristicas y coeficientes

feature_coefs = list(zip(df.columns, coefs))
print(feature_coefs)

# Ordenarlos de manera descendente 
# y buscamos valor absoluto

feature_coefs.sort(key = lambda x: abs(x[1]), reverse = True)

print('\nCaracteristicas mas importantes:\n')

for feature, coef in feature_coefs:
	print(f'{feature}: {coef}')