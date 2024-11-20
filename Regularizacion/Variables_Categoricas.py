
# Librerías
import pandas as pd
import seaborn as sns

# Cargar el dataset
df = sns.load_dataset("tips")

# Separar columnas numéricas y categóricas
df_numerico = df.select_dtypes(include = ["number"])
df_categoricas = df.select_dtypes(exclude = ["number"])

# Convertir variables categóricas a numéricas usando One-Hot Encoding
df_categoricas_numerico = pd.get_dummies(df_categoricas, 
	drop_first = True, dtype = int)

# Combinar las columnas numéricas y categóricas transformadas
df_final = pd.concat([df_numerico, df_categoricas_numerico], axis = 1)

# Mostrar el DataFrame final
print(df_final)