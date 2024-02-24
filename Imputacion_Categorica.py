
# # Feature_Engineering
# # Librerias

# import pandas as pd 
# import numpy as np
# from sklearn.impute import SimpleImputer

# # Dataset

# url = 'C:/Users/Bradalis/Desktop/LenguajesDeProgramacion/Datasets/CSV/all_bikez_curated.csv'
# df = pd.read_csv(url)
# df = df.iloc[:, [num for num in range(0, 5)]]

# print('DF: \n', df)

# # Separar df en 
# # numericas y categoricas

# df_categorico = df[['Model']]

# # Aplicar distintas imputaciones
# # Imputar con un valor constante

# imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', 
# 	fill_value = 'S/D')

# imputer.fit(df_categorico[['Model']])
# df_categorico['Constante'] = imputer.transform(df_categorico[['Model']])

# # Imputar con un valor mas frecuente

# imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
# imputer.fit(df_categorico[['Model']])
# df_categorico['Mas_Frecuente'] = imputer.transform(df_categorico[['Model']])

# print(df_categorico)