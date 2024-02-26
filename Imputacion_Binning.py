
# Feature Engineering
# Librerias

import pandas as pd 
from sklearn.impute import SimpleImputer

# Dataset

url = 'C:/Users/Bradalis/Desktop/LenguajesDeProgramacion/Datasets/CSV/all_bikez_curated.csv'
df = pd.read_csv(url)
df = df.iloc[:, [num for num in range(0, 5)]]

print('DF: \n', df)
print(df.columns)

# Generamos 10 bins

df['Rating_bins'] = pd.qcut(df['Rating'], q = 10)
print(df[['Rating', 'Rating_bins']])