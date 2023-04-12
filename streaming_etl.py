
# Importar librearias esenciales

# %%
import pandas as pd

# Cargar todos los csv (Amazon Prime, Netflix, Disney Plus, Hulu)

# %%
amzdf = pd.read_csv('./datasets/amazon_prime_titles.csv')
netdf = pd.read_csv('./datasets/netflix_titles.csv')
disdf = pd.read_csv('./datasets/disney_plus_titles.csv')
hudf = pd.read_csv('./datasets/hulu_titles.csv')

# Cambiar los indices de cada CSV con la inicial de su plataforma

# %%
amzdf['show_id'] = amzdf['show_id'].apply(lambda x: 'a' + x)
amzdf.set_index('show_id', inplace=True)
netdf['show_id'] = netdf['show_id'].apply(lambda x: 'n' + x)
netdf.set_index('show_id', inplace=True)
disdf['show_id'] = disdf['show_id'].apply(lambda x: 'd' + x)
disdf.set_index('show_id', inplace=True)
hudf['show_id'] = hudf['show_id'].apply(lambda x: 'h' + x)
hudf.set_index('show_id', inplace=True)

# Juntamos todos los csv dentro de un solo dataframe

# %%
streaming = pd.concat([amzdf, netdf, disdf, hudf])

# Ingestamos 'G' dentro de todos los valores nulos de la columna 'Rating'

# %%
streaming['rating'].fillna('G', inplace=True)


# Cambiamos el formato de fecha de la columna 'date_added' de 'Mes Dia, Anio' a 'AAAA-DD-MM'

# %%
streaming['date_added'] = pd.to_datetime(amzdf.date_added)

# Separamos en dos columnas ('duration_int' y 'duration_type') la columna 'duration'

# %%
streaming[['duration_int', 'duration_type']
          ] = streaming['duration'].str.split(' ', 1, expand=True)
streaming.drop('duration', axis=1, inplace=True)

# Llenamos vacios con 1 y convertimos la columna a type int

# %%
streaming['duration_int'].fillna(1, inplace=True)
streaming['duration_int'].astype(int)

# Hacemos el dataset completo en minusculas

# %%
streaming = streaming.applymap(lambda s: s.lower() if type(s) == str else s)
streaming.head()
