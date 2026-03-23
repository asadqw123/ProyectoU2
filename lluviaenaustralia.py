import pandas as pd


# Cargar los datos 
df = pd.read_csv('weatherAUS.csv')

# Ejemplo de filtrado por ciudad 
ciudades = ['Sydney', 'Perth', 'Canberra']
df_proyecto = df[df['Location'].isin(ciudades)]

# Limpieza de nulos para las variables de interés [cite: 8, 23]
df_proyecto = df_proyecto.dropna(subset=['Cloud3pm', 'RainTomorrow'])

print(f"Registros listos para procesar: {len(df_proyecto)}")