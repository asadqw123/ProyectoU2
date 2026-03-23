import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pgmpy.models import DiscreteBayesianNetwork as DBN
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# carga de dataset rain in australia
# variables: cloud3pm, humidity3pm, pressure3pm, raintoday y raintomorrow
df = pd.read_csv('weatherAUS.csv')
cols = ['Cloud3pm', 'Humidity3pm', 'Pressure3pm', 'RainToday', 'RainTomorrow', 'Location']
# limpieza de datos: eliminacion de valores nulos para asegurar calculos correctos
df = df[cols].dropna()

# fase 1: teorema de bayes
# validacion de la teoria mediante frecuencias observadas en el dataset
print("fase 1: calculos manuales de bayes")

def calcular(ciudad):
    # seleccion de estacion meteorologica especifica
    data = df[df['Location'] == ciudad]
    total = len(data)
    
    # subconjuntos para conteo de registros
    lluvia_si = data[data['RainTomorrow'] == 'Yes']
    nubes_si = data[data['Cloud3pm'] > 5]
    nubes_y_lluvia = data[(data['Cloud3pm'] > 5) & (data['RainTomorrow'] == 'Yes')]
    
    # a. prior p(lluviatomorrow): probabilidad base sin conocer variables
    prior = len(lluvia_si) / total
    
    # b. likelihood p(nubes|lluviatomorrow): nublado dado que mañana llovio
    likelihood = len(nubes_y_lluvia) / len(lluvia_si)
    
    # c. evidencia p(nubes): probabilidad total de estar nublado hoy
    evidencia = len(nubes_si) / total
    
    # d. teorema de bayes para encontrar la probabilidad posterior
    # p(lluvia|nubes) = (p(nubes|lluvia) * p(lluvia)) / p(nubes)
    posterior = (likelihood * prior) / evidencia
    
    print(f"ciudad: {ciudad}")
    print(f"prior: {prior:.4f}, likelihood: {likelihood:.4f}, evidencia: {evidencia:.4f}, posterior: {posterior:.4f}")

# ejecucion para tres estaciones distintas (sydney, perth, melbourne)
for c in ['Sydney', 'Perth', 'Melbourne']:
    calcular(c)

# fase 2: implementacion de clasificador naive bayes
# automatizacion de la prediccion con multiples variables de entrada
print("\nfase 2: modelo naive bayes")

# preprocesamiento: codificacion de variables categoricas (no/yes a 0/1)
df_nb = df.drop(columns=['Location']).copy()
df_nb['RainToday'] = df_nb['RainToday'].map({'No': 0, 'Yes': 1})
df_nb['RainTomorrow'] = df_nb['RainTomorrow'].map({'No': 0, 'Yes': 1})

# preparacion de datos para entrenamiento
x = df_nb.drop('RainTomorrow', axis=1)
y = df_nb['RainTomorrow']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# entrenamiento: implementacion de modelo gaussiannb con scikit-learn
nb = GaussianNB()
nb.fit(x_train, y_train)

# evaluacion: reporte de precision, f1-score y matriz de confusion
pred = nb.predict(x_test)
print(f"precision: {accuracy_score(y_test, pred)}")
print("matriz de confusion:")
print(confusion_matrix(y_test, pred))
print("reporte de clasificacion:")
print(classification_report(y_test, pred))

# fase 3: modelado con redes bayesianas
# diseño de estructura de dependencias para superar limitaciones de naive bayes
print("\nfase 3: red bayesiana y consultas")

# preprocesamiento: discretizacion de datos (necesario para pgmpy)
df_rb = df_nb.copy()
df_rb['Cloud3pm'] = pd.cut(df_rb['Cloud3pm'], bins=[-1, 2, 5, 9], labels=[0, 1, 2])
df_rb['Humidity3pm'] = pd.qcut(df_rb['Humidity3pm'], q=3, labels=[0, 1, 2])
df_rb['Pressure3pm'] = pd.qcut(df_rb['Pressure3pm'], q=3, labels=[0, 1, 2])

# 1. diseño de la estructura: grafo dirigido con sentido fisico
# presion influye en nubes y raintomorrow
# nubes influye en humedad
# humedad influye en raintomorrow
red = DBN([
    ('Pressure3pm', 'Cloud3pm'),
    ('Cloud3pm', 'Humidity3pm'),
    ('Humidity3pm', 'RainTomorrow'),
    ('Pressure3pm', 'RainTomorrow')
])

# 2. aprendizaje de parametros: ajuste de tablas de probabilidad condicional (cpd)
# utilizando el metodo de maxima verosimilitud
red.fit(df_rb, estimator=MaximumLikelihoodEstimator)
infer = VariableElimination(red)

# 3. inferencia probabilistica: consultas al modelo
# a. probabilidad de lluvia si la presion es baja (0) pero no hay nubes (0)
print("consulta a: p(lluvia | presion baja y sin nubes)")
print(infer.query(variables=['RainTomorrow'], evidence={'Pressure3pm': 0, 'Cloud3pm': 0}))

# b. identificar variable mas probable como causa principal si sabemos que llovio
print("consulta b: p(humedad | llovio)")
print(infer.query(variables=['Humidity3pm'], evidence={'RainTomorrow': 1}))