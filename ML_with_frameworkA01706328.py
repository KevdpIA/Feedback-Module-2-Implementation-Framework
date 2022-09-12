##    Matricula: A01706328
##    Nombre: Kevin Joan Delgado Perez



###      Librerías utilizadas      ###

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers



sns.set(rc={"figure.figsize": (10,6)})              # tamano de la grafica

df = pd.read_csv('SeoulBikeData.csv', encoding= 'unicode_escape')                   # leer el archivo

df_cut = df[['Rented Bike Count', 'Hour', 'Temperature(°C)','Humidity(%)','Wind speed (m/s)']].copy()   # duplicar dataframe con las variables a usar

df_dummies = pd.get_dummies(df['Seasons'])                            # generar dummies en variable independiente categorica
df_dummies = df_dummies.assign(ind = df_dummies['Winter'].index)      # obtener el index para realizar merge entre la tabla de variables y las dummies
df_cut = df_cut.assign(ind = df_cut['Hour'].index)                    # hacer lo mismo con la tabla de dummies

df_cut = pd.merge(df_cut, df_dummies,on= 'ind')                       
df_cut = df_cut.drop(columns='ind')                             # al realizar el merge de acuerdo al index, eliminar esa columna para obtener la tabla final

tr_df = df_cut.sample(frac=0.75,random_state=0)                 # obtener datos de entrenamientos (75% por cierto de los datos)
test_df = df_cut.drop(tr_df.index)                        # aquellos datos que sobraron seran los datos de prueba (relacion de 75-25)

tr_stats = tr_df.describe()                               # tabla de estadisticas descriptivas
tr_stats.pop("Rented Bike Count")                         # separar variable dependiente
tr_stats = tr_stats.transpose()                           # estadisticas descriptivas de datos de entrenamiento
print(tr_stats)

tr_labels = tr_df.pop('Rented Bike Count')
test_labels = test_df.pop('Rented Bike Count')            # separar variable a predecir para normalizar datos

def normalize(x):                                         # normalizacion para obtener mejor rendimiento en los algoritmos
  return (x - tr_stats['mean']) / tr_stats['std']         # formula de normalizacion
normed_tr = normalize(tr_df)
normed_test = normalize(test_df)                          # normalizar datos de entrenamiento y prueba

model = keras.Sequential([                                # diseno completo de la red neuronal y configuraciones
  layers.Dense(64, activation='relu', input_shape=[len(tr_df.keys())]),
  layers.Dense(64, activation='relu'),
  layers.Dense(64, activation='relu'),
  layers.Dense(8, activation='relu'),
  layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)  # optimizador de parametros para la red

model.compile(loss='mse',                                 # comparador de errores para el modelo
              optimizer=optimizer,
              metrics=['mae', 'mse'])

model.summary()                         # detalles del modelo

EPOCHS = 100                            # numero de epocas

history = model.fit(                    # iniciar proceso de entrenamiento
  normed_tr, tr_labels,
  epochs=EPOCHS, validation_split = 0.25, verbose=1,
                    )


hist = pd.DataFrame(history.history)    # abstraer datos de entrenamiento 
hist['epoch'] = history.epoch           # obtener el historial segun las epocas
print(hist.tail(10))                    # imprimir informacion de las ultimas 10 epocas

plt.figure, axis = plt.subplots(2)      # generar dos graficas

axis[0].set_title('Mean Abs Error ( Rented Bikes Count )')        # grafica del MAE
axis[0].plot(hist['epoch'], hist['mae'], label='Train Error')     # linea de los errores de entrenamiento
axis[0].plot(hist['epoch'], hist['val_mae'], label = 'Validation Error') # linea de los errores de validacion

plt.ylim([0,500])
axis[0].legend()              # graficar limites y leyendas

axis[1].set_title('Mean Square Error ( Rented Bikes Count^2 )')   # grafica del MSE
axis[1].plot(hist['epoch'], hist['mse'], label='Train Error')     # errores de entrenamiento
axis[1].plot(hist['epoch'], hist['val_mse'], label = 'Validation Error') # errores de validacion
plt.ylim([0,500000])
axis[1].legend()          # limites y leyendas

for ax in axis.flat:
  ax.set(xlabel='Epochs', ylabel='Mean Error')      # tipo de datos en eje-x (epocas) y eje-y (error promedio)

for ax in axis.flat:                                
  ax.label_outer()                                  # solo generar los labels en los lados exteriores

plt.show()