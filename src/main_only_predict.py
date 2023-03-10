from utils.train import train_model
from utils.functions import data_processing 
import pandas as pd

'''ESTE MAIN ESTA DEDICADO ÚNICAMENTE A LA PREDICCIÓN DE RESULTADOS'''

#Carga de datos

df_final = pd.read_csv('data/processed_files/df_datos_completos.csv')

#Creación de datos nuevos. 
#Para crear los datos nuevos hay que darle valor a una serie de variables. Se muestra un ejemplo, varían por partido, y no es necesario pasar una lista completa de lesionados
#y alineaciones pero mejorará el desempeño del modelo. Los nombres del estadio y el árbitro deben estar correctos y 100% igual escritos. Para sacarlos se puede llamar a las funciones 
# buscar_estadio y buscar_arbitro del archivo functions.py . También se pueden encontrar ids de equipos y jugadores con sus respectivas funciones. (Hay que pasar siempre id de equipo,
# y jugador)
#Para este ejemplo se usarán los siguientes datos (Lugo - Real Zaragoza)
id_equipo_local = 727  
id_equipo_visitante = 538
odd_1 = 2.6
odd_x = 3.1
odd_2 = 2.88
arbitro = 'José Sánchez' 
estadio = 'Estadio El Sadar'  
season = 2022  
ids_lesionados = [46658,47391,2464,162712,2433]  
ids_titulares = [
 181421,46746,47579,47448,46653,47574,46662,67939,64309,1825,
 1926,47445,47435,19026,47566,2032,47432,47277,67955,182504,47427
]
#Instacia de la clase data_preprocessing
data_processing = data_processing()

#Creación de datos nuevos
datos_nuevos = data_processing.creacion_datos_nuevos(df_final,id_equipo_local, id_equipo_visitante,odd_1, odd_x, odd_2, arbitro, estadio, season, ids_lesionados, ids_titulares)

#Entrenamiento del modelo. La línea de código estará comentada, se descomentará para poder reentrenar cuando haya datos nuevos

#Instacia de la clase train_model
train_model = train_model()

#Importación del modelo

modelo = train_model.importar_modelo('model/football_predictor.pkl')

#Prediccion

train_model.prediccion_modelo(modelo, datos_nuevos)



