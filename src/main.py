from utils.train import train_model
from utils.functions import data_processing 
import pandas as pd

'''UNICAMENTE USARÉ ESTE main SI HAY QUE REENTRENAR EL MODELO CON NUEVAS ESTADÍSTICAS. HAY OTRO MAIN UNICAMENTE DEDICADO A PREDECIR'''

#Instancio la clase data_processing
data_processing = data_processing()

#Carga de datos
df_datos_generales = pd.read_csv('data/raw_files/datos_generales_fx.csv')
df_estadisticas = pd.read_csv('data/raw_files/df_estadisticas.csv')
df_alineaciones = pd.read_csv('data/raw_files/datos_alineaciones.csv')
df_lesionados = pd.read_csv('data/raw_files/datos_lesionados.csv')
df_dicc_equipos = pd.read_csv('data/raw_files/df_dicc_equipos.csv')
cuotas = data_processing.ruta_cuotas()


#Procesado de datos

#Procesamiento datos generales de partidos
df_datos_generales_procesado = data_processing.procesado_datos_generales(df_datos_generales)

#Procesamiento de las estadisticas de los partidos
df_estadisticas_procesado = data_processing.procesado_estadisticas(df_estadisticas)
#Procesamiento de las alineaciones de los partidos
df_alineaciones_procesado = data_processing.procesado_titulares(df_alineaciones)
#Procesamiento de los lesionados de los partidos
df_lesionados_procesado = data_processing.procesado_lesionados(df_lesionados)
#Procesamiento de los datos de las cuotas
df_cuotas_procesado = data_processing.procesado_cuotas(cuotas,df_dicc_equipos)
#Unión de los 4 dataframes anteriores, y realización de limpieza e imputación de missings si los hubiera
df_union_procesado = data_processing.creacion_df_final(df_lesionados=df_lesionados_procesado, 
                                                       df_alineaciones=df_alineaciones_procesado,
                                                        df_datos_partidos=df_datos_generales_procesado,
                                                        df_estadisticas=df_estadisticas_procesado,
                                                        df_cuotas = df_cuotas_procesado)
#Creación de nuevas variables interesantes para el desempeño del modelo
df_final = data_processing.creacion_nuevas_variables(df_union_procesado)

df_final.to_csv('data/processed_files/df_datos_completos.csv', index=False)
#Creación de datos nuevos. 
#Para crear los datos nuevos hay que darle valor a una serie de variables. Se muestra un ejemplo, varían por partido, y no es necesario pasar una lista completa de lesionados
#y alineaciones pero mejorará el desempeño del modelo. Los nombres del estadio y el árbitro deben estar correctos y 100% igual escritos. Para sacarlos se puede llamar a las funciones 
# buscar_estadio y buscar_arbitro del archivo functions.py . También se pueden encontrar ids de equipos y jugadores con sus respectivas funciones. (Hay que pasar siempre id de equipo,
# y jugador)
#Para este ejemplo se usarán los siguientes datos (Lugo - Real Zaragoza)
id_equipo_local = 716  
id_equipo_visitante = 732
odd_1 = 3.6
odd_x = 3
odd_2 = 2.25  
arbitro = 'Raul Martin Gonzalez Frances, Spain' 
estadio = 'Anxo Carro'  
season = 2022  
ids_lesionados = [2352,47379,182786,58,161590,46732]  
ids_titulares = [
   15575, 47190,46765,182261,46909,310609,46981,3365,194519,47177,104894,
   47044,380261,104916,47332,47053,107158,47065,162249,47553,323935,22098
]

datos_nuevos = data_processing.creacion_datos_nuevos(df_final,id_equipo_local, id_equipo_visitante,odd_1, odd_x, odd_2, arbitro, estadio, season, ids_lesionados, ids_titulares)

#Instancio la clase entrenamiento del modelo
train_model = train_model()

train_model.train_xgbc(df_final) #Realizará la exportación también de un .pkl que importaremos para realizar la predicción

#Importación del modelo
modelo = train_model.importar_modelo('model/football_predictor.pkl')

#Prediccion
train_model.prediccion_modelo(modelo, datos_nuevos)



