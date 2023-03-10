import category_encoders as ce
import numpy as np
import pandas as pd
from unidecode import unidecode


class data_processing():

    def __init__(self):
        pass

    def procesado_lesionados(self, df):
        '''Coge el dataframe de lesionados y le aplica un OneHotEncoder, pero sin usar la librería. Para tener en cuenta que jugadores
        han participado en el encuentro de inicio o no. Es representativo ya que la no presencia de un jugador puede afectar en el resultado
        de un partido'''
        #Añade una fila de 1 para identificar que ese jugador ha estado lesionado en algún momento
            
        df['lesionados'] = pd.Series(np.ones(len(df)), index=df.index)
        
        
        df_lesionados_id = df[['fixture_id', 'id_lesionado', 'lesionados']]
        
        #Elimino fila si hay missings en la columna de id_jugador_titular
        df_lesionados_id = df_lesionados_id.dropna(subset=['id_lesionado'])
        
        #Pivota la tabla para convertir en una variable cada jugador. Para los partidos que el jugador no ha estado lesionado,
        #se rellenará con un '0'. Dejo el índice de la forma correcta.
        df_lesionados_id = df_lesionados_id.pivot( index= 'fixture_id', 
                                                columns = 'id_lesionado', 
                                                values = 'lesionados').fillna(0).reset_index()
        df_lesionados_id.columns.name = None
        
        #Transformo los valores '1' y '0' a int.
        df_lesionados_id.iloc[:,1:] =df_lesionados_id.iloc[:,1:].astype(int)
        
        #Añado al nombre de las variables de los id de jugadores 'les-' para identificar que es la variable de lesionados.
        df_lesionados_id = df_lesionados_id.rename(columns={col: f'les-{col}' for col in df_lesionados_id.iloc[:,1:]})
        
        return df_lesionados_id


    def procesado_titulares(self, df):
        '''Coge el dataframe de alineaciones y le aplica un OneHotEncoder, pero sin usar la librería. Para tener en cuenta que jugadores
        han participado en el encuentro de inicio o no. Es representativo ya que la no presencia de un jugador puede afectar en el resultado
        de un partido'''    

        #Añade una fila de 1 para identificar que ese jugador ha estado lesionado en algún momento
        df['titular'] = pd.Series(np.ones(len(df)), index=df.index)
        df_alineaciones_id = df[['fixture_id', 'id_jugador_titular', 'titular']]
        #Elimino fila si hay missings en la columna de id_jugador_titular
        df_alineaciones_id = df_alineaciones_id.dropna(subset=['id_jugador_titular'])
        #Pivota la tabla para convertir en una variable cada jugador. Para los partidos que el jugador no ha estado lesionado,
        #se rellenará con un '0'. Dejo el índice de la forma correcta.
        df_alineaciones_id = df_alineaciones_id.pivot( index= 'fixture_id', 
                                                columns = 'id_jugador_titular', 
                                                values = 'titular').fillna(0).reset_index()
        df_alineaciones_id.columns.name = None
        
        #Me cargo un jugador con id nulo (hay que revisarlo después del procesado)
        df_alineaciones_id = df_alineaciones_id.drop(df_alineaciones_id.columns[1], axis=1)
        
        #Transformo los valores '1' y '0' a int.
        df_alineaciones_id.iloc[:,1:]=df_alineaciones_id.iloc[:,1:].astype(int)
        
        #Añado al nombre de las variables de los id de jugadores 'les-' para identificar que es la variable de lesionados.
        df_alineaciones_id = df_alineaciones_id.rename(columns={col: f'titu-{col}' for col in df_alineaciones_id.iloc[:,1:]})
        
        return df_alineaciones_id
    
    def procesado_estadisticas(self, df):
        '''Procesado de todas las estadisticas que se han extraido, y que ocurren dentro de un partido.'''
        #Elimino las filas en las que la API no me devuelve un solo valor(ha pasado)
        rows_with_all_missing = df.iloc[:, :-1].isna().all(axis=1)
        df = df[~rows_with_all_missing]
        #Renombro dos columnas mal nombradas (no es el dato que dice la columna)
        df = df.rename(columns={'pass_precision_local': 'total_pass_local',
                                'pass_precision_away': 'total_pass_away',
                            'fixture_id_2': 'fixture_id'})
        #Transformo los datos de posesion a float para poder usarlos de forma más sencilla
        df['ball_possession_local'] = df['ball_possession_local'].str.replace('%','').astype(float)
        df['ball_possession_away'] = df['ball_possession_away'].str.replace('%', '').astype(float)

        df['ball_possession_local'] = df['ball_possession_local']/100
        df['ball_possession_away'] = df['ball_possession_away']/100
        # Convertir columna a tipo numérico. Esto es porque en las tarjetas amarillas habia datos erroneos (con porcntaje)
        df['yellow_cards_local'] = pd.to_numeric(df['yellow_cards_local'], errors='coerce')
        df['yellow_cards_away'] = pd.to_numeric(df['yellow_cards_away'], errors='coerce')

        # Filtro filas con NaN en la columna en cuestión y las elimino
        rows_with_nan = df['yellow_cards_local'].isna() | df['yellow_cards_away'].isna()
        df = df[~rows_with_nan]
        #Cambio los missings por 0, ya que cuando el valor es 0 la api devuelve null.
        df.fillna(0, inplace = True)
        #Cambio todas las columnas que quiero que sean número entero para trabajar mejor con ellos
        cols_to_int = ['shots_on_goal_local', 'shots_on_goal_away', 'shots_off_goal_local', 'shots_off_goal_away', 
                'total_shots_local', 'total_shots_away', 'blocked_shots_local', 'blocked_shots_away', 
                'shots_insidebox_local', 'shots_insidebox_away', 'shots_outsidebox_local', 'shots_outsidebox_away', 
                'fouls_local', 'fouls_away', 'corners_local', 'corners_away', 'offsides_local', 'offsides_away', 
                'yellow_cards_local', 'yellow_cards_away', 'red_cards_local', 'red_cards_away', 'goalkeeper_saves_local', 
                'goalkeeper_saves_away', 'total_pass_local', 'total_pass_away']
        df[cols_to_int] = df[cols_to_int].astype(int)
        
        return df
        
    def procesado_datos_generales(self, df):
        '''Eliminación de missings en los goles (ya que si no hay goles lo considera como missing), y cambiar el tipo de los goles
        a favor/en contra'''
        #Sustituimos los missings por 0, ya que esos missings significa que ha habido 0 goles
        df['goles_descanso_local'] = df['goles_descanso_local'].fillna(0)
        df['goles_descanso_visitante'] = df['goles_descanso_visitante'].fillna(0)
        #Cambio el tipo de float a int, ya que no puede haber goles decimales
        df['goles_descanso_local'] = df['goles_descanso_local'].astype(int)
        df['goles_descanso_visitante'] = df['goles_descanso_visitante'].astype(int)
        
        return df
    
    def ruta_cuotas(self):
        return [
        'data/raw_files/cuotas/SP1-2012.csv',
        'data/raw_files/cuotas/SP2-2012.csv',
        'data/raw_files/cuotas/SP1-2013.csv',
        'data/raw_files/cuotas/SP2-2013.csv',
        'data/raw_files/cuotas/SP1-2014.csv',
        'data/raw_files/cuotas/SP2-2014.csv',
        'data/raw_files/cuotas/SP1-2015.csv',
        'data/raw_files/cuotas/SP2-2015.csv',
        'data/raw_files/cuotas/SP1-2016.csv',
        'data/raw_files/cuotas/SP2-2016.csv',
        'data/raw_files/cuotas/SP1-2017.csv',
        'data/raw_files/cuotas/SP2-2017.csv',
        'data/raw_files/cuotas/SP1-2018.csv',
        'data/raw_files/cuotas/SP2-2018.csv',
        'data/raw_files/cuotas/SP1-2019.csv',
        'data/raw_files/cuotas/SP2-2019.csv',
        'data/raw_files/cuotas/SP1-2020.csv',
        'data/raw_files/cuotas/SP2-2020.csv',
        'data/raw_files/cuotas/SP1-2021.csv',
        'data/raw_files/cuotas/SP2-2021.csv',
        'data/raw_files/cuotas/SP1-2022.csv',
        'data/raw_files/cuotas/SP2-2022.csv'
    ]
    
    def procesado_cuotas(self, file_names, df_ids):
        equivalencia_nombres = {
            'Celta':'Celta Vigo',
            'Mallorca':'Mallorca',
            'Sevilla': 'Sevilla',
            'Ath Bilbao': 'Athletic Club',
            'Barcelona':'Barcelona',
            'Levante':'Levante',
            'Real Madrid': 'Real Madrid',
            'La Coruna': 'Deportivo La Coruna',
            'Vallecano': 'Rayo Vallecano',
            'Zaragoza': 'Zaragoza',
            'Betis': 'Real Betis',
            'Espanol':'Espanyol',
            'Malaga':'Malaga',
            'Sociedad':'Real Sociedad',
            'Getafe':'Getafe',
            'Granada':'Granada CF',
            'Osasuna':'Osasuna',
            'Valencia':'Valencia',
            'Ath Madrid':'Atletico Madrid',
            'Valladolid':'Valladolid',
            'Barcelona B':'Barcelona B',
            'Mirandes':'Mirandes',
            'Villarreal':'Villarreal',
            'Girona':'Girona',
            'Lugo':'Lugo',
            'Xerez':'Xerez',
            'Alcorcon':'Alcorcon',
            'Elche':'Elche',
            'Numancia':'Numancia',
            'Santander':'Racing Santander',
            'Murcia':'Real Murcia',
            'Almeria':'Almeria',
            'Guadalajara':'Guadalajara',
            'Huesca':'Huesca',
            'Las Palmas':'Las Palmas',
            'Ponferradina':'Ponferradina',
            'Real Madrid B':'Real Madrid II',
            'Recreativo':'Recreativo Huelva',
            'Sabadell':'Sabadell',
            'Sp Gijon':'Sporting Gijon',
            'Cordoba':'Cordoba',
            'Hercules':'Hércules',
            'Jaen':'Real Jaén',
            'Alaves':'Alaves',
            'Eibar':'Eibar',
            'Tenerife':'Tenerife',
            'Albacete':'Albacete',
            'Leganes':'Leganes',
            'Llagostera':'Llagostera',
            'Gimnastic':'Gimnastic',
            'Oviedo':'Oviedo',
            'Ath Bilbao B':'Athletic Club II',
            'Sevilla B':'Sevilla Atletico',
            'Reus Deportiu':'Reus',
            'Cadiz':'Cadiz',
            'UCAM Murcia':'Ucam Murcia',
            'Lorca':'Lorca',
            'Leonesa':'Cultural Leonesa',
            'Extremadura UD':'Extremadura',
            'Rayo Majadahonda':'Rayo Majadahonda',
            'Fuenlabrada':'Fuenlabrada',
            'Castellon':'Castellón',
            'Cartagena':'FC Cartagena',
            'Logrones':'UD Logroñés',
            'Sociedad B':'Real Sociedad II',
            'Ibiza':'Ibiza',
            'Amorebieta':'Amorebieta',
            'Burgos':'Burgos',
            'Villarreal B':'Villarreal II',
            'Andorra':'FC Andorra'
        }
    
        def select_columns_and_add_season(df, file_name):
            # Extraer año del nombre del archivo
            year = file_name.split('-')[1][:4]

            # Crear un diccionario que contenga los nombres de los equipos como claves y sus IDs como valores
            equipo_id = {}
            for index, row in df_ids.iterrows():
                equipo_id[row['equipo_jugador']] = row['id_equipo']

            # Reemplazar los nombres de los equipos por sus IDs correspondientes utilizando el diccionario de equivalencias y el diccionario equipo_id
            df['HomeTeam'] = df['HomeTeam'].map(equivalencia_nombres).map(equipo_id)
            df['AwayTeam'] = df['AwayTeam'].map(equivalencia_nombres).map(equipo_id)

            # Seleccionar columnas requeridas
            df_selected = df[['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']]

            # Añadir columna "season" con el año extraído
            df_selected['season'] = int(year)

            # Eliminar filas con valores NaN
            df_selected.dropna(inplace=True)

            return df_selected

        # Lista para guardar los dataframes procesados
        processed_dfs = []

        # Iterar sobre los nombres de archivo
        for file_name in file_names:
            # Leer archivo CSV en un dataframe
            df = pd.read_csv(file_name)

            # Aplicar la función select_columns_and_add_season y renombrar las columnas
            df_processed = select_columns_and_add_season(df, file_name).rename(columns={'B365H': 'odd_1', 'B365D': 'odd_x', 'B365A': 'odd_2'})

            # Agregar el dataframe procesado a la lista
            processed_dfs.append(df_processed)

        # Concatenar todos los dataframes procesados en uno solo
        final_df = pd.concat(processed_dfs, ignore_index=True)
        final_df = final_df.dropna(how='any')
        final_df['HomeTeam'] = final_df['HomeTeam'].astype(int)
        final_df['AwayTeam'] = final_df['AwayTeam'].astype(int)

        return final_df
        
    def creacion_df_final(self, df_lesionados, df_alineaciones, df_datos_partidos, df_estadisticas, df_cuotas):
        '''Esta función hace un merge de todos los datos sacados anteriormente'''
        #Comenzamos la unión de dataframes, empezando por los datos de partidos y estadísticas
        df_final = pd.merge(df_datos_partidos, df_estadisticas, on='fixture_id', how='left')
        
        #Elimino las filas en las que la API no me devuelve un solo valor(ha pasado)
        rows_with_all_missing = df_final.loc[:, 'shots_on_goal_local':].isna().all(axis=1)
        df_final = df_final[~rows_with_all_missing]
        
        #Unimos el df resultante con el de lesionados
        df_final = pd.merge(df_final,df_lesionados, on='fixture_id', how = 'left')
        #Relleno los missings con 0, ya que significa que en esos partidos no ha habido lesionados
        df_final = df_final.fillna(0)
        
        #Unimos el df_final con el de alineaciones, que es el que faltaría.
        df_final = pd.merge(df_final, df_alineaciones, on='fixture_id', how='left')
        #Relleno los missings con 0, ya que significa que en esos partidos no habría participado ese jugador
        df_final = df_final.fillna(0)
        
        #Unimos el df_final con el de odds
        df_final = pd.merge(df_final, df_cuotas, left_on=['id_equipo_local', 'id_equipo_visitante', 'season'], 
                            right_on=['HomeTeam', 'AwayTeam', 'season'], how='inner')
        
        #Eliminamos filas con NaN
        df_final = df_final.dropna(how='any')
        df_final = df_final.drop(['HomeTeam','AwayTeam'], axis=1)
        
        df_final = df_final.reset_index()

        #Para agilizar tiempos en métedos que necesitan esta tabla para usarse, ya que tarda un poco en ejecutarse.
        #df_final.to_csv('df_partidos_completo.csv', index=False)
        
        return df_final
        
    def creacion_nuevas_variables(self, df_final):
        '''Esta función creará una nueva variable que se me ha ocurrido: los lanzamientos necesarios para marcar gol'''
        #Se cogen la suma de los goles y lanzamientos de los tres ultimos partidos como local/visitante para calcular el número de 
        #lanzamientos que se necesitan para marcar gol.
        df_final['goles_local_previos'] =   df_final.groupby('id_equipo_local')['goles_local'].shift(1) + \
                                    df_final.groupby('id_equipo_local')['goles_local'].shift(2) + \
                                    df_final.groupby('id_equipo_local')['goles_local'].shift(3)
        
        df_final['tiros_local_previos'] =   df_final.groupby('id_equipo_local')['total_shots_local'].shift(1) + \
                                    df_final.groupby('id_equipo_local')['total_shots_local'].shift(2) + \
                                    df_final.groupby('id_equipo_local')['total_shots_local'].shift(3)
        #En el caso de nulos, se coge el siguiente partido(en rara ocasión habrá nulos)
        df_final['goles_local_previos'] = df_final['goles_local_previos'].fillna(df_final.groupby('id_equipo_local')['goles_local'].shift(-1))
        df_final['tiros_local_previos'] = df_final['tiros_local_previos'].fillna(df_final.groupby('id_equipo_local')['total_shots_local'].shift(-1))


        df_final['tiros_para_marcar_local'] = np.where(df_final['goles_local_previos'] == 0, 
                                            df_final['tiros_local_previos'], 
                                            df_final['tiros_local_previos'] / df_final['goles_local_previos'])
        
        df_final['goles_away_previos'] =    df_final.groupby('id_equipo_visitante')['goles_visitante'].shift(1) + \
                                    df_final.groupby('id_equipo_visitante')['goles_visitante'].shift(2) + \
                                    df_final.groupby('id_equipo_visitante')['goles_visitante'].shift(3)

        
        df_final['tiros_away_previos'] =    df_final.groupby('id_equipo_visitante')['total_shots_away'].shift(1) + \
                                    df_final.groupby('id_equipo_visitante')['total_shots_away'].shift(2) + \
                                    df_final.groupby('id_equipo_visitante')['total_shots_away'].shift(3)

        df_final['goles_away_previos'] = df_final['goles_away_previos'].fillna(df_final.groupby('id_equipo_visitante')['goles_visitante'].shift(-1))
        df_final['tiros_away_previos'] = df_final['tiros_away_previos'].fillna(df_final.groupby('id_equipo_visitante')['total_shots_away'].shift(-1))


        df_final['tiros_para_marcar_away'] = np.where(df_final['goles_away_previos'] == 0, 
                                            df_final['tiros_away_previos'], 
                                            df_final['tiros_away_previos'] / df_final['goles_away_previos'])

        df_final = df_final.drop(['tiros_away_previos','goles_away_previos','tiros_local_previos','goles_local_previos'], axis=1)
        
        df_final = df_final.sort_values(by='fecha_timestamp', ascending=True)
        
        df_final['tiros_para_marcar_local'] = df_final['tiros_para_marcar_local'].fillna(df_final['tiros_para_marcar_local'].mean())
        df_final['tiros_para_marcar_away'] = df_final['tiros_para_marcar_away'].fillna(df_final['tiros_para_marcar_away'].mean())

        
        return df_final



    #Las funciones siguientes tendrán únicamente la utilidad de crear datos nuevos.
    def buscar_jugador(self, id_equipo, temporada_equipo):
        ''' Esta función únicamente será llamada para localizar los ids de jugadores y poder crear los datos nuevos'''

        df = pd.read_csv("df_diccionario_jugadores.csv")

        # Filtre el DataFrame original utilizando los valores de los parámetros
        filtro = (df['id_equipo'] == id_equipo) & (df['temporada_equipo'] == temporada_equipo)
        df_filtrado = df[filtro]

        # Devuelva el DataFrame filtrado
        return df_filtrado

    def buscar_equipo(self, nombres):
        ''' Esta función únicamente será llamada para localizar los ids delos equipos y poder crear los datos nuevos. Acepta una lista de nombres o
         un único nombre '''
        # Cargo el diccionario de ids que tengo y fue descargado. Es el mismo que de jugadores
        equipos = pd.read_csv("df_diccionario_jugadores.csv")
        # Elimino acentos de los nombres de los equipos en el DataFrame aplicando unidecode. También se quedan en minúsculas
        equipos['nombre_equipo'] = equipos['equipo_jugador'].apply(lambda x: unidecode(x.lower()))
        
        if isinstance(nombres, str):
            # Eliminar las marcas diacríticas del nombre introducido
            equipo = unidecode(nombres.lower())
            # Busco los equipos cuyo nombre contenga la cadena de texto introducida como parámetro
            equipos_coincidentes = equipos[equipos['nombre_equipo'].str.contains(nombres, case=False)]
            #Elimino los jugadores duplicados, porque pueden salir jugadores repetidos si participaron más de 1 temporada
            equipos_coincidentes = equipos_coincidentes.drop_duplicates(subset='id_equipo')
            # Devuelve una tabla con los nombres y los ids de los jugadores encontrados
            return equipos_coincidentes[['nombre_equipo', 'id_equipo']]
        
        elif isinstance(nombres, list):
            resultados = []
            # Busco cada id de los jugadores en la lista
            for n in nombres:
                # Elimino acentos del nombre introducido
                equipo = unidecode(n.lower())
                # Busco los jugadores cuyo nombre contenga la cadena de texto introducida como parámetro
                equipos_coincidentes = equipos[equipos['nombre_equipo'].str.contains(n, case=False)]
                # Elimino jugadores duplicados en función del id
                equipos_coincidentes = equipos_coincidentes.drop_duplicates(subset='id_equipo')
                # Añado los resultados a la lista de resultados
                for i, row in equipos_coincidentes.iterrows():
                    resultados.append([row['nombre_equipo'], row['id_equipo']])
            # Devuelvo una tabla con los nombres y los IDs de los jugadores encontrados
            return pd.DataFrame(resultados, columns=['nombre_equipo', 'id_equipo'])
        
        else: return 'Introduce una lista de nombres o un nombre único'

    
    def nombre_arbitro_correcto(self, nombre):

        arbitros = pd.read_csv('df_partidos_completo.csv')
        # Elimino los acentos del nombre introducido
        arbitro = unidecode(nombre.lower())
        # Busco los equipos cuyo nombre contenga la cadena de texto introducida como parámetro
        arbitros_coincidentes = arbitros[arbitros['arbitro'].str.contains(nombre, case=False)]
        #Elimino los arbitros duplicados, porque pueden salir árbitros repetidos si participaron más de 1 temporada
        arbitros_coincidentes = arbitros_coincidentes.drop_duplicates(subset='arbitro')
        if len(arbitros_coincidentes) == 0:
            return None
        else:
            # Obtengo el índice de la fila correspondiente al árbitro
            indice = arbitros_coincidentes.index[0]
            # Obtengo el nombre del árbitro con el formato adecuado
            nombre_completo = arbitros.loc[indice, 'arbitro']
            # Devuelve el nombre del árbitro con el formato adecuado
            return nombre_completo


    def nombre_estadio_correcto(self, nombre):
        estadios = pd.read_csv('df_partidos_completo.csv')
        # Elimino los acentos del nombre introducido
        estadio = unidecode(nombre.lower())
        # Busco los equipos cuyo nombre contenga la cadena de texto introducida como parámetro
        estadios_coincidentes = estadios[estadios['estadio'].str.contains(nombre, case=False)]
        #Elimino los arbitros duplicados, porque pueden salir estadios repetidos si participaron más de 1 temporada
        estadios_coincidentes = estadios_coincidentes.drop_duplicates(subset='estadio')
        if len(estadios_coincidentes) == 0:
            return None
        else:
            # Obtengo el índice de la fila correspondiente al estadio
            indice = estadios_coincidentes.index[0]
            # Obtengo el nombre del estadio con el formato adecuado
            nombre_completo = estadios.loc[indice, 'estadio']
            # Devuelve el nombre del estadio con el formato adecuado
            return nombre_completo
    
    def creacion_datos_nuevos(self, df_partidos,id_equipo_local, id_equipo_visitante,odd_1, odd_x, odd_2, arbitro, estadio, season, ids_lesionados, ids_titulares):
        #Leo el csv donde estan todos los datos completos de los partidos. Se ha creado con la función creacion_df_final()
        
        #Creo los datos de estadisticas que se preveen con la media de datos de los últimos 3 partidos en casa o de visitante
        shots_on_goal_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_on_goal_local'].shift(1) +  \
                                df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_on_goal_local'].shift(2) +  \
                                df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_on_goal_local'].shift(3))
        shots_on_goal_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_on_goal_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_on_goal_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_on_goal_away'].shift(3))
        shots_off_goal_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_off_goal_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_off_goal_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_off_goal_local'].shift(3))
        shots_off_goal_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_off_goal_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_off_goal_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_off_goal_away'].shift(3))
        total_shots_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'total_shots_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'total_shots_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'total_shots_local'].shift(3))
        total_shots_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'total_shots_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'total_shots_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'total_shots_away'].shift(3))
        blocked_shots_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'blocked_shots_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'blocked_shots_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'blocked_shots_local'].shift(3))
        blocked_shots_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'blocked_shots_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'blocked_shots_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'blocked_shots_away'].shift(3))
        shots_insidebox_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_insidebox_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_insidebox_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_insidebox_local'].shift(3))
        shots_insidebox_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_insidebox_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_insidebox_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_insidebox_away'].shift(3))
        shots_outsidebox_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_outsidebox_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_outsidebox_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_outsidebox_local'].shift(3))
        shots_outsidebox_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_outsidebox_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_outsidebox_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_outsidebox_away'].shift(3))
        fouls_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'fouls_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'fouls_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'fouls_local'].shift(3))
        fouls_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'fouls_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'fouls_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'fouls_away'].shift(3))
        corners_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'corners_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'corners_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'corners_local'].shift(3))
        corners_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'corners_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'corners_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'corners_away'].shift(3))
        offsides_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'offsides_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'offsides_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'offsides_local'].shift(3))
        offsides_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'offsides_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'offsides_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'offsides_away'].shift(3))
        ball_possession_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'ball_possession_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'ball_possession_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'ball_possession_local'].shift(3))
        ball_possession_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'ball_possession_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'ball_possession_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'ball_possession_away'].shift(3))
        yellow_cards_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'yellow_cards_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'yellow_cards_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'yellow_cards_local'].shift(3))
        yellow_cards_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'yellow_cards_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'yellow_cards_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'yellow_cards_away'].shift(3))
        red_cards_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'red_cards_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'red_cards_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'red_cards_local'].shift(3))
        red_cards_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'red_cards_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'red_cards_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'red_cards_away'].shift(3))
        goalkeeper_saves_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'goalkeeper_saves_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'goalkeeper_saves_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'goalkeeper_saves_local'].shift(3))
        goalkeeper_saves_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'goalkeeper_saves_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'goalkeeper_saves_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'goalkeeper_saves_away'].shift(3))
        total_pass_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'total_pass_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'total_pass_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'total_pass_local'].shift(3))
        total_pass_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'total_pass_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'total_pass_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'total_pass_away'].shift(3))
        
        #Creo un dataframe de solo estadísticas más los datos de arbitro, estadio, ids, season (todos menos lesionados y titulares)
        df_datos_nuevos = pd.DataFrame({
                                        'id_equipo_local': id_equipo_local,
                                            'id_equipo_visitante': id_equipo_visitante,
                                            'arbitro': arbitro,
                                            'estadio': estadio,
                                            'season': season,
                                        'shots_on_goal_local':shots_on_goal_local,
                                        'shots_on_goal_away':shots_on_goal_away,
                                        'shots_off_goal_local':shots_off_goal_local,
                                        'shots_off_goal_away':shots_off_goal_away,
                                        'total_shots_local':total_shots_local,
                                        'total_shots_away':total_shots_away,
                                        'blocked_shots_local':blocked_shots_local,
                                        'blocked_shots_away':blocked_shots_away,
                                        'shots_insidebox_local':shots_insidebox_local,
                                        'shots_insidebox_away':shots_insidebox_away,
                                        'shots_outsidebox_local':shots_outsidebox_local,
                                        'shots_outsidebox_away':shots_outsidebox_away,
                                        'fouls_local':fouls_local,
                                        'fouls_away':fouls_away,
                                        'corners_local':corners_local,
                                        'corners_away':corners_away,
                                        'offsides_local':offsides_local,
                                        'offsides_away':offsides_away,
                                        'ball_possession_local':ball_possession_local,
                                        'ball_possession_away':ball_possession_away,
                                        'yellow_cards_local':yellow_cards_local,
                                        'yellow_cards_away':yellow_cards_away,
                                        'red_cards_local':red_cards_local,
                                        'red_cards_away':red_cards_away,
                                        'goalkeeper_saves_local':goalkeeper_saves_local,
                                        'goalkeeper_saves_away':goalkeeper_saves_away,
                                        'total_pass_local':total_pass_local,
                                        'total_pass_away':total_pass_away
                                        }, index = [0])
        
        #Creo un dataframe solo con las columnas de lesionados y las relleno con 0 todas
        #Primero localizo todas las columnas de df_partidos
        columns_les = []
        for col in df_partidos.columns:
            if 'les-' in col:
                columns_les.append(col)
        #Relleno con 0 y creo el dataframe de lesionados
        valores = {col: 0 for col in columns_les}
        df_lesionados_nuevos = pd.DataFrame([valores])
        
        #Hago el mismo proceso con un dataframe de titulares
        columns_titus = []
        for col in df_partidos.columns:
            if 'titu-' in col:
                columns_titus.append(col)
        valores_titu = {col: 0 for col in columns_titus}
        df_titulares_nuevos = pd.DataFrame([valores_titu])
        
        #Concateno los 3 dataframe para obtener el dataframe de datos final
        df_datos_nuevos_final = pd.concat([df_datos_nuevos, df_lesionados_nuevos,df_titulares_nuevos], axis = 1)
        
        #Añado los prefijos y sufijos necesarios para localizar los ids de lesionados y titulares en la tabla
        ids_lesionado_prefijo = ['les-{}'.format(id) for id in ids_lesionados]
        ids_titular_prefijo = ['titu-{}{}'.format(id,'.0') for id in ids_titulares]
        
        #Y sustituyo el valor correspondiente por 1, ya que o estan lesionados en ese partido o van a jugar
        for id_les in ids_lesionado_prefijo:
            df_datos_nuevos_final.loc[0, id_les] = 1
        for id_titu in ids_titular_prefijo:
            df_datos_nuevos_final.loc[0, id_titu] = 1
            
        #Aquí añado las nuevas variables que me parecieron interesantes siguiendo el mismo código que como las cree en el método anterior    
        df_datos_nuevos_final['goles_local_previos'] = df_partidos.groupby('id_equipo_local')['goles_local'].shift(1) + \
                                    df_partidos.groupby('id_equipo_local')['goles_local'].shift(2) + \
                                    df_partidos.groupby('id_equipo_local')['goles_local'].shift(3)

        df_datos_nuevos_final['tiros_local_previos'] = df_partidos.groupby('id_equipo_local')['total_shots_local'].shift(1) + \
                                        df_partidos.groupby('id_equipo_local')['total_shots_local'].shift(2) + \
                                        df_partidos.groupby('id_equipo_local')['total_shots_local'].shift(3)

        df_datos_nuevos_final['goles_local_previos'] = df_datos_nuevos_final['goles_local_previos'].fillna(df_partidos.groupby('id_equipo_local')['goles_local'].shift(-1))
        df_datos_nuevos_final['tiros_local_previos'] = df_datos_nuevos_final['tiros_local_previos'].fillna(df_partidos.groupby('id_equipo_local')['total_shots_local'].shift(-1))

        df_datos_nuevos_final['tiros_para_marcar_local'] = np.where(df_datos_nuevos_final['goles_local_previos'] == 0, 
                                                df_datos_nuevos_final['tiros_local_previos'], 
                                                df_datos_nuevos_final['tiros_local_previos'] / df_datos_nuevos_final['goles_local_previos'])

        df_datos_nuevos_final['goles_away_previos'] = df_partidos.groupby('id_equipo_visitante')['goles_visitante'].shift(1) + \
                                        df_partidos.groupby('id_equipo_visitante')['goles_visitante'].shift(2) + \
                                        df_partidos.groupby('id_equipo_visitante')['goles_visitante'].shift(3)

        df_datos_nuevos_final['tiros_away_previos'] = df_partidos.groupby('id_equipo_visitante')['total_shots_away'].shift(1) + \
                                        df_partidos.groupby('id_equipo_visitante')['total_shots_away'].shift(2) + \
                                        df_partidos.groupby('id_equipo_visitante')['total_shots_away'].shift(3)

        df_datos_nuevos_final['goles_away_previos'] = df_datos_nuevos_final['goles_away_previos'].fillna(df_partidos.groupby('id_equipo_visitante')['goles_visitante'].shift(-1))
        df_datos_nuevos_final['tiros_away_previos'] = df_datos_nuevos_final['tiros_away_previos'].fillna(df_partidos.groupby('id_equipo_visitante')['total_shots_away'].shift(-1))

        df_datos_nuevos_final['tiros_para_marcar_away'] = np.where(df_datos_nuevos_final['goles_away_previos'] == 0, 
                                                df_datos_nuevos_final['tiros_away_previos'], 
                                                df_datos_nuevos_final['tiros_away_previos'] / df_datos_nuevos_final['goles_away_previos'])

        df_datos_nuevos_final = df_datos_nuevos_final.drop(['tiros_away_previos','goles_away_previos','tiros_local_previos','goles_local_previos'], axis=1)

        df_datos_nuevos_final['tiros_para_marcar_local'] = df_datos_nuevos_final['tiros_para_marcar_local'].fillna(df_datos_nuevos_final['tiros_para_marcar_local'].mean())
        df_datos_nuevos_final['tiros_para_marcar_away'] = df_datos_nuevos_final['tiros_para_marcar_away'].fillna(df_datos_nuevos_final['tiros_para_marcar_away'].mean())

        df_datos_nuevos_final['odd_1'] = odd_1
        df_datos_nuevos_final['odd_x'] = odd_x
        df_datos_nuevos_final['odd_2'] = odd_2

        return df_datos_nuevos_final
    
    def creacion_datos_nuevos_redes(self, df_partidos,id_equipo_local, id_equipo_visitante,odd_1, odd_x, odd_2, arbitro, estadio, season):
        #Leo el csv donde estan todos los datos completos de los partidos. Se ha creado con la función creacion_df_final()
        
        #Creo los datos de estadisticas que se preveen con la media de datos de los últimos 3 partidos en casa o de visitante
        shots_on_goal_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_on_goal_local'].shift(1) +  \
                                df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_on_goal_local'].shift(2) +  \
                                df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_on_goal_local'].shift(3))
        shots_on_goal_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_on_goal_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_on_goal_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_on_goal_away'].shift(3))
        shots_off_goal_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_off_goal_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_off_goal_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_off_goal_local'].shift(3))
        shots_off_goal_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_off_goal_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_off_goal_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_off_goal_away'].shift(3))
        total_shots_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'total_shots_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'total_shots_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'total_shots_local'].shift(3))
        total_shots_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'total_shots_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'total_shots_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'total_shots_away'].shift(3))
        blocked_shots_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'blocked_shots_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'blocked_shots_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'blocked_shots_local'].shift(3))
        blocked_shots_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'blocked_shots_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'blocked_shots_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'blocked_shots_away'].shift(3))
        shots_insidebox_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_insidebox_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_insidebox_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_insidebox_local'].shift(3))
        shots_insidebox_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_insidebox_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_insidebox_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_insidebox_away'].shift(3))
        shots_outsidebox_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_outsidebox_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_outsidebox_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'shots_outsidebox_local'].shift(3))
        shots_outsidebox_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_outsidebox_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_outsidebox_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'shots_outsidebox_away'].shift(3))
        fouls_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'fouls_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'fouls_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'fouls_local'].shift(3))
        fouls_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'fouls_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'fouls_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'fouls_away'].shift(3))
        corners_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'corners_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'corners_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'corners_local'].shift(3))
        corners_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'corners_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'corners_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'corners_away'].shift(3))
        offsides_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'offsides_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'offsides_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'offsides_local'].shift(3))
        offsides_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'offsides_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'offsides_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'offsides_away'].shift(3))
        ball_possession_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'ball_possession_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'ball_possession_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'ball_possession_local'].shift(3))
        ball_possession_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'ball_possession_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'ball_possession_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'ball_possession_away'].shift(3))
        yellow_cards_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'yellow_cards_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'yellow_cards_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'yellow_cards_local'].shift(3))
        yellow_cards_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'yellow_cards_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'yellow_cards_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'yellow_cards_away'].shift(3))
        red_cards_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'red_cards_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'red_cards_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'red_cards_local'].shift(3))
        red_cards_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'red_cards_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'red_cards_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'red_cards_away'].shift(3))
        goalkeeper_saves_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'goalkeeper_saves_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'goalkeeper_saves_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'goalkeeper_saves_local'].shift(3))
        goalkeeper_saves_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'goalkeeper_saves_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'goalkeeper_saves_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'goalkeeper_saves_away'].shift(3))
        total_pass_local = np.mean(df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'total_pass_local'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'total_pass_local'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_local'] == id_equipo_local, 'total_pass_local'].shift(3))
        total_pass_away = np.mean(df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'total_pass_away'].shift(1) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'total_pass_away'].shift(2) +  \
                                    df_partidos.loc[df_partidos['id_equipo_visitante'] == id_equipo_visitante, 'total_pass_away'].shift(3))
        
        #Creo un dataframe de solo estadísticas más los datos de arbitro, estadio, ids, season (todos menos lesionados y titulares)
        df_datos_nuevos_final = pd.DataFrame({
                                        'id_equipo_local': id_equipo_local,
                                            'id_equipo_visitante': id_equipo_visitante,
                                            'arbitro': arbitro,
                                            'estadio': estadio,
                                            'season': season,
                                        'shots_on_goal_local':shots_on_goal_local,
                                        'shots_on_goal_away':shots_on_goal_away,
                                        'shots_off_goal_local':shots_off_goal_local,
                                        'shots_off_goal_away':shots_off_goal_away,
                                        'total_shots_local':total_shots_local,
                                        'total_shots_away':total_shots_away,
                                        'blocked_shots_local':blocked_shots_local,
                                        'blocked_shots_away':blocked_shots_away,
                                        'shots_insidebox_local':shots_insidebox_local,
                                        'shots_insidebox_away':shots_insidebox_away,
                                        'shots_outsidebox_local':shots_outsidebox_local,
                                        'shots_outsidebox_away':shots_outsidebox_away,
                                        'fouls_local':fouls_local,
                                        'fouls_away':fouls_away,
                                        'corners_local':corners_local,
                                        'corners_away':corners_away,
                                        'offsides_local':offsides_local,
                                        'offsides_away':offsides_away,
                                        'ball_possession_local':ball_possession_local,
                                        'ball_possession_away':ball_possession_away,
                                        'yellow_cards_local':yellow_cards_local,
                                        'yellow_cards_away':yellow_cards_away,
                                        'red_cards_local':red_cards_local,
                                        'red_cards_away':red_cards_away,
                                        'goalkeeper_saves_local':goalkeeper_saves_local,
                                        'goalkeeper_saves_away':goalkeeper_saves_away,
                                        'total_pass_local':total_pass_local,
                                        'total_pass_away':total_pass_away
                                        }, index = [0])
        
           
        #Aquí añado las nuevas variables que me parecieron interesantes siguiendo el mismo código que como las cree en el método anterior    
        df_datos_nuevos_final['goles_local_previos'] = df_partidos.groupby('id_equipo_local')['goles_local'].shift(1) + \
                                    df_partidos.groupby('id_equipo_local')['goles_local'].shift(2) + \
                                    df_partidos.groupby('id_equipo_local')['goles_local'].shift(3)

        df_datos_nuevos_final['tiros_local_previos'] = df_partidos.groupby('id_equipo_local')['total_shots_local'].shift(1) + \
                                        df_partidos.groupby('id_equipo_local')['total_shots_local'].shift(2) + \
                                        df_partidos.groupby('id_equipo_local')['total_shots_local'].shift(3)

        df_datos_nuevos_final['goles_local_previos'] = df_datos_nuevos_final['goles_local_previos'].fillna(df_partidos.groupby('id_equipo_local')['goles_local'].shift(-1))
        df_datos_nuevos_final['tiros_local_previos'] = df_datos_nuevos_final['tiros_local_previos'].fillna(df_partidos.groupby('id_equipo_local')['total_shots_local'].shift(-1))

        df_datos_nuevos_final['tiros_para_marcar_local'] = np.where(df_datos_nuevos_final['goles_local_previos'] == 0, 
                                                df_datos_nuevos_final['tiros_local_previos'], 
                                                df_datos_nuevos_final['tiros_local_previos'] / df_datos_nuevos_final['goles_local_previos'])

        df_datos_nuevos_final['goles_away_previos'] = df_partidos.groupby('id_equipo_visitante')['goles_visitante'].shift(1) + \
                                        df_partidos.groupby('id_equipo_visitante')['goles_visitante'].shift(2) + \
                                        df_partidos.groupby('id_equipo_visitante')['goles_visitante'].shift(3)

        df_datos_nuevos_final['tiros_away_previos'] = df_partidos.groupby('id_equipo_visitante')['total_shots_away'].shift(1) + \
                                        df_partidos.groupby('id_equipo_visitante')['total_shots_away'].shift(2) + \
                                        df_partidos.groupby('id_equipo_visitante')['total_shots_away'].shift(3)

        df_datos_nuevos_final['goles_away_previos'] = df_datos_nuevos_final['goles_away_previos'].fillna(df_partidos.groupby('id_equipo_visitante')['goles_visitante'].shift(-1))
        df_datos_nuevos_final['tiros_away_previos'] = df_datos_nuevos_final['tiros_away_previos'].fillna(df_partidos.groupby('id_equipo_visitante')['total_shots_away'].shift(-1))

        df_datos_nuevos_final['tiros_para_marcar_away'] = np.where(df_datos_nuevos_final['goles_away_previos'] == 0, 
                                                df_datos_nuevos_final['tiros_away_previos'], 
                                                df_datos_nuevos_final['tiros_away_previos'] / df_datos_nuevos_final['goles_away_previos'])

        df_datos_nuevos_final = df_datos_nuevos_final.drop(['tiros_away_previos','goles_away_previos','tiros_local_previos','goles_local_previos'], axis=1)

        df_datos_nuevos_final['tiros_para_marcar_local'] = df_datos_nuevos_final['tiros_para_marcar_local'].fillna(df_datos_nuevos_final['tiros_para_marcar_local'].mean())
        df_datos_nuevos_final['tiros_para_marcar_away'] = df_datos_nuevos_final['tiros_para_marcar_away'].fillna(df_datos_nuevos_final['tiros_para_marcar_away'].mean())

        df_datos_nuevos_final['odd_1'] = odd_1
        df_datos_nuevos_final['odd_x'] = odd_x
        df_datos_nuevos_final['odd_2'] = odd_2

        return df_datos_nuevos_final