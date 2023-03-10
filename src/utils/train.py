import numpy as np
import os
import pandas as pd
import pickle
import xgboost as xgb
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


class train_model():
    def __init__(self):
        pass

    def train_xgbc(self,df):
        #Dividimos en los datos de entrenamiento y la clasificación de los datos de entrenamiento que usaremos para entrenar el modelo
        X = df.drop(['index','fixture_id','resultado', 'goles_local', 'goles_visitante','goles_descanso_local','goles_descanso_visitante','fecha_timestamp' ], axis=1)
        y = df['resultado']

        # Pipeline para codificar la columna 'arbitro' con OneHotEncoder
        arbitro_pipeline = Pipeline([
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])

        # Pipeline para codificar la columna 'estadio' con TargetEncoder
        estadio_pipeline = Pipeline([
            ('target', TargetEncoder())
        ])

        # ColumnTransformer para aplicar los pipelines a las columnas correspondientes
        preprocessor = ColumnTransformer([
            ('arbitro', arbitro_pipeline, ['arbitro']),
            ('estadio', estadio_pipeline, ['estadio']),
            ], remainder = "passthrough")

        # Pipeline final con el preprocesamiento y el modelo RandomForestClassifier
        pipeline_xgb = Pipeline([
            ('preprocessor', preprocessor),
            ('pca', PCA()),
            ('xgb', xgb.XGBClassifier())
        ])

        xgb_param = {
        'pca__n_components': [25,30,35],
        'xgb__n_estimators': [300, 500, 700],
        'xgb__learning_rate': [0.1],
        'xgb__max_depth': [27,25],
        'xgb__subsample': [0.5, 0.8],
        'xgb__colsample_bytree': [0.5, 0.6],
        'xgb__min_child_weight': [1, 2],
        'xgb__gamma': [0]
        }

        gs_xgb = GridSearchCV(
                                pipeline_xgb,
                                xgb_param,
                                cv=3,
                                scoring="accuracy",
                                verbose=1,
                                n_jobs=-1
                            )
        
        modelo = gs_xgb.fit(X, y)

        with open(os.path.join('model','football_predictor.pkl'), 'wb') as file:
            pickle.dump(modelo, file)

        return "Modelo entrenado con éxito y guardado en 'football_predictor.pkl'."


    def importar_modelo(self, ruta_modelo):
        with open(ruta_modelo, 'rb') as archivo:
            gs_xgb = pickle.load(archivo)
        return gs_xgb


    def prediccion_modelo(self, modelo, datos_nuevos):
        resultado = modelo.predict(datos_nuevos)
        probabilidades = modelo.predict_proba(datos_nuevos)
        return print(f'El resultado del partido será {resultado[0]}. Las probabilidades son de X - {probabilidades[0][0]*100}%, 1 - {probabilidades[0][1]*100} y 2 - {probabilidades[0][2]*100}')
