# IMPORTAR LIBRERIAS
import os
import pickle

#Load and manipulate data
import pandas as pd
from pandas_summary import DataFrameSummary

#Calculate with data (mean, std)
import numpy as np

#ML PREPROCESSING _ ANALISIS DE DATOS PREDICTIVO
# scaling
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#SPLIT TRAIN-TEST
from sklearn.model_selection import train_test_split

#ML PROCESSING
## MODELS
### classification_other
from sklearn.tree import DecisionTreeClassifier
### classification_ensambles
from sklearn.ensemble import RandomForestClassifier

## OPTIMIZATIONS (proceso/resultado): CROSS VALIDATION, GRIDSEARCH
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

## CLASSIFICATION_EVALUATION
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

### classification_evaluation_models_visualization 
from sklearn.metrics import plot_confusion_matrix

# IMPORTAR FUNCIONES
from utils.funciones import train_model, model_predictions, save_model

# IMPORTAR DATASETS + CREAR 1 DATASET
print("**********IMPORTAR DATOS EN DATAFRAME************")
## importar dataset 1 & 2
#************************with open(os.path.join(os.path.dirname(__file__), 'model\\new_model2'), "wb") as archivo_out:
data1 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data\\raw\\dataset_names_data_train.txt'), names = ("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "result"))
data2 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data\\raw\\dataset_names_data_train.txt'), names = ("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "result"))

## añadir columna extra con identificación dataset original
data1["origin"] = "dataset1_train"
data2["origin"] = "dataset2_test"
## juntar 2 datasets en 1 - indice nuevo para dataset juntado
data_concat = pd.concat([data1,data2], axis=0, ignore_index = True)
## renombrar dataset
data = data_concat 
print()

# TRATAR DATOS 
print("**************LIMPIAR DATOS**********************")
## limpiar target: de 4 > 2 valores + transformar catg > num
data["result"] = data["result"].str.replace(".","")
data["result_01"] = np.where(data["result"]==" <=50K",0,1)

## extraer columnas num potencialmente relevantes 
datadef_parte1 = data.loc[:,["age","education_num","hours_per_week","capital_gain", "result_01"]]
## extraer columnas catg potencialmente relevantes + convertir catg > num
datadef_parte2 = data.loc[:,['workclass', 'education','marital_status', 'occupation','relationship','race','sex']]

## fusión parte1 + parte2 > 1 dataframe
datadef = pd.merge(datadef_parte1, datadef_parte2, left_index=True, right_index = True)

## 'drop na': eliminar columnas con valor " ?" + indización nueva
datadef_rowslimp = datadef.drop(datadef[(datadef["workclass"] == " ?") | (datadef["occupation"] == " ?")].index)
datadef_rowslimp = datadef_rowslimp.reset_index(drop=True)

## 'drop dupl': eliminar columna catg con valores distintivos equivalentes a otra (numérica)
datadef_rowscolslimp = datadef_rowslimp.drop("education", axis=1)

## 'drop' otras columnas no relavantes
data_antestransform = datadef_rowscolslimp
data_antestransform = data_antestransform.drop(columns=["marital_status","workclass","occupation","race"])

## encoding: convertir columnas catg relevantes > num
data_transform1 = data_antestransform
data_transform1["sex"] = LabelEncoder().fit_transform(data_antestransform["sex"])

## disminuir nº de valores de features catg de cara a su conversión > num:
data_transform2 = data_transform1
data_transform2["relationship"] = data_transform2["relationship"].replace([" Husband"," Wife"], "Spouse")
## conversion1 catg > num mediante OneHotEncoder
encoder = OneHotEncoder()
encoder_data_transform2 = pd.DataFrame(encoder.fit_transform(data_transform2[["relationship"]]).toarray())
data_antestransform_encod_rel = data_transform2.join(encoder_data_transform2)
data_antestransform_encod_rel = data_antestransform_encod_rel.rename(columns={0:"NotinFam",1:"OtherRelative",2:"Own_Child",3:"Unmarr",4:"Spouse"})
data_antestransform_encod_rel = data_antestransform_encod_rel.drop("relationship", axis=1)

## limpieza layout: 'drop' y desplaza columna target al final
finaldata = data_antestransform_encod_rel.drop(columns = ["result_01"], axis = 1)   #,"relationship","marital_status"], axis=1)
finaldata = pd.merge(finaldata, data_antestransform_encod_rel["result_01"],left_index=True, right_index=True)
finaldata = finaldata.rename(columns={"NotinFam":"REL_NotInFam", "OtherRelative":"REL_OtherRelat", "Own_Child":"REL_OwnChild","Unmarr":"REL_Unmarr","Spouse":"REL_Spouse"})

## reducir volumen dataset final:
for col in finaldata.columns:
    finaldata[col] = finaldata[col].astype(int)

## renombrar dataset tratado:
data_proc= finaldata
print(data_proc.sample(4))

## guardar dataset tratado
#data_proc.to_csv("C:\\Users\\piovr\\Documents\\BOOTCAMP\\Alumno\\3-Machine_Learning\\Entregas\\E3\\src\\data\\processed\\data_proc_from_py.csv", index=False)
#(os.path.join(os.path.dirname(__file__), 'model\\new_model')
data_proc.to_csv(os.path.join(os.path.dirname(__file__), 'data\\processed\\dataset_to_start_ML_from_py.csv'))

## SPLIT FEATURES(X) vs. TARGET(y) con garantía de proporción idéntica de los valores target
X = data_proc.drop(columns = ["result_01"], axis =1)
y = data_proc["result_01"]

## SPLIT TRAIN vs. TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify = y)  
print()

# ENTRENAR MODELOS
print("************ENTRENAR MODELOS*******************")

print("---modelo 1 (mejor modelo): RandomForestClassifier---")
## instanciar modelo y parametros 
model = RandomForestClassifier(class_weight="balanced")
params = [{"n_estimators": [25,50,75,100,150,200], "max_depth": [2,3,4,5,6,8]}]
## aplicar gridsearch 
gridsearch_rf = GridSearchCV(model, params, cv=5, scoring="recall",n_jobs=-1) 
## entrenar parte train                                            
gridsearch_rf.fit(X_train,y_train)       
## predicciones sobre test
y_pred = gridsearch_rf.predict(X_test)
## mejores parametros del modelo
print(gridsearch_rf.best_params_)
print(gridsearch_rf.best_estimator_)
## scores del modelo con mejor resultado recall:
print("RandomForest: resultados del entrenamiento sobre el grupo test")
print("RECALL SCORE:",recall_score(y_test,y_pred))
recall_scoreRFMEJOR = recall_score(y_test,y_pred)
print("F1_score:", f1_score(y_test,y_pred))
print("roc_auc_score:", roc_auc_score(y_test,y_pred))
print("Classification Report:", classification_report(y_test,y_pred))
print()

print("---modelo 2: RandomForest via pipeline y gridsearch----")
## instanciar modelo y parámetros mediante Pipeline
pl_rf_clas = Pipeline(steps = [("mod_rf_clas", RandomForestClassifier())])
params_mod_rf_clas = {"mod_rf_clas__n_estimators":[15,30,45,60], "mod_rf_clas__min_samples_leaf":[10,20,50], "mod_rf_clas__class_weight": [None, "balanced"], "mod_rf_clas__n_jobs": [-1]}
gs_mod_rf_clas = RandomizedSearchCV(pl_rf_clas,
                         params_mod_rf_clas,
                         cv = 8,
                         scoring = 'recall',
                         verbose = 1)
## entrenar parte train
gs_mod_rf_clas.fit(X_train,y_train)
## predicciones sobre test
y_pred = gs_mod_rf_clas.predict(X_test)
## scores del modelo con mejor restultado recall:
print("RandomForest: resultados del entrenamiento sobre el grupo test")
print("RECALL SCORE:",recall_score(y_test,y_pred))
print("f1_score:", f1_score(y_test,y_pred))
print("roc_auc_score:", roc_auc_score(y_test,y_pred))
print("classification report:", classification_report(y_test,y_pred))

#GUARDAR MEJOR MODELO
with open(os.path.join(os.path.dirname(__file__), 'model\\new_model'), "wb") as archivo_out:
    pickle.dump(gridsearch_rf, archivo_out)

print()
print("El mejor modelo ha sido ", gridsearch_rf.best_estimator_ , "con un score de recall de ", round(recall_scoreRFMEJOR,4),".")
print()
print("Script finished!")