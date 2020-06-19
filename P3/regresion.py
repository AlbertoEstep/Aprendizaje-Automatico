# Importamos las librerías necesarias
import numpy as np              # arrays
import pandas as pd             # dataframe
import matplotlib.pyplot as plt # visualizacion
import seaborn as sns           # visualizacion
import math                     # Para la cota y valores NaN
import plotly.graph_objects as go # visualizacion

from sklearn.preprocessing import StandardScaler # Escalado
from sklearn.pipeline import Pipeline            # Pipelines
from sklearn.model_selection import train_test_split # Separamos los datos

from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge  # Regresión lineal Ridge
from sklearn.linear_model import SGDRegressor # Regresión lineal SGD
from sklearn.svm import LinearSVR   # SVM regresor
from sklearn.model_selection import GridSearchCV    # Para cross-validation
import warnings                 # Para warnings

warnings.filterwarnings('ignore') # Para tratar los warnings
# Fijamos la semilla
np.random.seed(1)

print("--------------------------------------------------------------\n")
print("----------              REGRESIÓN                  -----------\n")
print("--------------------------------------------------------------\n")

# Create a switcher class that works for any estimator
# (codigo extraído de https://stackoverflow.com/questions/50285973/pipeline-multiple-classifiers
# y adaptado a nuestro problema)
class ClfSwitcher(BaseEstimator):
    def __init__(
        self,
        estimator = Ridge(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """

        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

# Calcula los valores perdidos del dataframe df
def valores_perdidos(df):
    total = df.isnull().sum().sort_values(ascending = False) # Columna recuento total
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending = False) # Columna porcentaje
    missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Porcentaje'])
    return missing_data

# Calcula la matriz de correlación
def matriz_correlaciones(datos):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = datos.corr(method = 'pearson')
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
               square=True, ax=ax, cbar_kws={'label': 'Correlación Pearson'})
    f.suptitle('Matriz Correlaciones')
    plt.show()

# Respuesta de Si o No
def quiere_visualizar():
    while True:
        respuesta = input().lower()
        if respuesta.startswith('s'):
          return True
        elif respuesta.startswith('n'):
          return False


print("Leyendo los datos.", end=" ", flush=True)
# Header
header = ['state', 'county', 'community', 'communityname', 'fold', 'population', 'householdsize',
          'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21',
          'agePct12t29', 'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 'medIncome',
          'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire',
          'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap',
          'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade',
          'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ',
          'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv',
          'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par',
          'PctWorkMomYoungKids', 'PctWorkMom', 'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent',
          'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5',
          'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', 'PctLargHouseFam',
          'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous',
          'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant',
          'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt',
          'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart',
          'RentLowQ', 'RentMedian', 'RentHighQ', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc',
          'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState',
          'PctSameHouse85', 'PctSameCity85', 'PctSameState85', 'LemasSwornFT', 'LemasSwFTPerPop',
          'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop',
          'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack',
          'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
          'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
          'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop',
          'ViolentCrimesPerPop']
        #ViolentCrimesPerPop: total number of violent crimes per 100K popuation - GOAL attribute (to be predicted)
df = pd.read_csv('datos/communities.data', names=header)

# Reemplazamos los valores ? por NaN en Python, para que funcione la función isnull()
df.replace({'?': math.nan}, inplace = True)
print("Lectura completada.\n")


# Estadísticas de los datos leídos
print("El número de instancias del dataset: {}".format(df.shape[0]))
print("El número de atributos del dataset: {}".format(df.shape[1]))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Visualización online de crímenes agrupados por estado normalizados\n")
# Agrupamos por estados los crímenes violentos y calculamos su media
crimedata_state = df.groupby('state').agg({'ViolentCrimesPerPop':'mean'})[['ViolentCrimesPerPop']].reset_index()

# Definimos los estados
codigo_estados = {1: 'AL', 2: 'AK', 3:'', 4:'AZ', 5:'AR', 6:'CA', 7:'', 8:'CO', 9:'CT', 10:'DE', 11:'DC',
            12:'FL', 13:'GA', 14:'', 15:'', 16:'ID', 17:'', 18:'IN', 19:'IA', 20:'KS', 21:'KY',
            22:'LA', 23:'ME', 24:'MD', 25:'MA', 26:'', 27:'MN', 28:'MS', 29:'MO', 30:'', 31:'',
            32:'NV', 33:'NH', 34:'NJ', 35:'NM', 36:'NY', 37:'NC', 38:'ND', 39:'OH', 40:'OK', 41:'OR',
            42:'PA', 43:'', 44:'RI', 45:'SC', 46:'SD', 47:'TN', 48:'TX', 49:'UT', 50:'VT', 51:'VA',
            52:'', 53:'WA', 54:'WV', 55:'WI', 56:'WY'}

array_codigo = np.array(list(codigo_estados.values()))

ViolentCrimesPerPop_porEstado = []
j = 0
for i in range (len(array_codigo)):
    if (array_codigo[i] == ''):
        ViolentCrimesPerPop_porEstado.append(None)
    else:
        ViolentCrimesPerPop_porEstado.append(crimedata_state['ViolentCrimesPerPop'][j])
        j = j + 1


fig = go.Figure(data=go.Choropleth(
    locations=array_codigo, # Spatial coordinates
    z = ViolentCrimesPerPop_porEstado, # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Ratio de crímenes violentos por estado",
    colorbar_titleside  = 'right',
    colorbar_ticks = 'outside'
))

fig.update_layout(
    title_text = '1990 US: Media de crímenes por condado agrupados por estado normalizados',
    geo_scope='usa', # limite map scope to USA
)

fig.show()

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

# Como el dataset incluía '?' para los valores perdidos, el tipo de la columna se ha mantenido
# Vamos a cambiar el tipo de dichas columnas, menos de 'communityname', que debe ser tipo cadena
columnas_objeto = df.select_dtypes(include=['object']).columns
columnas_objeto = columnas_objeto.delete(np.where(columnas_objeto == 'communityname'))
# Modificamos el tipo de las columnas con valores categóricos, interpretando los errores por NaN
df[columnas_objeto] = df[columnas_objeto].apply(pd.to_numeric, errors = 'coerce')
print("Tipo de los atributos: \n{}".format(df.dtypes.value_counts()))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")



missing_data = valores_perdidos(df)
print('Valores perdidos original:\n')
print(missing_data[missing_data['Total'] > 0])

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

# Vemos cual es el estado donde tenemos el valor perdido de OtherPerCap
estado = df[df['OtherPerCap'].isna()]['state'].values[0]
# Obtenemos todos los valores de OtherPerCap de dicho estado
valores_estado = df[df['state'] == estado]['OtherPerCap'].values
# Eliminamos el valor NaN de los valores obtenidos
valores_estado = valores_estado[~pd.isnull(valores_estado)]
valores_estado = valores_estado.astype(np.float)
# Calculamos la media de dichos valores
media = valores_estado.mean()
# Rellenamos el valor perdido con la media calculada
indice_fila = df[df['OtherPerCap'].isna()].index[0]
pd.set_option('mode.chained_assignment', None)
df.loc[indice_fila, 'OtherPerCap'] = media
# Comprobamos que se ha rellenado dicho valor:
missing_data = valores_perdidos(df)
print('Valores perdidos tras rellenar con la media el valor faltante de OtherPerCap:\n')
print(missing_data[missing_data['Total'] > 0])
columnas_eliminadas = missing_data[missing_data['Total'] > 0].index

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

# Eliminamos ahora todas las demas columnas con valores perdidos:
df.dropna(axis = 1, inplace = True)
# Comprobamos que se han eliminado:
missing_data = valores_perdidos(df)
print('Valores perdidos finales:\n')
print(missing_data[missing_data['Total'] > 0])

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

# Comprobamos ahora el número de atributos e instancias de nuestro dataset en este momento del preprocesamiento:
print("\nEl número de instancias en este momento es de: {}".format(df.shape[0]))
print("El número de atributos en este momento es de: {}".format(df.shape[1]))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

df.drop(['state', 'communityname', 'fold'], axis=1, inplace = True)
# Comprobamos ahora el número de atributos e instancias de nuestro dataset en este momento del preprocesamiento:
print("\nEl número de instancias tras eliminar los atributos no predictivos es de: {}".format(df.shape[0]))
print("El número de atributos tras eliminar los atributos no predictivos es de: {}".format(df.shape[1]))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Visualización de la matriz de correlaciones original\n")
matriz_correlaciones(df)

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

corr_matrix = df.corr(method = 'pearson').abs() # Matriz de correlación en valor absoluto

# Seleccionamos la matriz triangular superior de la matriz de correlación anterior
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Buscamos la columnas con una correlacion mayor que 0.95 con alguna otra
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print("¿Quiere visualizar las matrices de correlaciones ampliada de los atributos con correlación > 0.95? Responda S o N:\n")

if(quiere_visualizar()):
    print("Visualización de las matrices de correlaciones ampliada de los atributos con correlación > 0.95:\n")
    # Matriz de correlación
    k = 5 # Número de variables.
    for i in to_drop:
        cols = corr_matrix.nlargest(k, i)[i].index
        cm = np.abs(np.corrcoef(df[cols].values.T))
        ax = plt.axes()
        sns.set(font_scale = 1.25)
        hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10},
                yticklabels = cols.values, xticklabels = cols.values, cbar_kws={'label': 'Correlación Pearson'})
        ax.set_title('M. de corr. ampliada de {}'.format(i))
        plt.show()
        print("Visualización de la matriz de correlaciones ampliada de {}\n".format(i))
        input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")


df.drop(['numbUrban', 'medFamInc', 'whitePerCap', 'PctOccupMgmtProf',
             'TotalPctDiv', 'PctFam2Par', 'PctRecImmig8', 'PctRecImmig5',
             'PctRecImmig10', 'PctLargHouseFam', 'PctPersOwnOccup',
             'OwnOccLowQuart', 'OwnOccHiQuart', 'RentMedian', 'RentHighQ'], axis=1, inplace = True)
# Comprobamos ahora el número de atributos e instancias de nuestro dataset en este momento del preprocesamiento:
print("\nEl número de instancias tras la eliminación de atributos muy correlados es de: {}".format(df.shape[0]))
print("El número de atributos tras la eliminación de atributos muy correlados es de: {}".format(df.shape[1]))
input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Visualización de la matriz de correlaciones tras la eliminación de atributos muy correlados\n")
matriz_correlaciones(df)


input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Separamos los conjuntos de train y test\n")


# Separamos el dataset original en atributos y etiquetas:
X = df.iloc[:,:-1] # Todas las columnas menos la última
y = df.iloc[:, -1] # Última columna

# Dividimos los conjuntos en test (25 %) y train (75 %)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Estadísticas de los datos leídos
n_train = X_train.shape[0]
n_test = X_test.shape[0]
porcentaje_train = 100*n_train/(n_test+n_train)
porcentaje_test = 100*n_test/(n_test+n_train)
print("El número de instancias de entrenamiento es de: {}".format(X_train.shape[0]))
print("El número de instancias de test es de: {}".format(X_test.shape[0]))
print("Porcentaje de train: {} y porcentaje de test: {}".format(
    porcentaje_train, porcentaje_test))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")


preprocesado = [("escalado", StandardScaler())]

preprocesador = Pipeline(preprocesado)

# Cross-validation para elegir los hiperparámetros
pipeline = Pipeline(steps=[('preprocesador', preprocesador),
                      ('clf', ClfSwitcher())])

parameters = [
    {
        'clf__estimator': [Ridge(solver = 'sparse_cg', tol = 1e-7)],
        'clf__estimator__alpha': [10.0, 2.0, 1.0, 0.1, 0.01, 0.001]
    },
    {
        'clf__estimator': [SGDRegressor(loss="squared_loss",
                           penalty="l2",
                           tol=1e-7,
                           learning_rate = 'invscaling',
                           max_iter = 2000)],
        'clf__estimator__alpha': [10.0, 2.0, 1.0, 0.1, 0.01, 0.001],
        'clf__estimator__eta0': [1.0, 0.5, 0.2, 0.1, 0.05],
        'clf__estimator__power_t': [0.3, 0.25, 0.2]
    }
]

grid = GridSearchCV(pipeline, parameters, cv=5)
# Entrenamos el modelo
print("Entrenando el modelo.", end=" ", flush=True)
grid.fit(X_train, y_train)
print("Entrenamiento completado\n")
clasificador = grid.best_estimator_
E_val = 1 - grid.best_score_
E_in = 1 - clasificador.score(X_train, y_train)

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")


parametros = clasificador['clf'].get_params
print("Parámetros del clasificador: {}".format(parametros))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("R^2: {}".format(clasificador.score(X_train, y_train)))
print("E_in: {}".format(E_in))
print('E_val: {}'.format(E_val))
E_test = 1 - clasificador.score(X_test, y_test)
print("E_test: {}".format(E_test))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("-------------- Ampliación -------------------\n")
# Definimos el clasificador

svr = LinearSVR(dual = False, loss = 'squared_epsilon_insensitive', tol = 1e-7)

svr_pipe = Pipeline(steps=[('clf', svr)])

params_svr = {'clf__C': [10.0, 2.0, 1.0, 0.1, 0.01, 0.001],
              'clf__epsilon': [0.0, 0.01, 0.05, 0.1]}

grid_svr = GridSearchCV(svr_pipe, params_svr, cv=5) # Cross-validation para elegir hiperparámetros
print("Entrenando el modelo SVM.", end=" ", flush=True)
grid_svr.fit(X_train, y_train)
print("Entrenamiento completado\n")
clasificador_svr = grid_svr.best_estimator_
E_val_svr = 1 - grid_svr.best_score_
E_in_svr = 1 - clasificador_svr.score(X_train, y_train)

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

parametros_svr = clasificador_svr['clf'].get_params
print("Parámetros del clasificador SVM: {}".format(parametros_svr))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("E_in SVM: {}".format(E_in_svr))
print('E_val SMV: {}'.format(E_val_svr))
E_test_svr = 1 - clasificador_svr.score(X_test, y_test)
print("E_test SMV: {}".format(E_test_svr))

input("\nFin. Pulse 'Enter' para terminar\n\n\n")
