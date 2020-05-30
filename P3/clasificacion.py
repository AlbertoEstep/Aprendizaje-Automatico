# Importamos las librerías necesarias
import numpy as np              # arrays
import pandas as pd             # dataframe
import matplotlib.pyplot as plt # visualizacion
import seaborn as sns           # visualizacion
import math                     # Para la cota

from sklearn.preprocessing import StandardScaler # Escalado
from sklearn.decomposition import PCA            # Reducimos dimensionalidad
from sklearn.pipeline import Pipeline            # Pipelines

from sklearn.linear_model import LogisticRegression # Regresion logística
from sklearn.svm import LinearSVC # SVM lineal para clasificación
from sklearn.model_selection import GridSearchCV    # Para cross-validation
import warnings                 # Para warnings

from sklearn.metrics import confusion_matrix    # Matriz de confusion
from sklearn.metrics import plot_confusion_matrix # visualizacion

warnings.filterwarnings('ignore') # Para tratar los warnings
# Fijamos la semilla
np.random.seed(1)

print("--------------------------------------------------------------\n")
print("----------              CLASIFICACIÓN              -----------\n")
print("--------------------------------------------------------------\n")

# Función para leer los datos
def leerDatos(file):
    datos = pd.read_csv(file, header=None)
    X = datos.iloc[:,:-1] # Todas las columnas menos la última
    y = datos.iloc[:, -1] # Última columna
    return X, y

def parseBooleano(boleano):
    if (boleano == True):
        return 'Sí'
    return 'No'

# Calcula la matriz de correlación
def matriz_correlaciones(datos):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = datos.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
               square=True, ax=ax, cbar_kws={'label': 'Correlación Pearson'})
    plt.xlabel('Características')
    plt.ylabel('Características')
    f.suptitle('Matriz Correlaciones')
    plt.show()

# Calcula la matriz de correlación
def matriz_correlaciones_procesados(datos):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = np.corrcoef(datos.T)
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
           square=True, ax=ax, cbar_kws={'label': 'Correlación Pearson'})
    plt.xlabel('Características')
    plt.ylabel('Características')
    f.suptitle('Matriz Correlaciones')
    plt.show()

# Estima la cota E_out
def cota_Eout(E_test, delta):
    N = X_test.shape[0]
    cota = E_test + math.sqrt((1/(2*N))*math.log(2/delta))
    return cota

print("Leyendo los datos.", end=" ", flush=True)
# Lectura de los datos de entrenamiento
X_train, y_train = leerDatos('datos/optdigits.tra')
# Lectura de los datos para el test
X_test, y_test = leerDatos('datos/optdigits.tes')
# Convertimos en dataframe
y_train_df = pd.DataFrame(data=y_train)
y_test_df = pd.DataFrame(data=y_test)
# Cambiamos el nombre de las columnas
X_train = X_train.add_prefix('Característica ')
X_test = X_test.add_prefix('Característica ')
y_train_df.rename(columns={64:'Dígito'}, inplace=True)
y_test_df.rename(columns={64:'Dígito'}, inplace=True)
print("Lectura completada.\n")

# Estadísticas de los datos leídos
n_train = X_train.shape[0]
n_test = X_test.shape[0]
porcentaje_train = 100*n_train/(n_test+n_train)
porcentaje_test = 100*n_test/(n_test+n_train)
print("El número de instancias de entrenamiento es de: {}".format(X_train.shape[0]))
print("El número de instancias de test es de: {}".format(X_test.shape[0]))
print("Porcentaje de train: {}, porcentaje de test: {}".format(porcentaje_train, porcentaje_test))
input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

# Aunque los datos están preprocesados y esperamos no tener valores perdidos, lo comprobamos:
print("Valores perdidos en el dataset de entrenamiento: {}".format(X_train.isnull().sum().sum()
                                                                    + y_train_df.isnull().sum().sum()))
print("Valores perdidos en el dataset de test: {}".format(X_test.isnull().sum().sum()
                                                                    + y_test_df.isnull().sum().sum()))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

# Comprobación de outliers: 'Comprobamos que los valores sean enteros y estén entre 0 y 16'
print("Todos los valores de entrenamiento son enteros: {}".format(
        parseBooleano(X_train.dtypes.unique() == type(0) and y_train_df.dtypes.unique() == type(0))))
print("Todos los valores de test son enteros: {}".format(
        parseBooleano(X_test.dtypes.unique() == type(0) and y_test_df.dtypes.unique() == type(0))))
print("Todos los valores de las características de entrenamiento pertenecen al intervalo: [{},{}]".format(
        X_train.values.min(), X_train.values.max()))
print("Todos los valores de las características de test pertenecen al intervalo: [{},{}]".format(
        X_test.values.min(), X_test.values.max()))
print("Todos los valores de las etiquetas de entrenamiento pertenecen al intervalo: [{},{}]".format(
        y_train_df.values.min(), y_train_df.values.max()))
print("Todos los valores de las características de test pertenecen al intervalo: [{},{}]".format(
        y_test_df.values.min(), y_test_df.values.max()))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

# Instancias por clase'
print("Dataset de entrenamiento")
for i in range (len(y_train_df['Dígito'].unique())):
    print("\t {} instancias del dígito {}".format(y_train_df['Dígito'].value_counts()[i],i))
print("\n")
print("Dataset de test")
for i in range (len(y_train_df['Dígito'].unique())):
    print("\t {} instancias del dígito {}".format(y_test_df['Dígito'].value_counts()[i],i))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Visualización de la matriz de correlaciones original:\n")
matriz_correlaciones(X_train)


input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

# Creamos el preprocesador
preprocesado = [("escalado", StandardScaler()),
                ("PCA", PCA(n_components=0.98))]

preprocesador = Pipeline(preprocesado)

datos_preprocesados = preprocesador.fit_transform(X_train)
print("Visualización de la matriz de correlaciones al aplicar el preprocesado:\n")
matriz_correlaciones_procesados(preprocesador.fit_transform(X_train))


input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")


# Convertimos en dataframe
X_train_procesados = pd.DataFrame(data=datos_preprocesados)

# Cambiamos el nombre de las columnas
X_train_procesados = X_train_procesados.add_prefix('Característica ')
# Estadísticas de los datos procesados
n_características = X_train_procesados.shape[1]
print("El número de características tras aplicar PAC es de: {}".format(n_características))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Entrenando el modelo.", end=" ", flush=True)
log = LogisticRegression(penalty='l2', # Regularización Ridge (L2)
                                    multi_class='multinomial', # Indicamos que la regresión logística es multinomial
                                    solver = 'lbfgs', # Algoritmo a utilizar en el problema de optimización, aunque es
                                                            # el dado por defecto
                                    max_iter = 1000)

log_pipe = Pipeline(steps=[('preprocesador', preprocesador),
                      ('clf', log)])

params_log = {'clf__C': [2.0, 1.0, 0.1, 0.01, 0.001]}

grid = GridSearchCV(log_pipe, params_log, scoring='accuracy', cv=5) # Cross-validation para elegir hiperparámetros
grid.fit(X_train, y_train)
print("Entrenamiento completado\n")

clasificador = grid.best_estimator_
E_val = 1 - grid.best_score_
E_in = 1 - clasificador.score(X_train, y_train)

parametros = clasificador['clf'].get_params
print("Parámetros del clasificador: {}".format(parametros))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Visualización de la matriz de confusión:\n")
# Predecimos los valores de salida del test
y_predict = clasificador.predict(X_test)
m = confusion_matrix(y_test, y_predict)
fig, ax = plt.subplots(figsize=(10, 8))
ax.set(title="Matriz de confusión")
plot_confusion_matrix(clasificador, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 ax = ax,
                                 values_format='.3g')
plt.show()


input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Visualización de la matriz de confusión normalizada:\n")

fig, ax = plt.subplots(figsize=(10, 8))
ax.set(title="Matriz de confusión normalizada")
plot_confusion_matrix(clasificador, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 ax = ax)
plt.show()


input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")


print("E_in: {}".format(E_in))
print('E_val: {}'.format(E_val))
E_test = 1 - clasificador.score(X_test, y_test)
print("E_test: {}".format(E_test))
print("E_out: {}".format(cota_Eout(E_test, delta = 0.05)))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("-------------- Ampliación -------------------\n")

np.random.seed(1)

# Definimos el clasificador
svc = LinearSVC(penalty='l2', # Regularización Ridge (L2)
                                    multi_class='crammer_singer',
                                    loss = 'hinge')

svc_pipe = Pipeline(steps=[('preprocesador', preprocesador),
                      ('clf', svc)])

params_svc = {'clf__C': [2.0, 1.0, 0.1, 0.01, 0.001]}

grid = GridSearchCV(svc_pipe, params_svc, scoring='accuracy', cv=5) # Cross-validation para elegir hiperparámetros
print("Entrenando el modelo SVM.", end=" ", flush=True)
grid.fit(X_train, y_train)
print("Entrenamiento completado\n")
clasificador = grid.best_estimator_
E_val = 1 - grid.best_score_
E_in = 1 - clasificador.score(X_train, y_train)

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

parametros = clasificador['clf'].get_params
print("Parámetros del clasificador SVM: {}".format(parametros))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")


print("E_in SVM: {}".format(E_in))
print('E_val SVM: {}'.format(E_val))
E_test_svr = 1 - clasificador.score(X_test, y_test)
print("E_test SVM: {}".format(E_test_svr))
print("E_out estimado con E_test de SMV: {}".format(cota_Eout(E_test_svr, delta = 0.05)))

input("\nFin. Pulse 'Enter' para terminar\n\n\n")
