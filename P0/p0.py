# -*- coding: utf-8 -*-

# AA - Práctica 0 - UGR
# Author: Alberto Estepa Fernández
# Date: 18/02/2020

from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# ---------------------------- PARTE 1 -----------------------------------------
# ------------------------------------------------------------------------------

# Leer la base de datos de iris que hay en scikit-learn.
iris = datasets.load_iris()

# Obtener las características (datos de entrada X) y la clase (y). Quedarse con
# las dos últimas características (2 últimas columnas de X).
x = iris.data[:, -2:]
y = iris.target

# Visualizar con un Scatter Plot los datos, coloreando cada clase con un color
# diferente (con rojo, verde y azul), e indicando con una leyenda la clase a la
# que corresponde cada color.

# Diccionario de colores {key = número de la clase, value = color de la clase}
colores = {0: 'red', 1: 'green', 2: 'blue'}
nombres = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginia'}
clases = np.unique(y)

# Recorrer las clases y las pintamos con un color diferente
for clase in clases:
    index = np.where(y == clase)
    plt.scatter(x[index, 0], x[index, 1], c=colores[clase], label=nombres[clase])

# Incluimos los nombres en los ejes, el título del gráfico y la leyenda, y la pintamos
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Iris dataset classification')
plt.legend()
plt.show()


# ------------------------------------------------------------------------------
# ---------------------------- PARTE 2 -----------------------------------------
# ------------------------------------------------------------------------------

# Separar en training (80 % de los datos) y test (20 %) aleatoriamente conservando
# la proporción de elementos en cada clase tanto en training como en test. Con
# esto se pretende evitar que haya clases infra-representadas en entrenamiento o
# test. Nota: el comando zip puede ser de utilidad en este apartado.

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in sss.split(x, y):
    training_x, test_x = x[train_index], x[test_index]
    training_y, test_y = y[train_index], y[test_index]
print(training_x)
print(training_y)
print(test_x)
print(test_y)


# ------------------------------------------------------------------------------
# ---------------------------- PARTE 3 -----------------------------------------
# ------------------------------------------------------------------------------

# Obtener 100 valores equiespaciados entre 0 y 2π.
z = np.linspace(0, 2*np.pi, 100)
print(z)

# Obtener el valor de sin(x), cos(x) y sin(x) + cos(x) para los 100 valores
# anteriormente calculados.
sin_z = np.sin(z)
print(sin_z)
cos_z = np.cos(z)
print(cos_z)
suma = sin_z + cos_z
print(suma)

# Visualizar las tres curvas simultáneamente en el mismo plot (con líneas
# discontinuas en negro, azul y rojo).

plt.plot(z, sin_z, color="black", dashes=[6, 2], label="seno")
plt.plot(z, cos_z, color="blue", dashes=[6, 2], label="coseno")
plt.plot(z, suma, color="red", dashes=[6, 2], label="seno + coseno")

plt.legend(loc='lower left')
plt.show()
