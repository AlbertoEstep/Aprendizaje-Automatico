# -*- coding: utf-8 -*-

# AA - Práctica 0 - UGR
# Author: Alberto Estepa Fernández
# Date: 18/02/2020

from sklearn import datasets
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
clases = np.unique(y)

# Recorrer las clases y las pintamos con un color diferente
for clase in clases:
    index = np.where(y == clase)
    plt.scatter(x[index, 0], x[index, 1], c=colores[clase], label='Clase {}'.format(clase))

# Incluimos los nombres en los ejes, el título del grafico y la leyenda, y la pintamos
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


#COPIADO
# Dividir la muestra proporcionalmente según el ratio
def stratify_sample(in_vec, labels, ratio=0.8):
    sample = np.c_[in_vec, labels]               # Juntar por columnas vec. características y etiquetas en una matriz
    group_set = np.unique(y)                     # Grupos únicos que existen
    np.random.shuffle(sample)                    # Mezclar la muestra para distribuirla aleatoriamente

    # Listas donde se guardarán las selecciones
    train_list = []
    test_list = []

    # De la muestra mezclada, escoger los n primeros elementos de cada grupo
    # y juntarlos en la lista de entrenamiento, el resto en la de test
    # Cada grupo de elementos  escogidos es una lista, por tanto
    # se tienen que combinar luego
    # n = num_elementos_grupo * ratio
    for group in group_set:
        elem_group = sample[sample[:, -1] == group]
        n_elem = elem_group.shape[0]
        n_selected_elem = int(n_elem * ratio)

        train_list.append(elem_group[:n_selected_elem, :])
        test_list.append(elem_group[n_selected_elem:, :])

    # Juntar las sub-listas en una única matriz
    training = np.concatenate(train_list)
    test = np.concatenate(test_list)

    # Volver a mezclar las muestras para distribuirlas aleatoriamente
    np.random.shuffle(training)
    np.random.shuffle(test)

    return training, test

training, test = stratify_sample(x_last_cols, y)

# Separar las muestras en x e y
training_x = training[:, :-1]
training_y = training[:, -1]
test_x = test[:, :-1]
test_y = test[:, -1]

# Mostrar información sobre la muestra de entrenamiento
# Mostrar número de elementos de cada grupo, el tamaño y la muestra de dos formas:
# - x,y juntos
# - x,y por separado
print('Number of Group 0 elements in training sample: {}'.format(np.count_nonzero(training[:, -1] == 0)))
print('Number of Group 1 elements in training sample: {}'.format(np.count_nonzero(training[:, -1] == 1)))
print('Number of Group 2 elements in training sample: {}'.format(np.count_nonzero(training[:, -1] == 2)))
print('Size of training sample: {}'.format(training.shape))
print('Training sample:\n{}'.format(training))
print('Training sample x:\n{}'.format(training_x))
print('Training sample y:\n{}'.format(training_y))

# Mostrar información sobre la muestra de pruebas
# Mostrar número de elementos de cada grupo, el tamaño y la muestra de dos formas:
# - x,y juntos
# - x,y por separado
print('Number of Group 0 elements in test sample: {}'.format(np.count_nonzero(test[:, -1] == 0)))
print('Number of Group 1 elements in test sample: {}'.format(np.count_nonzero(test[:, -1] == 1)))
print('Number of Group 2 elements in test sample: {}'.format(np.count_nonzero(test[:, -1] == 2)))
print('Size of test sample: {}'.format(test.shape))
print('Test sample:\n{}'.format(test))
print('Test sample x:\n{}'.format(test_x))
print('Test sample y:\n{}'.format(test_y))
