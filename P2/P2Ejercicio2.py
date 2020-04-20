# -*- coding: utf-8 -*-

# AA - Práctica 2 - UGR
# Author: Alberto Estepa Fernández
# Date: 25/04/2020

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle # Obtener estado semilla


#----------------------------------------------------------------------------#
#----------------------------- Modelos lineales -----------------------------#
#----------------------------------------------------------------------------#

# Aclaramos que hemos usado la plantilla proporcionada por el profesor del
# grupo 1 de prácticas.

##############################################################################
# EJERCICIO 1: Necesario para obtener los datos para este nuevo ejercicio

# Función que calcula una lista de N vectores de dimensión dim. Cada vector
# contiene dim números aleatorios uniformes en el intervalo rango.
def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

# Función que simula de forma aleatoria los parámetros, v = (a, b) de una
# recta, y = ax + b, que corta al cuadrado [−Intervalo[0], Intervalo[1]] ×
# [−Intervalo[0], Intervalo[1]].
def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.

    return a, b

def funcion_signo(x):
	if x >= 0:
		return 1
	else:
		return -1

def asigna_etiquetas(x, y, a, b):
	return funcion_signo(y - a*x - b)

def ejercio_anterior():
    # Obtengo el estado de la semilla en el fichero
    with open('./estado_semilla.dat', 'rb') as fich:
        estado = pickle.load(fich)

    np.random.set_state(estado)
    N = 100
    dim = 2
    rango = [-50, 50]
    color = {1: 'b', -1: 'g'}

    x = simula_unif(N = N, dim = dim, rango = rango)
    a, b = simula_recta(rango)
    y = []
    for punto in x:
        y.append(asigna_etiquetas(punto[0], punto[1], a, b))
    y = np.array(y)
    etiquetas = np.unique(y)

    # Pintar puntos
    for etiqueta in etiquetas:
        indice = np.where(y == etiqueta)
        plt.scatter(x[indice, 0], x[indice, 1], c=color[etiqueta],
    		label='Etiqueta {}'.format(etiqueta))
    # Pintar recta
    puntos = np.array([np.min(x[:, 0]), np.max(x[:, 0])])
    plt.plot(puntos, a * puntos + b, c='r', label='Recta de simulación')
    titulo = "Nube de 100 puntos con distribucción uniforme y recta de " + \
    				"simulación"
    plt.title(titulo)
    plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 2A')
    plt.xlabel('Eje $x_1$')
    plt.ylabel('Eje $x_2$')
    plt.legend()
    plt.show()
    return x, y
##############################################################################

##############################################################################
##############################################################################
##############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

# Algoritmo PLA
def adjust_PLA(datos, label, max_iter, vini):
    w = np.copy(vini)
    continuar = True
    epoca = 0
    while continuar:
        continuar = False
        epoca += 1
        for x, y in zip(datos, label):
            y_predicha = funcion_signo(w.dot(x.reshape(-1, 1)))
            if y_predicha != y:
                w += y * x
                continuar = True
        if epoca == max_iter:
            break

    return w, epoca

def perceptron():
    x, y = ejercio_anterior()
    # Crear el conjunto de datos añadiendo una columna con unos a los x
    data = np.c_[np.ones((x.shape[0], 1), dtype=np.float64), x]
    # Crear array de zeros
    zeros = np.array([0.0, 0.0, 0.0])
    print('Algoritmo PLA con w_0 = [0.0, 0.0, 0.0]\n')
    # Lanzar el algoritmo PLA con w = [0, 0, 0] y guardar la información
    w, iter = adjust_PLA(data, y, 10000, zeros)
    print('Valor w: {} \tNum. iteraciones: {}'.format(w, iter))
    input("\n--- Pulsar tecla para continuar ---\n")

    # Random initializations
    iterations = []
    for i in range(0,10):
        #CODIGO DEL ESTUDIANTE
        True

    print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))
    input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def sgdRL():
    #CODIGO DEL ESTUDIANTE
    return w



#CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")



# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")








##############################################################################
###########                                                     ##############
###########                 Funcion principal                   ##############
###########                                                     ##############

# Función principal del programa
def ejercicio2():
	perceptron()

###########                                                     ##############
##############################################################################

if __name__ == "__main__":
    ejercicio2()














###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM

#CODIGO DEL ESTUDIANTE




input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
