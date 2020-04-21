# -*- coding: utf-8 -*-

# AA - Práctica 2 - UGR
# Author: Alberto Estepa Fernández
# Date: 25/04/2020

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


##############################################################################
##############################################################################
##############################################################################
#BONUS: Clasificación de Dígitos

# Fijamos la semilla
np.random.seed(3)

def funcion_signo(x):
	if x >= 0:
		return 1
	else:
		return -1

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

def muestra_datos():
    print ('\nEJERCICIO SOBRE CLASIFICACIÓN DE DÍGITOS\n')
    print ('-------------------- Bonus ---------------------------------\n')
    # Lectura de los datos de entrenamiento
    x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
    # Lectura de los datos para el test
    x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8],
                                [-1,1])


    # Mostramos los datos
    fig, ax = plt.subplots()
    ax.plot(np.squeeze(x[np.where(y == -1),1]),
                np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
    ax.plot(np.squeeze(x[np.where(y == 1),1]),
                np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
    ax.set(xlabel='Intensidad promedio', ylabel='Simetria',
                title='Digitos Manuscritos (TRAINING)')
    ax.set_xlim((0, 1))
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]),
                np.squeeze(x_test[np.where(y_test == -1),2]),
                'o', color='red', label='4')
    ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]),
                np.squeeze(x_test[np.where(y_test == 1),2]),
                'o', color='blue', label='8')
    ax.set(xlabel='Intensidad promedio', ylabel='Simetria',
                title='Digitos Manuscritos (TEST)')
    ax.set_xlim((0, 1))
    plt.legend()
    plt.show()

    input("\n--- Pulsar 'Enter' para continuar ---\n")
    return x, y, x_test, y_test

#LINEAR REGRESSION FOR CLASSIFICATION Práctica 1

def pseudoinverse(x, y):
	# Aplicamos el algoritmo paso a paso
    x_traspuesta = x.transpose()
    y_traspuesta = y.reshape(-1, 1)
    w = np.linalg.inv(x_traspuesta.dot(x))
    w = w.dot(x_traspuesta)
    w = w.dot(y_traspuesta)
    return w

def Err(datos, labels, w):
    #print(w)
    recta = lambda x: w[0]*x[:,0] + w[1]*x[:, 1] + w[2]*x[:, 2]
    signos = labels*recta(datos)
    aciertos = 100*len(signos[signos >= 0])/len(labels)
    #print(aciertos)
    return 100 - aciertos


def PLAPocket(datos, labels, max_iter, vini):
    w = vini.copy()
    mejor_w = w.copy()
    error_minimo = Err(datos, labels, mejor_w)

    for iteracion in range(1, max_iter + 1):
        w_antiguo = w.copy()
        for dato, etiqueta in zip(datos, labels):
            if funcion_signo(w.dot(dato)) != etiqueta:
                w += etiqueta * dato
        error_actual = Err(datos, labels, w)
        print(error_actual, error_minimo)
        if error_actual < error_minimo:
            mejor_w = w.copy()
            error_minimo = error_actual

        if np.all(w == w_antiguo):
            return mejor_w

    return mejor_w

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def ejecucion(x, y, x_test, y_test):
    print("Cálculamos la regresión lineal mediante la pseudoinversa.",
            end=" ", flush=True)
    w_rl = pseudoinverse(x, y)
    print("Completado.\n")

    print("Cálculamos los coeficientes de la regresion lineal mediante " +
            "el algoritmo de PLAPocket.", end=" ", flush=True)
    w = np.random.rand(3)
    w_pla = PLAPocket(x, y, 1000, w_rl.reshape(-1,))
    print("Completado.\n")
    print(w_rl, w_pla)
    w_rl = normalize(w_rl)
    w_pla = normalize(w_pla)
    print(w_rl, w_pla)

    # Pintamos la regresión obtenida
    datos = x[:, 1:]
    plt.xlim(np.min(datos[:, 0]), np.max(datos[:, 0]))
    plt.ylim(np.min(datos[:, 1]), np.max(datos[:, 1]))
    color = {1: 'b', -1: 'g'}
    for etiqueta, nombre in [(-1, "Etiqueta -1"), (1, "Etiqueta 1")]:
        d = datos[y == etiqueta]
        plt.scatter(d[:, 0], d[:, 1], c=color[etiqueta], label=nombre)
    x_recta = np.array([np.min(datos[:, 0]), np.max(datos[:, 0])])
    plt.plot(x_recta, (-w_rl[1] * x_recta - w_rl[0]) / w_rl[2], label='Regresión lineal')
    plt.plot(x_recta, (-w_pla[1] * x_recta - w_pla[0]) / w_pla[2], label='PLAPocket')
    plt.legend()
    plt.title('Digitos Manuscritos (TRAINING) y comparación de rectas')
    plt.show()

    datos = x_test[:, 1:]
    plt.xlim(np.min(datos[:, 0]), np.max(datos[:, 0]))
    plt.ylim(np.min(datos[:, 1]), np.max(datos[:, 1]))
    color = {1: 'b', -1: 'g'}
    for etiqueta, nombre in [(-1, "Etiqueta -1"), (1, "Etiqueta 1")]:
        d = datos[y_test == etiqueta]
        plt.scatter(d[:, 0], d[:, 1], c=color[etiqueta], label=nombre)
    x_recta = np.array([np.min(datos[:, 0]), np.max(datos[:, 0])])
    plt.plot(x_recta, (-w_rl[1] * x_recta - w_rl[0]) / w_rl[2], label='Regresión lineal')
    plt.plot(x_recta, (-w_pla[1] * x_recta - w_pla[0]) / w_pla[2], label='PLAPocket')
    plt.legend()
    plt.title('Digitos Manuscritos (TEST) y comparación de rectas')
    plt.show()

    input("\n--- Pulsar 'Enter' para continuar ---\n")
    print(w_rl, w_pla)
    print("E_in {}: \t{}".format('Regresión lineal', Err(x, y, w_rl)))
    print("E_test {}: \t{}".format('Regresión lineal', Err(x_test, y_test, w_rl)))
    print("E_in {}: \t{}".format('PLAPocket', Err(x, y, w_pla)))
    print("E_test {}: \t{}".format('PLAPocket', Err(x_test, y_test, w_pla)))


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE


##############################################################################
###########                                                     ##############
###########                 Funcion principal                   ##############
###########                                                     ##############

# Función principal del programa
def bonus():
    x, y, x_test, y_test = muestra_datos()
    ejecucion(x, y, x_test, y_test)


###########                                                     ##############
##############################################################################

if __name__ == "__main__":
    bonus()
