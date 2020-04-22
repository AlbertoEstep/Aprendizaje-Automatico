# -*- coding: utf-8 -*-

# AA - Práctica 2 - UGR
# Author: Alberto Estepa Fernández
# Date: 25/04/2020

import numpy as np
import matplotlib.pyplot as plt
import pickle # Obtener estado semilla


#----------------------------------------------------------------------------#
#----------------------------- Modelos lineales -----------------------------#
#----------------------------------------------------------------------------#

# Aclaramos que hemos usado la plantilla proporcionada por el profesor del
# grupo 1 de prácticas.

# Obtengo el estado de la semilla en el fichero
def recuperar_semilla(ruta_fichero):
    with open(ruta_fichero, 'rb') as fich:
        estado = pickle.load(fich)
    np.random.set_state(estado)

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


# Introducimos ruido en las etiquetas
def introduce_ruido(y, porcentaje = 0.1):
	y_ruido = np.copy(y)
    # Primero calculamos el número de etiquetas que tenemos que cambiarle el
	# signo (ruido) y luego obtenemos una muestra de forma aleatoria de dicho
	# tamaño y le cambiamos el signo.
	y_positivos = np.where(y == 1)[0]
	n_ruido_positivos = round(y_positivos.shape[0] * porcentaje)
	i_positivos = np.random.choice(y_positivos, n_ruido_positivos,
									replace=False)

	y_negativos = np.where(y == -1)[0]
	n_ruido_negativos = round(y_negativos.shape[0] * porcentaje)
	i_negativos = np.random.choice(y_negativos, n_ruido_negativos,
									replace=False)

	y_ruido[i_positivos] = -1
	y_ruido[i_negativos] = 1
	return y_ruido



def ejercio_anterior():
    # Obtengo el estado de la semilla en el fichero
    recuperar_semilla('./estado_semilla.dat')
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
    y_ruido = introduce_ruido(y)

    # Pintar puntos y comprobamos que son los mismos que el ejercicio anterior
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

    # Pintar puntos y comprobamos que son los mismos que el ejercicio anterior
    for etiqueta in etiquetas:
        indice = np.where(y_ruido == etiqueta)
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
    return x, y, y_ruido
##############################################################################

##############################################################################
##############################################################################
##############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

# Algoritmo PLA
def adjust_PLA(datos, label, max_iter, vini):
    w = np.copy(vini)
    n_iteraciones = 0
    continuar = True
    while continuar:
        continuar = False
        n_iteraciones += 1
        for x, y in zip(datos, label):
            y_predicha = funcion_signo(w.dot(x.reshape(-1, 1)))
            if y_predicha != y:
                w += y * x
                continuar = True
        if n_iteraciones == max_iter:
            break

    return w, n_iteraciones

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def perceptron():
    x, y, y_ruido = ejercio_anterior()
    datos = np.c_[np.ones((x.shape[0], 1), dtype=np.float64), x]
    vector_cero = np.array([0.0, 0.0, 0.0])

    # Con los datos del ejercicio 1.2.a
    print("------------------- DATOS SIN RUIDO ---------------------------\n")
    print("------ Algoritmo del perceptron partiendo del vector cero -----\n")
    w, n_iteraciones = adjust_PLA(datos, y, 1000, vector_cero)
    print('Valor w: {} - Número de iteraciones: {}'.format(w, n_iteraciones))
    w = normalize(w)
    print('Valor w normalizado: {}'.format(w))
    input("\n--- Pulsar 'Enter' para continuar ---\n")
    print("-- Algoritmo del perceptron partiendo de vectores aleatorios --\n")
    n_iteraciones = []
    w_aleatorios = []
    for _ in range(10):
        w_aleatorio = simula_unif(3, 1, [0.0, 1.0]).reshape(-1,)
        w, iteracion = adjust_PLA(datos, y, 1000, w_aleatorio)
        n_iteraciones.append(iteracion)
        w_aleatorios.append(w_aleatorio)
        print('w_0 = {}'.format(w_aleatorio))
        print('Valor w: {} - Número de iteraciones: {}'.format(w, iteracion))
        w = normalize(w)
        print('Valor w normalizado: {}'.format(w))

    print('Valor medio de iteraciones necesario para converger: {}'.format(
            np.mean(np.asarray(n_iteraciones))))
    input("\n--- Pulsar 'Enter' para continuar ---\n")

    # Con los datos del ejercicio 1.2.b
    print("------------------- DATOS CON RUIDO ---------------------------\n")
    print("------ Algoritmo del perceptron partiendo del vector cero -----\n")
    w, n_iteraciones = adjust_PLA(datos, y_ruido, 1000, vector_cero)
    print('Valor w: {} - Número de iteraciones: {}'.format(w, n_iteraciones))
    w = normalize(w)
    print('Valor w normalizado: {}'.format(w))
    input("\n--- Pulsar 'Enter' para continuar ---\n")
    print("-- Algoritmo del perceptron partiendo de vectores aleatorios --\n")
    n_iteraciones = []
    for w_aleatorio in w_aleatorios:
        w, iteracion = adjust_PLA(datos, y_ruido, 1000, w_aleatorio)
        n_iteraciones.append(iteracion)
        print('w_0 = {}'.format(w_aleatorio))
        print('Valor w: {} - Número de iteraciones: {}'.format(w, iteracion))
        w = normalize(w)
        print('Valor w normalizado: {}'.format(w))

    print('Valor medio de iteraciones necesario para converger: {}'.format(
        np.mean(np.asarray(n_iteraciones))))
    input("\n--- Pulsar 'Enter' para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.2: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

# Función gradiente
def gradiente(x, y, w):
    return -y * x/(1 + np.exp(y * w.dot(x)))

# Función para ajustar un clasificador basado en regresión lineal mediante
# el algoritmo SGD
def sgdRL(datos, etiquetas, umbral=0.01, lr=0.01):
    N, dimension = datos.shape
    w = np.zeros(dimension)
    delta = np.inf
    while delta > umbral:
        indices = np.random.permutation(N)
        w_anterior = np.copy(w)
        for indice in indices:
            w = w - lr * gradiente(datos[indice], etiquetas[indice], w)
        delta = np.linalg.norm(w_anterior - w)
    return w


def regresion_logistica_sgd():
    # Establecemos una nueva semilla
    np.random.seed(1)

    # Calculamos los coeficcientes de la recta y simulamos los datos
    intervalo = [0, 2]
    a, b = simula_recta(intervalo)
    N = 100
    datos = simula_unif(N, 2, intervalo)
    matriz_datos = np.hstack((np.ones((N, 1)), datos))
    etiquetas = np.empty((N, ))
    for i in range(N):
        etiquetas[i] = asigna_etiquetas(datos[i, 0], datos[i, 1], a, b)

    # Calculamos los coeficientes del modelos mediante la regresión logistica
    # con gradiente descendente que hemos implementado
    w = sgdRL(matriz_datos, etiquetas)

    # Pintamos la regresión obtenida
    plt.xlim(np.min(datos[:, 0]), np.max(datos[:, 0]))
    plt.ylim(np.min(datos[:, 1]), np.max(datos[:, 1]))
    color = {1: 'b', -1: 'g'}
    for etiqueta, nombre in [(-1, "Etiqueta -1"), (1, "Etiqueta 1")]:
        d = datos[etiquetas == etiqueta]
        plt.scatter(d[:, 0], d[:, 1], c=color[etiqueta], label=nombre)
    x = np.array([np.min(datos[:, 0]), np.max(datos[:, 0])])
    for w, nombre in zip([w], ["sgdRL"]):
        plt.plot(x, (-w[1]*x - w[0])/w[2], c='r', label=nombre)
    # Pintar recta
    puntos = np.array([np.min(datos[:, 0]), np.max(datos[:, 0])])
    plt.plot(puntos, a * puntos + b, c='b', label='Recta de simulación')
    plt.gcf().canvas.set_window_title('Ejercicio 2 - Apartado 2B')
    plt.xlabel('Eje $x_1$')
    plt.ylabel('Eje $x_2$')
    plt.title('Regresión logística aplicada a los datos obtenidos')
    plt.legend()
    plt.show()

    input("\n--- Pulsar 'Enter' para continuar ---\n")

    return intervalo, a, b, w


# Usar la muestra de datos etiquetada para encontrar nuestra solución g
# y estimar Eout usando para ello un número suficientemente grande de nuevas
# muestras (>999).

def error_sgdRL(w, x, y):
  return np.mean(np.log(1 + np.exp(-y * x.dot(w))))

def error_mal_clasificados(w, datos, labels):
    recta = lambda x: w[0]*x[:,0] + w[1]*x[:, 1] + w[2]*x[:, 2]
    signos = labels*recta(datos)
    aciertos = len(signos[signos >= 0])/len(labels)
    return 1 - aciertos

def estudio_error(intervalo, a, b, w):
    N = 1000
    datos_test = simula_unif(N, 2, intervalo)
    matriz_datos_test = np.hstack((np.ones((N, 1)), datos_test))
    etiquetas_test = np.empty((N, ))

    for i in range(N):
        etiquetas_test[i] = asigna_etiquetas(datos_test[i, 0],
                                            datos_test[i, 1], a, b)

    print("Error (de la regresión): {}".format(error_sgdRL(w, matriz_datos_test,
                                            etiquetas_test)))
    print("Error (mal clasificados): {}".format(error_mal_clasificados(w, matriz_datos_test,
                                            etiquetas_test)))
    input("\n--- Pulsar 'Enter' para continuar ---\n")


##############################################################################
###########                                                     ##############
###########                 Funcion principal                   ##############
###########                                                     ##############

# Función principal del programa
def ejercicio2():
    perceptron()
    intervalo, a, b, w = regresion_logistica_sgd()
    estudio_error(intervalo, a, b, w)

###########                                                     ##############
##############################################################################

if __name__ == "__main__":
    ejercicio2()
