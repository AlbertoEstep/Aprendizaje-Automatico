# -*- coding: utf-8 -*-

# AA - Práctica 2 - UGR
# Author: Alberto Estepa Fernández
# Date: 25/04/2020

import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------#
#--------------- Ejercicio sobre la complejidad de H y el ruido -------------#
#----------------------------------------------------------------------------#

# Fijamos la semilla
np.random.seed(1)

# Función que calcula una lista de N vectores de dimensión dim. Cada vector
# contiene dim números aleatorios uniformes en el intervalo rango.
def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

# Función que calcula una lista de longitud N de vectores de dimensión dim,
# donde cada posición del vector contiene un número aleatorio extraido de una
# distribucción Gaussiana de media 0 y varianza dada, para cada dimension, por
# la posición del vector sigma.
def simula_gaus(N, dim, sigma):
    media = 0
    out = np.zeros((N,dim),np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para
        # la primera columna se usará una N(0,sqrt(sigma[0])) y para la
		# segunda N(0,sqrt(simga[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

    return out

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


#------------------------------Apartado 1 -----------------------------------#

print ('\nEJERCICIO SOBRE LA COMPLEJIDAD DE H Y EL RUIDO\n')
print ('-------------------- Ejercicio 1 ---------------------------------\n')

def apartado1():
	N = 50
	dim = 2
	rango_unif = [-50, 50]
	sigma_gaus = [5, 7]

	# Pintamos el gráfico de dispersion con los datos obtenidos
	train_x_unif = simula_unif(N = N, dim = dim, rango = rango_unif)
	plt.scatter(train_x_unif[:, 0], train_x_unif[:, 1], c='g')
	plt.title('Nube de puntos con N = 50, distribucción uniforme')
	plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 1A')
	plt.xlabel('Eje $x_1$')
	plt.ylabel('Eje $x_2$')
	plt.show()

	input("\n--- Pulsar tecla para continuar ---\n\n")

	# Pintamos el gráfico de dispersion con los datos obtenidos
	train_x_gaus = simula_gaus(N = N, dim = dim, sigma = sigma_gaus)
	plt.scatter(train_x_gaus[:, 0], train_x_gaus[:, 1], c='g')
	plt.title('Nube de puntos con N = 50, distribucción gaussiana')
	plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 1B')
	plt.xlabel('Eje $x_1$')
	plt.ylabel('Eje $x_2$')
	plt.show()

	input("\n--- Pulsar tecla para continuar ---\n\n")

#------------------------------Apartado 2 -----------------------------------#

def funcion_signo(x):
	if x >= 0:
		return 1
	else:
		return -1

def asigna_etiquetas(x, y, a, b):
	return funcion_signo(y - a*x - b)

# Introducimos ruido en las etiquetas
def introduce_ruido(y, porcentaje = 0.1):
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

	y[i_positivos] = -1
	y[i_negativos] = 1

def apartado2():
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

	input("\n--- Pulsar tecla para continuar ---\n\n")

	# Inreoducimos ruido
	introduce_ruido(y)
	# Pintar puntos
	for etiqueta in etiquetas:
	    indice = np.where(y == etiqueta)
	    plt.scatter(x[indice, 0], x[indice, 1], c=color[etiqueta],
			label='Etiqueta {}'.format(etiqueta))
	# Pintar recta
	puntos = np.array([np.min(x[:, 0]), np.max(x[:, 0])])
	plt.plot(puntos, a * puntos + b, c='r', label='Recta de simulación')
	titulo = "Nube de 100 puntos con distribucción uniforme con ruido \n" + \
 				"y recta de simulación"
	plt.title(titulo)
	plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 2B')
	plt.xlabel('Eje $x_1$')
	plt.ylabel('Eje $x_2$')
	plt.legend()
	plt.show()

	input("\n--- Pulsar tecla para continuar ---\n\n")

#------------------------------Apartado 3 -----------------------------------#

# Funcion 1
def f1(x, y):
	return (x-10)**2 + (y-20)**2 - 400

# Funcion 2
def f2(x, y):
	return 0.5 * (x+10)**2 + (y-20)**2 - 400

# Funcion 3
def f3(x, y):
	return 0.5 * (x-10)**2 - (y+20)**2 - 400

# Funcion 4
def f4(x, y):
	return y - 20*x**2 - 5*x + 3


def apartado3():
	N = 100
	dim = 2
	rango = [-50, 50]
	color = {1: 'b', -1: 'g'}

	x = simula_unif(N = N, dim = dim, rango = rango)
	a, b = simula_recta(rango)
	y = []

	input("\n--- Pulsar tecla para continuar ---\n\n")



##############################################################################
###########                                                     ##############
###########                 Funcion principal                   ##############
###########                                                     ##############

# Función principal del programa
def ejercicio1():
	apartado1()
	apartado2()
	apartado3()

###########                                                     ##############
##############################################################################

if __name__ == "__main__":
    ejercicio1()
