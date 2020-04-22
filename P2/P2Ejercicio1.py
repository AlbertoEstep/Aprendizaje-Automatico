# -*- coding: utf-8 -*-

# AA - Práctica 2 - UGR
# Author: Alberto Estepa Fernández
# Date: 25/04/2020

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle # Guardar estado semilla

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

	input("\n--- Pulsar 'Enter' para continuar ---\n\n")

	# Pintamos el gráfico de dispersion con los datos obtenidos
	train_x_gaus = simula_gaus(N = N, dim = dim, sigma = sigma_gaus)
	plt.scatter(train_x_gaus[:, 0], train_x_gaus[:, 1], c='g')
	plt.title('Nube de puntos con N = 50, distribucción gaussiana')
	plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 1B')
	plt.xlabel('Eje $x_1$')
	plt.ylabel('Eje $x_2$')
	plt.show()

	input("\n--- Pulsar 'Enter' para continuar ---\n\n")

#------------------------------Apartado 2 -----------------------------------#

# Funcion signo, si es positivo o cerose le asigna 1 y si es negativo -1
def funcion_signo(x):
	if x >= 0:
		return 1
	else:
		return -1

# Función para asignar etiquetas en función de la función signo de una recta
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
	# Guardo el estado de la semilla en el fichero, que usare en el ejercicio 2
	with open('./estado_semilla.dat', 'wb') as fich:
		pickle.dump(np.random.get_state(), fich)
	N = 100
	dim = 2
	rango = [-50, 50]
	color = {1: 'b', -1: 'g'}
	# Genero datos
	x = simula_unif(N = N, dim = dim, rango = rango)
	a, b = simula_recta(rango)
	y = []
	# Genero etiquetas
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

	input("\n--- Pulsar 'Enter' para continuar ---\n\n")

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

	input("\n--- Pulsar 'Enter' para continuar ---\n\n")
	return a, b, x, y

#------------------------------Apartado 3 -----------------------------------#

# Funcion 1
def f1(x):
	return (x[:,0] - 10)**2 + (x[:,1] - 20)**2 - 400

# Funcion 2
def f2(x):
	return 0.5 * (x[:,0] + 10)**2 + (x[:,1] - 20)**2 - 400

# Funcion 3
def f3(x):
	return 0.5 * (x[:,0] - 10)**2 - (x[:,1] + 20)**2 - 400

# Funcion 4
def f4(x):
	return x[:,1] - 20 * x[:,0]**2 - 5 * x[:,0] + 3

# Función proporcionada por el profesor del grupo 1 de práticas
# Funcion que dibuja las gráficas de las funciones pedidas y los puntos
# de nuestra muestra.
def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis',
						yaxis='y axis'):
	#Preparar datos
	min_xy = X.min(axis=0)
	max_xy = X.max(axis=0)
	border_xy = (max_xy-min_xy)*0.01

	#Generar grid de predicciones
	xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+
						0.001:border_xy[0],
	                  min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+
					  	0.001:border_xy[1]]
	grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
	pred_y = fz(grid)
	# pred_y[(pred_y>-1) & (pred_y<1)]
	pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)

	#Plot
	f, ax = plt.subplots(figsize=(8, 6))
	contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
	ax_c = f.colorbar(contour)
	ax_c.set_label('$f(x, y)$')
	ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
	ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2,
	            cmap="RdYlBu", edgecolor='white')

	XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),
							X.shape[0]),np.linspace(round(min(min_xy)),
							round(max(max_xy)),X.shape[0]))
	positions = np.vstack([XX.ravel(), YY.ravel()])
	ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0],
												colors='black')

	ax.set(
	   xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]),
	   ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
	   xlabel=xaxis, ylabel=yaxis)
	plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 3')
	plt.title(title)
	plt.show()

# Creamos la matriz de confusión
def conf_mat(x, y, funcion):
	# Calculamos las etiquetas devueltas
	etiquetas_devueltas = funcion(x)
	# Normalizamos la etiquetas
	prediccion = etiquetas_devueltas/abs(etiquetas_devueltas)
	# Creamos la matriz de confusion con pandas
	y_real = pd.Series(y, name='Real')
	y_pred = pd.Series(prediccion, name='Predecido')
	df_confusion = pd.crosstab(y_real, y_pred)
	return df_confusion

# Creamos la matriz de confusión
def conf_mat_recta(x, y, a, b):
	# Calculamos las etiquetas devueltas
	etiquetas_devueltas = []
	for elemento in x:
		etiquetas_devueltas.append(asigna_etiquetas(elemento[0],
													elemento[1], a, b))
	prediccion = np.array(etiquetas_devueltas)
	# Creamos la matriz de confusion con pandas
	y_real = pd.Series(y, name='Real')
	y_pred = pd.Series(prediccion, name='Predecido')
	df_confusion = pd.crosstab(y_real, y_pred)
	return df_confusion


def apartado3(x, y):
	plot_datos_cuad(x, y, f1, "$f(x, y) = (x - 10)^2 + (y - 20)^2 - 400$")
	input("\n--- Pulsar 'Enter' para continuar ---\n\n")
	plot_datos_cuad(x, y, f2, "$f(x, y) = 0,5(x + 10)^2 + (y - 20)^2 - 400$")
	input("\n--- Pulsar 'Enter' para continuar ---\n\n")
	plot_datos_cuad(x, y, f3, "$f(x, y) = 0,5(x - 10)^2 - (y + 20)^2 - 400$")
	input("\n--- Pulsar 'Enter' para continuar ---\n\n")
	plot_datos_cuad(x, y, f4, "$f(x, y) = y - 20x^2 - 5x + 3$")

	input("\n--- Pulsar 'Enter' para continuar ---\n\n")

def evaluar_rendimiento(x, y, a, b):
	print("\nEvaluacion del rendimiento:\n")

	print("\nMatriz de confusión de la recta de clasificación\n")
	print(conf_mat_recta(x, y, a, b))

	print("\nMatriz de confusión de la función 1\n")
	print(conf_mat(x, y, f1))

	print("\nMatriz de confusión de la función 2\n")
	print(conf_mat(x, y, f2))

	print("\nMatriz de confusión de la función 3\n")
	print(conf_mat(x, y, f3))

	print("\nMatriz de confusión de la función 4\n")
	print(conf_mat(x, y, f4))

	input("\n--- Pulsar 'Enter' para continuar ---\n\n")


##############################################################################
###########                                                     ##############
###########                 Funcion principal                   ##############
###########                                                     ##############

# Función principal del programa
def ejercicio1():
	apartado1()
	a, b, x, y = apartado2()
	apartado3(x, y)
	evaluar_rendimiento(x, y, a, b)

###########                                                     ##############
##############################################################################

if __name__ == "__main__":
    ejercicio1()
