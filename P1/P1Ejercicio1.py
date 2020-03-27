# -*- coding: utf-8 -*-

# AA - Práctica 1 - UGR
# Author: Alberto Estepa Fernández
# Date: 24/03/2020

import numpy as np
from math import pi, sin, cos, exp, inf
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------#
#------------- Ejercicio sobre la búsqueda iterativa de óptimos ----------------#
#-------------------------------------------------------------------------------#


#------------------------------Apartado 1 -------------------------------------#

# Fijamos la semilla

def E(w):
	return (w[0] * exp(w[1]) - 2 * w[1] * exp(-w[0]))**2

# Derivada parcial de E respecto de u
def Eu(w):
	return 2 * exp(-2 * w[0]) * (w[0] * exp(w[0]+w[1]) - 2 * w[1]) * (exp(w[0]+w[1]) +2*w[1])

# Derivada parcial de E respecto de v
def Ev(w):
	return 2 * exp(-2 * w[0]) * (w[0] * exp(w[0]+w[1]) - 2) * (w[0] * exp(w[0]+w[1]) -2*w[1])

# Gradiente de E
def gradE(w):
	return np.array([Eu(w), Ev(w)])

def gd(w, lr, grad_fun, fun, epsilon, max_iters = 1000):
	"""
	- w = punto inicial
	- lr = learning rate
	- grad_fun = gradiente de la funcion fun
	- fun = funcion
	- epsilon = epsilon
	- max_iters = numero maximo de iteraciones
	"""
	it = 0
	while it < max_iters:
		it += 1
		w = w - lr * grad_fun(w)
		if fun(w) < epsilon:
			break
	return w, it


print ('\nGRADIENTE DESCENDENTE')
print ('\nEjercicio 1\n')
w, num_ite = gd(w = [1,1], lr = 0.1, grad_fun = gradE, fun = E, epsilon = 10**(-14))
print ('Numero de iteraciones: ', num_ite)
input("\n--- Pulsar tecla para continuar ---\n")
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

input("\n--- Pulsar tecla para continuar ---\n")

#------------------------------Ejercicio 2 -------------------------------------#

def f(w):
	return pow(w[0] - 2, 2) + 2 * pow(w[1] + 2, 2) + 2 * sin(2*pi*w[0]) * sin(2*pi*w[1])

# Derivada parcial de f respecto de u
def fx(w):
	return 2 * (2 * pi * cos(2 * pi * w[0]) * sin(2 * pi * w[1]) + w[0] - 2)

# Derivada parcial de f respecto de v
def fy(w):
	return 4 * (pi * sin(2 * pi * w[0]) * cos(2 * pi * w[1]) + w[1] + 2)

# Gradiente de f
def gradf(w):
	return np.array([fx(w), fy(w)])

# a) Usar gradiente descendente para minimizar la función f, con punto inicial (1,-1)
# tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1

def gd_grafica(w, lr, grad_fun, fun, max_iters = 50):
	vector_evaluaciones = []
	it = 0
	while it < max_iters:
		vector_evaluaciones.append(fun(w))
		it += 1
		w = w - lr * grad_fun(w)

	plt.plot(range(0,max_iters), vector_evaluaciones, 'b-o', label=r"$\eta$ = {}".format(lr))
	plt.title("Valor de la función")
	plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 3A')
	plt.xlabel('Iteraciones')
	plt.ylabel('f(x,y)')
	plt.legend()
	plt.show()


print ('Resultados ejercicio 2\n')
print ('\nGrafica con learning rate igual a 0.01')
gd_grafica(w = [1,-1], lr = 0.01, grad_fun = gradf, fun = f, max_iters = 50)
print ('\nGrafica con learning rate igual a 0.1')
gd_grafica(w = [1,-1], lr = 0.1, grad_fun = gradf, fun = f, max_iters = 50)
input("\n--- Pulsar tecla para continuar ---\n")


# b) Obtener el valor minimo y los valores de (x,y) con los
# puntos de inicio siguientes:

def min_evualuacion(par):
	return par[1]

def gd_minimo(w, lr, grad_fun, fun, max_iters = 1000):
	"""
	- w = punto inicial
	- lr = learning rate
	- grad_fun = gradiente de la funcion fun
	- fun = funcion
	- max_iters = numero maximo de iteraciones
	"""
	vector_coordenadas_evaluaciones = []
	it = 0
	while it < max_iters:
		vector_coordenadas_evaluaciones.append([w, fun(w)])
		it += 1
		w = w - lr * grad_fun(w)
	minimo = min(vector_coordenadas_evaluaciones, key=min_evualuacion)
	return minimo[0], minimo[1], w

# learning rate = 0.01
print('\n--------------LEARNING RATE = 0.01------------------\n')

coordenadas_minimo, minimo, w = gd_minimo(w = [2.1, -2.1], lr = 0.01, grad_fun = gradf, fun = f, max_iters = 1000)
print ('Punto de inicio: (2.1, -2.1)\n')
print ('Coordenadas mínimo: (x,y) = (', coordenadas_minimo[0], ', ', coordenadas_minimo[1],')')
print ('Valor minimo: ', minimo, '\n')
print ('Coordenadas valor devuelto por el algoritmo: (x,y) = (', w[0], ', ', w[1],')')
print ('Valor minimo devuelto por el algoritmo: ', f(w), '\n')

input("\n--- Pulsar tecla para continuar ---\n")

coordenadas_minimo, minimo, w = gd_minimo(w = [3.0, -3.0], lr = 0.01, grad_fun = gradf, fun = f, max_iters = 1000)
print ('Punto de inicio: (3.0, -3.0)\n')
print ('Coordenadas mínimo: (x,y) = (', coordenadas_minimo[0], ', ', coordenadas_minimo[1],')')
print ('Valor minimo: ', minimo, '\n')
print ('Coordenadas valor devuelto por el algoritmo: (x,y) = (', w[0], ', ', w[1],')')
print ('Valor minimo devuelto por el algoritmo: ', f(w), '\n')

input("\n--- Pulsar tecla para continuar ---\n")

coordenadas_minimo, minimo, w = gd_minimo(w = [1.5, 1.5], lr = 0.01, grad_fun = gradf, fun = f, max_iters = 1000)
print ('Punto de inicio: (1.5, 1.5)\n')
print ('Coordenadas mínimo: (x,y) = (', coordenadas_minimo[0], ', ', coordenadas_minimo[1],')')
print ('Valor minimo: ', minimo, '\n')
print ('Coordenadas valor devuelto por el algoritmo: (x,y) = (', w[0], ', ', w[1],')')
print ('Valor minimo devuelto por el algoritmo: ', f(w), '\n')

input("\n--- Pulsar tecla para continuar ---\n")

coordenadas_minimo, minimo, w = gd_minimo(w = [1.0, -1.0], lr = 0.01, grad_fun = gradf, fun = f, max_iters = 1000)
print ('Punto de inicio: (1.0, -1.0)\n')
print ('Coordenadas mínimo: (x,y) = (', coordenadas_minimo[0], ', ', coordenadas_minimo[1],')')
print ('Valor minimo: ', minimo, '\n')
print ('Coordenadas valor devuelto por el algoritmo: (x,y) = (', w[0], ', ', w[1],')')
print ('Valor minimo devuelto por el algoritmo: ', f(w), '\n')

input("\n--- Pulsar tecla para continuar ---\n")

#--------------------Pruebas Ejercicio 2 -------------------------------------#

# learning rate = 0.1
print('\n--------------LEARNING RATE = 0.1------------------\n')

coordenadas_minimo, minimo, w = gd_minimo(w = [2.1, -2.1], lr = 0.1, grad_fun = gradf, fun = f, max_iters = 1000)
print ('Punto de inicio: (2.1, -2.1)\n')
print ('Coordenadas mínimo: (x,y) = (', coordenadas_minimo[0], ', ', coordenadas_minimo[1],')')
print ('Valor minimo: ', minimo, '\n')
print ('Coordenadas valor devuelto por el algoritmo: (x,y) = (', w[0], ', ', w[1],')')
print ('Valor minimo devuelto por el algoritmo: ', f(w), '\n')

input("\n--- Pulsar tecla para continuar ---\n")

coordenadas_minimo, minimo, w = gd_minimo(w = [3.0, -3.0], lr = 0.1, grad_fun = gradf, fun = f, max_iters = 1000)
print ('Punto de inicio: (3.0, -3.0)\n')
print ('Coordenadas mínimo: (x,y) = (', coordenadas_minimo[0], ', ', coordenadas_minimo[1],')')
print ('Valor minimo: ', minimo, '\n')
print ('Coordenadas valor devuelto por el algoritmo: (x,y) = (', w[0], ', ', w[1],')')
print ('Valor minimo devuelto por el algoritmo: ', f(w), '\n')

input("\n--- Pulsar tecla para continuar ---\n")

coordenadas_minimo, minimo, w = gd_minimo(w = [1.5, 1.5], lr = 0.1, grad_fun = gradf, fun = f, max_iters = 1000)
print ('Punto de inicio: (1.5, 1.5)\n')
print ('Coordenadas mínimo: (x,y) = (', coordenadas_minimo[0], ', ', coordenadas_minimo[1],')')
print ('Valor minimo: ', minimo, '\n')
print ('Coordenadas valor devuelto por el algoritmo: (x,y) = (', w[0], ', ', w[1],')')
print ('Valor minimo devuelto por el algoritmo: ', f(w), '\n')

input("\n--- Pulsar tecla para continuar ---\n")

coordenadas_minimo, minimo, w = gd_minimo(w = [1.0, -1.0], lr = 0.1, grad_fun = gradf, fun = f, max_iters = 1000)
print ('Punto de inicio: (1.0, -1.0)\n')
print ('Coordenadas mínimo: (x,y) = (', coordenadas_minimo[0], ', ', coordenadas_minimo[1],')')
print ('Valor minimo: ', minimo, '\n')
print ('Coordenadas valor devuelto por el algoritmo: (x,y) = (', w[0], ', ', w[1],')')
print ('Valor minimo devuelto por el algoritmo: ', f(w), '\n')

input("\n--- Pulsar tecla para continuar ---\n")
