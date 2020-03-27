# -*- coding: utf-8 -*-

# AA - Práctica 1 - UGR
# Author: Alberto Estepa Fernández
# Date: 27/03/2020

import numpy as np
from math import pi, sin, cos, exp, inf
import matplotlib.pyplot as plt

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

# Derivada segunda de f respecto a x x
def fxx(w):
    return 2 - 8 * pi**2 * sin(2 * pi * w[0]) * sin(2 * pi * w[1])

# Derivada segunda de f respecto a y y
def fyy(w):
    return 4 - 8 * pi**2 * sin(2 * pi * w[0]) * sin(2 * pi * w[1])

# Derivada segunda de f respecto a x y
def fxy(w):
    return 8 * pi**2 * cos(2 * pi * w[0]) * cos(2 * pi * w[1])

# Matriz Hessiana
def hessiana_f(w):
    return np.array([[fxx(w), fxy(w)], [fxy(w), fyy(w)]])

# Método de Newton
def metodo_newton(w_0, max_iteraciones = 100, lr = None):
    w = np.copy(w_0)              # Copiar los w iniciales para no modificarlos
    i = 0                         # Iniciar las iteraciones a 0
    lista_w = []                   # Se inicializa una lista vacía con los valors de w
    lista_evaluaciones = []                   # Se inicializa una lista vacía con los valors de la función
    # Mientras el número de iteraciones no supere el máximo, calcular
    # la hessiana, invertirla, calcular el gradiente y ajustar w
    # Añadir además a las listas correspondientes los valores de w y de w evaluado en f
    while i < max_iteraciones:
        i += 1
        # Calcular la Hessiana, invertirla (pseudoinversa) y calcular el gradiente
        hessiana = hessiana_f(w)
        hessiana = np.linalg.inv(hessiana)
        gradiente = gradf(w)
        # Calcular theta (producto vectorial del Hessiana invertida y el gradiente)
        theta = hessiana.dot(gradiente.reshape(-1, 1))
        theta = theta.reshape(-1,)                      # Hacer que theta sea un vector fila
        # Multiplicar theta por lr, en caso de que se haya especificado
        if lr:
            theta = lr * theta
        # Actualizar w
        w = w - theta
        # Añadir w y su valor a las listas correspondientes
        lista_w.append(w)
        lista_evaluaciones.append(f(w))
    return w, i, np.array(lista_w), np.array(lista_evaluaciones)

def newton_grafica(lista_evaluaciones, max_iters = 50, lr = 0.01):
	plt.plot(range(max_iters), lista_evaluaciones, 'b-o', label=r"$\eta$ = {}".format(lr))
	plt.title("Valor de la función")
	plt.gcf().canvas.set_window_title('Bonus - Apartado A')
	plt.xlabel('Iteraciones')
	plt.ylabel('f(x,y)')
	plt.legend()
	plt.show()

print('\n-----------------------BONUS: METODO DE NEWTON --------------------\n\n')

print ('\nGrafica con learning rate igual a 0.01')

w, iteracion, lista_w, lista_evaluaciones = metodo_newton(w_0 = [1.0, -1.0], max_iteraciones = 50, lr = 0.01)
newton_grafica(lista_evaluaciones, max_iters = 50, lr = 0.01)

input("\n--- Pulsar tecla para continuar ---\n")

print ('\nGrafica con learning rate igual a 0.1')

w, iteracion, lista_w, lista_evaluaciones = metodo_newton(w_0 = [1.0, -1.0], max_iteraciones = 50, lr = 0.1)
newton_grafica(lista_evaluaciones, max_iters = 50, lr = 0.1)

input("\n--- Pulsar tecla para continuar ---\n")

w, iteracion, lista_w, lista_evaluaciones = metodo_newton(w_0 = [2.1, -2.1], max_iteraciones = 100)
print ('Punto de inicio: (2.1, -2.1)\n')
print ('Coordenadas obtenidas: (x,y) = (', w[0], ', ', w[1],')\n')

w, iteracion, lista_w, lista_evaluaciones = metodo_newton(w_0 = [3.0, -3.0], max_iteraciones = 100)
print ('\nPunto de inicio: (3.0, -3.0)\n')
print ('Coordenadas obtenidas: (x,y) = (', w[0], ', ', w[1],')\n')

w, iteracion, lista_w, lista_evaluaciones = metodo_newton(w_0 = [1.5, 1.5], max_iteraciones = 100)
print ('\nPunto de inicio: (1.5, 1.5)\n')
print ('Coordenadas obtenidas: (x,y) = (', w[0], ', ', w[1],')\n')

w, iteracion, lista_w, lista_evaluaciones = metodo_newton(w_0 = [1.0, -1.0], max_iteraciones = 100)
print ('\nPunto de inicio: (1.0, -1.0)\n')
print ('Coordenadas obtenidas: (x,y) = (', w[0], ', ', w[1],')\n')

input("\n--- Pulsar tecla para continuar ---\n")