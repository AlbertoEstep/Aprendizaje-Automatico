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

# Algoritmo Gradiente Descendente
def gd(w_0, max_iteraciones = 100, lr = 1):
    w = np.copy(w_0)
    vector_evaluaciones = []
    it = 0
    while it < max_iteraciones:
    	vector_evaluaciones.append(f(w))
    	it += 1
    	w = w - lr * gradf(w)
    return w, np.array(vector_evaluaciones)

# Método de Newton
def metodo_newton(w_0, max_iteraciones = 100, lr = 1):
	w = np.copy(w_0)
	vector_evaluaciones = []
	i = 0
	# Mientras no se cumpla el límite de iteraciones
	while i < max_iteraciones:
	    i += 1
	    # Calculamos la inversa de la hessiana
	    inv_hessiana = np.linalg.inv(hessiana_f(w))
	    # Calculamos el producto vectorial de la inversa de la hessiana y el gradiente
	    aux = inv_hessiana.dot(gradf(w).reshape(-1, 1))
	    aux = aux.reshape(-1,)
	    w = w - lr * aux
	    vector_evaluaciones.append(f(w))
	return w, i-1, np.array(vector_evaluaciones)

# Obtenemos la grafica del método de Newton
def newton_grafica(lista_evaluaciones, max_iters = 50, lr = 0.01):
	plt.plot(range(max_iters), lista_evaluaciones, 'b-o', label=r"$\eta$ = {}".format(lr))
	plt.title("Valor de la función")
	plt.gcf().canvas.set_window_title('Bonus - Apartado A')
	plt.xlabel('Iteraciones')
	plt.ylabel('f(x,y)')
	plt.legend()
	plt.show()

print('\n-----------------------BONUS: METODO DE NEWTON --------------------\n\n')

w, iteracion, lista_evaluaciones = metodo_newton(w_0 = [1.0, -1.0], max_iteraciones = 50, lr = 0.01)
print ('Punto de inicio: (1.0, -1.0)\n')
print ('Coordenadas obtenidas: (x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor obtenido: f(x,y) =', lista_evaluaciones[iteracion], '\n')
print ('Grafica con learning rate igual a 0.01')
newton_grafica(lista_evaluaciones, max_iters = 50, lr = 0.01)

input("\n--- Pulsar tecla para continuar ---\n")

print ('\nGrafica con learning rate igual a 0.1')

w, iteracion, lista_evaluaciones = metodo_newton(w_0 = [1.0, -1.0], max_iteraciones = 50, lr = 0.1)
print ('Punto de inicio: (1.0, -1.0)\n')
print ('Coordenadas obtenidas: (x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor obtenido: f(x,y) =', lista_evaluaciones[iteracion], '\n')
print ('Grafica con learning rate igual a 0.1')
newton_grafica(lista_evaluaciones, max_iters = 50, lr = 0.1)

input("\n--- Pulsar tecla para continuar ---\n")

w, iteracion, lista_evaluaciones = metodo_newton(w_0 = [2.1, -2.1], max_iteraciones = 1000)
print ('Punto de inicio: (2.1, -2.1)\n')
print ('Coordenadas obtenidas: (x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor obtenido: f(x,y) =', lista_evaluaciones[iteracion], '\n')

w, iteracion, lista_evaluaciones = metodo_newton(w_0 = [3.0, -3.0], max_iteraciones = 1000)
print ('\nPunto de inicio: (3.0, -3.0)\n')
print ('Coordenadas obtenidas: (x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor obtenido: f(x,y) =', lista_evaluaciones[iteracion], '\n')

w, iteracion, lista_evaluaciones = metodo_newton(w_0 = [1.5, 1.5], max_iteraciones = 1000)
print ('\nPunto de inicio: (1.5, 1.5)\n')
print ('Coordenadas obtenidas: (x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor obtenido: f(x,y) =', lista_evaluaciones[iteracion], '\n')

w, iteracion, lista_evaluaciones = metodo_newton(w_0 = [1.0, -1.0], max_iteraciones = 1000)
print ('\nPunto de inicio: (1.0, -1.0)\n')
print ('Coordenadas obtenidas: (x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor obtenido: f(x,y) =', lista_evaluaciones[iteracion], '\n')

input("\n--- Pulsar tecla para continuar ---\n")

print('\n\n----------------------- Comparación con GD --------------------\n\n')

w_newton, iteracion, lista_evaluaciones_newton = metodo_newton(w_0 = [1.0, -1.0], max_iteraciones = 100, lr = 0.01)
w_gd, lista_evaluaciones_gd = gd(w_0 = [1.0, -1.0], max_iteraciones = 100, lr = 0.01)
plt.plot(np.linspace(0, 100, 100), lista_evaluaciones_newton, 'r-', label="Metodo de Newton")
plt.plot(np.linspace(0, 100, 100), lista_evaluaciones_gd, 'g-', label='Gradiente descendente')
plt.xlabel('Iteración')
plt.ylabel('Valor de la función')
plt.gcf().canvas.set_window_title('Bonus - Apartado B')
plt.title("Comparación Newton vs Gradiente Descendente en el punto (1, -1)")
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

w_newton, iteracion, lista_evaluaciones_newton = metodo_newton(w_0 = [2.1, -2.1], max_iteraciones = 100)
w_gd, lista_evaluaciones_gd = gd(w_0 = [2.1, -2.1], max_iteraciones = 100, lr = 0.01)
plt.plot(np.linspace(0, 100, 100), lista_evaluaciones_newton, 'r-', label="Metodo de Newton")
plt.plot(np.linspace(0, 100, 100), lista_evaluaciones_gd, 'g-', label='Gradiente descendente')
plt.xlabel('Iteración')
plt.ylabel('Valor de la función')
plt.gcf().canvas.set_window_title('Bonus - Apartado B')
plt.title("Comparación Newton vs Gradiente Descendente en el punto (2.1, -2.1)")
plt.legend()
plt.show()

input("\n--- Pulsar tecla para finalizar ---\n")
