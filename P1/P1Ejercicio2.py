# -*- coding: utf-8 -*-

# AA - Práctica 1 - UGR
# Author: Alberto Estepa Fernández
# Date: 24/03/2020

#############################
#####     LIBRERIAS     #####
#############################

import numpy as np

#------------------------------------------------------------------------------#
#---------------------- Ejercicio sobre regresión lineal ----------------------#
#------------------------------------------------------------------------------#

#------------------------------Ejercicio 1 ------------------------------------#


# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []

	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y

# Funcion para calcular el error
def Err(x, y, w):
	##COPIADO
	error = np.square(x.dot(w) - y.reshape(-1, 1))        # Calcular el error cuadrático para cada vector de características
    error = error.mean()                                  # Calcular la media de los errors cuadráticos (matriz con una columna)
    return error

def diff_Err(x,y,w):
    d_error = x.dot(w) - y.reshape(-1, 1)           # Calcular producto vectorial de x*w y restarle y
    d_error =  2 * np.mean(x * d_error, axis=0)     # Realizar la media del producto escalar de x*error y la media en el eje 0
    d_error = d_error.reshape(-1, 1)                # Cambiar la forma para que tenga 3 filas y 1 columna
    return d_error

# Gradiente Descendente Estocastico
def sgd(x, y, lr, max_iters, tam_minibatch):
    w = np.zeros((3, 1), np.float64)
    n = x.shape[0]
    i = 0

	while i < max_iters:
        i += 1
        # Escoger valores aleatorios de índices sin repeticiones y obtener los elementos
        index = np.random.choice(n, tam_minibatch, replace=False)
        minibatch_x = x[index]
        minibatch_y = y[index]
        # Actualizar w
        w = w - lr * diff_Err(minibatch_x, minibatch_y, w)
	# HASTA AQUI COPIADO
	return w

# Algoritmo pseudoinversa
def pseudoinverse(x, y):

	return w

# Lectura de los datos de entrenamiento
x, y = readData()
# Lectura de los datos para el test
x_test, y_test = readData()

print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')
# Gradiente descendente estocastico

w = sgd()

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err())
print ("Eout: ", Err())

input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo Pseudoinversa

w = pseudoinverse(x, y)

print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err())
print ("Eout: ", Err())


#------------------------------Ejercicio 2 -------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return

# EXPERIMENTO
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]

print ('Ejercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')




# -------------------------------------------------------------------

# d) Ejecutar el experimento 1000 veces

print ('Errores Ein y Eout medios tras 1000reps del experimento:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para continuar ---\n")
