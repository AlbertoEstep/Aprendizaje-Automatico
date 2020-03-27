# -*- coding: utf-8 -*-

# AA - Práctica 1 - UGR
# Author: Alberto Estepa Fernández
# Date: 24/03/2020

#############################
#####     LIBRERIAS     #####
#############################

import numpy as np
import matplotlib.pyplot as plt


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

# Calculamos el error mediante la expresión matricial disponible en la diapositiva
# 8 del tema 2: E(w) = (1/N)||Xw-y||^2
def Err(x, y, w):
	# Calculamos el error cuadrático para cada vector de características
	err = np.square(x.dot(w) - y.reshape(-1, 1))
	# Calculamos la media de los errores cuadráticos y la devolvemos
	return err.mean()

# Calculamos la derivada del error mediante la expresión matricial disponible en
# la diapositiva 10 del tema 2: dE(w) = (2/N)X^t(Xw-y)
def dErr(x,y,w):
	# Aplicamos la formula paso a paso
    de = x.dot(w) - y.reshape(-1, 1)
    de =  2 * np.mean(x * de, axis=0)
    return de.reshape(-1, 1)

# Calculamos el Gradiente Descendente Estocastico
def sgd(x, y, lr = 0.01, max_iters = 1000, tam_minibatch = 32):
	# Rellenamos la solucion con ceros
	w = np.zeros((3, 1), np.float64)
	for i in range(max_iters):
		# Cogemos índices de forma aleatoria sin repeticiones.
		indice = np.random.choice(x.shape[0], tam_minibatch, replace=False)
		# Calculamos los elementos correspondientes a dichos índices.
		minibatch_x = x[indice]
		minibatch_y = y[indice]
		# Actualizar w
		w = w - lr * dErr(minibatch_x, minibatch_y, w)
	return w

# Calculamos la pseudoinversa mediante el algoritmo disponible en la diapositiva
# 12 del tema 2
def pseudoinverse(x, y):
	# Aplicamos el algoritmo paso a paso
    x_traspuesta = x.transpose()
    y_traspuesta = y.reshape(-1, 1)
    w = np.linalg.inv(x_traspuesta.dot(x))
    w = w.dot(x_traspuesta)
    w = w.dot(y_traspuesta)
    return w


# DATOS:
# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')
# Etiquetas
etiqueta1 = -1
etiqueta5 = 1
etiquetas = (etiqueta1, etiqueta5)
color = {etiqueta1: 'b', etiqueta5: 'g'}
valores = {etiqueta1: 1, etiqueta5: 5}

# Ejercicio 1
print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')

# Gradiente descendente estocastico
w = sgd(x, y, lr = 0.01, max_iters = 1000, tam_minibatch = 32)
# Pintamos las soluciones obtenidas junto con los datos usados en el ajuste
for etiqueta in etiquetas:
    indice = np.where(y == etiqueta)
    plt.scatter(x[indice, 1], x[indice, 2], c=color[etiqueta], label='{}'.format(valores[etiqueta]))
# Para pintar la recta de separación de los datos despejamos de la ecuación:
# 0 = w0 + w1 * x1 + w2 * x2, x2 a partir de x1
plt.plot([0, 1], [-w[0]/w[2], -(w[0] + w[1])/w[2]], 'r-')
plt.title('Regresión lineal con SGD')
plt.gcf().canvas.set_window_title('Ejercicio 2 - Apartado 1A')
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.legend()
plt.show()

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo Pseudoinversa
w = pseudoinverse(x, y)
# Pintamos las soluciones obtenidas junto con los datos usados en el ajuste
for etiqueta in etiquetas:
    indice = np.where(y == etiqueta)
    plt.scatter(x[indice, 1], x[indice, 2], c=color[etiqueta], label='{}'.format(valores[etiqueta]))
# Para pintar la recta de separación de los datos despejamos de la ecuación:
# 0 = w0 + w1 * x1 + w2 * x2, x2 a partir de x1
plt.plot([0, 1], [-w[0]/w[2], -(w[0] + w[1])/w[2]], 'r-')
plt.title('Regresión lineal con el algoritmo de la pseudoinversa')
plt.gcf().canvas.set_window_title('Ejercicio 2 - Apartado 1B')
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.legend()
plt.show()

print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

#------------------------------Ejercicio 2 -------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size, size, (N,d))

# EXPERIMENTO
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]

print ('Ejercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')

# Datos
N = 1000
d = 2
size = 1

# Pintamos el gráfico de dispersion con los datos obtenidos
train_x = simula_unif(N, d, size)
plt.scatter(train_x[:, 0], train_x[:, 1], c='g')
plt.title('Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]')
plt.gcf().canvas.set_window_title('Ejercicio 2 - Apartado 2A')
plt.xlabel('Eje $x_1$')
plt.ylabel('Eje $x_2$')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# -------------------------------------------------------------------

# b) asignamos las etiquetas a cada punto con la función dada e introducimos ruido

print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1] con ruido y definidas las etiquetas')


# Calculamos f(x_1 , x_2) = sign((x_1 - 0,2)^2 + x_2^2 - 0,6)
def asigna_etiquetas(x):
    return np.sign(np.square(x[:, 0] - 0.2) + np.square(x[:, 1]) - 0.6)

# Introducimos ruido en las etiquetas de la muestra
def introduce_ruido(train_y, porcentaje = 0.1):
    # Primero calculamos el número de etiquetas que tenemos que cambiarle el
	# signo (ruido) y luego obtenemos una muestra de forma aleatoria de dicho
	# tamaño y le cambiamos el signo.
    n = train_y.shape[0]
    n_ruido = int(n * porcentaje)
    i = np.random.choice(np.arange(n), n_ruido, replace=False)
    train_y[i] = - train_y[i]

# Asignamos las etiquetas a la muestra
train_y = asigna_etiquetas(train_x)
# Introducimos el ruido en las etiquetas
introduce_ruido(train_y, 0.1)
# Definimos el color y las etiquetas que usaremos para pintar los datos
color = {1: 'b', -1: 'g'}
etiquetas = np.unique(train_y)
# Pintamos las soluciones obtenidas junto con los datos usados en el ajuste
for etiqueta in etiquetas:
    indice = np.where(train_y == etiqueta)
    plt.scatter(train_x[indice, 0], train_x[indice, 1], c=color[etiqueta], label='Etiqueta {}'.format(etiqueta))

plt.title('Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1], con ruido')
plt.gcf().canvas.set_window_title('Ejercicio 2 - Apartado 2B')
plt.xlabel('Eje $x_1$')
plt.ylabel('Eje $x_2$')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# -------------------------------------------------------------------

# c ) Ajustamos un modelo de regresion lineal al conjunto de datos generado

print ('Ajustamos un modelo de regresion lineal al conjunto de datos generado')

# Columna de unos
columna_unos = np.ones((1000, 1), dtype=np.float64)
# Concatenamos la columna de 1 con los valores de entrenamiento para así obtener
# el vector de características (1, x1, x2)
train_x = np.c_[columna_unos, train_x]
# Gradiente descendente estocastico
w = sgd(train_x, train_y, lr = 0.01, max_iters = 1000, tam_minibatch = 32)
# Pintamos las soluciones obtenidas junto con los datos usados en el ajuste
for etiqueta in etiquetas:
	indice = np.where(train_y == etiqueta)
	plt.scatter(train_x[indice, 1], train_x[indice, 2], c=color[etiqueta], label='{}'.format(etiqueta))
# Para pintar la recta de separación de los datos despejamos de la ecuación:
# 0 = w0 + w1 * (x1 = -1) + w2 * x2, x2 a partir de x1 y de 0 = w0 + w1 * (x1 = 1) + w2 * x2
plt.plot([-1, 1], [(-w[0] + w[1])/w[2], -(w[0] + w[1])/w[2]], 'r-')
plt.title('Regresión lineal con SGD')
plt.gcf().canvas.set_window_title('Ejercicio 2 - Apartado 2C')
plt.xlabel('Eje $x_1$')
plt.ylabel('Eje $x_2$')
plt.ylim(-1.1, 1.1)
plt.legend()
plt.show()

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(train_x ,train_y, w))

input("\n--- Pulsar tecla para continuar ---\n")

# -------------------------------------------------------------------

# d) Ejecutar el experimento 1000 veces
'''
lista_error_in = []
lista_error_out = []

for _ in range(1000):
    # Generamos los datos de entrenamiento con ruido y los de test sin ruido
    train_x = simula_unif(N, d, size)
    train_y = asigna_etiquetas(train_x)
    train_x = np.c_[columna_unos, train_x]
    introduce_ruido(train_y, 0.1)
    test_x = simula_unif(N, d, size)
    test_y = asigna_etiquetas(test_x)
    test_x = np.c_[columna_unos, test_x]
	# Aplicamos el SGD
    w = sgd(train_x, train_y, lr = 0.01, max_iters = 1000, tam_minibatch = 32)
    # Calculamos E_in y E_out
    lista_error_in.append(Err(train_x, train_y, w))
    lista_error_out.append(Err(test_x, test_y, w))

# Calculamos la media de los vectores E_in y E_out
Ein = np.array(lista_error_in)
Eout = np.array(lista_error_out)
Ein_media = Ein.mean()
Eout_media = Eout.mean()

print ('Errores Ein y Eout medios tras 1000reps del experimento:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para continuar ---\n")
'''
# -------------------------------------------------------------------

# NO LINEAL

print ('Ajustamos un modelo de regresion lineal al conjunto de datos generado')

train_x = simula_unif(N, d, size)
train_y = asigna_etiquetas(train_x)

def creamos_matriz(train_x, N, d, size):
    columna_unos = np.ones((N, 1), dtype=np.float64)
    columnas_normal = train_x
    columna_producto = columnas_normal[:,0] * columnas_normal[:,1]
    columna_primera_c_cuadrado = columnas_normal[:,0] * columnas_normal[:,0]
    columna_segunda_c_cuadrado = columnas_normal[:,1] * columnas_normal[:,1]
    caracteristicas = np.c_[columna_unos, columnas_normal, columna_producto.reshape(-1,1), columna_primera_c_cuadrado.reshape(-1,1), columna_segunda_c_cuadrado.reshape(-1,1)]
    return caracteristicas

train_x = creamos_matriz(train_x, N = 1000, d = 2, size = 1)
introduce_ruido(train_y, 0.1)


'''
# Gradiente descendente estocastico
w = sgd(train_x, train_y, lr = 0.01, max_iters = 1000, tam_minibatch = 32)
# Pintamos las soluciones obtenidas junto con los datos usados en el ajuste
for etiqueta in etiquetas:
	indice = np.where(train_y == etiqueta)
	plt.scatter(train_x[indice, 1], train_x[indice, 2], c=color[etiqueta], label='{}'.format(etiqueta))
# Para pintar la recta de separación de los datos despejamos de la ecuación:
# 0 = w0 + w1 * (x1 = -1) + w2 * x2, x2 a partir de x1 y de 0 = w0 + w1 * (x1 = 1) + w2 * x2
plt.plot([-1, 1], [(-w[0] + w[1])/w[2], -(w[0] + w[1])/w[2]], 'r-')
plt.title('Regresión lineal con SGD')
plt.gcf().canvas.set_window_title('Ejercicio 2 - Apartado 2C')
plt.xlabel('Eje $x_1$')
plt.ylabel('Eje $x_2$')
plt.ylim(-1.1, 1.1)
plt.legend()
plt.show()

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(train_x ,train_y, w))
'''
input("\n--- Pulsar tecla para continuar ---\n")
