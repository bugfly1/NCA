import tensorflow as tf
from src.loss import softmin
import matplotlib.pylab as pl
import numpy as np
import os

os.environ['XLA_FLAGS'] = ' --xla_gpu_deterministic_ops=true'

def plot(x, y, name):
  pl.figure(figsize=(10, 4))
  pl.title(name)
  pl.plot(x, y, '.', alpha=0.1)
  pl.savefig(f"{name}.jpg")
  #pl.show()
  pl.close()

# Realiza un grafico comparando el menor numero que tiende a 0 con un determinado valor de b.
# eje x: b, eje y: x
# Basicamente ve sumando de a 0.01 (o con mas 0 para mas presicion) guarda la dupla (b,min x) y aumenta b en 1,
# Finalmente grafica en tipico grafico

b = 200
px = np.array([])
py = np.array([])

"""
step = 0.001
intervalo = [86, 90]

dif = intervalo[1] - intervalo[0]
iter_n = int(dif*(1/step))
for i in range(iter_n):
    x = tf.constant(intervalo[0] + i*step, dtype=tf.float32)
    y = tf.exp(-x)
    px = np.append(px, x.numpy())
    py = np.append(py, y.numpy())



#plot(px, py, "Valores posibles para e^(-x).jpg")
"""

for i in range(3000):
    x = -1 + i * 0.001
    y = tf.constant(softmin(np.array([x])))
    px = np.append(px, x)
    py = np.append(py, y)

plot(px, py, f"Valores posibles para softmin (b={b})")
