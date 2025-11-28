import tensorflow as tf
from keras.layers import Conv2D, Conv1D, MaxPooling2D, Dense
import numpy as np
import os

from src.parameters import *
from src.CAModel import CA1DModel

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['XLA_FLAGS'] = ' --xla_gpu_deterministic_ops=true'


## Trainer SetUp
lr = 2e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [2000], [lr, lr*0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)



R = 2
w = 100
seed = np.zeros([1, w, CHANNEL_N], dtype=NP_PRECISION)
seed[0, w//4, 3:] = 1.0
 
x = np.repeat(seed[None, ...], 1, 0)
ca = CA1DModel(R=R)
n_iter = 1000

T = 20

B = tf.keras.Sequential([
    Conv1D(128, 1, activation=tf.nn.relu),
    Conv1D(CHANNEL_N, 
           kernel_size = 2,
           strides = 2,
           activation=None,
           kernel_initializer=tf.zeros_initializer),
])


def loss_f(x, x_prime):
    return tf.reduce_mean(tf.square(x[0, w//2:(w//2) + R]-x_prime), [-2, -3, -1])

def train_step(x, x_prime):
    new_boundary = np.zeros([1, w, CHANNEL_N])
    x = ca(x)
    x_prime = ca(x_prime)
    #izq = x[0, 0,:w//2]
    #izq_R = x[0, 0, :(w//2) + R]
    #izq_R_minus_izq = x[0, 0, w//2 :(w//2) + R]
    Boundary_R = x[:, 0, (w//2) - R : (w//2) + R + 1]
    
    with tf.GradientTape() as g:    #TODO: Esto no funciona, hay que hacerlo diferenciable
        updated_boundary = B(Boundary_R)
        new_boundary[0, w//2:(w//2) + R] = updated_boundary
        new_boundary = tf.cast(new_boundary, tf.float32)
        #new_boundary = tf.zeros([1, w, CHANNEL_N], dtype=tf.float32)
        x_prime = x_prime + new_boundary
        loss = loss_f(x[0], x_prime)
        #loss = tf.reduce_sum(tf.abs(x - x_prime))
    
    
    grads = g.gradient(loss, B.weights)
    grads = [g/(tf.norm(g)+1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, B.weights))
    return x, x_prime, loss


N_train_steps = 2000
x = np.repeat(seed[None, ...], 1, 0)
x_prime = np.repeat(seed[None, ...], 1, 0)
for i in range(N_train_steps):
    x, x_prime, loss = train_step(x, x_prime)
    print('\r step: %d, log10(loss): %.3f'%(i, np.log10(loss)), end='')

    
B.save_weights("Pesos_B.weights.h5")
    