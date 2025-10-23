import tensorflow as tf
import numpy as np
from src.Utils import to_rgba, to_rgb
from src.parameters import PRECISION, delta, BATCH_SIZE, b, ROLL, TAU, SERIE_CORTA, ALPHA

## Mediciones de error
@tf.function
def pixelWiseHuberLoss(x, target):
    if tf.reduce_mean(tf.abs(to_rgba(x) - target)) <= delta:
        return pixelWiseMSE(x, target)
    else:
        return tf.reduce_mean(delta * (tf.abs(to_rgba(x) - target) - (0.5 * delta)))

@tf.function
def pixelWiseMEA(x, target):
    return tf.reduce_mean(tf.abs(to_rgba(x)-target), [-2,-3,-1])

@tf.function
def pixelWiseMSE(x, target):
    if ALPHA:
        return tf.reduce_mean(tf.square(to_rgba(x)-target), [-2, -3, -1])
    else:
        return tf.reduce_mean(tf.square(to_rgb(x)-target), [-2, -3, -1])            # TESTEA

## Softmin
@tf.function
def softmin(x):
    return -(1/b) * tf.math.log(tf.reduce_sum(tf.exp(-b*x)))

@tf.function
def EstableSoftmin(x):
    m = tf.reduce_min(x) # minimo
    return m - (1/b) * tf.math.log(tf.reduce_sum(tf.exp(-b*(x - m))))              # TESTEA

## ============ Serie =====================
# Principalmente para DEBUG
def loss_serie(serie_CA, serie_extendida):
    n_frames = serie_CA.shape[-4]
    lista_serie = tf.TensorArray(dtype=tf.float32, size=BATCH_SIZE)
    for i in tf.range(BATCH_SIZE):
        batch = serie_CA[i]
        error = [tf.reduce_mean(pixelWiseMSE(batch, tf.roll(serie_extendida, tbar, axis=0)), axis=[-1]) for tbar in tf.range(n_frames)]
        
        tf.print("\n\nError por tbar:", error)
        error = tf.convert_to_tensor(error, dtype=tf.float32)
        
        batch_MSE = softmin(error)
        tf.print('softmin:', batch_MSE)
        lista_serie.write(i,batch_MSE).mark_used()
    
    lista_tensor = lista_serie.stack()
    tf.print("Error por batchs", lista_tensor)

    loss = tf.reduce_mean(lista_tensor)
    tf.print("Loss:", loss)
    return loss

@tf.function
def loss_serie_tf(serie_CA, serie_extendida, n_frames):
    if SERIE_CORTA:
        MSE = tf.map_fn(lambda tbar: tf.reduce_mean(pixelWiseMSE(serie_CA, tf.roll(serie_extendida, tbar, axis=0)[:2*TAU]), axis=[-1]), tf.range(n_frames), fn_output_signature=PRECISION)
    else:
        MSE = tf.map_fn(lambda tbar: tf.reduce_mean(pixelWiseMSE(serie_CA, tf.roll(serie_extendida, tbar, axis=0)), axis=[-1]), tf.range(n_frames), fn_output_signature=PRECISION)
    return softmin(MSE)

@tf.function
def loss_batch_tf(serie_CA, serie_extendida):
    n_frames = serie_CA.shape[-4]
    # Separamos por batch para evaluar el softmin de cada uno
    new_serie = tf.map_fn(fn=lambda t: loss_serie_tf(t, serie_extendida, n_frames), elems=serie_CA, fn_output_signature=PRECISION)
    return tf.reduce_mean(new_serie)



# ====================== Original =====================
@tf.function    
def loss_f(x, pad_target=None, n_frames=None):
    if not ROLL:
        return pixelWiseMSE(x, pad_target)
    else:
        MSE = pixelWiseMSE(x, np.roll(to_rgba(pad_target), -1, axis=0))
        return tf.reduce_mean(MSE)


