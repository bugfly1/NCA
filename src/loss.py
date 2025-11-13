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
def loss_serie(Batch_CA, serie_extendida, tbar_fijo=None):
    n_frames = Batch_CA.shape[-4]
    lista_series = tf.TensorArray(dtype=tf.float32, size=BATCH_SIZE)
    lista_tbar = tf.TensorArray(dtype=tf.float32, size=BATCH_SIZE)
    
    for i in tf.range(BATCH_SIZE):
        serie_CA = Batch_CA[i]
        if tbar_fijo == None:
            error = [tf.reduce_mean(pixelWiseMSE(serie_CA, tf.roll(serie_extendida, tbar, axis=0)), axis=[-1]) for tbar in tf.range(n_frames)]
        else:
            error = tf.reduce_mean(pixelWiseMSE(serie_CA, tf.roll(serie_extendida, tbar_fijo, axis=0)), axis=[-1])
        
        #tf.print("\n\nError por tbar:", error)
        error = tf.convert_to_tensor(error, dtype=tf.float32)
        #print(tf.argmin(input=error).numpy().item())
        
        #serie_MSE = softmin(error)
        serie_MSE = tf.reduce_min(error)
        min_tbar = tf.argmin(input=serie_MSE)
        
        lista_tbar.write(i, min_tbar).mark_used()
        lista_series.write(i,serie_MSE).mark_used()
    
    Batch_MSE = lista_series.stack()
    tbar_batch = lista_tbar.stack()
    #tf.print("Error por batchs", lista_tensor)

    loss = tf.reduce_mean(Batch_MSE)
    tbar_step = tf.reduce_mean(tbar_batch)
    #tf.print("Loss:", loss)
    return loss, tbar_step


def get_min_tbar(serie_CA, serie_extendida, n_frames):
    MSE = tf.map_fn(lambda tbar: tf.reduce_mean(pixelWiseMSE(serie_CA, tf.roll(serie_extendida, tbar, axis=0)), axis=[-1]), tf.range(n_frames), fn_output_signature=PRECISION)
    return tf.argmin(input=MSE, output_type=tf.int32)
    
    
@tf.function
def loss_serie_tf(serie_CA, serie_extendida, n_frames, tbar_fijo=None):
    if tbar_fijo != None:
        return tf.reduce_mean(pixelWiseMSE(serie_CA, tf.roll(serie_extendida, tbar_fijo, axis=0)), axis=[-1])
    
    if SERIE_CORTA:
        MSE = tf.map_fn(lambda tbar: tf.reduce_mean(pixelWiseMSE(serie_CA, tf.roll(serie_extendida, tbar, axis=0)[:2*TAU]), axis=[-1]), tf.range(n_frames), fn_output_signature=PRECISION)
    else:        
        MSE = tf.map_fn(lambda tbar: tf.reduce_mean(pixelWiseMSE(serie_CA, tf.roll(serie_extendida, tbar, axis=0)), axis=[-1]), tf.range(n_frames), fn_output_signature=PRECISION)
    
    return MSE

"""
def loss_batch_tf(Batch_CA, serie_extendida, tbar_fijo_seed=None, dtype=PRECISION):
    n_frames = Batch_CA.shape[-4]
    # Si tbar de seed verdaderamente se mantiene constante, escoje uno con tbar fijo y evalualo por separado
    seed_MSE = loss_serie_tf(Batch_CA[0], serie_extendida, n_frames, tbar_fijo=tbar_fijo_seed)
    Batch_MSE = tf.map_fn(fn=lambda serie_CA: loss_serie_tf(serie_CA, serie_extendida, n_frames, tbar_fijo=None), elems=Batch_CA[1:], fn_output_signature=dtype)
    #Batch_MSE = tf.map_fn(fn=lambda serie_CA: loss_serie_tf(serie_CA, serie_extendida, n_frames, tbar_fijo=None), elems=Batch_CA, fn_output_signature=dtype)
    
    min_tbar_seed = tf.argmin(seed_MSE, output_type=tf.int32)
    min_tbars = tf.map_fn(fn=lambda serie_MSE: tf.argmin(input=serie_MSE, output_type=tf.int32), elems=Batch_MSE, fn_output_signature=tf.int32)
    
    tf.print(Batch_MSE.dtype)
    
    min_tbars = tf.concat([tf.expand_dims(min_tbar_seed, 0), min_tbars],axis=0)
    Batch_MSE = tf.concat([tf.expand_dims(seed_MSE, 0), Batch_MSE],axis=0)
    
    Batch_MSE = tf.map_fn(fn=lambda serie_MSE: tf.reduce_min(serie_MSE), elems=Batch_MSE, fn_output_signature=dtype)
    
    return tf.reduce_mean(Batch_MSE), min_tbars
    #return tf.reduce_mean(tf.reduce_mean(Batch_MSE, tf.reduce_min(seed_MSE)))
"""

@tf.function
def loss_batch_tf(Batch_CA, serie_extendida, dtype=PRECISION):
    n_frames = Batch_CA.shape[-4]
    Batch_MSE = tf.map_fn(fn=lambda serie_CA: loss_serie_tf(serie_CA, serie_extendida, n_frames, tbar_fijo=None), elems=Batch_CA, fn_output_signature=dtype)
    min_tbars = tf.map_fn(fn=lambda serie_MSE: tf.argmin(input=serie_MSE, output_type=tf.int32), elems=Batch_MSE, fn_output_signature=tf.int32)
    Batch_MSE = tf.map_fn(fn=lambda serie_MSE: tf.reduce_min(serie_MSE), elems=Batch_MSE, fn_output_signature=dtype)
    return tf.reduce_mean(Batch_MSE), min_tbars

def loss_batch_tf_tbar_fijo_seed(Batch_CA, serie_extendida, tbar_fijo_seed, dtype=PRECISION):
    n_frames = Batch_CA.shape[-4]
    seed_MSE = loss_serie_tf(Batch_CA[0], serie_extendida, n_frames, tbar_fijo=tbar_fijo_seed)
    Batch_MSE = tf.map_fn(fn=lambda serie_CA: loss_serie_tf(serie_CA, serie_extendida, n_frames, tbar_fijo=None), elems=Batch_CA[1:], fn_output_signature=dtype)
    
    min_tbar_seed = tbar_fijo_seed
    min_tbars = tf.map_fn(fn=lambda serie_MSE: tf.argmin(input=serie_MSE, output_type=tf.int32), elems=Batch_MSE, fn_output_signature=tf.int32)
    
    min_tbars = tf.concat([tf.expand_dims(min_tbar_seed, 0), min_tbars],axis=0)
    
    Batch_MSE = tf.map_fn(fn=lambda serie_MSE: tf.reduce_min(serie_MSE), elems=Batch_MSE, fn_output_signature=dtype)
    Batch_MSE = tf.concat([tf.expand_dims(seed_MSE, 0), Batch_MSE],axis=0)
    return tf.reduce_mean(Batch_MSE), min_tbars
    


# ====================== Original =====================
@tf.function    
def loss_f(x, pad_target=None, n_frames=None):
    if not ROLL:
        return pixelWiseMSE(x, pad_target)
    else:
        MSE = pixelWiseMSE(x, np.roll(to_rgba(pad_target), -1, axis=0))
        return tf.reduce_mean(MSE)


