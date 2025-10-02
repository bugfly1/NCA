import os
import numpy as np
import tensorflow as tf
from src.Utils import (load_target, imwrite, to_rgba, make_circle_masks, save_loss, load_training,
                      export_model, visualize_batch, visualize_target, visualize_series, visualize_step_seed, plot_loss, 
                      generate_pool_figures, to_rgb, save_rolls, export_ca_to_webgl_demo, save_params)
from src.CAModel import CAModel
from src.SamplePooling import SamplePool
from src.parameters import *
from math import isnan, isinf

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['XLA_FLAGS'] = ' --xla_gpu_deterministic_ops=true'

if not os.path.isdir("train_log"):
    os.mkdir("train_log")

## Trainer SetUp
lr = 2e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [2000], [lr, lr*0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)

ca = CAModel()
loss_log = np.array([])

os.system('clear')

# TODO:
# - Probar distinta arquitectura de modelo
# - Probar entregar parametros constantes como input
# - Probar softmin con perdida en log 10
# - Probar el agregar una medicion de diferencia entre elementos de la sequencia en la funcion de perdida
# - En perdida, comparar por rgb en vez de rgba. El que haya estado asi todo este rato significa que 
#   le estabamos pidiendo que todas las celulas estuvieran vivas (puse que todo el alpha fuera 1), talvez eso causa el
#   crecimiento descontrolado al principio

# ============== Initialize Trainig ==================

# Input loading
pad_target = load_target(SRC_TARGET)
n_frames = 1

### Load and pad target Image
if VIDEO:
    n_frames = pad_target.shape[0]

h, w, _ = pad_target.shape[-3:]

# We add invisible parameters to CA
seed = np.zeros([h, w, CHANNEL_N], dtype=np.float32)
# Set center cell alive for seed
seed[h//2, w//2, 3:] = 1.0
pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))

visualize_target(pad_target)

if ROLL:
    video_seed = np.pad(pad_target, [(0,0), (0,0),(0,0), (0, CHANNEL_N - 4)]).astype(np.float32)
    pool = SamplePool(x = np.repeat(video_seed, int(POOL_SIZE / n_frames), 0))
    video_seed = tf.cast(np.roll(video_seed, -1, axis=0), tf.float32)

### Load Checkpoint
if START_TRAINING_FROM_SAVE_POINT:
    begining = SAVE_POINT
    ca, loss_log = load_training()
else:
    begining = 0

### Loss Functions
@tf.function
def softmin(x):
    return -(1/b) * tf.math.log(tf.reduce_sum(tf.exp(-b*x)))

def LogSumExp(x):
    return tf.math.log(tf.reduce_sum(tf.exp(x)))

delta = 300
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
    return tf.reduce_mean(tf.square((to_rgba(x)-target)), [-2, -3, -1]) 

@tf.function
def loss_serie(serie_CA, serie_extendida):
    error = [tf.reduce_mean(pixelWiseMSE(serie_CA, tf.roll(serie_extendida, tbar, axis=0)), axis=[-1,-2]) for tbar in tf.range(n_frames)]
    #tf.print("\n\nError por t:", error)
    MSE = tf.convert_to_tensor(error, dtype=tf.float32)
    return softmin(MSE)

# Version que funciona :)
def loss_serie_2(serie_CA, serie_extendida):
    lista_serie = tf.TensorArray(dtype=tf.float32, size=BATCH_SIZE)
    for i in tf.range(BATCH_SIZE):
        batch = serie_CA[i]
        error = [tf.reduce_mean(pixelWiseMSE(batch, tf.roll(serie_extendida, tbar, axis=0)), axis=-1) for tbar in tf.range(n_frames)]
        error = tf.convert_to_tensor(error, dtype=tf.float32)
        batch_loss = softmin(error)
        lista_serie.write(i,batch_loss).mark_used()
        
    lista_tensor = lista_serie.stack()    
    loss = tf.reduce_mean(lista_tensor)
    return loss
    
@tf.function    
def loss_f(x):
    if not ROLL:
        return pixelWiseMSE(x, pad_target)
    else:
        MSE = tf.convert_to_tensor([pixelWiseMSE(x, np.roll(to_rgba(pad_target), k, axis=0)) for k in range(n_frames)], dtype=tf.float32)
        return softmin(MSE)



if SERIE:
    serie_temporal_extendida = np.repeat(pad_target, TAU, axis=0)
    # Duplica el video
    #serie_temporal_extendida = np.append(serie_temporal_extendida, serie_temporal_extendida, axis=0)
    n_frames = len(serie_temporal_extendida)
    visualize_target(serie_temporal_extendida)
    
def train_serie(x):
    with tf.GradientTape() as g:
        lista_serie = tf.TensorArray(dtype=tf.float32, size=n_frames)
        for j in tf.range(n_frames):
            for i in tf.range(T):
                x = ca(x)
            lista_serie.write(j,x).mark_used()
        serie_CA = lista_serie.stack()
        # Changes shape from (n_frames, n_batch, h, w, channels) to
        # (n_batch, n_frames, h, w, channels)
        serie_CA = tf.transpose(serie_CA, perm=[1,0,2,3,4])
        loss = loss_serie_2(serie_CA, serie_temporal_extendida)
    grads = g.gradient(loss, ca.weights)
    grads = [g/(tf.norm(g)+1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, ca.weights))
    return x, loss, serie_CA
 

### Training function
@tf.function
def train_step(x):
    iter_n = tf.random.uniform([], N_ITER_CA[0], N_ITER_CA[1], tf.int32)
    with tf.GradientTape() as g:
        for i in tf.range(iter_n):
            x = ca(x)
        
        if ROLL:
            loss = tf.reduce_max(loss_f(x))
        else:
            loss = tf.reduce_mean(loss_f(x))
    
    grads = g.gradient(loss, ca.weights)
    grads = [g/(tf.norm(g)+1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, ca.weights))
    return x, loss

save_params()

x = 0
# ========================= Training Loop =====================
for i in range(begining, 10000+1):
  ### Generate input grids for CA

    if ROLL and USE_PATTERN_POOL:
        batch = pool.sample(n_frames)
        x0 = batch.x
    
    elif not VIDEO and USE_PATTERN_POOL:
        # Sample a batch from pool
        batch = pool.sample(BATCH_SIZE)
        x0 = batch.x
        
        # We sort the batch by loss
        loss_rank = loss_f(x0).numpy().argsort()[::-1]
        x0 = x0[loss_rank]
        
        # The one with less loss gets changed with the seed
        x0[:1] = seed
        
        if DAMAGE_N:
            damage = 1.0-make_circle_masks(DAMAGE_N, h, w).numpy()[..., None]
            x0[-DAMAGE_N:] *= damage
    
    elif SERIE and USE_PATTERN_POOL:
        batch = pool.sample(BATCH_SIZE)
        x0 = batch.x
        x0[:1] = seed
        
        if DAMAGE_N:
            damage = 1.0-make_circle_masks(DAMAGE_N, h, w).numpy()[..., None]
            x0[-DAMAGE_N:] *= damage
       
    else:
        x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)
    

    ## Train
    if SERIE:
        x, loss, serie_CA = train_serie(x0)
    else:    
        x, loss = train_step(x0)

    if USE_PATTERN_POOL:
        batch.x[:] = x
        batch.commit()

    if isinf(loss) or isnan(loss):
        print("Exploto la perdida", "b:", b)
        exit(1)
    
    step_i = i
    loss_log = np.append(loss_log, loss.numpy())

    ### Save Training Data
    if step_i%200 == 0:
        if not os.path.isdir(f"train_log/{step_i:04d}"):
            os.mkdir(f"train_log/{step_i:04d}")
        
        plot_loss(loss_log, step_i)
        export_model(ca, step_i)
        save_loss(loss_log, step_i)
        
        if SERIE:
            visualize_series(serie_CA, step_i)
            #visualize_step_seed(x0, step_i)
        else:
            visualize_batch(x0, x, step_i)
        
        if USE_PATTERN_POOL:
            generate_pool_figures(pool, step_i)
        
        # Un pool de 1024 son 300-500 MB
        #save_pool(pool, step_i)

    #print('\r step: %d, log10(loss): %.3f'%(i, np.log10(loss)), end='')
    print('\r step: %d, loss: %f'%(i, loss), end='')


#  ======================= Export =======================
with open("ex_user.json", "w") as f:
    f.write(export_ca_to_webgl_demo(ca))