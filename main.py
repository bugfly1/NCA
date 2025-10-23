import os
import numpy as np
import tensorflow as tf
from src.Utils import (imwrite, to_rgba, make_circle_masks, save_loss, load_training,
                      export_model, visualize_batch, visualize_target, visualize_series, plot_loss, 
                      generate_pool_figures, to_rgb, save_params)
from src.load_target import load_target
from src.CAModel import CAModel
from src.SamplePooling import SamplePool
from src.parameters import *
from src.loss import loss_batch_tf, loss_serie, loss_f
from math import isnan, isinf

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['XLA_FLAGS'] = ' --xla_gpu_deterministic_ops=true'

if PRECISION == tf.float64:
    tf.keras.backend.set_floatx('float64')
    tf.keras.mixed_precision.set_global_policy("float64")

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
# - Probar entregar parametros constantes como input
# - Probar el agregar una medicion de diferencia entre elementos de la sequencia en la funcion de perdida
# - En perdida, comparar por rgb en vez de rgba. El que haya estado asi todo este rato significa que 
#       le estabamos pidiendo que todas las celulas estuvieran vivas (puse que todo el alpha fuera 1), talvez eso causa el
#       crecimiento descontrolado al principio
# - Usar como seed el ultimo frame y empezar desde ahi
# - Agregar lo que sea que hicieron aqui cuando lo liberen https://cells2pixels.github.io/
# - Probar traslacion desde la dimension de frecuencia (DFT de las imagenes de input) https://tuprints.ulb.tu-darmstadt.de/29695/1/DemocratizingLearning_JohnKalkhof.pdf#page=14.62
#                                                                                       (Instant Global Communication through the Fourier Space)
# - El agregar el laplaciano rompe el codigo de export_ca_webgl_demo
# - Probar la norma euclidiana como funcion de perdida
# ============== Initialize Trainig ==================

# Input loading
pad_target = load_target(SRC_TARGET)
n_frames = 1

### Load and pad target Image
if VIDEO:
    n_frames = pad_target.shape[0]

h, w, _ = pad_target.shape[-3:]

# We add invisible parameters to CA
seed = np.zeros([h, w, CHANNEL_N], dtype=NP_PRECISION)
# Set center cell alive for seed
seed[h//2, w//2, 3:] = 1.0
#seed[:,:,:4] = pad_target[-1]
pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))

save_params()
visualize_target(pad_target)

if SERIE:
    serie_temporal_extendida = np.repeat(pad_target, TAU, axis=0)
    # Duplica el video
    #serie_temporal_extendida = np.append(serie_temporal_extendida, serie_temporal_extendida, axis=0)
    n_frames = len(serie_temporal_extendida)
    visualize_target(serie_temporal_extendida)

### Load Checkpoint
if START_TRAINING_FROM_SAVE_POINT:
    begining = SAVE_POINT
    ca, loss_log = load_training()
else:
    begining = 0

def train_serie(x):
    if SERIE_CORTA:
        n_frames_local = min(n_frames, 2*TAU)
    else:
        n_frames_local = n_frames
    iter_n = T
    with tf.GradientTape() as g:
        lista_serie = tf.TensorArray(dtype=PRECISION, size=n_frames_local)
        for j in tf.range(n_frames_local):
            for i in tf.range(iter_n):
                x = ca(x)
            lista_serie.write(j,x).mark_used()
        serie_CA = lista_serie.stack()
        # Changes shape from (n_frames, n_batch, h, w, channels) to
        # (n_batch, n_frames, h, w, channels)
        serie_CA = tf.transpose(serie_CA, perm=[1,0,2,3,4])
        #loss = loss_serie(serie_CA, serie_temporal_extendida)
        loss = loss_batch_tf(serie_CA, serie_temporal_extendida)

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


x = 0
# ========================= Training Loop =====================
for i in range(begining, 10000+1):
  ### Generate input grids for CA

    if ROLL and USE_PATTERN_POOL:
        batch = pool.sample(n_frames)
        for i in range(n_frames):
            k = pool.sample()
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
    
    
    #start = time.time()
    ## Train
    if SERIE:
        x, loss, serie_CA = train_serie(x0)
    else:    
        x, loss = train_step(x0)

    #end = time.time()
    #print("elapsed time: ", end-start)

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
#with open("ex_user.json", "w") as f:
#    f.write(export_ca_to_webgl_demo(ca))