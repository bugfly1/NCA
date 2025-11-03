from src.CAModel import CAModel
from src.Utils import *
from src.load_target import load_target
from src.loss import loss_batch_tf
from src.SamplePooling import SamplePool
from math import isnan, isinf
from src.parameters import SRC_TARGET
import os

# TODO:
#   - parent: 2f, target: 3f. Le cuesta aumentar?
#   - Completamente distintas imagenes

parent = "2f_rgb_min"
target_name = SRC_TARGET
model_name = parent + "_" + target_name

pad_target = load_target(SRC_TARGET)

n_frames = pad_target.shape[0]
h, w, _ = pad_target.shape[-3:]

inicio = 10000
ca = CAModel()
ca.load_weights(f"{parent}/{inicio}/{inicio}.weights.h5")

## Trainer SetUp
lr = 2e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [2000], [lr, lr*0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)
prev_log = load_loss_log(f"{parent}/{inicio}/{inicio}_loss.npy")
loss_log = prev_log
# We add invisible parameters to CA
seed = np.zeros([h, w, CHANNEL_N], dtype=NP_PRECISION)
# Set center cell alive for seed
seed[h//2, w//2, 3:] = 1.0
#seed[:,:,:4] = pad_target[-1]
pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))

if SERIE:
    serie_temporal_extendida = np.repeat(pad_target, TAU, axis=0)
    # Duplica el video
    #serie_temporal_extendida = np.append(serie_temporal_extendida, serie_temporal_extendida, axis=0)
    n_frames = len(serie_temporal_extendida)
    visualize_target(serie_temporal_extendida)

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

# ========================= Training Loop =====================
for i in range(inicio, 20000+1):
  ### Generate input grids for CA

    if ROLL and USE_PATTERN_POOL:
        batch = pool.sample(n_frames)
        for i in range(n_frames):
            k = pool.sample()
        x0 = batch.x
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
    
    x, loss, serie_CA = train_serie(x0)



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
