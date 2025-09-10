import os
import zipfile
import numpy as np
import tensorflow as tf
import cv2

from src.Utils import (load_target, load_user_image, imwrite, load_images_as_video, to_rgba, make_circle_masks, save_loss, load_loss_log, save_pool, load_pool,
                      export_model, visualize_batch, plot_loss, generate_pool_figures, export_ca_to_webgl_demo)
from src.CAModel import CAModel
from src.SamplePooling import SamplePool

from src.parameters import *

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


if not os.path.isdir(f"train_log"):
  os.mkdir(f"train_log")

# TODO:
# - Experimento: RNN
# - ¿Y si agregamos la semilla como frame 1?
# - Probar el colocar parametros constantes
# - Decidir la forma de entregar los frames del video
# - Poder manipular la seed de la visualizacion

# ============== Initialize Trainig ==================

pad_target = load_target(SRC_TARGET)
pad_target = tf.cast(pad_target, tf.float32)

n_frames = 1

### Load and pad target Image
if VIDEO:
  n_frames = pad_target.shape[0]
  
  if n_frames > BATCH_SIZE: # Sacar esto apenas se logre comparar la tupla de T en loss
    pad_target = pad_target[:BATCH_SIZE,:,:,:]

h, w, _ = pad_target.shape[-3:]

# We add invisible parameters to CA
seed = np.zeros([h, w, CHANNEL_N], np.float32)
# Set center cell alive for seed
seed[h//2, w//2, 3:] = 1.0
pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))

if VIDEO and ROLL:
  video_seed = np.pad(pad_target, [(0,0), (0,0),(0,0), (0, CHANNEL_N - 4)]).astype(np.float32)
  pool = SamplePool(x = np.repeat(video_seed, int(POOL_SIZE / n_frames), 0))
  video_seed = tf.cast(np.roll(video_seed, -1, axis=0), tf.float32)


ca = CAModel()
loss_log = np.array([])

## Trainer SetUp
lr = 2e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [2000], [lr, lr*0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)

def softmin(x):
  b = 100
  return (1/-b) * tf.math.log(tf.reduce_sum(tf.exp(-b*x), [-1]))


def pixelWiseMSE(x, target):
  return tf.reduce_mean(tf.square(to_rgba(x)-target), [-2, -3, -1])

### Loss Function
def loss_f(x):
  if not ROLL:
    return pixelWiseMSE(x, pad_target)
  else:
    rolls = np.array([np.roll(to_rgba(pad_target), k) for k in range(n_frames)])
    MSE = tf.convert_to_tensor([pixelWiseMSE(x, rolls[k]) for k in range(n_frames)], dtype=tf.float32)
    return softmin(MSE)
    

### Training functions
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
        
      #print("loss", loss)
  grads = g.gradient(loss, ca.weights)
  grads = [g/(tf.norm(g)+1e-8) for g in grads]
  trainer.apply_gradients(zip(grads, ca.weights))
  return x, loss


### Load Checkpoint
if START_TRAINING_FROM_SAVE_POINT:
  begining = SAVE_POINT
  ca.load_weights(f"train_log/{SAVE_POINT}/{SAVE_POINT}.weights.h5")
  loss_log = load_loss_log(f"train_log/{SAVE_POINT}/{SAVE_POINT}_loss.npy")
  x_pool = load_pool(f"train_log/{SAVE_POINT}/{SAVE_POINT}_pool.npy")
  pool = SamplePool(x=x_pool)
else:
  begining = 0

  
# ========================= Training Loop =====================
for i in range(begining, 8000+1):
  ### Generate input grids for CA
  
  if ROLL and USE_PATTERN_POOL:
    batch = pool.sample(n_frames)
    x0 = batch.x
    if DAMAGE_N:
      damage = 1.0-make_circle_masks(DAMAGE_N, h, w).numpy()[..., None]
      x0[-DAMAGE_N:] *= damage
    
  elif USE_PATTERN_POOL:
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
      
  
  else:
    x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)
    

  ## Train
  x, loss = train_step(x0)

  if USE_PATTERN_POOL:
    batch.x[:] = x
    batch.commit()

  step_i = i
  loss_log = np.append(loss_log, loss.numpy())

  ### Save Training Data
  if step_i%100 == 0:
    if not os.path.isdir(f"train_log/{step_i:04d}"):
      os.mkdir(f"train_log/{step_i:04d}")
    
    generate_pool_figures(pool, step_i)
    visualize_batch(x0, x, step_i)
    plot_loss(loss_log, step_i)
    export_model(ca, step_i)
    save_loss(loss_log, step_i)
    
    # Un pool de 1024 son 300-500 MB
    #save_pool(pool, step_i)

  print('\r step: %d, log10(loss): %.3f'%(i, np.log10(loss)), end='')

#  ======================= Export =======================
with open("ex_user.json", "w") as f:
  f.write(export_ca_to_webgl_demo(ca))