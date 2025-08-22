
import os
import base64
import zipfile
import json
import numpy as np
import tensorflow as tf

from src.Utils import (load_user_image, load_loss_log, to_rgba, save_loss, export_model,
                      visualize_batch, plot_loss, generate_pool_figures)
from src.CAModel import CAModel
from src.SamplePooling import SamplePool, make_circle_masks

from src.parameters import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# ============== Initialize Trainig ==================
#@title Initialize Training { vertical-output: true}

if (USER_VIDEO):
  frames = os.listdir(USER_VIDEO)
  target_video = []
  for frame in frames:
    target_video.append(os.path.join(USER_VIDEO, frame))

target_img = load_user_image(USER_IMAGE)

p = TARGET_PADDING
pad_target = tf.pad(target_img, [(p, p), (p, p), (0, 0)])

print(pad_target.shape)


# height and width of image
h, w = pad_target.shape[:2]

# We add invisible parameters to CA
seed = np.zeros([h, w, CHANNEL_N], np.float32)

# Set Alpha value off center to 1 (??????)
seed[h//2, w//2, 3:] = 1.0


def loss_f(x):
  return tf.reduce_mean(tf.square(to_rgba(x)-pad_target), [-2, -3, -1])

def loss_video(x, pad_target):
  return tf.reduce_mean(tf.square(to_rgba(x)-pad_target), [-2, -3, -1])

ca = CAModel()


loss_log = np.array([])

lr = 2e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [2000], [lr, lr*0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)

loss0 = loss_f(seed).numpy()
pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))


@tf.function
def train_step(x):
  iter_n = tf.random.uniform([], 64, 96, tf.int32)
  with tf.GradientTape() as g:
    for i in tf.range(iter_n):
      x = ca(x)
    loss = tf.reduce_mean(loss_f(x))
  grads = g.gradient(loss, ca.weights)
  grads = [g/(tf.norm(g)+1e-8) for g in grads]
  trainer.apply_gradients(zip(grads, ca.weights))
  return x, loss


@tf.function
def train_step_video(x0,frame):
  target_img = load_user_image(target_video[frame])
  p = TARGET_PADDING
  pad_target = tf.pad(target_img, [(p, p), (p, p), (0, 0)])
  
  time = 0
  iter_n = tf.random.uniform([], 64, 96, tf.int32)
  with tf.GradientTape() as g:
    for i in tf.range(iter_n):
      x = ca(x)
      time += 1
    loss = tf.reduce_mean(loss_video(x, pad_target))
  grads = g.gradient(loss, ca.weights)
  grads = [g/(tf.norm(g)+1e-8) for g in grads]
  trainer.apply_gradients(zip(grads, ca.weights))
  return x, loss


if START_TRAINING_FROM_SAVE_POINT:
  begining = int(SAVE_POINT)
  ca.load_weights(f"train_log/{SAVE_POINT}/{SAVE_POINT}.weights.h5")
  loss_log = load_loss_log(f"train_log/{SAVE_POINT}/{SAVE_POINT}.npy")
else:
  begining = 0



if not os.path.isdir(f"train_log"):
  os.mkdir(f"train_log")
  

first_loop = True
# ========================= Training Loop =====================
for i in range(begining, 8000+1):
  if USE_PATTERN_POOL:
    batch = pool.sample(BATCH_SIZE)
    x0 = batch.x
    print(x0.shape)
    loss_rank = loss_f(x0).numpy().argsort()[::-1]
    x0 = x0[loss_rank]
    x0[:1] = seed
    if DAMAGE_N:
      damage = 1.0-make_circle_masks(DAMAGE_N, h, w).numpy()[..., None]
      x0[-DAMAGE_N:] *= damage
  else:
    x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)
  
  if USER_VIDEO:
    loss = 0
    x = x0
    steps = 0
    for frame in range(len(target_video)):
      x, loss_step = train_step_video(x, frame)
      steps += tf.random.uniform([], 64, 96, tf.int32)
      loss += loss_step
  else:
    x, loss = train_step(x0)
    


  if USE_PATTERN_POOL:
    batch.x[:] = x
    batch.commit()

  step_i = i
  loss_log = np.append(loss_log, loss.numpy())

  if step_i%100 == 0 and not first_loop:
    if not os.path.isdir(f"train_log/{step_i:04d}"):
        os.mkdir(f"train_log/{step_i:04d}")
        
    generate_pool_figures(pool, step_i)
    visualize_batch(x0, x, step_i)
    plot_loss(loss_log, step_i)
    export_model(ca, f'train_log/{step_i:04d}/{step_i:04d}.weights.h5')
    save_loss(f"train_log/{step_i:04d}/{step_i:04d}.npy", loss_log)

  print('\r step: %d, log10(loss): %.3f'%(i, np.log10(loss)), end='')
  first_loop = False
  

#  ======================= Export =======================

def pack_layer(weight, bias, outputType=np.uint8):
  in_ch, out_ch = weight.shape
  assert (in_ch%4==0) and (out_ch%4==0) and (bias.shape==(out_ch,))
  weight_scale, bias_scale = 1.0, 1.0
  if outputType == np.uint8:
    weight_scale = 2.0*np.abs(weight).max()
    bias_scale = 2.0*np.abs(bias).max()
    weight = np.round((weight/weight_scale+0.5)*255)
    bias = np.round((bias/bias_scale+0.5)*255)
  packed = np.vstack([weight, bias[None,...]])
  packed = packed.reshape(in_ch+1, out_ch//4, 4)
  packed = outputType(packed)
  packed_b64 = base64.b64encode(packed.tobytes()).decode('ascii')
  return {'data_b64': packed_b64, 'in_ch': in_ch, 'out_ch': out_ch,
          'weight_scale': float(weight_scale), 'bias_scale': float(bias_scale),
          'type': outputType.__name__}

def export_ca_to_webgl_demo(ca, outputType=np.uint8):
  # reorder the first layer inputs to meet webgl demo perception layout
  chn = ca.channel_n
  w1 = ca.weights[0][0, 0].numpy()
  w1 = w1.reshape(chn, 3, -1).transpose(1, 0, 2).reshape(3*chn, -1)
  layers = [
      pack_layer(w1, ca.weights[1].numpy(), outputType),
      pack_layer(ca.weights[2][0, 0].numpy(), ca.weights[3].numpy(), outputType)
  ]
  return json.dumps(layers)

with zipfile.ZipFile('webgl_models8.zip', 'w') as zf:
  zf.writestr("ex_user.json", export_ca_to_webgl_demo(ca))
