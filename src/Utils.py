import numpy as np
import base64
import tensorflow as tf
import json
import matplotlib.pylab as pl
from IPython.display import Image
from tensorflow.python.framework import convert_to_constants
from google.protobuf.json_format import MessageToDict

import os
import cv2
import imageio
from src.CAModel import CAModel
from src.SamplePooling import SamplePool

from src.parameters import *
 
 
def load_training(checkpoint=SAVE_POINT):
    ca = CAModel()
    ca.load_weights(f"train_log/{SAVE_POINT}/{SAVE_POINT}.weights.h5")
    loss_log = load_loss_log(f"train_log/{SAVE_POINT}/{SAVE_POINT}_loss.npy")
    x_pool = load_pool(f"train_log/{SAVE_POINT}/{SAVE_POINT}_pool.npy")
    pool = SamplePool(x=x_pool)
    return ca, loss_log, pool

# Loading

## Especificamente hecho para mp4
def load_user_video(path, max_size=TARGET_SIZE, padding=TARGET_PADDING):
  frames = []
  cap = cv2.VideoCapture(path)
  
  if not cap.isOpened():
    raise ValueError("Error: Could not open video file")
  
  ret = True
  dims = (max_size, max_size)
  while ret:
    ret, img = cap.read()
    if ret:
      img = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)
      frames.append(img)
    video = np.stack(frames, axis=0)

  # Agregamos el canal Alpha
  video = np.pad(video, [(0, 0), (0, 0), (0, 0), (0,1)])
  video[:,:,:,3] = 255  # colocamos todo como alpha = 1 ya que mp4 no posee alpha channel
  
  p = padding
  video = np.pad(video, [(0, 0), (p, p), (p, p), (0,0)])
  video = np.float32(video) / 255
  
  return video

def load_images_as_video(dirpath, max_size=TARGET_SIZE, padding=TARGET_PADDING):
  images = os.listdir(dirpath)
  n_frames = len(images)
  
  dims = (max_size, max_size)
  p = padding
  target_video = np.zeros((n_frames, TARGET_SIZE+2*TARGET_PADDING, TARGET_SIZE+2*TARGET_PADDING, 4))
  for img_path, i in zip(images, range(n_frames)):
    path = os.path.join(dirpath, img_path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)
    
    if img.shape[-1] == 3:
      img = np.pad(img, [(0, 0), (0, 0), (0,1)])
      img[:,:,3] = 255
    
    img = np.float32(img) / 255
    
    target_video[i] = np.pad(img, [(p, p), (p, p), (0, 0)])
    
  return target_video

def load_gif(path, max_size=TARGET_SIZE, padding=TARGET_PADDING):
  gif = imageio.mimread(path)
  p = padding
  for i in range(len(gif)):
    gif[i] = cv2.resize(gif[i], (max_size, max_size))
    gif[i] = np.pad(gif[i], [(p, p), (p, p), (0, 0)])
  
    if gif[i].shape[-1] != 4:
      alpha_channel = np.full(gif[i].shape[:-1], 255, dtype=np.float32)
      img_rgba = np.dstack((gif[i], alpha_channel))
      gif[i] = img_rgba
      

  temp = np.zeros([len(gif), *gif[0].shape])
  for i in range(len(gif)):
    temp[i] = gif[i]
  
  gif = temp
  gif = np.float32(gif) / 255

  return gif

def load_user_image(image, max_size=TARGET_SIZE, padding=TARGET_PADDING):
  img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
  img = cv2.resize(img, (max_size, max_size))
  p = padding
  img = np.pad(img, [(p, p), (p, p), (0, 0)])
  img = np.float32(img) / 255.0
  return img

def imwrite(path, img):
  if img.shape[-1] > 4:
    img = to_rgba(img)
  img = img * 255
  cv2.imwrite(path, img)
  return


def load_target(path):
    if path.endswith(".png"):
        return load_user_image(path)
    elif path.endswith(".mp4"):
        return load_user_video(path)
    elif path.endswith(".gif"):
        return load_gif(path)
    elif os.path.isdir(path):
        return load_images_as_video(path)
    else:
        raise ValueError("Target format not supported")
    
    
# Visual utilities
def to_rgba(x):
  return x[..., :4]

def to_alpha(x):
  return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

def to_rgb(x):
  # assume rgb premultiplied by alpha
  rgb, a = x[..., :3], to_alpha(x)
  return 1.0-a+rgb



def make_seed(size, n=1):
  x = np.zeros([n, size, size, CHANNEL_N], np.float32)
  x[:, size//2, size//2, 3:] = 1.0
  return x

def tile2d(a, w=None):
  a = np.asarray(a)
  if w is None:
    w = int(np.ceil(np.sqrt(len(a))))
  th, tw = a.shape[1:3]
  pad = (w-len(a))%w
  a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
  h = len(a)//w
  a = a.reshape([h, w]+list(a.shape[1:]))
  a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
  return a

def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img


@tf.function
def make_circle_masks(n, h, w):
  x = tf.linspace(-1.0, 1.0, w)[None, None, :]
  y = tf.linspace(-1.0, 1.0, h)[None, :, None]
  center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
  r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
  x, y = (x-center[0])/r, (y-center[1])/r
  mask = tf.cast(x*x+y*y < 1.0, tf.float32)
  return mask

## ============ Save Training Data ============
def generate_pool_figures(pool, step_i):
  tiled_pool = tile2d(to_rgb(pool.x[:49]))
  fade = np.linspace(1.0, 0.0, 72)
  ones = np.ones(72)
  tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None]
  tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
  tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
  tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]
  imwrite('train_log/%04d/%04d_pool.jpg'%(step_i, step_i), tiled_pool)

def visualize_batch(x0, x, step_i):
  vis0 = np.hstack(to_rgb(x0).numpy())
  vis1 = np.hstack(to_rgb(x).numpy())
  vis = np.vstack([vis0, vis1])
  imwrite('train_log/%04d/batches_%04d.jpg'%(step_i, step_i), vis)
  #print('batch (before/after):')
  #imshow(vis)


def visualize_target(target):
  vis = np.hstack(to_rgb(target).numpy())
  imwrite('train_log/target.jpg', vis)
  #print('batch (before/after):')
  #imshow(vis)

def visualize_series(serie_CA, step_i):
  vis = np.hstack(to_rgb(serie_CA).numpy())
  imwrite('train_log/%04d/serie_%04d.jpg'%(step_i, step_i), vis)
  cv2.destroyAllWindows()
  #print('batch (before/after):')
  #imshow(vis)
  
def visualize_step_seed(seed, step_i):
  seed = to_rgb(seed[0]).numpy()
  imwrite('train_log/%04d/seed_%04d.jpg'%(step_i, step_i), seed)
  cv2.destroyAllWindows()
  
 
  
  
def save_rolls(serie_temporal):
    n_frames = len(serie_temporal)
    for k in range(n_frames):
        visualize_series(tf.roll(serie_temporal, k, axis=0), k)
        

def plot_loss(loss_log, step_i):
  pl.figure(figsize=(10, 4))
  pl.title('Loss history (log10)')
  pl.plot(np.log10(loss_log), '.', alpha=0.1)
  pl.savefig('train_log/%04d/%04d_loss.jpg'%(step_i, step_i))
  pl.close()
  #pl.show()

def export_model(ca, step_i):
  base_fn = f'train_log/{step_i:04d}/{step_i:04d}.weights.h5'
  ca.save_weights(base_fn)

  cf = ca.call.get_concrete_function(
      x=tf.TensorSpec([None, None, None, CHANNEL_N]),
      fire_rate=tf.constant(0.5),
      angle=tf.constant(0.0),
      step_size=tf.constant(1.0))
  cf = convert_to_constants.convert_variables_to_constants_v2(cf)
  graph_def = cf.graph.as_graph_def()
  graph_json = MessageToDict(graph_def)
  graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
  model_json = {
      'format': 'graph-model',
      'modelTopology': graph_json,
      'weightsManifest': [],
  }
  with open(base_fn+'.json', 'w') as f:
    json.dump(model_json, f)

def save_loss(loss_log, step_i):
  path = f"train_log/{step_i:04d}/{step_i:04d}_loss.npy"
  with open(path, 'wb') as f:
    np.save(f, loss_log)
    
def save_pool(pool, step_i):
  path = f"train_log/{step_i:04d}/{step_i:04d}_loss.npy"
  with open(path, 'wb') as f:
    np.save(f, pool.x)
    
def load_pool(file):
  with open(file, 'rb') as f:
    x = np.load(file)
  return x
    
def load_loss_log(file):
  with open(file, 'rb') as f:
    a = np.load(file)
  return a


### ============ Export to WebGL ============
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