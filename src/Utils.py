import numpy as np
import PIL
import io
import base64
import tensorflow as tf
import json
import matplotlib.pylab as pl
from IPython.display import Image
from tensorflow.python.framework import convert_to_constants
from google.protobuf.json_format import MessageToDict

import os
import cv2
import random
import imageio

from src.parameters import *


 
# Loading

## Especificamente hecho para mp4
def load_user_video(path, max_size=TARGET_SIZE, max_frames=MAX_FRAMES):
  frames = []
  cap = cv2.VideoCapture(path)
  ret = True
  dims = (max_size, max_size)
  while ret and len(frames) < max_frames:
    ret, img = cap.read()
    if ret:
      img = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)
      frames.append(img)
    video = np.stack(frames, axis=0)

  n_frames, h, w, channels = video.shape

  # Agregamos el canal Alpha
  video = np.pad(video, [(0, 0), (0, 0), (0, 0), (0,1)])
  video[:,:,:,3] = 255  # colocamos todo como alpha = 1 ya que mp4 no posee alpha channel
  video = np.float32(video) / 255
  return video

def load_images_as_video(dirpath=SRC_VIDEO, max_size=TARGET_SIZE, padding=TARGET_PADDING, max_frames=MAX_FRAMES):
  images = os.listdir(dirpath)
  n_frames = max_frames
  if len(images) < max_frames:
    n_frames = len(images)
  
  dims = (max_size, max_size)
  p = padding
  target_video = np.zeros((n_frames, TARGET_SIZE+2*TARGET_PADDING, TARGET_SIZE+2*TARGET_PADDING, 4))
  for img_path, i in zip(images, range(n_frames)):
    path = os.path.join(SRC_VIDEO, img_path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)
    
    if img.shape[-1] == 3:
      img = np.pad(img, [(0, 0), (0, 0), (0,1)])
      img[:,:,3] = 255
    
    img = np.float32(img) / 255
    
    target_video[i] = tf.pad(img, [(p, p), (p, p), (0, 0)])
    
  return target_video

def load_gif(path=SRC_VIDEO, max_size=TARGET_SIZE, padding=TARGET_PADDING):
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

def load_user_image_cv2(image, max_size=TARGET_SIZE):
  img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
  img = cv2.resize(img, (max_size, max_size))
  img = np.float32(img) / 255.0
  return img

def imwrite_cv2(path, img):
  if img.shape[-1] > 4:
    img = to_rgba(img)
  img = img * 255
  cv2.imwrite(path, img)
  return

# El original usa imagenes de 128x128
def load_user_image(image, max_size=TARGET_SIZE):
  with open(image, "rb") as f:
    user_image = f.read()

  # Supuestamente forza la conversion a imagenes con canal alpha
  img = PIL.Image.open(io.BytesIO(user_image)).convert("RGBA") 
  img.thumbnail((max_size, max_size), PIL.Image.LANCZOS)
  img = np.float32(img) / 255.0
  # premultiply RGB by Alpha
  img[..., :3] *= img[..., 3:]
  return img


# Visual utilities
def to_rgba(x):
  return x[..., :4]

def to_alpha(x):
  return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

def to_rgb(x):
  # assume rgb premultiplied by alpha
  rgb, a = x[..., :3], to_alpha(x)
  return 1.0-a+rgb

def get_living_mask(x):
  alpha = x[:, :, :, 3:4]
  # Cell is considered empty if there is no alpha > 0.1 cell in its
  # 3x3 neightborhood  
  return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 0.1

def make_seed(size, n=1):
  x = np.zeros([n, size, size, CHANNEL_N], np.float32)
  x[:, size//2, size//2, 3:] = 1.0
  return x


def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

def im2url(a, fmt='jpeg'):
  encoded = imencode(a, fmt)
  base64_byte_string = base64.b64encode(encoded).decode('ascii')
  return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

def imshow(a, fmt='jpeg'):
  Image(data=imencode(a, fmt))

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

def plot_loss(loss_log, step_i):
  pl.figure(figsize=(10, 4))
  pl.title('Loss history (log10)')
  pl.plot(np.log10(loss_log), '.', alpha=0.1)
  pl.savefig('train_log/%04d/%04d_loss.jpg'%(step_i, step_i))
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