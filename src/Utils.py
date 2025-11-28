import numpy as np
import base64
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from tensorflow.python.framework import convert_to_constants
from google.protobuf.json_format import MessageToDict

import cv2
from src.CAModel import CAModel

from src.parameters import *
 
  
# Visual utilities
def to_rgba(x) -> np.ndarray:
    return x[..., :4]

def to_alpha(x) -> tf.Tensor:
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

def to_rgb(x) -> tf.Tensor:
    return tf.cast(x[..., :3], PRECISION) # Hacemos cast solamente porque la funcion to_rgb entrega un tensor

def to_rgb_premultiplied(x) -> tf.Tensor: # Esta era la funcion original, la mantenemos porsiacaso
    if x.shape[-1] == 3:
        return to_rgb(x)
    # assume rgb premultiplied by alpha
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0-a+rgb
    
def imwrite(path, img):
    if img.shape[-1] > 4:
        img = to_rgba(img)
    img = img * 255
    cv2.imwrite(path, img)
    return

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

## ============ Load Training Data ============
   
def load_pool(file):
    with open(file, 'rb') as f:
        x = np.load(file)
    return x
    
def load_loss_log(file):
    with open(file, 'rb') as f:
        a = np.load(file)
    return a

def load_training(checkpoint=SAVE_POINT):
    ca = CAModel()
    ca.load_weights(f"train_log/{checkpoint}/{checkpoint}.weights.h5")
    loss_log = load_loss_log(f"train_log/{checkpoint}/{checkpoint}_loss.npy")
    #x_pool = load_pool(f"train_log/{SAVE_POINT}/{SAVE_POINT}_pool.npy")
    #pool = SamplePool(x=x_pool)
    return ca, loss_log#,pool



## ============ Save Training Data ============
def generate_pool_figures(pool, step_i):
    if ALPHA:
        tiled_pool = tile2d(to_rgb_premultiplied(pool.x[:49]))
    else:
        tiled_pool = tile2d(to_rgb(pool.x[:49]))

    fade = np.linspace(1.0, 0.0, 72)
    ones = np.ones(72)
    tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None]
    tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
    tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
    tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]
    imwrite('train_log/%04d/%04d_pool.jpg'%(step_i, step_i), tiled_pool)

def visualize_batch(x0, x, step_i):
    if x.shape[-3] != 1:
        if ALPHA:
            vis0 = np.hstack(to_rgb_premultiplied(x0).numpy())
            vis1 = np.hstack(to_rgb_premultiplied(x).numpy())
        else:
            vis0 = np.hstack(to_rgb(x0).numpy())
            vis1 = np.hstack(to_rgb(x).numpy())
        
        vis = np.vstack([vis0, vis1])
    else:
        vis0 = np.vstack(to_rgb(x0).numpy())
        vis1 = np.vstack(to_rgb(x).numpy())
        vis = np.hstack([vis0, vis1])
        
    imwrite('train_log/%04d/batches_%04d.jpg'%(step_i, step_i), vis)
    #print('batch (before/after):')
    #imshow(vis)


def visualize_target(target):
    if target.shape[-3] != 1:
        if ALPHA:
            vis = np.hstack(to_rgb_premultiplied(target).numpy())
        else:
            vis = np.hstack(to_rgb(target).numpy())
    else:
        vis = np.vstack(to_rgb(target).numpy())
            
    imwrite('train_log/target.jpg', vis)

def visualize_series(serie_CA, step_i):
    n_batches, n_frames, h, w, channels = serie_CA.shape
    vis = np.zeros((n_batches, h, n_frames*w, 3))
    for batch in range(n_batches):
        if ALPHA:
            line = np.hstack(to_rgb_premultiplied(serie_CA[batch]).numpy())
        else:
            line = np.hstack(to_rgb(serie_CA[batch]).numpy())
        vis[batch] = line
    vis = np.vstack(vis)

    imwrite('train_log/%04d/serie_%04d.jpg'%(step_i, step_i), vis)
  
def visualize_step_seed(seed, step_i):
    if ALPHA:
        seed = to_rgb_premultiplied(seed[0]).numpy()
    else:
        seed = to_rgb(seed[0]).numpy()
        
    imwrite('train_log/%04d/seed_%04d.jpg'%(step_i, step_i), seed)
    
  
def plot_loss(loss_log, step_i):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    
    if any(point <= 0 for point in loss_log):
        # No se puede graficar con el valor de log
        plt.plot(loss_log, '.', alpha=0.1)
        plt.savefig('train_log/%04d/%04d_loss.jpg'%(step_i, step_i))
    else:
        plt.plot(np.log10(loss_log), '.', alpha=0.1)
        plt.savefig('train_log/%04d/%04d_loss(log10).jpg'%(step_i, step_i))
    
    plt.ylabel("Loss")
    plt.xlabel("Train step")
    plt.close()
    
def plot_tbar_no_seed(tbar_log, step_i):
    steps = np.arange(step_i + 1)   # Llegan aca antes de imprimir, por eso es como medio raro
    steps = np.repeat(steps, BATCH_SIZE - 1, axis=0)
    plt.figure(figsize=(10, 4))
    plt.scatter(steps, tbar_log, s=np.repeat(2, len(steps)))
    plt.title('Valores de tbar por paso de entrenamiento')
    plt.xlabel("Train step")
    plt.ylabel("Tbar values")
    plt.savefig('train_log/%04d/%04d_Valores_tbar.jpg'%(step_i, step_i))
    plt.close()

def plot_tbar_seed(tbar_seed_log, step_i):
    steps = np.arange(step_i + 1)   # Llegan aca antes de imprimir, por eso es como medio raro
    steps = np.repeat(steps, 1, axis=0)
    plt.figure(figsize=(10, 4))
    plt.scatter(steps, tbar_seed_log, s=np.repeat(2, len(steps)))
    plt.title('Valores de tbar por paso de entrenamiento de secuencia seed')
    plt.xlabel("Train step")
    plt.ylabel("Tbar values")
    plt.savefig('train_log/%04d/%04d_Valores_tbar_seed.jpg'%(step_i, step_i))
    plt.close()
    pass

def plot_tbar(tbar_log, tbar_seed_log, step_i):
    plot_tbar_seed(tbar_seed_log, step_i)
    plot_tbar_no_seed(tbar_log, step_i)
    return
    

def export_model(ca, step_i):
    base_fn = f'train_log/{step_i:04d}/{step_i:04d}.weights.h5'
    ca.save_weights(base_fn)
    return

    cf = ca.call.get_concrete_function(
        x=tf.TensorSpec([None, None, None, CHANNEL_N], dtype=PRECISION),
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

def save_params():
    path = f"train_log/description.txt"
    if PRECISION == tf.float64:
        string_precision = "tf.float64"
    else:
        string_precision = "tf.float32"
        
    lines = [
        f"SRC_TARGET={SRC_TARGET.split("/")[-1]}",
        "",
        f"CHANNEL_N={CHANNEL_N}",
        f"TARGET_PADDING={TARGET_PADDING}",
        f"TARGET_SIZE={TARGET_SIZE}",
        "",
        f"b={b}",
        f"TAU={TAU}",
        f"T={T}",
        "",
        f"BATCH_SIZE={BATCH_SIZE}",
        f"POOL_SIZE={POOL_SIZE}",
        "",
        f"USE_PATTERN_POOL={bool(USE_PATTERN_POOL)}",
        f"DAMAGE_N={DAMAGE_N}",
        "",
        f"PRECISION={string_precision}",
        "",
        f"ALPHA={ALPHA}"
    ]
    
    lines_string = "\n".join(lines)
    
    with open(path, 'w') as f:
        f.write(lines_string)

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