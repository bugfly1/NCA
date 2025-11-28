import numpy as np
import tensorflow as tf
import cv2
import os
import imageio
from src.parameters import TARGET_SIZE, TARGET_PADDING, PRECISION, NP_PRECISION, ALPHA, D1, VIDEO

## Especificamente hecho para mp4
def load_user_video(path, max_size=TARGET_SIZE) -> np.ndarray:
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
            video = np.stack(frames, axis=0, dtype=NP_PRECISION)
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

def load_user_image(image, max_size=TARGET_SIZE, padding=TARGET_PADDING) -> np.ndarray:
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (max_size, max_size))
    p = padding
    img = np.pad(img, [(p, p), (p, p), (0, 0)])
    img = np.float32(img) / 255.0
    return img


def add_alpha(x: np.ndarray) -> np.ndarray:
    n_canales = x.shape[-1]
    if n_canales > 4:
        raise ValueError("El input tiene mas de 4 canales")
    
    if n_canales == 4:
        return x
    
    axis_padding = [(0,0)] * len(x.shape)
    axis_padding[-1] = (0, 1)
    while n_canales != 3:   # Si faltan de rgb agregamos canales con todo valor = 0
        x = np.pad(x, axis_padding)
        n_canales = x.shape[-1]

    
    x = np.pad(x, axis_padding, mode="constant")
    x[..., 3] = 255  # colocamos todo como alpha = 1 ya que mp4 no posee alpha channel (Falta normalizarlo)
    return x
    
def add_padding(x: np.ndarray, p=TARGET_PADDING) -> tf.Tensor:
    axis_padding = [(0,0)] * len(x.shape)
    axis_padding[-3] = (p, p) # Height axis
    axis_padding[-2] = (p, p) # Width axis
    x = tf.pad(x, axis_padding, "CONSTANT")
    return x

def create_1D_target(w=100):
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    RED = (0,0,255)

    n_frames = 2
    COLORS = [BLACK, WHITE]

    if VIDEO:
        pad_target = np.zeros([n_frames, 1, w, 3], dtype=np.float32)
        
        for i in range(w):
            for n in range(n_frames):
                pad_target[n, 0, i, :3] = COLORS[(i + n) % n_frames]
    else:
        pad_target = np.zeros([1, w, 3], dtype=np.float32)
        
        for i in range(w):
            pad_target[0, i, :3] = COLORS[i % n_frames]
    
    return pad_target

def load_target(path: str) -> tf.Tensor:
    target = None
    if D1:
        return create_1D_target()
    
    if path.endswith(".png"):
        return load_user_image(path)
    elif path.endswith(".mp4"):
        target = load_user_video(path)
    elif path.endswith(".gif"):
        return load_gif(path)
    elif os.path.isdir(path):
        return load_images_as_video(path)
    else:
        raise ValueError("Target format not supported")
    
    if type(target) == type(None):
        raise ValueError(f"No se pudo cargar el input (Imagen, Video, GIF, etc.): {path}")    
    
    if ALPHA:
        target = add_alpha(target)

    target = add_padding(target)
    
    # Normalizamos los valores de todos los canales
    if PRECISION == tf.float64:
        target = np.float64(target) / 255
    else:
        target = np.float32(target) / 255
    
    return target
    