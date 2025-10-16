import cv2
import numpy as np
import os
from src.parameters import *
from src.CAModel import CAModel
from src.Utils import to_rgb, imwrite, to_rgba
from src.loss import pixelWiseMSE
import matplotlib.pylab as pl


def plot(loss_log, name):
  pl.figure(figsize=(10, 4))
  #pl.title('Loss history (log10)')
  #pl.plot(np.log10(loss_log), '.', alpha=0.1)
  pl.title(name)
  pl.plot(loss_log, '.', alpha=0.1)
  pl.savefig(f"{name}.jpg")
  #pl.show()
  pl.close()



model_path="models/first_periodic/8000/8000.weights.h5"

ca1 = CAModel()
ca1.load_weights(model_path)
ca2 = CAModel()
ca2.load_weights(model_path)


grid_h, grid_w = 2*TARGET_PADDING + TARGET_SIZE, 2*TARGET_PADDING + TARGET_SIZE
seed = np.zeros([grid_h, grid_w, CHANNEL_N], dtype=np.float32)
seed[grid_h//2, grid_w//2, 3:] = 1.0

x1 = np.repeat(seed[None, ...], 1, 0)
x2 = np.repeat(seed[None, ...], 1, 0)

error_log = np.array([])

for i in range(1000):
    x1 = ca1(x1)
    x2 = ca2(x2)
    error = pixelWiseMSE(x1, to_rgba(x2))
    #tf.print(f"{i}:", error)
    error_log = np.append(error_log, error.numpy())

plot(error_log, "Diferencia con mismo S (firerate = 1)")
    

    
    

