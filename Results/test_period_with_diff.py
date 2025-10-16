import cv2
import numpy as np
import os
from src.parameters import *
from src.CAModel import CAModel
from src.Utils import to_rgb, imwrite, to_rgba, load_target, imwrite
from src.loss import pixelWiseMSE
import matplotlib.pylab as pl

def plot(loss_log, name):
  pl.figure(figsize=(10, 4))
  pl.title(name)
  pl.plot(loss_log, '.', alpha=0.1)
  pl.savefig(f"{name}.jpg")
  #pl.show()
  pl.close()

pad_target = load_target(SRC_TARGET)
f0 = pad_target[0]
model_path="models/first_periodic/8000/8000.weights.h5"
model_path="3frames_real/10000/10000.weights.h5"


ca = CAModel()
ca.load_weights(model_path)


grid_h, grid_w = 2*TARGET_PADDING + TARGET_SIZE, 2*TARGET_PADDING + TARGET_SIZE
seed = np.zeros([grid_h, grid_w, CHANNEL_N], dtype=np.float32)
seed[grid_h//2, grid_w//2, 3:] = 1.0

x = np.repeat(seed[None, ...], 1, 0)

error_log = np.array([])

iter_n = 100_000_000
for i in range(iter_n):
    x = ca(x)
    error = pixelWiseMSE(x[0], f0)
    print(f"\r step: {i} , Porcentaje: {i/iter_n}", end="")
    error_log = np.append(error_log, error.numpy())

imwrite("f0.jpg", f0.numpy())
plot(error_log, "Diferencia al primer fotograma (3f)")