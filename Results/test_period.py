import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from src.parameters import *
from src.CAModel import CAModel
from src.Utils import to_rgba, imwrite, load_target
from src.loss import pixelWiseMSE

# Crea imagenes de estados i en lista_estados de la secuencia del NCA, ademas de imagen de la secuencia
def create_states_images(model_path: str, nombre_modelo: str, image_per_state: bool, lista_estados: list):
    ca = CAModel()
    ca.load_weights(model_path)
    grid_h, grid_w = 2*TARGET_PADDING + TARGET_SIZE, 2*TARGET_PADDING + TARGET_SIZE
    seed = np.zeros([grid_h, grid_w, CHANNEL_N], dtype=np.float32)
    seed[grid_h//2, grid_w//2, 3:] = 1.0
    x = np.repeat(seed[None, ...], 1, 0)
    
    
    lista_vis = np.zeros((len(lista_estados), grid_h, grid_w, 4), dtype=np.float32)

    j = 0
    for i in range(max(lista_estados) + 1):
        x = ca(x)
        if i in lista_estados:
            vis = to_rgba(x[0]).numpy()
            if image_per_state: imwrite(f"{nombre_modelo}/{i}.jpg", vis)
            lista_vis[j] = vis
            j += 1
            
    return lista_vis

SRC_TARGET = "data/Videos/heavy_diff.mp4"
pad_target = load_target(SRC_TARGET)
imwrite("target_period.jpg", to_rgba(np.hstack(pad_target.numpy())))




puntos = []
x =[]
y = []
for i in range(-5, 6):
    tau = 3
    T = 8 + i
    lista_vis = create_states_images(
                model_path="models/first_periodic/8000/8000.weights.h5",
                nombre_modelo=f"first_periodic_T={T}",
                image_per_state = False,
                #lista_estados=[3*tau*T, 4*tau*T, 5*tau*T, 6*tau*T, 7*tau*T]
                lista_estados=[i*tau*T for i in range(0,100)],
            )

    sum_error = 0
    for img in lista_vis:
        sum_error += tf.reduce_min([pixelWiseMSE(img, f) for f in pad_target])
        sum_error = sum_error.numpy().item()

    x.append(T)
    y.append(sum_error)
    print(f"T={T}, error:", sum_error)
    
"""
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_aspect('equal')
ax.grid(True, which='both')

plt.show()
"""