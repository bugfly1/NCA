import cv2
import numpy as np
from src.parameters import *
from src.CAModel import CAModel
from src.Utils import to_rgb

ca = CAModel()
ca.load_weights(f"train_log/10000/10000.weights.h5")
grid_h, grid_w = 2*TARGET_PADDING + TARGET_SIZE, 2*TARGET_PADDING + TARGET_SIZE
seed = np.zeros([grid_h, grid_w, CHANNEL_N], dtype=np.float32)
seed[grid_h//2, grid_w//2, 3:] = 1.0
x = np.repeat(seed[None, ...], 1, 0)

output_file = "NCA.mp4"
frame_width, frame_height = 1280,720
fps = 15.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

def pad_until_video_resolution(grid):
    h, w, _ = grid.shape    
    grid = np.pad(grid, [((frame_height - h)//2, (frame_height - h)//2),
                         ((frame_width - w)//2, (frame_width - w)//2),
                         (0,0)]
                  )
    return grid

"""
for i in range(50000):
    x = ca(x)
    print(i)
"""

for i in range(500):
    x = ca(x)
    rgb = to_rgb(x[0]).numpy()
    rgb = cv2.normalize(rgb, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # ahora mismo el video es de 40 x 40
    rgb = np.repeat(rgb, 720 // grid_h, axis=0)
    rgb = np.repeat(rgb, 720 // grid_w, axis=1)
    # ahora 720x720
    
    vis = pad_until_video_resolution(rgb)
    out.write(vis)


out.release()
print("Video saved as", output_file)
