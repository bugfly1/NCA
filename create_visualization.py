import cv2
import numpy as np
import os
from src.parameters import *
from src.CAModel import CAModel, CA1DModel
from src.Utils import to_rgb, to_rgb_premultiplied, imwrite

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

@tf.function
def make_circle_masks(n, h, w):
    x = tf.linspace(-1.0, 1.0, w)[None, None, :]
    y = tf.linspace(-1.0, 1.0, h)[None, :, None]
    center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
    r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
    x, y = (x-center[0])/r, (y-center[1])/r
    mask = tf.cast(x*x+y*y < 1.0, tf.float32)
    return mask


# https://stackoverflow.com/questions/54607447/opencv-how-to-overlay-text-on-video
def __draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_SIMPLEX
   scale = 0.4
   color = BLACK
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   end_x = pos[0] + txt_size[0][0] + margin
   end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


def create_video(model_path, n_iters_before, video_dims, fps, n_iters_video):
    ca = CAModel() if not ca_1d else CA1DModel
    ca.load_weights(model_path)
    modelo = model_path.split("/")[-3]
    output_file = f"{modelo}.mp4"

    if not ca_1d:
        grid_h, grid_w = 2*TARGET_PADDING + TARGET_SIZE, 2*TARGET_PADDING + TARGET_SIZE
    else:
        grid_h, grid_w = 1, 2*TARGET_PADDING + TARGET_SIZE
        
    n = grid_h
    seed = np.zeros([grid_h, grid_w, CHANNEL_N], dtype=np.float32)
    seed[grid_h//2, grid_w//2, 3:] = 1.0
    #seed = np.random.rand(n, n, CHANNEL_N)
    x = np.repeat(seed[None, ...], 1, 0)

    frame_width, frame_height = video_dims

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    def pad_until_video_resolution(grid):
        h, w, _ = grid.shape    
        grid = np.pad(grid, [((frame_height - h)//2, (frame_height - h)//2),
                            ((frame_width - w)//2, (frame_width - w)//2),
                            (0,0)]
                    )
        return grid

    # Realizamos iteraciones sin grabar (para revisar periodicidad)
    for j in range(n_iters_before):
        x = ca(x)

    # Grabamos el video
    for i in range(n_iters_video):
        x = ca(x)
        rgb = to_rgb(x[0]).numpy()
        
        if i == 1000 or i == 2000:
            damage = 1.0-make_circle_masks(1, grid_h, grid_w).numpy()[..., None]
            x = x.numpy()
            x[:1] *= damage
            x = tf.cast(x, tf.float32)
            
        rgb = np.uint8(rgb.clip(0,1)*255)
        #rgb = cv2.normalize(rgb, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # ahora mismo el video es de 40 x 40
        rgb = np.repeat(rgb, frame_height // grid_h, axis=0)
        rgb = np.repeat(rgb, frame_height // grid_w, axis=1)
        
        vis = pad_until_video_resolution(rgb)
        __draw_label(vis, f"step: {i+n_iters_before}", (frame_height, 10), WHITE)
        out.write(vis)
        
        print(f"\r step {i}/{n_iters_video}", end="")
        

    out.release()
    print("Video saved as", output_file)


if __name__ == "__main__":
    path = "5f_rgb_min/10000/10000.weights.h5"
    ca_1d = False
    create_video(
        model_path= path,
        n_iters_before=0,
        video_dims = (640, 480),
        fps=60,
        n_iters_video=3000
    )
        
    
