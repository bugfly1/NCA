import cv2
import numpy as np
import os
from src.parameters import *
from src.CAModel import CAModel
from src.Utils import to_rgb, to_rgb_premultiplied, imwrite

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

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
   

# Crea imagenes de estados i en lista_estados de la secuencia del NCA, ademas de imagen de la secuencia
def create_states_images(model_path: str, nombre_modelo: str, image_per_state: bool, lista_estados: list):
    ca = CAModel()
    ca.load_weights(model_path)
    grid_h, grid_w = 2*TARGET_PADDING + TARGET_SIZE, 2*TARGET_PADDING + TARGET_SIZE
    seed = np.zeros([grid_h, grid_w, CHANNEL_N], dtype=np.float32)
    seed[grid_h//2, grid_w//2, 3:] = 1.0
    x = np.repeat(seed[None, ...], 1, 0)
    
    if not os.path.isdir(nombre_modelo):
        os.mkdir(nombre_modelo)
    
    lista_vis = np.zeros((len(lista_estados), grid_h, grid_w, 3))

    j = 0
    for i in range(max(lista_estados) + 1):
        x = ca(x)
        if i in lista_estados:
            vis = to_rgb(x[0]).numpy()
            if image_per_state: imwrite(f"{nombre_modelo}/{i}.jpg", vis)
            lista_vis[j] = vis
            j += 1
            
    vis = np.hstack(lista_vis)
    imwrite(f"{nombre_modelo}/serie.jpg", vis)  

    return



def create_video(model_path, n_iters_before, video_dims, fps, output_file, n_iters_video):
        
    ca = CAModel()
    ca.load_weights(model_path)
    grid_h, grid_w = 2*TARGET_PADDING + TARGET_SIZE, 2*TARGET_PADDING + TARGET_SIZE
    seed = np.zeros([grid_h, grid_w, CHANNEL_N], dtype=np.float32)
    seed[grid_h//2, grid_w//2, 3:] = 1.0
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
        rgb = cv2.normalize(rgb, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # ahora mismo el video es de 40 x 40
        rgb = np.repeat(rgb, frame_height // grid_h, axis=0)
        rgb = np.repeat(rgb, frame_height // grid_w, axis=1)
        
        vis = pad_until_video_resolution(rgb)
        __draw_label(vis, f"step: {i+n_iters_before}", (frame_height, 10), WHITE)
        out.write(vis)
        


    out.release()
    print("Video saved as", output_file)


if __name__ == "__main__":
    
    nombre_modelo = "3f_FireRate=1"
    create_video(
        model_path= f"{nombre_modelo}/9000/9000.weights.h5",
        n_iters_before=0,
        video_dims = (640, 480),
        fps=30,
        output_file=f"{nombre_modelo}.mp4",
        n_iters_video=5000
    )
    
    
    
    """
    T = 4
    tau = 3
    offset = 24
    create_states_images(
        model_path="models/first_periodic/8000/8000.weights.h5",
        nombre_modelo=f"first_periodic_T={T}",
        image_per_state = False,
        #lista_estados=[3*tau*T, 4*tau*T, 5*tau*T, 6*tau*T, 7*tau*T]
        lista_estados=[i*tau*T + offset for i in range(0,50)],
    )
    """
        
    