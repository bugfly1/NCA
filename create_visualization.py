import cv2
import numpy as np
from src.parameters import *
from src.CAModel import CAModel
from src.Utils import to_rgb, imwrite

# Crea imagenes de estados i en lista_estados de la secuencia del NCA, ademas de imagen de la secuencia
def create_states_images(model_path: str, nombre_modelo: str, image_per_state: bool, lista_estados: list):
    ca = CAModel()
    ca.load_weights(model_path)
    grid_h, grid_w = 2*TARGET_PADDING + TARGET_SIZE, 2*TARGET_PADDING + TARGET_SIZE
    seed = np.zeros([grid_h, grid_w, CHANNEL_N], dtype=np.float32)
    seed[grid_h//2, grid_w//2, 3:] = 1.0
    x = np.repeat(seed[None, ...], 1, 0)
    
    lista_vis = np.zeros((len(lista_estados), grid_h, grid_w, 3))

    j = 0
    for i in range(max(lista_estados) + 1):
        x = ca(x)
        if i in lista_estados:
            vis = to_rgb(x[0]).numpy()
            if image_per_state: imwrite(f"{nombre_modelo}_i={i}.jpg", vis)
            lista_vis[j] = vis
            j += 1
            
    vis = np.hstack(lista_vis)
    imwrite(f"{nombre_modelo}.jpg", vis)  

    return



def create_video(model_path, n_iters_before, fps, output_file, n_iters_video):
        
    ca = CAModel()
    ca.load_weights(model_path)
    grid_h, grid_w = 2*TARGET_PADDING + TARGET_SIZE, 2*TARGET_PADDING + TARGET_SIZE
    seed = np.zeros([grid_h, grid_w, CHANNEL_N], dtype=np.float32)
    seed[grid_h//2, grid_w//2, 3:] = 1.0
    x = np.repeat(seed[None, ...], 1, 0)

    frame_width, frame_height = 1280,720

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
    for i in range(n_iters_before):
        x = ca(x)

    # Grabamos el video
    for i in range(n_iters_video):
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


if __name__ == "__main__":
    """
    create_video(
        model_path="models/no_dup_video/10000/10000.weights.h5",
        n_iters_before=0,
        fps=15,
        output_file="NCA.mp4",
        n_iters_video=500
    )
    """
    

    T = 16 - 1
    tau = 3
    create_states_images(
        model_path="models/first_periodic/8000/8000.weights.h5",
        nombre_modelo="first_periodic",
        image_per_state = False,
        #lista_estados=[3*tau*T, 4*tau*T, 5*tau*T, 6*tau*T, 7*tau*T]
        lista_estados=[i*tau*T for i in range(3,20)]
    )
    
    