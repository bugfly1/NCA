import numpy as np
import tensorflow as tf
import cv2
from src.CAModel import CAModel
from src.Utils import imwrite, to_rgb_premultiplied, to_rgb
from src.parameters import *
import matplotlib.pyplot as plt

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)



# https://www.geeksforgeeks.org/python/how-to-generate-2-d-gaussian-array-using-numpy/
def gaussian_filter(kernel_size, sigma=1, muu=0):
    # Initializing value of x, y as grid of kernel size in the range of kernel size
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2 + y**2)

    # Normal part of the Gaussian function
    normal = 1 / (2 * np.pi * sigma**2)

    # Calculating Gaussian filter
    gauss = np.exp(-((dst - muu)**2 / (2.0 * sigma**2))) * normal

    return gauss  # Return the calculated Gaussian filter


# ChatGPT
def center_decay_matrix(size, decay='linear'):
    """
    Create a matrix where values diminish from the center.
    
    Args:
        size (int or tuple): Size of the matrix (e.g., 5 or (5, 7))
        decay (str): 'linear' or 'exponential' decay pattern.
    
    Returns:
        np.ndarray: The resulting matrix.
    """
    # Handle square or rectangular matrices
    if isinstance(size, int):
        rows = cols = size
    else:
        rows, cols = size
    
    # Create grid of coordinates
    y, x = np.ogrid[:rows, :cols]
    
    # Compute distance from center
    cy, cx = (rows - 1) / 2, (cols - 1) / 2
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Normalize distance to [0, 1]
    distance /= distance.max()
    
    # Apply decay function
    if decay == 'linear':
        matrix = 1 - distance
    elif decay == 'exponential':
        matrix = np.exp(-3 * distance)  # you can tune the 3 for steeper/softer decay
    else:
        raise ValueError("decay must be 'linear' or 'exponential'")
    
    return matrix

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
   

def plot_grid_values(x):
    grid_state = x[0].numpy()
    y = grid_state.flatten()
    plt.figure(figsize=(10, 4))
    plt.plot(y, alpha=0.1, linewidth=2.0)
    plt.show()
    plt.close()


def join_ca_outputs(x, ca1, ca2, mask, i):
    x1, x2 = ca1(x), ca2(x)
    if USE_MASK:
        x = x1 + (x2 - x1)*mask
    elif USE_TIME:
        x = x1 + (x2 - x1)*np.clip(((i-100)/1000.0), 0, 1)
    else:
        print("\nELIGE UNA FORMA DE GRAFTING")
        exit(1)
    
    return x

def create_video_grafting(ca1, ca2, mask, grid_dims, n_iters_before, video_dims, fps, output_file, n_iters_video):        
    grid_h, grid_w = grid_dims
    seed = np.zeros([grid_h, grid_w, CHANNEL_N], dtype=np.float32)
    ratio_w_h = w//h
    seed[grid_h//2, grid_w//(2*ratio_w_h), 3:] = 1.0
    seed[grid_h//2, (4*ratio_w_h-1)*grid_w//(4*ratio_w_h), 3:] = 1.0
    
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
        x = join_ca_outputs(x, ca1, ca2, mask, j)

    show = True
    # Grabamos el video
    for i in range(n_iters_video):
        # Grafteamos
        x = join_ca_outputs(x, ca1, ca2, mask, i)

        
        #if i % 500 == 0:
        #    plot_grid_values(x)   
        
        rgb = to_rgb(x[0]).numpy()
        rgb = cv2.normalize(rgb, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # ahora mismo el video es de 40 x 40
        rgb = np.repeat(rgb, frame_height // grid_h, axis=0)
        rgb = np.repeat(rgb, frame_height // grid_w, axis=1)
        # ahora 720x720
        
        vis = pad_until_video_resolution(rgb)
        __draw_label(vis, f"step: {i+n_iters_before}", (frame_height, 10), WHITE)
        out.write(vis)
        
        print(f"\r step {i}/{n_iters_video}", end="")


    out.release()
    print("\nVideo saved as", output_file)

# ChatGPT (Es correcto, lo revise)
def mask_from_video(size, radius=0.7, decay=8.0):
    # Create a squared linspace (same as torch.linspace(-1, 1, W)**2)
    r = tf.linspace(-1.0, 1.0, size) ** 2

    # Equivalent of (r + r[:, None]).sqrt()
    r_matrix = tf.sqrt(r + tf.reshape(r, (-1, 1)))

    # Equivalent of ((0.6 - r) * 8.0).sigmoid()
    mask = tf.sigmoid((radius - r_matrix) * decay)

    # Esto lo hace un circulo, en ves de una distribucion sigmoid
    #mask = tf.cast(mask < 0.6, tf.float32)
    # Convert to numpy for plotting
    return mask.numpy()

def show_mask(mask):
    plt.contourf(mask)
    plt.colorbar()
    plt.axis('equal')
    plt.show()
    return

Experiments = {
    "original": [
        "models/original/8000_control/8000.weights.h5", "models/original/lizard_original/10000/10000.weights.h5"
    ],
    "series": [
        "models/2frames/2f_rgb_min/10000/10000.weights.h5", "models/3frames/3f_rgb_min/10000/10000.weights.h5"
    ],
    "gol": [
        "2f_gol_3segments/10000/10000.weights.h5", "2f_gol/10000/10000.weights.h5"
    ]
}

CAs = Experiments["gol"]
ca1 = CAModel()
ca1.load_weights(CAs[0])

ca2 = CAModel()
ca2.load_weights(CAs[1])


h, w, CHANNEL_N = 80, 80, CHANNEL_N
#w *= 2

#mask = center_decay_matrix(40)
#imwrite("mask.jpg", mask)
#mask = np.zeros([h, w, CHANNEL_N], dtype=np.float32)
#mask[:,w//2:] = 1
#mask[:,w//2 - 5: w//2 + 5] = 0.5
#mask = center_decay_matrix(h, decay='exponential')
mask = mask_from_video(h, 0.7, 20.0)


#show_mask(mask)

mask = mask[None, ..., None]


USE_MASK = True
USE_TIME = False

print("Comienza Grafting")
create_video_grafting(
    ca1 = ca1,
    ca2 = ca2,
    mask = mask,
    grid_dims = (h, w),
    n_iters_before=0,
    video_dims = (640, 480),
    fps=30,
    output_file="grafting_vis.mp4",
    n_iters_video=2000
)