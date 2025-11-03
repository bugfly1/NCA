import numpy as np
import tensorflow as tf
import cv2
from src.CAModel import CAModel
from src.Utils import imwrite, to_rgb_premultiplied, to_rgb
from src.parameters import *

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
   


def create_video_grafting(ca1, ca2, mask, grid_dims, n_iters_before, video_dims, fps, output_file, n_iters_video):        
    grid_h, grid_w = grid_dims
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
        x1, x2 = ca1(x), ca2(x)
        x = x1 + (x2 - x1)*mask

    # Grabamos el video
    for i in range(n_iters_video):
        # Grafteamos
        x1, x2 = ca1(x), ca2(x)
        #print("shapes:", x1.shape, x2.shape, mask.shape)
        x = x1 + (x2 - x1)*mask
        rgb = to_rgb_premultiplied(x[0]).numpy()
        rgb = cv2.normalize(rgb, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # ahora mismo el video es de 40 x 40
        rgb = np.repeat(rgb, frame_height // grid_h, axis=0)
        rgb = np.repeat(rgb, frame_height // grid_w, axis=1)
        # ahora 720x720
        
        vis = pad_until_video_resolution(rgb)
        __draw_label(vis, f"step: {i+n_iters_before}", (frame_height, 10), WHITE)
        out.write(vis)


    out.release()
    print("Video saved as", output_file)




ca1 = CAModel()
ca1.load_weights("models/8000_control/8000.weights.h5")
ca2 = CAModel()
ca2.load_weights("models/2frames/2f_rgb_min/10000/10000.weights.h5")

h, w, CHANNEL_N = 40+32, 40+32, 16
# We add invisible parameters to CA
seed = np.zeros([h, w, CHANNEL_N], dtype=np.float32)
# Set center cell alive for seed
seed[h//2, w//2, 3:] = 1.0
#seed[:,:,:4] = pad_target[-1]


#mask = center_decay_matrix(40)
#imwrite("mask.jpg", mask)
mask = np.zeros([h, w, CHANNEL_N], dtype=np.float32)
mask[:,w//2:] = 1
#mask = center_decay_matrix(44, decay='exponential')
#mask = np.repeat(mask[..., None], CHANNEL_N, -1)

x = np.repeat(seed[None, ...], 1, 0)


create_video_grafting(
    ca1 = ca1,
    ca2 = ca2,
    mask = mask,
    grid_dims = (h, w),
    n_iters_before=0,
    video_dims = (640, 480),
    fps=30,
    output_file="grafting_vis.mp4",
    n_iters_video=5000
)