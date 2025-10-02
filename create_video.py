import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img[:,:,:3]

output_file = "square_to_circle(white).mp4"
frame_width, frame_height = 128, 128
fps = 2.0
path_img1 = "data/images/dcc_comprimido.png"
path_img2 = "data/images/charmander.png"
path_img3 = "data/images/cpu.png"

img1 = load_image(path_img1)
img2 = load_image(path_img2)
img3 = load_image(path_img3)
img3 = cv2.resize(img3, (frame_height, frame_width), interpolation=cv2.INTER_AREA)



fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
"""
for i in range(1):
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    frame[:,:,:] = 0
    
    cv2.rectangle(frame, (40, 40), (80, 80), (255,255,255), -1)
    
    out.write(frame)
    
for i in range(1):
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    frame[:,:,:] = 0
    
    cv2.circle(frame, (60, 60), 20, (255,255,255), -1)
    
    out.write(frame)
"""

out.write(img1)
out.write(img2)
out.write(img3)


out.release()
print("Video saved as", output_file)
