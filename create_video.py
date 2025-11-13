import cv2
import numpy as np
import os


WHITE = (255, 255, 255)
BLACK = (0,0,0)

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img[:,:,:3]

output_file = "translation.mp4"
frame_width, frame_height = 128, 128
fps = 2.0
image_dir = "data/images"

paths = [
    "water1.jpg",
    "water2.jpg"
]


for i in range(len(paths)):
    paths[i] = os.path.join(image_dir, paths[i])

print(paths)

images = []
for img_path in paths:
    images.append(load_image(img_path))

for i in range(len(images)):
    images[i] = cv2.resize(images[i], (frame_height, frame_width), interpolation=cv2.INTER_AREA)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
frame[:,:] = WHITE
   
#
#frame = cv2.circle(frame, (20, 110), 10, BLACK, -1)
#out.write(frame)
#
#frame[:,:,:] = 255
#frame = cv2.circle(frame, (110, 20), 10, BLACK, -1)
#out.write(frame)
#
#out.write(frame)


for i in range(len(images)):
    out.write(images[i])

out.release()
print("Video saved as", output_file)
