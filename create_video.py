import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img[:,:,:3]

output_file = "translation.mp4"
frame_width, frame_height = 128, 128
fps = 2.0
path_img1 = "data/images/catterpillar.png"
path_img2 = "data/images/butterfly.png"
path_img3 = "data/images/cpu.png"
path_img4 = "data/images/ccc_robot.png"

#img1 = load_image(path_img1)
#img2 = load_image(path_img2)
#img3 = load_image(path_img3)
#img4 = load_image(path_img4)
#img1 = cv2.resize(img1, (frame_height, frame_width), interpolation=cv2.INTER_AREA)
#img2 = cv2.resize(img2, (frame_height, frame_width), interpolation=cv2.INTER_AREA)
#img3 = cv2.resize(img3, (frame_height, frame_width), interpolation=cv2.INTER_AREA)
#img4 = cv2.resize(img4, (frame_height, frame_width), interpolation=cv2.INTER_AREA)



fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

white = (255, 255, 255)
black = (0,0,0)

frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
frame[:,:] = white
   

frame = cv2.circle(frame, (20, 110), 10, black, -1)
out.write(frame)

frame[:,:,:] = 255
frame = cv2.circle(frame, (110, 20), 10, black, -1)
out.write(frame)


    
#out.write(frame)


#out.write(img1)
#out.write(img2)
#out.write(img3)
#out.write(img4)

out.release()
print("Video saved as", output_file)
