import cv2
import numpy as np

output_file = "square_to_circle(white).mp4"
frame_width, frame_height = 120, 120
fps = 20.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

for i in range(10):
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    frame[:,:,:] = 0
    
    cv2.rectangle(frame, (40, 40), (80, 80), (255,255,255), -1)
    
    out.write(frame)
    
for i in range(10):
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    frame[:,:,:] = 0
    
    cv2.circle(frame, (60, 60), 20, (255,255,255), -1)
    
    out.write(frame)


out.release()
print("Video saved as", output_file)
