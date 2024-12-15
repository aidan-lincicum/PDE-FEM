import os
import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

image_files = []

for img_number in range(900): 
    image_files.append('imgs/fig' + str(img_number) + '.png') 

fps = 30

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('imgs/my_new_video.mp4')  