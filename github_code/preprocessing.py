import os
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance

from utils.data_augument import img_contrast, img_shift, img_rotation, gaussain_blur, gaussain_noise, avg_blur, motion_blur


_max_filiter_size = 12  # for avg_blur and gaussain_blur
_sigma = 0  # for gaussain_blur

_mean = 0  # for gaussain_noise
_var = 164.1  # for gaussain_noise


_generate_quantity = 10

def random_noiseCV(img, varmax):
    mean = 0
    var = varmax * random.random()    
    sigma = var ** 0.5
    #if random.randint(0, 100)>90:
    #    return img #
        
    if img.min() < -0.01:
        low_clip = -1
    else:
        low_clip = 0    
        
    if img.max() > 2:
        max_clip = 255
    else:
        max_clip = 1    
        
    gaussian = np.random.normal(mean, sigma, img.shape) 

    #noisy_image = np.zeros(img.shape, np.float32)
    
    noisy_image = img + gaussian
    noisy_image =np.clip(noisy_image,low_clip,max_clip)
    
    # cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX)
    # noisy_image = noisy_image.astype(np.uint8)    
    return noisy_image



data_dir = 'data/img'
img_lst = os.listdir(data_dir)


for name in img_lst:
    abs_path = os.path.join(data_dir, name)
    print(abs_path)
    img = cv2.imread(abs_path)
    #prefix, suffix, _ = abs_path.split('.')
    
    suffix = 'jpg'
    for i in range(1,3):
        
        img2 = motion_blur(img, _max_filiter_size)    
        img2 = gaussain_blur(img2, _max_filiter_size)    
        img2 = motion_blur(img2, _max_filiter_size)    
        
        cv2.imwrite('%s_%s.%s' % (abs_path, f'noise{i}', suffix), random_noiseCV(img2,  _var))
    
    
