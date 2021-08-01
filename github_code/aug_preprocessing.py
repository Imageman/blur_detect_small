import os
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance

import imageio
import imgaug as ia
from imgaug import augmenters as iaa

_generate_quantity = 3


data_dir = 'data/img'
img_lst = os.listdir(data_dir)

blur_s = iaa.OneOf([
    iaa.GaussianBlur(sigma=(1.5, 2.5)), # blur images with a sigma of 0 to 3.0
    iaa.AverageBlur(k=(3, 6)),
    iaa.MedianBlur(k=(5, 7)),
    iaa.Resize((2.0, 2.5)),
])

seq_rescale = iaa.Sequential([
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.001*255, 0.02*255), per_channel=0.5),
    iaa.Sometimes(
        0.5, iaa.MotionBlur(k=(3, 5)),
    ),
    iaa.Resize((1.9, 2.5)),
    iaa.Sometimes(
        0.5, iaa.MotionBlur(k=(3, 5)),
    ),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.00*255, 0.02*255), per_channel=0.5),
])

seq = iaa.Sequential([
    #iaa.Affine(rotate=(-5, 5)),
    
    blur_s,    
    
    iaa.Sharpen(alpha=(0, 0.70), lightness=(0.95, 1.05)),
    
    iaa.AdditiveGaussianNoise(scale=(0, 15)),
    
     # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.00*255, 0.07*255), per_channel=0.5),
    
    iaa.Sometimes(
        0.5,
        iaa.MotionBlur(k=(4, 7)),
        iaa.MotionBlur(k=(4, 7)),
    ),
    
    iaa.Sometimes(
        0.5,
        iaa.ElasticTransformation(alpha=50, sigma=19),  # water-like effect
    ),
], random_order=True)

seq_test = iaa.Sequential([
    iaa.MotionBlur(k=(4, 4)),
], random_order=True)

for name in img_lst:
    abs_path = os.path.join(data_dir, name)
    print(abs_path)
    image = imageio.imread(abs_path)
    #print("Original:")
    #ia.imshow(image)
    
    #image_aug = seq(image=image)
    # print(image_aug)
    #ia.imshow(image_aug)
    
    suffix = 'jpg'
    for i in range(0,_generate_quantity):
        image_aug = seq_rescale(image=image)     
        #image_aug = seq_test(image=image)     
        cv2.imwrite('%s_%s.%s' % (abs_path, f'noise{i}', suffix), cv2.cvtColor(image_aug,cv2.COLOR_BGR2RGB) , [cv2.IMWRITE_JPEG_QUALITY, random.randint(75,99)])
    
    
