import cv2
import numpy as np
import os
import random
from matplotlib import pyplot as plt

# create by Feng, edit by Feng 2017 / 08 / 14
# this project is doing image data augmentation for any DL/ML algorithm
#


def bad_radial_blur_bad(img, max_filiter_size=0.001):
    w, h = img.shape[:2]

    center_x = w / 2 #+ random.randint(-w //13, +w //13)
    center_y = h / 2 #+ random.randint(-h //13, +h //13)
    blur = max_filiter_size*random.random()
    iterations = 5

    growMapx = np.tile(np.arange(h) + ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    shrinkMapx = np.tile(np.arange(h) - ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    growMapy = np.tile(np.arange(w) + ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
    shrinkMapy = np.tile(np.arange(w) - ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
    growMapx, growMapy = np.abs(growMapx), np.abs(growMapy)
    img_blur = img.copy()

    for i in range(iterations):
        tmp1 = cv2.remap(img_blur, growMapx, growMapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        tmp2 = cv2.remap(img_blur, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img_blur = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)
    return img_blur

# avg blur minimum filter size is 3
def avg_blur(img, max_filiter_size = 3) :
	img = img.astype(np.uint8)
	if max_filiter_size >= 3 :
		filter_size = random.randint(3, max_filiter_size)
		if filter_size % 2 == 0 :
			filter_size += 1
		out = cv2.blur(img, (filter_size, filter_size))
	return out

# gaussain blur minimum filter size is 3
# when sigma = 0 gaussain blur weight will compute by program
# when the sigma is more large the blur effect more obvious
def gaussain_blur(img, max_filiter_size = 3, sigma = cv2.BORDER_DEFAULT) :
	img = img.astype(np.uint8)
	if max_filiter_size >= 3 :
		filter_size = random.randint(3, max_filiter_size)
		if filter_size % 2 == 0 :
			filter_size += 1
		#print ('size = %d'% filter_size)
		out = cv2.GaussianBlur(img, (filter_size, filter_size), sigma)
	return out
    
#size - in pixels, size of motion blur
#angel - in degrees, direction of motion blur
def motion_blur(image, max_filiter_size=12, angle=0):
    if max_filiter_size >= 3 :
        size = random.randint(3, max_filiter_size)
    angle = random.randint(0, 360)
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
    k = k * ( 1.0 / np.sum(k) )        
    return cv2.filter2D(image, -1, k)     

def gaussain_noise(img, mean = 0, var = 0.1) :
	img = img.astype(np.uint8)
	h, w, c = img.shape
	sigma = var ** 0.5
	gauss = np.random.normal(mean, sigma, (h, w, c))
	gauss = gauss.reshape(h, w, c).astype(np.uint8)
	noisy = img + gauss
	return noisy

# fill_pixel is 0(black) or 255(white)

def img_shift(img, x_min_shift_piexl = -1, x_max_shift_piexl = 1, y_min_shift_piexl = -1, y_max_shift_piexl = 1, fill_pixel = 0) :
	img = img.astype(np.uint8)
	h, w, c = img.shape
	out = np.zeros(img.shape)
	if fill_pixel == 255 :
		out[:, :] = 255
	out = out.astype(np.uint8)
	move_x = random.randint(x_min_shift_piexl, x_max_shift_piexl)
	move_y = random.randint(y_min_shift_piexl, y_max_shift_piexl)
	#print (('move_x = %d')% (move_x))
	#print (('move_y = %d')% (move_y))
	if move_x >= 0 and move_y >= 0 :
		out[move_y:, move_x: ] = img[0: (h - move_y), 0: (w - move_x)]
	elif move_x < 0 and move_y < 0 :
		out[0: (h + move_y), 0: (w + move_x)] = img[ - move_y:, - move_x:]
	elif move_x >= 0 and move_y < 0 :
		out[0: (h + move_y), move_x:] = img[ - move_y:, 0: (w - move_x)]
	elif move_x < 0 and move_y >= 0 :
		out[move_y:, 0: (w + move_x)] = img[0 : (h - move_y), - move_x:]
	return out

# In img_rotation func. rotation center is image center

def img_rotation(img, min_angel = 0, max_angel = 0, min_scale = 1, max_scale = 1, fill_pixel = 0) :
	img = img.astype(np.uint8)
	h, w, c = img.shape
	_angel = random.randint(min_angel, max_angel)
	_scale = random.uniform(min_scale, max_scale)
	rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), _angel, _scale)
	out = cv2.warpAffine(img, rotation_matrix, (w, h))
	if fill_pixel == 255 :
		mask = np.zeros(img.shape)
		mask[:, :, :] = 255
		mask = mask.astype(np.uint8)
		mask = cv2.warpAffine(mask, rotation_matrix, (w, h))
		for i in range (h) :
			for j in range(w) :
				if mask[i, j, 0] == 0 and mask[i, j, 1] == 0 and mask[i, j, 2] == 0 :
					out[i, j, :] = 255
	return out

# In img_flip func. it will random filp image
# when flip factor is 1 it will do hor. flip (Horizontal)
#					  0            ver. flip (Vertical)
#					 -1			   hor. + ver flip

def img_flip(img) :
	img = img.astype(np.uint8)
	flip_factor = random.randint(-1, 1)
	out = cv2.flip(img, flip_factor)
	return out

# Zoom image by scale

def img_zoom(img, min_scale = 1, max_scale = 1) :
	img = img.astype(np.uint8)
	h, w, c = img.shape
	scale = random.uniform(min_scale, max_scale)
	h = int(h * scale)
	w = int(w * scale)
	out = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
	return out

# change image contrast by hsv

def img_contrast(img, min_s, max_s, min_v, max_v) :
	img = img.astype(np.uint8)
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	_s = random.randint(min_s, max_s)
	_v = random.randint(min_v, max_v)
	if _s >= 0 :
		hsv_img[:, :, 1] += _s
	else :
		_s = - _s
		hsv_img[:, :, 1] -= _s
	if _v >= 0 :
		hsv_img[:, :, 2] += _v
	else :
		_v = - _v
		hsv_img[:, :, 2] += _v
	out = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
	return out

# change image color by hsv

def img_color(img, min_h, max_h) :
	img = img.astype(np.uint8)
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	_h = random.randint(min_h, max_h)
	hsv_img[:, :, 0] += _h
	out = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
	return out




























