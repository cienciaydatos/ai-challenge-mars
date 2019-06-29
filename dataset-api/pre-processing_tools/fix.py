# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:53:54 2019

@author: SEBASTIAN LAVERDE
"""

import rasterio
from rasterio.plot import show
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tools
from tqdm import tqdm_notebook as tqdm
from numpy.lib.stride_tricks import as_strided
from warnings import warn
from PIL import Image
from scipy import ndimage

def view_as_blocks(arr_in, block_shape):
 
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length "
                         "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")

    # -- restride the array to build the block view

    if not arr_in.flags.contiguous:
        warn(RuntimeWarning("Cannot provide views on a non-contiguous input "
                            "array without copying."))

    arr_in = np.ascontiguousarray(arr_in)

    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out

#%%
img_file = 'C:/Users/SEBASTIAN LAVERDE/Documents/Unterlagen/SoSe2019/mars/python/sample-images/sample-images/ESP_016793_2485_RED.JP2'
chunk_size=256

#%%
with rasterio.open(img_file) as src:
    
    for block_index, window in tqdm(src.block_windows(1)):
      block_array = src.read(window=window)
      # print('Block array', block_array.shape)
      
      block_array = np.moveaxis(block_array, 0, -1)
      # print('Move axis', block_array.shape)
      
      if block_array.shape[2] != 1:
        block_array = cv2.cvtColor(block_array, cv2.COLOR_RGB2GRAY)
      else:
        block_array = np.squeeze(block_array)
      block_array_shape = block_array.shape
      
      # plt.imshow(block_array, cmap='gray')      
      # print('Grayscale Block Shape', block_array_shape)

      if (block_array_shape[0] % chunk_size == 0 and block_array_shape[1] % chunk_size == 0):
        result_blocks = view_as_blocks(block_array, block_shape=(chunk_size, chunk_size))
#        write_result_blocks(result_blocks, window, product_name, chunk_size, save_dir, skip_black_images, align_images, vectorized_chunks)  
        
#%%
#Resizing jp2 file or saving as jpg
img = rasterio.open(img_file)
print(type(img))

img_np = img.read(1)
print(type(img_np))
    
print('shape = {}'.format(img_np.shape))
print('Resolution =', img.width, 'x', img.height)
print('Estimated number of iterations =', ((img.width * img.height) / (1024 * 1024))*1.085)
#plt.imshow(img_np)
#plt.show()
#cv2.imwrite("jpg of jp2.jpg", img_np)
#%%
#result = tools.align_and_crop(img_np) #memory error, try resizing and it should be equivalent
#cv2.imwrite("align_and_crop", result)
res = cv2.resize(img_np, dsize=(2048, 2048), interpolation=cv2.INTER_CUBIC)
#plt.imshow(res)
#plt.show()
#cv2.imwrite("resized.jpg", res)
#%%
path = 'C:/Users/SEBASTIAN LAVERDE/Documents/Unterlagen/SoSe2019/mars/python/sample-images/sample-images/'
img2 = Image.open(path + 'ESP_029670_1530_MIRB.browse.jpg')
#result = tools.align_and_crop(res) #memory error, try resizing and it should be equivalent
#cv2.imwrite("align_and_crop.jpg", result)
plt.imshow(res)
plt.show()
aligned = tools.align_image(res)
plt.imshow(aligned)
plt.show()
#aligned = tools.align_image(img2)
#plt.imshow(aligned)
#plt.show()
#%%
img_rotated = ndimage.rotate(res, 27)

plt.imshow(res)
plt.show()
plt.imshow(img_rotated)
plt.show()
img_rotated = ndimage.rotate(res, 90)
plt.imshow(img_rotated)
plt.show()
#%%
path = 'C:/Users/SEBASTIAN LAVERDE/Documents/Unterlagen/SoSe2019/mars/python/1024x1024/'
img = Image.open(path + 'chameleon.jpg')

print("\nwithout .5\n")
img_rotated = ndimage.rotate(img, 25) #It's generating an error everytime the angles has a .5 decimal
tools.align_image(img_rotated)
print("\nwith .5\n")
img_rotated = ndimage.rotate(img, 25.5) #It's generating an error everytime the angles has a .5 decimal
tools.align_image(img_rotated)
print("\nwithout .5\n")
img_rotated = ndimage.rotate(img, 26) #It's generating an error everytime the angles has a .5 decimal
tools.align_image(img_rotated)
print("\nwith .5\n")
img_rotated = ndimage.rotate(img, 26.5) #It's generating an error everytime the angles has a .5 decimal
tools.align_image(img_rotated)
print("\nwithout .5 = 27\n")
img_rotated = ndimage.rotate(img, 27) #It's generating an error everytime the angles has a .5 decimal
tools.align_image(img_rotated)
print("\nwith .5 = 27.5\n")
img_rotated = ndimage.rotate(img, 27.5) #It's generating an error everytime the angles has a .5 decimal
tools.align_image(img_rotated)
print("\nwith .1 = 27.1\n")
img_rotated = ndimage.rotate(img, 27.1) #It's generating an error everytime the angles has between 0.2 and 0.7

aligned = tools.align_image(img_rotated)
plt.imshow(img)
plt.show()
plt.imshow(img_rotated)
plt.show()
plt.imshow(aligned)
plt.show()
#%% test with the jpg images-- working when margin exist all sides
path = 'C:/Users/SEBASTIAN LAVERDE/Documents/Unterlagen/SoSe2019/mars/python/sample-images/sample-images/'
#ESP_029670_1530_COLOR.abrowse
#ESP_029670_1530_IRB.NOMAP.browse
#ESP_029670_1530_MIRB.browse
#ESP_029670_1530_MRGB.abrowse
#ESP_029670_1530_RGB.NOMAP.browse

#img = cv2.imread(path + 'ESP_029670_1530_COLOR.abrowse.jpg')
img = Image.open(path + 'ESP_029670_1530_RGB.NOMAP.browse.jpg')

#result = tools.align_and_crop(img)
#plt.imshow(result)
#plt.show()
#cv2.imwrite(path + 'test1/ESP_029670_1530_RGB.NOMAP.browse.jpg', result)
#show(img.read(), transform=img.transform)