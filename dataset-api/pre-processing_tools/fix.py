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
from pystackreg import StackReg
from skimage import io

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

#_--------------------------------HOMOGRAPHY------------------------------------------------

#%% image alignment with template (homography)
img_rotated = cv2.imread('resized.jpg')
#img_rotated = ndimage.rotate(img, 27.5) #It's generating an error everytime the angles has a .5 decimal
aligned = tools.align_image(img_rotated)

template = np.zeros((img_rotated.shape[0],img_rotated.shape[1],3), dtype = 'uint8')
print(img_rotated.shape)
template = cv2.rectangle(template,(200,200),(img_rotated.shape[0]-200,img_rotated.shape[1]-200),(125,0,125),-1) #top-left corner and bottom-right corner of rectangle.
#cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
plt.imshow(img_rotated)
plt.show()
plt.imshow(template)
plt.show()
gray_test = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_test)
plt.show()
#print(template)
#%% not working but cool...
 
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
 
 
def alignImages_homography(im1, im2): #specify format and dimensions
 
  MAX_FEATURES = 500
  GOOD_MATCH_PERCENT = 0.15
    
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = im2[:,:,0]
  
  #im1Gray = im1.copy()
  #im2Gray = im2.copy()
  
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  print("Aligning images ...")
  print(im1.shape, im2.shape)
  print(type(im1), type(im2))
  
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches_resized.jpg", imMatches)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
  
  plt.imshow(im1Reg)
  plt.show()
  print("Estimated homography : \n",  h)
   
  return im1Reg, h
 #%%
# Read reference image
imReference = cv2.imread('bilinear.jpg')
plt.imshow(imReference)
plt.show()

im = cv2.imread('resized.jpg')
imReference = cv2.resize(imReference,(im.shape[1], im.shape[0]))
plt.imshow(imReference)
plt.show()
plt.imshow(im)
plt.show()

print(type(im))
print(im.dtype)
print(im.ndim)
#tuple object is not callable FIX! : reboot kernel compile libraries and write again...cv2.imwrite('rotated.jpg', im)
#%%
#apply thresholding first
#ret, im = cv2.threshold(im, 1, 125, cv2.THRESH_BINARY)
#plt.imshow(im)
#plt.show()

imReg, h = tools.alignImages_homography(im, imReference)
 
outFilename = "homography_aligned_resized.jpg"

#print("Saving aligned image : ", outFilename); 
cv2.imwrite(outFilename, imReg)


#_------------------------------------------------------------------------------------------------
#%%
from pystackreg import StackReg
from skimage import io

img_rotated = cv2.imread('type2.jpg')
#img_rotated = ndimage.rotate(img, 27.5) #It's generating an error everytime the angles has a .5 decimal
aligned = tools.align_image(img_rotated)

template = np.zeros((img_rotated.shape[0],img_rotated.shape[1],3), dtype = 'uint8')
print(img_rotated.shape)
template = cv2.rectangle(template,(200,200),(img_rotated.shape[1]-200,img_rotated.shape[0]-200),(220,15,88),-1) #top-left corner and bottom-right corner of rectangle.
#cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
plt.imshow(img_rotated)
plt.show()
plt.imshow(template)
plt.show()
gray_test = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_test)
plt.show()

#%%

#load reference and "moved" image
ref = template
mov = img_rotated

#Translational transformation
sr = StackReg(StackReg.TRANSLATION)
out_tra = sr.register_transform(ref[:,:,0], mov[:,:,0])

#Rigid Body transformation
sr = StackReg(StackReg.RIGID_BODY)
out_rot = sr.register_transform(ref[:,:,0], mov[:,:,0])

#Scaled Rotation transformation
sr = StackReg(StackReg.SCALED_ROTATION)
out_sca = sr.register_transform(ref[:,:,0], mov[:,:,0])

#Affine transformation
sr = StackReg(StackReg.AFFINE)
out_aff = sr.register_transform(ref[:,:,0], mov[:,:,0])

#Bilinear transformation
sr = StackReg(StackReg.BILINEAR)
out_bil = sr.register_transform(ref[:,:,0], mov[:,:,0])

#%%
img = cv2.imread('mars.jpg')
#templ = tools.generate_template(img)

#%%
trans, _ = tools.register_image(img, 'bilinear.jpg')
#%%
print(type(trans))
#%%
plt.imshow(trans[0])
plt.show()
plt.imshow(trans[1])
plt.show()
plt.imshow(trans[2])
plt.show()
plt.imshow(trans[3])
plt.show()
plt.imshow(trans[4])
plt.show()
#%%
cv2.imwrite('alignment_out_tra_mars_bilinear.jpg',trans[0])
cv2.imwrite('alignment_out_rot_mars_bilinear.jpg',trans[1])
cv2.imwrite('alignment_out_sca_mars_bilinear.jpg',trans[2])
cv2.imwrite('alignment_out_aff_mars_bilinear.jpg',trans[3])
cv2.imwrite('alignment_out_bil_mars_bilinear.jpg',trans[4])
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