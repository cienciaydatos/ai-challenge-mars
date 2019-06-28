# -*- coding: utf-8 -*-
"""
Created on June 15, 2019
@author: SEBASTIAN LAVERDE
"""

import numpy as np
import cv2
import math
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import imutils

"""Tools for satellite imagery pre-processing"""

def sharp(img,level=3): #level[1:5]
    
    def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened
    
    if level == 1: #low
        sharpened = unsharp_mask(img)
        
    elif level == 2: #med_low
        kernel_sharp = np.array([[0, -1, 0], 
                                 [-1, 5, -1], 
                                 [0, -1, 0]])
        sharpened = cv2.filter2D(img, -1, kernel_sharp)
        sharpened = cv2.bilateralFilter(img, 3, 75 ,75)

    elif level == 3: #med. Best result on average
        kernel_sharp = np.array([[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, 0]])
        sharpened = cv2.filter2D(img, -1, kernel_sharp)

    elif level == 4: #med_high
        kernel_sharpening = np.array([[-1,-1,-1], 
                                      [-1, 9,-1],
                                      [-1,-1,-1]])
        sharpened = cv2.filter2D(img, -1, kernel_sharpening)

    elif level == 5: #high
        kernel_sharp = np.array([[-2, -2, -2], 
                                 [-2, 17, -2], 
                                 [-2, -2, -2]])
        sharpened = cv2.filter2D(img, -1, kernel_sharp)
    
    else:
        sharpened = img
        print("image didn't change...")
    
    return sharpened

def align_image(img):
  img = np.uint8(img)
  img_edges = cv2.Canny(img, 100, 100, apertureSize=3)
  lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
  if lines is None:
    return img
  
  # print("lines: ", lines)
  angles = []
  
  for x1, y1, x2, y2 in lines[0]:
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)

  median_angle = np.median(angles)
  img_rotated = ndimage.rotate(img, median_angle)
  return img_rotated

def crop_black_margin(img):
  #gray scale conversion first
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #define threshold in 1 for almost black
  ret, thresh = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
  
  #find the maximum area contour of the filtered image
  _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  
  filter_area = []
  for c in contours:
    if (cv2.contourArea(c) < 200000):
      continue #TODO: set automatically with a max function
    filter_area.append(c)

  if (len(filter_area) == 0):
      return None
      
  contour = filter_area[0]
  rects = [cv2.boundingRect(cnt) for cnt in contour]
  height,width = img_gray.shape

  #Calculate the combined bounding rectangle points.
  bottom_x = min([x for (x, y, height, width) in rects])
  bottom_y = min([y for (x, y, height, width) in rects])
  top_x = max([x+width for (x, y, height, width) in rects])
  top_y = max([y+height for (x, y, height, width) in rects])

  cropped = img[height - top_y : height - bottom_y, bottom_x : top_x]
  
  return cropped

def align_and_crop(img):
  return crop_black_margin(align_image(img))

def augment_simple(img): #flip, rollback, rotate 90 and rotate 180
    flipped = np.fliplr(img)
    rolled = np.rollaxis(img, 1)
    rotated90 = imutils.rotate(img, 90)
    rotated180 = imutils.rotate(img, 180)
    augmentations = [flipped, rolled, rotated90, rotated180]
    return augmentations


def augment_random(img, generations=5): #random augmentation
    #img = np.rollaxis(np_im, 1) #useful for simple data augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='output', save_prefix='augmentation', save_format='jpeg'):
        i += 1
        if i >= generations:
            break  # otherwise the generator would loop indefinitely
    return True

def stretch_8bit(bands, lower_percent=2, higher_percent=98): #Image enhancement linear contrast stretch
    out = np.zeros_like(bands)
    for i in range(3):
        a = 0 #np.min(band)
        b = 255  #np.max(band)
        c = np.percentile(bands[:,:,i], lower_percent)
        d = np.percentile(bands[:,:,i], higher_percent)        
        t = a + (bands[:,:,i] - c) * (b - a) / (d - c)    
        t[t<a] = a
        t[t>b] = b
        out[:,:,i] =t
    return out.astype(np.uint8)