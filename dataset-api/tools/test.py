# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:29:41 2019

@author: SEBASTIAN LAVERDE
"""

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tools
import numpy as np

#images
#PSP_001414_1780_RED_img_row_33792_col_12288_w_1024_h_1024_x_0_y_0
#PSP_001414_1780_RED_img_row_32768_col_15360_w_1024_h_1024_x_0_y_0
#PSP_001414_1780_RED_img_row_32768_col_14336_w_1024_h_1024_x_0_y_0
#PSP_001414_1780_RED_img_row_32768_col_13312_w_1024_h_1024_x_0_y_0
#PSP_001414_1780_RED_img_row_9216_col_11264_w_1024_h_1024_x_0_y_0

path = "C:/Users/SEBASTIAN LAVERDE/Documents/Unterlagen/SoSe2019/mars/python/1024x1024/"
img = cv2.imread(path + 'PSP_001414_1780_RED_img_row_33792_col_12288_w_1024_h_1024_x_0_y_0.jpg')
im = Image.open(path + 'PSP_001414_1780_RED_img_row_33792_col_12288_w_1024_h_1024_x_0_y_0.jpg')
np_im = np.array(im)

sharpened = tools.sharp(np_im, 3)
stretched = tools.stretch_8bit(np_im)
enhanced = tools.stretch_8bit(sharpened)

#im.save('original.jpg')
#cv2.imwrite('enhanced.jpg', enhanced)
#cv2.imwrite('stretched.jpg', stretched)
#cv2.imwrite('sharpened.jpg', sharpened)

#_________________ create function with this _________________________
list_im = ['original.jpg','sharpened.jpg','stretched.jpg','enhanced.jpg']
imgs    = [ Image.open(i) for i in list_im ]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

# save that beautiful picture
imgs_comb = Image.fromarray( imgs_comb)
imgs_comb.save( 'test_hor.jpg' )    

# for a vertical stacking it is simple: use vstack
imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
imgs_comb = Image.fromarray( imgs_comb)
imgs_comb.save( 'test_ver.jpg' )
#_______________________________________________________________________

list_im = ['original.jpg','enhanced.jpg']
imgs    = [ Image.open(i) for i in list_im ]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

# save that beautiful picture
imgs_comb = Image.fromarray( imgs_comb)
imgs_comb.save( 'orginal_final.jpg' )