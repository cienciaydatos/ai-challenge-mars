# -*- coding: utf-8 -*-
"""
Created on June 15, 2019
@author: SEBASTIAN LAVERDE
"""

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from scipy import ndimage
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import imutils
from pystackreg import StackReg
from skimage import io
from PIL import Image

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

def align_image(img): #img not NoneType
    
    """This functions works with discrete angles or decimal points not betwenn 0.2 and 0.7"""
    img = np.uint8(img)
    img_edges = cv2.Canny(img, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    
    if lines is None:
        return img
  
    #print("lines: ", lines)
    angles = []
  
    for x1, y1, x2, y2 in lines[0]:
        print("x1 : {}, y1 : {}, x2 : {}, y2 : {},".format(x1,y1,x2,y2))
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
        
    median_angle = np.median(angles)
    print("Angle detected: {}".format(median_angle))
    print("angles: ",angles)
      
    img_rotated = ndimage.rotate(img, median_angle)
    return img_rotated

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
  cv2.imwrite("matches2.jpg", imMatches)
   
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

def random_pixel(low=5, high=250):
    return np.random.randint(5,250)

def random_color(low=5, high=250):
    color = [random_pixel(),random_pixel(),random_pixel()]
    return color

def generate_template(img, color = None): #rotar 90
        
    ref = np.zeros((img.shape[0],img.shape[1],3), dtype = 'uint8') #before 1,0
    margin = int(min(img.shape[0], img.shape[1])/10) #a tenth of the minimum lenght
    print("margin: ", margin)

    for i in range(0,img.shape[0]-2*margin): #before = 1
        i+=1
        #print("new i: ",i) #debug
        for j in range(0,img.shape[1]-2*margin): #before = 0
            colour = random_color() if color == None else color
            #print(j) #debug
            ref[margin+i,margin+j,:] = colour
            j+=1
    return ref

def register_image(img, ref = None):  #img must be 3 channels, ref could be None which is random noise wiht a margin, a string with the file name and or path, or the word 'solid' which is a solid color with a margin
    
    """pystackreg library"""
    
    transformations = []
    
    if ref == None:
        ref = generate_template(img) #default color
    elif ref == 'solid': #random color between 5 - 250
        ref = generate_template(img, color=random_color())
    elif ref == 'gray': 
        ref = generate_template(img, color=[125,125,125])
    else:
        ref = cv2.imread(ref)
        
    print("\nshowing original image...\n")    
    plt.imshow(img)
    plt.show()    
        
    print("\nshowing reference image...\n")    
    plt.imshow(ref)
    plt.show()
    #Translational transformation
    sr = StackReg(StackReg.TRANSLATION)

    min_shape = [min(ref.shape[0],img.shape[0]), min(ref.shape[1],img.shape[1])]
    img = cv2.resize(img, (min_shape[1], min_shape[0]))
    ref = cv2.resize(ref, (min_shape[1], min_shape[0]))
    out_tra = sr.register_transform(ref[:,:,0], img[:,:,0])
    transformations.append(out_tra)

    #Rigid Body transformation
    sr = StackReg(StackReg.RIGID_BODY)
    out_rot = sr.register_transform(ref[:,:,0], img[:,:,0])
    transformations.append(out_rot)
    
    #Scaled Rotation transformation
    sr = StackReg(StackReg.SCALED_ROTATION)
    out_sca = sr.register_transform(ref[:,:,0], img[:,:,0])
    transformations.append(out_sca)
    
    #Affine transformation
    sr = StackReg(StackReg.AFFINE)
    out_aff = sr.register_transform(ref[:,:,0], img[:,:,0])
    transformations.append(out_aff)

    #Bilinear transformation
    sr = StackReg(StackReg.BILINEAR)
    out_bil = sr.register_transform(ref[:,:,0], img[:,:,0])    
    transformations.append(out_bil)
    
    return transformations, ref

def concatenate(imgflnames, from_file = False): #filename: 'name.jpg', returns class PIL.Image.Image

    if from_file:
        images = [cv2.imread(i) for i in imgflnames]
    else:
        images = imgflnames
        
    min_shape = sorted( [(np.sum(i.shape), i.shape ) for i in images])[0][1]
    imgs_comb = np.hstack( (np.asarray(cv2.resize(i,(min_shape[1], min_shape[0]))) for i in images ) ) #height and width are inverted, check this

    return Image.fromarray( imgs_comb)

def crop_black_margin(img, show_contour = False):
  #gray scale conversion first
  base = img.copy()
  
  if len(img.shape) == 3:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #define threshold in 1 for almost black
  ret, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
  
  #find the maximum area contour of the filtered image
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  
  filter_area = []
  for c in contours:
    if (cv2.contourArea(c) < 200000): #modify for the biggest pictures...define calculation
      continue #TODO: set automatically with a max function
    filter_area.append(c)

  if (len(filter_area) == 0):
      return None
      
  contour = filter_area[0]
  rects = [cv2.boundingRect(cnt) for cnt in contour]
  
  if show_contour:
      plt.imshow(contour)
      plt.show()
      
  height,width = img.shape

  #Calculate the combined bounding rectangle points.
  bottom_x = min([x for (x, y, height, width) in rects])
  bottom_y = min([y for (x, y, height, width) in rects])
  top_x = max([x+width for (x, y, height, width) in rects])
  top_y = max([y+height for (x, y, height, width) in rects])

  cropped = base[height - top_y : height - bottom_y, bottom_x : top_x]
  
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