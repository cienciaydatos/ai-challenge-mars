#  AI Challenge - Mars

The project is about finding out the craters in the surface of the Mars. This project is under the category of supervised algorithm where the data is been trained on the below set of images in order to do a small PoC. 

The model was trained on Google Colab using Mask _RCNN technique. 

## Image Slicer
A demo for identification of craters on the mars surface.
## Datasets
For a PoC purpose used the following two images for training, and one for testing.
- Train
-- https://photojournal.jpl.nasa.gov/catalog/PIA00179 
- Validation
-- https://photojournal.jpl.nasa.gov/catalog/PIA00179 
-- https://photojournal.jpl.nasa.gov/catalog/PIA00180 
-Test
-- https://photojournal.jpl.nasa.gov/catalog/PIA00181
For training the image was split into the 6 small image using the library - image_slicer
For testing we again slice the .tiff image

###  Credits 
https://github.com/matterport/Mask_RCNN

