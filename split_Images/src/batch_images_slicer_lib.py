import image_slicer
from PIL import Image
import os
from split_Images.src import utils

### Define some variables:


listInputFiles = utils.get_list_files_folder('../input')

for element in listInputFiles:

    if element.endswith('.jpg'):

        imgInput = os.path.join('../input', str(element))  # 'Image input

        im = Image.open(imgInput) # Import image to the variable 'im' to get the size of the image
        width, height = im.size

        ### 1 cm =~ 37.7952755906 pixels
        numCM = 6 # number of centimeters that we will like to have per slice on the original photo

        numColumns = round(width / (numCM * 37.7952755906)) # calculate the number of columns that we will have
        numLines = round(height / (numCM * 37.7952755906)) #calculate the number of lines that we will have

        numSlices = numColumns * numLines #calculate the number of slices that we will have

        ### Split the image from the 'input' folder
        tiles = image_slicer.slice(imgInput, numSlices, save=False)

        prefix = str(element).replace(".jpg", "")

        ### Save the multiples parts on the 'output' folder
        image_slicer.save_tiles(tiles, directory='../output/sliced_images', prefix=prefix)

        ### Resize the images
        listFiles = utils.get_list_files_folder('../output/sliced_images')

        for element in listFiles:
            if element.endswith('.png'):
                pathImageInput = os.path.join('../output/sliced_images')
                pathImageOutput = os.path.join('../output/resized_images')
                utils.resize_image(pathImageInput, element, pathImageOutput)