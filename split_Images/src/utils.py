from os import listdir
from os.path import isfile, join
import image_slicer
from PIL import Image
import os

def get_list_files_folder(pathFolder):

    try:
        listFiles = [f for f in listdir(pathFolder) if isfile(join(pathFolder, f))]
        return listFiles

    except Exception as e:
        print(e)

def split_multiple_Images(pathInput, pathOutput):

    try:

        listImageInputs = get_list_files_folder(pathInput)

        for element in listImageInputs:

            imgInput = os.path.join(pathInput, element)  # 'Image input

            im = Image.open(imgInput)  # Import image to the variable 'im' to get the size of the image
            width, height = im.size

            ### 1 cm =~ 37.7952755906 pixels
            numCM = 6  # number of centimeters that we will like to have per slice on the original photo

            numColumns = round(width / (numCM * 37.7952755906))  # calculate the number of columns that we will have
            numLines = round(height / (numCM * 37.7952755906))  # calculate the number of lines that we will have

            numSlices = numColumns * numLines  # calculate the number of slices that we will have

            ### Split the image from the 'input' folder
            tiles = image_slicer.slice(imgInput, numSlices, save=False)

            ### Save the multiples parts on the 'output' folder
            image_slicer.save_tiles(tiles, directory=pathOutput, prefix=element.replace(".jpg", ""))

    except Exception as e:
        print(e)

def resize_image(pathImageInput, imgName, pathImageOutput, width=224, height=224): # TO-DO: interpolationType
    try:

        im1 = Image.open(os.path.join(pathImageInput, imgName))

        # use one of these filter options to resize the image
        #im2 = im1.resize((width, height), Image.NEAREST)  # use nearest neighbour
        #im3 = im1.resize((width, height), Image.BILINEAR)  # linear interpolation in a 2x2 environment
        im4 = im1.resize((width, height), Image.BICUBIC)  # cubic spline interpolation in a 4x4 environment
        #im5 = im1.resize((width, height), Image.ANTIALIAS)  # best down-sizing filter

        im4.save(os.path.join(pathImageOutput, imgName))

    except Exception as e:
        print(e)