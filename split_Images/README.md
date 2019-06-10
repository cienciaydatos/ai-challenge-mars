# AI Challenge Mars

<p align="center">
  <img src="https://github.com/cienciaydatos/ai-challenge-trees/blob/master/imgs/logos.png">
</p>

[Ciencia & Datos](https://www.cienciaydatos.org/)

[Omdena](https://omdena.com/)

in this repo you will find the code to address the following pull-request:

'Simple script (in python) to split one image into multiples ones'

## How to activate the virtual env

1. Go to:

```
<path for your source>/split_Images
```

2. Activate the 'virtual env'

```
source venv/bin/activate
```

## Quick overview about the script

1. Define some variables:

prefix = 'example_img_01' # prefix for the 'output' file (that's the same that we use as an input)

imgInput = os.path.join('../input', prefix + ".jpg") # 'Image input

2. Split the image from the 'input' folder

3. Save the multiples parts on the 'output' folder

4. Resize the images #The image is save here:

../output/sliced_images

../output/resized_images

