#  AI Challenge - Mars

The project is about finding out the different anomalies on the surface of Mars. This project is under the category of a supervised algorithm where the data is been trained on the below set of images in order to do a small PoC.

# Project Structure
```
MASK_RCNN/
├── assets/
├── datasets/  Datasets used in every model for training/
├── modules/                
│   ├── anomaly/     Model 2 (Anomalies -- `crater` and `sand dune`)
│   ├── crater/      Model 1 (Anomaly -- `crater` )
│   ├── mars_anomaly/  Model 3  (Anomalies -- `slope streak,  bright dune,  impact ejecta, swiss cheese`)
├── mrcnn  Main files for configuration, model, utilities and visualization.
├── notebooks/
│   ├── anomaly_2_classes_demo/     Demo for identification of crater and sand dune.
│   ├── crater_demo/      Demo for identification of crater.
│   ├── mars_anomaly_demo/  Demo for identification of 4 anomalies.
│   ├── 1_image_slicer.ipynb  A notebook demonstrating to slice the given image.
│   ├── 2_image_annotations.ipynb  Empty Notebook.
│   ├── 3_Mask_RCNN_Training.ipynb A notebook for training the model.
├── weights/
├── README.md
├── requirements.txt
├── setup.cfg
└── setup.py
```
# Data Preparation
* Data was annotated using [VIA] tool, the version was used is [1.0.6]
* We used polygon shape for the outliners for the anomaly.
* For every class we have two `.json` file for each of the two directories consisting of 250 images for training and 50 images for validation.

# Training
- The model was trained on [Google Colab] with the help of GPU. 

# Model
The project is done on 3 different datasets with different no of an anomaly.
- Model 1: Identify crater which is trained on a very small dataset [1].
    * The model was trained for 30 epochs having 100 batch steps per epoch is 100.
    * Data was distributed into `train`, `validation` and `test`
    * The model was created in order to try out the object detection and segmentation for a single class of anomalies on Mars Surface.
    * The accuracy isn't so good as the data to which it was trained was few.

- Model 2: Identify crater and sand dune which is trained on a very small dataset [1].
    * The model was trained for 30 epochs having 100 batch steps per epoch is 100.
    * Data was distributed into `train`, `validation` and `test`.
    * The model was created in order to try out the object detection and segmentation for two classes (Crater and Sand-Dune) of anomalies on Mars Surface.
    * The accuracy isn't so good as the data to which it was trained was few.

- Model 3: Identify crater which is trained on a `Mars orbital image (HiRISE) labeled data set version 3`  [dataset]
    * The total datasets compormises of label data with the following classes,
    
    | Class-Id  | Anomaly Name  | No of Images  |
    |---|---|---|
    |  0 | others  | 61k  |  
    |  1 | crater  | 4.9k  |   
    |  2 | dark dune  | 1.1k  |   
    |  3 | slope streak | 2.3k  |
    |  4 | bright dune |  1.7k |
    |  5 | impact ejecta | 231  |
    |  6 | swiss cheese | 1.1k  |
    |  7 | spider |  476 |
    * This model is trained on `3,4,5,6` class-ids each of having 250 images for training (except for `impact ejecta`, due to less count) and 50 for  validation. 
    * The model was trained for 30 epochs having 100 batch steps per epoch is 100.
    * Data was distributed into "train", 'validation' and "test".

[VIA]: <http://www.robots.ox.ac.uk/~vgg/software/via/>
[1.0.6]: <http://www.robots.ox.ac.uk/~vgg/software/via/downloads/via-1.0.6.zip>
[dataset]: <https://zenodo.org/record/2538136#.XPZriogzbIV>
[Google Colab]: <https://colab.research.google.com/>

## Datasets
For a Model 1 and Model 2 we used the following two images for training, and one for testing.
- Train
    * https://photojournal.jpl.nasa.gov/catalog/PIA00179 
- Validation
    * https://photojournal.jpl.nasa.gov/catalog/PIA00179 
    * https://photojournal.jpl.nasa.gov/catalog/PIA00180 
-Test
    * https://photojournal.jpl.nasa.gov/catalog/PIA00181
For training, the image was split into the 6 small images using the library - image_slicer.
For testing, we again slice the .tiff image

# Sample Outputs:
- [Model 1] - Anomaly - crater
- [Model 2] - Anomaly - crater and sand dune
- [Model 3] - Anomaly - slope streak,  bright dune,  impact ejecta, swiss cheese

[Model 1]: <https://github.com/amandalmia14/ai-challenge-mars/blob/master/Mask_RCNN/notebooks/crater_demo/4_Mask_RCNN_Crater_Demo.ipynb>
[Model 2]: <https://github.com/amandalmia14/ai-challenge-mars/blob/master/Mask_RCNN/notebooks/anomaly_2_classes_demo/4_Mask_RCNN_Anomaly_Crater_Sand-Dune_Demo.ipynb>
[Model 3]: <https://github.com/amandalmia14/ai-challenge-mars/blob/master/Mask_RCNN/notebooks/mars_anomaly_demo/4_Mask_RCNN_Multiple_Anomaly.ipynb>



###  References
https://github.com/matterport/Mask_RCNN

[1]: 
- Train
-- https://photojournal.jpl.nasa.gov/catalog/PIA00179 
- Validation
-- https://photojournal.jpl.nasa.gov/catalog/PIA00179 
-- https://photojournal.jpl.nasa.gov/catalog/PIA00180 
- Test
-- https://photojournal.jpl.nasa.gov/catalog/PIA00181