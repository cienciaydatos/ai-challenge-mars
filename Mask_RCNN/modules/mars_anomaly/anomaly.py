import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from numpy  import array
import argparse

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class AnomalyConfig(Config):
    """
    @function : Configuration for training on the anomaly  dataset.
    @input: None
    @return: None
    """
    NAME = "anomaly"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + anomaly
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200
    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.8


############################################################
#  Dataset
############################################################

class AnomalyDataset(utils.Dataset):

    def load_anomaly(self, dataset_dir, subset):
        """
        @function : Load a subset of the Anomaly dataset. 
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val

        @input: dataset_dir - Dataset Directory
                subset - subset of the diretory - train | val | test

        @return: None
        """
        # Add classes. We have only one class to add.
        self.add_class("anomaly", 1, "crater")
        self.add_class("anomaly", 2, "dark-dune")
        self.add_class("anomaly", 3, "slope-streak")
        self.add_class("anomaly", 4, "bright-dune")
        self.add_class("anomaly", 5, "impact-ejecta")
        self.add_class("anomaly", 6, "swiss-cheese")
        self.add_class("anomaly", 7, "spider")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys
        annotations = [a for a in annotations if a['regions']]

        # Add images
        try:
            for a in annotations:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                names_objects = [s['region_attributes'] for s in a['regions'].values()]
                num_ids=[]
                num_ids = [int(n['anomaly']) for n in names_objects]
                image_path = os.path.join(dataset_dir, a['filename'])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                self.add_image(
                    "anomaly",
                    image_id=a['filename'],  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons,
                    num_ids=num_ids)
        except Exception as e:
            print("There is an Error in Annotated Data, please rectify it")
            print(e)
            print()
            print(a)
        

    def load_mask(self, image_id):
        """
        @function: Generate instance masks for an image.
        @input: image_id - Id of the image to which mask need to be prepared.
        subset - subset of the diretory - train | val | test

        @return: masks - A bool array of shape [height, width, instance count] 
                        with one mask per instance.
                 class_ids - The class ids present in the image.
        """
        # If not a anomaly dataset image, delegate to parent class.

        # Note: In VIA 2.0, regions was changed from a dict to a list.
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            try:
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            except:
                print("Other Shape Found, Please Check !!")
                print("I  ", i)
                print("P  ", p)
                break
            mask[rr, cc, i] = 1

        class_ids = array(info['num_ids'])
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """
        @function: Return the path of the image.
        @input: image_id - Id of the image to which mask need to be prepared.
        @return: Return the path of the image
        """
        info = self.image_info[image_id]
        if info["source"] == "anomaly":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """
    @function: Train the model
    @input: model - Model for training for the train datsets and to validate
            as well
    @return: None
    """
    # Training dataset.
    dataset_train = AnomalyDataset()
    dataset_train.load_anomaly(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = AnomalyDataset()
    dataset_val.load_anomaly(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


# def color_splash(image, mask):
#     """Apply color splash effect.
#     image: RGB image [height, width, 3]
#     mask: instance segmentation mask [height, width, instance count]
#     Returns result image.
#     """
#     # Make a grayscale copy of the image. The grayscale copy still
#     # has 3 RGB channels, though.
#     gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
#     # Copy color pixels from the original color image where mask is set
#     if mask.shape[-1] > 0:
#         # We're treating all instances as one, so collapse the mask into one layer
#         mask = (np.sum(mask, -1, keepdims=True) >= 1)
#         splash = np.where(mask, image, gray).astype(np.uint8)
#     else:
#         splash = gray.astype(np.uint8)
#     return splash


# def detect_and_color_splash(model, image_path=None, video_path=None):
#     assert image_path or video_path

#     # Image or video?
#     if image_path:
#         # Run model detection and generate the color splash effect
#         print("Running on {}".format(args.image))
#         # Read image
#         image = skimage.io.imread(args.image)
#         # Detect objects
#         r = model.detect([image], verbose=1)[0]
#         # Color splash
#         splash = color_splash(image, r['masks'])
#         # Save output
#         file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
#         skimage.io.imsave(file_name, splash)
#     elif video_path:
#         import cv2
#         # Video capture
#         vcapture = cv2.VideoCapture(video_path)
#         width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = vcapture.get(cv2.CAP_PROP_FPS)

#         # Define codec and create video writer
#         file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
#         vwriter = cv2.VideoWriter(file_name,
#                                   cv2.VideoWriter_fourcc(*'MJPG'),
#                                   fps, (width, height))

#         count = 0
#         success = True
#         while success:
#             print("frame: ", count)
#             # Read next image
#             success, image = vcapture.read()
#             if success:
#                 # OpenCV returns images as BGR, convert to RGB
#                 image = image[..., ::-1]
#                 # Detect objects
#                 r = model.detect([image], verbose=0)[0]
#                 # Color splash
#                 splash = color_splash(image, r['masks'])
#                 # RGB -> BGR to save image to video
#                 splash = splash[..., ::-1]
#                 # Add image to video writer
#                 vwriter.write(splash)
#                 count += 1
#         vwriter.release()
#     print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect anomalys.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/anomaly/dataset/",
                        help='Directory of the Anomaly dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    # parser.add_argument('--video', required=False,
    #                     metavar="path or URL to video",
    #                     help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = AnomalyConfig()
    else:
        class InferenceConfig(AnomalyConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. " "Use 'train' ".format(args.command))