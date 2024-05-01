import mrcnn
import mrcnn.config
from mrcnn.config import Config
import mrcnn.model
import mrcnn.visualize
import cv2
import os

ROOT_DIR = os.getcwd()
TRAINED_MODEL_FILE = "mask_rcnn_object_0013.h5"
SAMPLE_IMAGE_FILE="XYZ.JPG"
CLASS_NAMES = ['car', 'parking_spot', 'person']
TRAINED_MODEL_PATH = os.path.join(ROOT_DIR,TRAINED_MODEL_FILE)
SAMPLE_IMAGE_PATH=os.path.join(ROOT_DIR,SAMPLE_IMAGE_FILE)

class CustomConfig(Config):
    """
    Configuration for training on Custom Dataset
    """

    NAME = "object"

    # Number of GPU to use, When using CPU set it to 1
    GPU_COUNT = 1

    # Images per GPU
    IMAGES_PER_GPU = 1

    # Number of Classes including background
    NUM_CLASSES = 1 + len(CLASS_NAMES)  # Background, Cars, Parking, Person

model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=CustomConfig(),
                             model_dir=ROOT_DIR)

model.load_weights(filepath=TRAINED_MODEL_PATH, 
                   by_name=True)


# load the input image, convert it from BGR to RGB channel
image = cv2.imread(SAMPLE_IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])