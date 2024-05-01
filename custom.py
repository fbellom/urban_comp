import os
import sys
import json
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
import imgaug

# ROOT Directory
ROOT_DIR = os.getcwd()
DATASET_DIR = os.path.join(ROOT_DIR,'dataset')

# import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained model weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and models checkpoints, if not provided
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


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
    NUM_CLASSES = 1 + 3  # Background, Cars, Parking, Person

    # Training steps or EPOCHS
    STEPS_PER_EPOCH = 5

    # Skip Detection < 90% confidence level
    DETECTION_MIN_CONFIDENCE = 0.9

    # Hyperparameters

    LEARNING_RATE = 0.001


###############################################
# Dataset
###############################################
class CustomDataset(utils.Dataset):
    """
    Load Custom Dataset
    """

    def load_custom(self, dataset_dir, subset):
        """
        Call the Dataset and process it
        """

        # Add Classes

        self.add_class("object", 1, "car")
        self.add_class("object", 2, "parking_spot")
        self.add_class("object", 3, "person")

        # Train or Validation
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        with open(os.join.path(dataset_dir,f"{subset}.json"), "r") as file:
            data_ann = json.loads(file.read())

        annotations = list(data_ann.values())

        annotations = [a for a in annotations if a["regions"]]

        # Add Images
        for a in annotations:
            polygons = [r["shape_attributes"] for r in a["regions"]]
            objects = [s["region_attributes"]["names"] for s in a["regions"]]

            print("objects", objects)
            name_dict = {"car": 1, "parking_spot": 2, "person": 3}

            # Tuples
            num_ids = [name_dict[a] for a in objects]

            print("numids", num_ids)
            image_path = os.path.join(dataset_dir, a["filename"])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=a["filename"],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                num_ids=num_ids,
            )

    def load_mask(self, image_id):
        """
        Load the Mask
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert Polygons to an image shape
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info["num_ids"]
        mask = np.zeros(
            [info["height"], info["width"], len(info["polygons"])], dtype=np.uint8
        )

        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
            rr = np.clip(rr, 0, mask.shape[0] - 1)
            cc = np.clip(cc, 0, mask.shape[1] - 1)
            mask[rr, cc, i] = 1

        # Return the mask
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """
        Return the path of the image
        """
        info = self.image_info["image_id"]
        if info["source"] == "object":
            return info["path"]
        else:
            return super(self.__class__, self).image_reference(image_id)


########################################
# TRAINING THE MODEL
########################################


def train(model):
    """
    Train the Model
    """

    # Training Dataset
    dataset_train = CustomDataset()
    dataset_train.load_custom(
        DATASET_DIR, "train"
    )
    dataset_train.prepare()

    # Validation
    dataset_val = CustomDataset()
    dataset_val.load_custom(
        DATASET_DIR, "val"
    )
    dataset_val.prepare()

    # Image Transform and Training
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=300,
        layers="heads",
        # augmentation=imgaug.augmenters.Sequential(
        #     [
        #         imgaug.augmenters.Fliplr(1),
        #         imgaug.augmenters.Flipud(1),
        #         imgaug.augmenters.Affine(rotate=(-45, 45)),
        #         imgaug.augmenters.Affine(rotate=(-90, 90)),
        #         imgaug.augmenters.Affine(scale=(0.5, 1.5)),
        #         imgaug.augmenters.Crop(px=(0, 10)),
        #         imgaug.augmenters.Grayscale(alpha=(0.0, 1.0)),
        #         imgaug.augmenters.AddToHueAndSaturation((-20, 20)),
        #         imgaug.augmenters.Add((-10, 10), per_channel=0.5),
        #         imgaug.augmenters.Invert(0.05, per_channel=True),
        #         imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
        #     ]
        # ),
    )


config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
weights_path = COCO_WEIGHTS_PATH
if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

model.load_weights(
    weights_path,
    by_name=True,
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
)

train(model)
