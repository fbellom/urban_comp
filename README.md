# Project Urban Computing

Use Tensorflow 2. Edit

## Install the libraries

pip install -r requirements.txt

## Install mrcnn matterport

```
cd repo
python setup.py install
```

## Check Paths in custom.py

```
ROOT_DIR=<PATH-TO-PROJECT>

annotations1 = json.loads(open("<PATH-TO>/dataset/train/train.json"))

dataset_train.load_custom("<PATH-TO>/dataset", "train")

dataset_val.load_custom("<PATH-TO>/dataset", "val")
```

## Download COCO weights here

https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
