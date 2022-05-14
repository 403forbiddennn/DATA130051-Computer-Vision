# This is the implementation of the midterm project Faster R-CNN.

## Document Details
+ [nets](./nets) contains networks required for this project, including [resnet](./nets/resnet50.py), [vgg16](./nets/vgg16.py), [rpn](./nets/rpn.py) and [Faster R_CNN](./nets/frcnn.py)
+ [predict.py](./predict.py) is used to predict an input image.
+ [get_map.py](./get_map.py) is used to evaluate the model, calculating mAP value of the predicted result.
+ [train.py](train.py) is the training file.
+ [utils](./utils) contains necessary utils for Faster R-CNN.

## Running this project
+ clone this project repo
+ download datasets from this [link](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and unzip it to this root, you will get a folder named VOCdevkit.
+ run [voc_annotation.py](./voc_annotation.py) to generate training and testing dataset.
+ run [train.py](./train.py) to train this model, and the model will be saved to logs directory. The default hyperparameters in this file is the hyperparameters used to train Faster R-CNN with ResNet-50 and is supposed to get almost the same result described in our report.
