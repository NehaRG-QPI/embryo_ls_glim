import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam as Adam
import numpy as np
from typing import *
sm.set_framework('tf.keras')
import train_utils

"""
network
Creates a segmentation neural network with pre-trained encoder weights
@:param input_size: the size of input tensor (H, W, C)
@:param l_rate: learning rate of the model
@:param num_classes: the number of channels in the output channel.
                        1 -- sigmoid activation, regression problem (float to float)
                        2 -- sigmoid activation, binary classification problem (float to [0,1])
                        3 -- softmax activation, multi-class classification problem (float to [0,1,2])
                        ...
"""


def network(
        network_name: str,
        backbone_name: str,
        input_size: tuple = (None, None, 1),
        l_rate: float = 1e-4,
        num_classes: int = 2,
        class_weights: List[float] = None,
        loss_functions: List[str] = None,
        loss_ratios: List[float] = None,
):
    if class_weights:
        assert len(class_weights) == num_classes, "Number of classes does not match with class weight length"
    model = None
    if network_name == "U-Net":
        assert input_size[2] is not None, "Channel dimension must be specified"
        if input_size[-1] == 3:
            activation_function = "softmax" if num_classes > 2 else "sigmoid"
            out_channel = num_classes if num_classes > 2 else 1
            model = sm.Unet(
                backbone_name,
                input_shape=input_size,
                classes=out_channel,
                encoder_weights='imagenet',
                activation=activation_function
            )
        else:
            print("Shouldn't see this for now")
            raise AssertionError("Shouldn't have 1-channel input specified for now")
            model_shape = tuple((list(input_size)[:-1] + [3]))
            base_model = sm.Unet(
                backbone_name,
                input_shape=model_shape,
                classes=num_classes,
                encoder_weights='imagenet'
            )
            inp = Input(shape=input_size)
            l1 = Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
            oup = base_model(l1)
            model = Model(inp, oup, name=base_model.name)
    else:
        raise Exception("Not implemented")

    opt = Adam(learning_rate=l_rate)

    total_loss = None
    for loss_function_name, loss_ratio in zip(loss_functions, loss_ratios):
        if not total_loss:
            total_loss = train_utils.proper_loss_function(loss_function_name, loss_ratio, class_weights)
        else:
            total_loss = total_loss + train_utils.proper_loss_function(loss_function_name, loss_ratio, class_weights)

    if num_classes == 1:
        # model_loss = tf.keras.losses.mean_squared_error
        model_logging_metrics = []
    else:
        # if class_weights:
        #     dice_loss = sm.losses.DiceLoss(class_weights=np.array(class_weights))
        # else:
        #     dice_loss = sm.losses.DiceLoss()
        # if num_classes > 2:
        #     focal_loss = sm.losses.CategoricalFocalLoss()
        # else:
        #     focal_loss = sm.losses.BinaryFocalLoss()
        # model_loss = dice_loss + focal_loss * f_to_d_ratio
        model_logging_metrics = [sm.metrics.iou_score, sm.metrics.FScore()]

    model.compile(
        optimizer=opt,
        loss=total_loss,
        metrics=model_logging_metrics,
    )
    return model
