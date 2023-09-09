import numpy as np
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras import backend as K
from typing import *
from segmentation_models.base import Loss

"""
correlation_coefficient_loss_v1
@:param y_true: ground truth tensor
@:param y_pred: prediction tensor
This function basically computes 1 - pho ^ 2 where pho is the pearson correlation between ground truth and prediction
"""


# https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras/46620771
def correlation_coefficient_loss_v1(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


"""
correlation_coefficient_loss_v2
@:param y_true: ground truth tensor
@:param y_pred: prediction tensor
This function basically computes (1 - pho) ^ 2 where pho is the pearson correlation between ground truth and prediction.
This loss function will force pho to be as close to 1 as possible. We don't really want negative pho.
"""


# https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras/46620771
def correlation_coefficient_loss_v2(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(1 - r)


def reverse_huber_loss(y_true, y_pred, delta):
    """
    reverse_huber_loss() computes the reversed Huber Loss between two tensors
    loss = 0.5 * x ^ 2 if | x | > d
    loss = 0.5 * d ^ 2 + d * (| x | - d) if | x | <= d
    Implementation follows the structure of Huber loss in TF: https://github.com/tensorflow/tensorflow/blob/85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/tensorflow/python/keras/losses.py#L1426

    :param y_true: ground truth tensor
    :param y_pred: prediction tensor
    :param delta: the thresholding value
    :return: a float (tensor) representing the loss
    """
    # type casting to be safe
    y_pred = tf.cast(y_pred, dtype=K.floatx())
    y_true = tf.cast(y_true, dtype=K.floatx())
    delta = tf.cast(delta, dtype=K.floatx())
    # error
    error = tf.subtract(y_pred, y_true)
    abs_error = tf.abs(error)
    half = tf.convert_to_tensor(0.5, dtype=abs_error.dtype)
    return K.mean(tf.where(abs_error > delta, half * tf.pow(error, 2), half * tf.pow(delta, 2) + delta * (delta - abs_error)))


"""
Loss Wrappers
All following classes are wrapper of common loss functions.
Some are not present in segmentation_models.base.losses
    - MSE()
    - MAE()
Some are wrapping around the tf.keras version
    - BCE()
    - CCE()
    - HuberLoss()
Some are purely custom
    - PearsonLoss()
    - BerhuLoss()
The pupose to create these wrappers is so that they can inherit the operator overloading defined for segmentation_models.loss.
This enables me to write PearsonLoss() + MAE() in a more systematic way.
"""
class customLoss(Loss):
    def __init__(self, max_val=1):
        super().__init__(name='Custom')
        self.max_val = max_val
 

    def __call__(self, gt, pr):
        eps=1e-6
        ssiml=1 - tf.image.ssim_multiscale(gt, pr, max_val=self.max_val)
        pearsonl=correlation_coefficient_loss_v1(gt, pr)
        msel=tf.keras.losses.mean_squared_error(gt, pr)
        mael=tf.keras.losses.mean_absolute_error(gt, pr)
        psnrl=tf.image.psnr(gt, pr, max_val=self.max_val, name=None)

        # mael*ssiml*pearsonl--1
        # mse+ssiml*pearsonl--2
        tl=mael*ssiml*pearsonl
        return tl


class SSIMLoss(Loss):
    def __init__(self, max_val=1):
        super().__init__(name='SSIM')
        self.max_val = max_val

    def __call__(self, gt, pr):
        return 1 - tf.image.ssim_multiscale(gt, pr, max_val=self.max_val)


class BerhuLoss(Loss):
    def __init__(self, delta=0.5):
        super().__init__(name='Berhu')
        self.delta = delta

    def __call__(self, gt, pr):
        return reverse_huber_loss(gt, pr, self.delta)


class HuberLoss(Loss):
    def __init__(self, delta=0.5):
        super().__init__(name='Huber')
        self.delta = delta

    def __call__(self, gt, pr):
        # loss = 0.5 * x ^ 2 if | x | <= d
        # loss = 0.5 * d ^ 2 + d * (| x | - d) if | x | > d
        huber_loss = tf.keras.losses.Huber(delta=self.delta)
        return huber_loss(gt, pr)


class PearsonLossV1(Loss):
    def __init__(self):
        super().__init__(name='pearson_loss_v1')

    def __call__(self, gt, pr):
        return correlation_coefficient_loss_v1(gt, pr)


class PearsonLossV2(Loss):
    def __init__(self):
        super().__init__(name='pearson_loss_v2')

    def __call__(self, gt, pr):
        return correlation_coefficient_loss_v2(gt, pr)


class MSE(Loss):
    def __init__(self):
        super().__init__(name='mse')

    def __call__(self, gt, pr):
        return tf.keras.losses.mean_squared_error(gt, pr)


class MAE(Loss):
    def __init__(self):
        super().__init__(name='mae')

    def __call__(self, gt, pr):
        return tf.keras.losses.mean_absolute_error(gt, pr)


class BCE(Loss):
    def __init__(self):
        super().__init__(name='bce_tf')

    def __call__(self, gt, pr):
        return tf.keras.losses.binary_crossentropy(gt, pr)


class CCE(Loss):
    def __init__(self):
        super().__init__(name='cce_tf')

    def __call__(self, gt, pr):
        return tf.keras.losses.sparse_categorical_crossentropy(gt, pr)


NO_PARAMS_LOSS_FUNCTIONS = {
    "MSE": MSE(),
    "mse": MSE(),
    "MAE": MAE(),
    "mae": MAE(),
    "BCE": BCE(),
    "CCE": CCE(),
    "BF": sm.losses.BinaryFocalLoss(),
    "CF": sm.losses.CategoricalFocalLoss(),
    "Pearson_v1": PearsonLossV1(),
    "Pearson_v2": PearsonLossV2(),
    "Huber": HuberLoss(),
    "Berhu": BerhuLoss(),
    "SSIM": SSIMLoss(),
    "Custom":customLoss()
}


def proper_loss_function(name: str, loss_ratio: float, class_weights: List[float] = None) -> Callable:
    if name in NO_PARAMS_LOSS_FUNCTIONS:
        if loss_ratio == 1:
            return (NO_PARAMS_LOSS_FUNCTIONS[name])
        else:
            return (NO_PARAMS_LOSS_FUNCTIONS[name]) *(loss_ratio)

    elif name == "DICE":
        if class_weights:
            return sm.losses.DiceLoss(class_weights=np.array(class_weights)) * loss_ratio
        else:
            return sm.losses.DiceLoss() * loss_ratio

    else:
        raise Exception(f"Unsupported loss function {name}")