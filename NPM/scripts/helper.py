import os
from typing import *
from math import pi as PI
import random
from datetime import datetime
import json
import socket
import tensorflow as tf

from tifffile import imread as tiff_imread
from tifffile import imsave as tiff_imsave
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects as sk_remove_small_objects

from tensorflow.keras.utils import Sequence, to_categorical
from scipy.ndimage import zoom as sci_zoom
from scipy.stats import pearsonr as sci_pearsonr

from skimage import filters as sk_filters
from sklearn.metrics import f1_score

"""
0. Constants
"""


def default_metrics(num_classes: int) -> List[str]:
    if num_classes == 1:
        return ['PSNR', 'Pearson']
    else:
        return ['F1']


def default_loss_functions(num_classes: int) -> List[str]:
    if num_classes == 1:
        return ['MSE']
    elif num_classes == 2:
        return ['BCE']
    else:
        return ['CCE']


"""
I. Training related
"""

"""
Custom_Generator
Used to load the training pair right before needed
The training data can be normalized here
"""


class CustomDataGenerator(Sequence):
    """
    __init__
    @:param in_dir : string, path to the directory holding all input files (images)
    @:param out_dir : string, path to the directory holding all output/target files (labels)
    @:param start : int, image with index [1, start-1] should be neglected
    @:param end : int, image with index larger than or equal to end should also be neglected ==> [start, end)
    @:param normal_x : bool, normalize the input image (phase image) from (-pi, pi) to (0,1) if true
    @:param normal_y: bool, normalize the output image (fl label) from (min, max) to (0,1) if true
    @:param batch_size : default to 1 for our image size and GPU
    @:param image_names : a list (from csv files) that gives the path to the input/output image
    """

    def __init__(
            self, in_dir, out_dir, count,
            normal_x=True, normal_y=False,
            x_in_range=None, x_out_range=None,
            y_in_range=None, y_out_range=None, y_shift=0, y_onehot=False, num_classes=1,
            augmentation=False, batch_size=1, image_names=None,
            random_sampling=False, sample_dim=(0, 0), network_mode='scratch', preprocessor=None,
            size_divisible=16
    ):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.count = count

        self.size_divisible = size_divisible
        self.normal_x = normal_x
        self.normal_y = normal_y
        self.x_in_range = x_in_range
        self.x_out_range = x_out_range
        if self.normal_x:
            assert self.x_out_range, "Invalid x out range"

        self.y_in_range = y_in_range
        self.y_out_range = y_out_range
        if self.normal_y:
            assert self.y_out_range, "Invalid y out range"
        self.aug = augmentation
        self.y_shift = y_shift
        self.y_onehot = y_onehot
        self.n_classes = num_classes

        self.batch_size = batch_size
        self.image_names = image_names
        self.sampling = random_sampling
        self.sample_dim = sample_dim
        self.network_mode = network_mode
        self.preprocessor = preprocessor

    def __len__(self):
        return (np.ceil(self.count / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for i in range(self.batch_size):
            img_idx = (idx * self.batch_size + i) % (self.count)
            read_ii = img_idx
            x_name = self.image_names[read_ii].strip().split(',')[0]
            y_name = self.image_names[read_ii].strip().split(',')[1]
            original_x = read_image_wrapper(os.path.join(self.in_dir, x_name))
            original_y = read_image_wrapper(os.path.join(self.out_dir, y_name)).astype(np.float)
            h, w = original_x.shape
            # sampling here
            if self.sampling:
                sh, sw = self.sample_dim
                assert sh % self.size_divisible == 0, "Sampled height {} not divisible by {}".format(sh,
                                                                                                     self.size_divisible)
                assert sw % self.size_divisible == 0, "Sampled weight {} not divisible by {}".format(sw,
                                                                                                     self.size_divisible)
                rs = random.randint(0, h - sh)
                cs = random.randint(0, w - sw)
                original_x = original_x[rs:rs + sh, cs:cs + sw]
                original_y = original_y[rs:rs + sh, cs:cs + sw]
            else:
                original_x = match_size_req(original_x, self.size_divisible)
                original_y = match_size_req(original_y, self.size_divisible)
            if self.preprocessor:
                original_x = np.stack((original_x,) * 3, axis=-1)
            batch_x.append(original_x)
            batch_y.append(original_y)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        if self.aug:
            #  apply a 50% flip no matter what
            flip_p = random.uniform(0, 1)
            if flip_p >= 0.5:
                batch_x = np.flip(batch_x, axis=2)
                batch_y = np.flip(batch_y, axis=2)
            rand_degree = random.randint(0, 90)
            # rand_zoom_factor = random.uniform(1, 1.2)
            # aug_x = zoom_in(rotate(original_x, rand_degree), rand_zoom_factor)
            # aug_y = zoom_in(rotate(original_y, rand_degree), rand_zoom_factor)
            aug_x = rotate(batch_x, rand_degree)
            aug_y = rotate(batch_y, rand_degree)
        if batch_x.ndim == 2:
            batch_x = np.expand_dims(batch_x, axis=3)
        if self.normal_x:
            if self.x_in_range is None:
                batch_x = conversion(np.min(batch_x), np.max(batch_x), self.x_out_range[0], self.x_out_range[1],
                                     batch_x)
            else:
                batch_x = conversion(self.x_in_range[0], self.x_in_range[1], self.x_out_range[0], self.x_out_range[1],
                                     batch_x)
        if batch_y.ndim == 2:
            batch_y = np.expand_dims(batch_y, axis=3)
        if self.normal_y:
            if self.y_in_range is None:
                batch_y = conversion(np.min(batch_y), np.max(batch_y), self.y_out_range[0], self.y_out_range[1],
                                     batch_y)
            else:
                batch_y = conversion(self.y_in_range[0], self.y_in_range[1], self.y_out_range[0], self.y_out_range[1],
                                     batch_y)
        if self.preprocessor:
            batch_x = self.preprocessor(batch_x)
        if self.network_mode == "scratch":
            if batch_x.ndim == 3:
                batch_x = np.expand_dims(batch_x, 3)
            if batch_y.ndim == 3:
                if self.y_onehot:
                    batch_y = to_categorical(batch_y, num_classes=self.n_classes)
                else:
                    batch_y = np.expand_dims(batch_y, 3)
            return batch_x, batch_y
        else:
            if self.y_onehot:
                batch_y = to_categorical(batch_y, num_classes=self.n_classes)
            else:
                batch_y = np.expand_dims(batch_y, 3)
            return batch_x, batch_y


"""
visuzlize_train_data
# For a batch, we only visualize the first one for now
@:param train_generator: training data generator
@:param fig_dir: where to save the visualization of the random training pairs
"""


def visualize_train_data(train_generator: CustomDataGenerator, fig_dir: str, num_classes: int):
    img_count = len(train_generator)
    sample_count = max(1, int(round(0.01 * img_count)))
    sample_idxes = set(random.sample(range(img_count), sample_count))

    idx = 0
    for x, y in train_generator:
        if idx in sample_idxes:
            if x.shape[-1] == 3:
                vis_x = x[0, :, :, :]  # nothing needs to be done on the input
            else:
                vis_x = x[0, :, :, 0]
 #           if train_generator.y_onehot:
 #              vis_y = np.argmax(y, axis=-1)[0, :, :, :]
 #           else:
 #               vis_y = y[0, :, :, 0]
            vis_y = y[0, :, :, 0]
            fig, axs = plt.subplots(1, 2)
            # fig.set_size_inches(8, 4)
            axs[0].imshow(vis_x)
            if num_classes >= 2:
                axs[1].imshow(vis_y, vmin=0, vmax=num_classes - 1, interpolation='none')
            else:
                axs[1].imshow(vis_y)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"train_data_vis_{idx}"))
            plt.clf()
        idx += 1


"""
Callback function after each epoch. Running the model on one fixed image
@:param out_base_dir : the root directory for all the output related to this training session
@:param model : the model we have after this epoch
@:param name: the name of the output image, typically "after_i" where i is the epoch number
"""


def evaluate_on_special(
        special_path, out_base_dir, model, name, num_classes,
        input_normalization=True, x_in_range=None, x_out_range=None, pre_processor=None,
        size_divisible=16,
):
    x_img = read_image_wrapper(special_path)
    x_img = match_size_req(x_img, size_divisible)
    if input_normalization:
        x_img = conversion(x_in_range[0], x_in_range[1], x_out_range[0], x_out_range[1], x_img)
    if pre_processor:
        x_img = np.stack((x_img,) * 3, axis=-1)
        x_img = pre_processor(x_img)
        print("In eval specail, x_img shape", x_img.shape)
        p_img = model.predict(np.reshape(x_img, (1, x_img.shape[0], x_img.shape[1], x_img.shape[2])))
    else:
        p_img = model.predict(np.reshape(x_img, (1, x_img.shape[0], x_img.shape[1], 1)))
    p_formatted = output_processor(p_img, num_classes)
    save_image_wrapper(os.path.join(out_base_dir, 'after' + name + '.tif'), p_formatted)


def preprocess_input(
        img_path: str,
        input_normalization: bool = True,
        x_in_range: tuple = None,
        x_out_range: tuple = None,
        pre_processor=None,
        size_divisible=16,
):
    assert os.path.isfile(img_path), "File does not exist: {}".format(img_path)
    x_img = read_image_wrapper(img_path)
    x_img = match_size_req(x_img, size_divisible)
    if input_normalization:
        x_img = conversion(x_in_range[0], x_in_range[1], x_out_range[0], x_out_range[1], x_img)
    if pre_processor:
        # RGB --> ImageNet/Other pre-defined preprocessing
        x_img = np.stack((x_img,) * 3, axis=-1)
        x_img = pre_processor(x_img)
        x_img = np.reshape(x_img, (1, x_img.shape[0], x_img.shape[1], x_img.shape[2]))
    else:
        x_img = np.reshape(x_img, (1, x_img.shape[0], x_img.shape[1], 1))
    return x_img


def preprocess_output(
        img_path: str,
        output_normalization: bool = False,
        y_in_range: tuple = None,
        y_out_range: tuple = None,
        size_divisible=16
):
    assert os.path.isfile(img_path), "File does not exist: {}".format(img_path)
    y_img = read_image_wrapper(img_path)
    y_img = match_size_req(y_img, size_divisible)
    if output_normalization:
        y_img = conversion(y_in_range[0], y_in_range[1], y_out_range[0], y_out_range[1], y_img)
    return y_img


"""
Loading all data into RAM beforehand is almost always faster than using a data generator.
So if you calculated and ensured that the RAM can fit all the data (remember augmentation takes memory too), use this.
"""


def load_all_data(
        data_path, in_dir, out_dir,
        scale_x, magic_x_range, scale_y, magic_y_range,
        train_names, val_names, augmenting_factor,
        out_txt
):
    x_train = []
    y_train = []
    x_val = []
    y_val = []

    for line in train_names:
        x_name, y_name = line.strip().split(',')
        x_img = read_image_wrapper(os.path.join(data_path, in_dir, x_name))
        if scale_x:
            x_img = conversion(magic_x_range[0], magic_x_range[1], 0, 1, x_img)
        x_img = np.reshape(x_img, (x_img.shape[0], x_img.shape[1], 1))
        x_train.append(x_img)
        y_img = read_image_wrapper(os.path.join(data_path, out_dir, y_name))
        if scale_y:
            y_img = conversion(magic_y_range[0], magic_y_range[1], 0, 1, y_img)
        y_img = np.reshape(y_img, (y_img.shape[0], y_img.shape[1], 1))
        y_train.append(y_img)

    for line in val_names:
        x_name, y_name = line.strip().split(',')
        x_img = read_image_wrapper(os.path.join(data_path, in_dir, x_name))
        if scale_x:
            x_img = conversion(magic_x_range[0], magic_x_range[1], 0, 1, x_img)
        x_img = np.reshape(x_img, (x_img.shape[0], x_img.shape[1], 1))
        x_val.append(x_img)
        y_img = read_image_wrapper(os.path.join(data_path, out_dir, y_name))
        if scale_y:
            y_img = conversion(magic_y_range[0], magic_y_range[1], 0, 1, y_img)
        y_img = np.reshape(y_img, (y_img.shape[0], y_img.shape[1], 1))
        y_val.append(y_img)

    for ai in range(augmenting_factor - 1):
        for line in train_names:
            x_name, y_name = line.strip().split(',')
            x_img = read_image_wrapper(os.path.join(data_path, in_dir, x_name))
            rand_degree = random.randint(1, 21)
            x_img = rotate(x_img, rand_degree)
            if scale_x:
                x_img = conversion(magic_x_range[0], magic_x_range[1], 0, 1, x_img)
            x_img = np.reshape(x_img, (x_img.shape[0], x_img.shape[1], 1))
            x_train.append(x_img)
            y_img = read_image_wrapper(os.path.join(data_path, out_dir, y_name))
            y_img = rotate(y_img, rand_degree)
            if scale_y:
                y_img = conversion(magic_y_range[0], magic_y_range[1], 0, 1, y_img)
            y_img = np.reshape(y_img, (y_img.shape[0], y_img.shape[1], 1))
            y_train.append(y_img)

    file_and_console("Loading x_train: %d, x_val: %d, y_train: %d, y_val: %d" % (
        len(x_train), len(x_val), len(y_train), len(y_val)), out_txt)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    return x_train, y_train, x_val, y_val


"""
II. Metric related
"""

"""
output_processor : convert output tensor (4D) to target, analyzable tensor (usually 2D, class label for instance)
@:param raw_tensor: a 4D tensor from model.predict(), I assumed batch = 1 here
@:param num_classes: number of classes
    1 -- regression (as it is)
    2 -- binary classification (>0.5)
    >=3 -- multiclass classification (argmax)
"""


def output_processor(raw_tensor, num_classes):
    if num_classes == 1:
        output = raw_tensor[0, :, :, 0]
        return output.astype(np.float32)
    elif num_classes == 2:
        output = raw_tensor[0, :, :, 0]
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        return output.astype(np.uint8)
    else:
        output = np.argmax(raw_tensor, axis=3)
        return output[0, :, :].astype(np.uint8)


def metric_wrapper(metric_name: str, original: np.ndarray, contrast: np.ndarray, num_classes: int = 1,
                   dynamic_range: float = 1.0) -> Union[float, Tuple[float, float]]:
    """
    metric_wrapper() returns the requested metric between two images

    :param metric_name: the name of the metric to compute
    :param original: ground truth image
    :param contrast: prediction (need to be preprocessed)
    :param num_classes: number of classes for computing f_1 score
    :param dynamic_range: dynamic range for computing SSIM
    :return:
    """
    if metric_name == "PSNR":
        return psnr(original, contrast, 1)
    if metric_name == 'F1':
        return f1_wrapper(original, contrast, num_classes)
    if metric_name == 'Pearson':
        return pearson_wrapper(original, contrast)[0]
    if metric_name == 'iou':
        return iou(original, contrast)
    if metric_name == "SSIM":
        return ssim(original, contrast, dynamic_range)
    if metric_name=="MSSSIM":
        return msssim_wrapper(original,contrast)

def f1_wrapper(original: np.ndarray, contrast: np.ndarray, num_classes: int = 0) -> Union[float, List[float]]:
    """
    f1_wrapper() computes the f1 score between two images for each possible class.

    :param original: the ground truth image
    :param contrast: the prediction image
    :param num_classes: number of classes (counting background)
    :return: a list of F-1 score from class 0 to class num_classes - 1
    """
    if num_classes == 0:
        num_classes = max(np.max(original), np.max(contrast))
    f1 = f1_score(original.flatten(), contrast.flatten(), labels=range(num_classes), average=None)
    return f1


def psnr(original: np.ndarray, contrast: np.ndarray, known_max: float = 0) -> float:
    """
    psnr() calculates peak signal to noise ratio between two images

    :param original: a numpy array representing input image 1
    :param contrast: a numpy array representing input image 2
    :param known_max: the maximum possible value in both original and contrast
    :return: a floating point number representing the ratio between two images
    """
    original = np.ravel(original)
    contrast = np.ravel(contrast)
    assert (original.shape == contrast.shape), "image shape must be the same"
    if known_max == 0:
        p_max = max(np.max(original), np.max(contrast))
    else:
        p_max = known_max
    mse = (np.square(original - contrast)).mean()
    psnr_val = 10 * np.log10(p_max ** 2 / mse)
    return psnr_val


def pearson_wrapper(original: np.ndarray, contrast: np.ndarray) -> Tuple[float, float]:
    """
    pearson_wrapper() computes pearson correlation coefficient between two images using the scipy verion

    :param original: the ground truth image
    :param contrast: the prediction image
    :return: the pearson correlation coefficient and the p-value
    """
    return sci_pearsonr(original.flatten(), contrast.flatten())

def msssim_wrapper(original: np.ndarray, contrast: np.ndarray) -> Tuple[float, float]:
    """
    computes multiscale ssim
    """
    p_max = max(np.max(original.flatten()), np.max(contrast.flatten()))
    gt1=tf.convert_to_tensor(np.expand_dims(original,2), dtype=float)
    pr1=tf.convert_to_tensor(np.expand_dims(contrast,2), dtype=float)
     
    return tf.image.ssim_multiscale(gt1, pr1, max_val=p_max)


def ssim(original: np.ndarray, contrast: np.ndarray, dynamic_range: float) -> float:
    """
    ssim() calculates structural similarity index between two images

    :param original: ground truth image
    :param contrast: prediction image
    :param dynamic_range: max_val - min_val in both images (usually is 1)
    :return: returning the structural similarity index measure between two images
    """
    #  Flatten to vector
    original = original.flatten()
    contrast = contrast.flatten()
    #  Computing the necessary quantities
    mu_x = np.mean(original)
    var_x = np.var(original)
    mu_y = np.mean(contrast)
    var_y = np.var(contrast)
    cov = np.cov(original, contrast)
    # computing covariance following https://en.wikipedia.org/wiki/Covariance
    inside = (original - mu_x) * (contrast - mu_y)
    cov_xy = np.mean(inside)
    L = dynamic_range
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    top = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    bot = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return top / bot


"""
iou: calculates the intersection over union score
@:param target: the ground truth image
@:param prediction: the prediction image

"""


def iou(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    iou() calculates the Intersection-over-Union score between two images
    :param target: the ground truth label map
    :param prediction: the predicted label map
    :return: the iou score between two maps
    Refernce: https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    """
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


"""
III. Morphological operation related
"""

"""
conversion_without_clipping: scaling the input image from (in_min, in_max) to (out_min, out_max)
PS: this function does NOT take care of outliers. Outliers remain outliers in output.
@:param in_min : minimal value of input array (or upper bound specific to data acquisition)
@:param in_max : maximal value of input array (or upper bound specific to data acquisition)
@:param out_min: minimal value of output array, typically 0
@:param out_max: maximal value of output array, typically 1
"""


def conversion_without_clipping(in_min, in_max, out_min, out_max, in_array):
    return (out_max - out_min) * (in_array - in_min) / (in_max - in_min) + out_min  


"""
conversion: scaling the input image from (in_min, in_max) to (out_min, out_max)
PS: this function takes care of outliers. Outliers will be clipped to out_min, out_max
@:param in_min : minimal value of input array (or upper bound specific to data acquisition)
@:param in_max : maximal value of input array (or upper bound specific to data acquisition)
@:param out_min: minimal value of output array, typically 0
@:param out_max: maximal value of output array, typically 1
"""


def conversion(in_min, in_max, out_min, out_max, in_array):
    converted = (out_max - out_min) * (in_array - in_min) / (in_max - in_min) + out_min
    converted[converted > out_max] = out_max
    converted[converted < out_min] = out_min
    return converted


"""
conversion with shift: shifting the input image first, then scale (Probably no one will use this?)
PS: this function takes care of outliers. Outliers will be clipped to out_min, out_max
@:param in_min : minimal value of input array (or upper bound specific to data acquisition)
@:param in_max : maximal value of input array (or upper bound specific to data acquisition)
@:param out_min: minimal value of output array, typically 0
@:param out_max: maximal value of output array, typically 1
"""


def conversion_with_shift(in_min, in_max, out_min, out_max, shift, in_array):
    in_array[in_array != 0] += shift
    in_max += shift
    in_min = min(0, in_min + shift)
    converted = (out_max - out_min) * (in_array - in_min) / (in_max - in_min) + out_min
    converted[converted > out_max] = out_max
    converted[converted < out_min] = out_min
    return converted


"""
otsu_thresholding_wrapper: wrapper function for sckit-image otsu threshoding
@:param in_img : numpy array representing the input image
@:return thresholded_img : numpy array of same shape as input image, in np.uint8, 0 for bg, 255 for fg
"""


def otsu_thresholding_wrapper(in_img):
    threshold_val = sk_filters.threshold_otsu(in_img)
    thresholded_img = (in_img >= threshold_val).astype(np.uint8) * 255
    return thresholded_img


"""
read_image_and_crop_into_four : read an image, pad it so that its dimension is multiple of 128
@:param f_name : the path to the image
@:return out_images : a list of numpy arrays, each representing 1/4 of the original image
"""


def read_image_and_crop_into_four(f_name):
    original = read_image_wrapper(f_name)
    #  We need to find the nearest dimension that's multiple of 128
    # out_h = (original.shape[0] // 128 + 1) * 128
    # out_w = (original.shape[1] // 128 + 1) * 128
    out_h = 1792
    out_w = 1792
    padded = randomly_pad_image(original, (out_h, out_w), 'Zero')
    #  Now we can crop
    bound_h = out_h // 2
    bound_w = out_w // 2
    out_images = [
        padded[0:bound_h, 0:bound_w],
        padded[bound_h:out_h, 0:bound_w],
        padded[0:bound_h, bound_w:out_w],
        padded[bound_h:out_h, bound_w:out_w]
    ]
    for i in out_images:
        assert i.shape == (bound_h, bound_w), "Shape is wrong"
    return out_images


"""
rotate
@:param img : ndarray, representing the original image, should be 2d
@:param degree : int, between 0 and 360, angle to rotate this original image
TODO: there might be faster implementation of this?
"""


def rotate(img, degree):
    assert (2 <= len(img.shape) <= 4), "Please check image dimension."
    if len(img.shape) == 2:
        rotated_img = ndimage.rotate(img, degree, mode='reflect', reshape=False, order=0)
    else:
        # assuming (n, h, w) or (n, h, w, c) here
        rotated_img = ndimage.rotate(img, degree, axes=(1, 2), mode='reflect', reshape=False, order=0)
    return rotated_img.astype(img.dtype)


"""
zoom_in
@:param img : ndarray, representing the original image, should be 2d
@:param factor : floating point number, how much to zoom the original image
"""


def zoom_in(img, factor=1):
    zoomed = sci_zoom(img, factor)
    top_left = (int(zoomed.shape[0] / 2 - img.shape[0] / 2), int(zoomed.shape[1] / 2 - img.shape[1] / 2))
    bottom_right = (int(zoomed.shape[0] / 2 + img.shape[0] / 2), int(zoomed.shape[1] / 2 + img.shape[1] / 2))
    a = zoomed[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    assert (a.shape == img.shape), "Zoomed image shape wrong!"
    return a


"""
randomlyPadImage()
Input:
    in_img_data: numpy array, representing the input image
    out_shape: tuple, (h,w) of the output image
    value: str, options of padded pixel values
        'Random' - padding random values
        'Zero' - padding zero 
        'One' - padding one
    range: tuple, if value is selected random
        (-min, max) - range of random values to be padded
Return:
    out_img_data: numpy array, of shape (h,w), representing output image
Updates on 07/12/2019:
    - Deleting the code to iterate through the image and copying pixel 
      value one by one. Replacing that portion with numpy built-in feature.
    - Hard-coding the starting point to be (0,0). Cause this function is 
      mainly used for evaulation now. Fixating the postion makes it easier
      to compare different model's performance.
"""


def randomly_pad_image(in_img_data, out_shape, value='Random', bound=None):
    in_shape = in_img_data.shape
    random_r = 0
    random_c = 0
    if value == 'Zero':
        padded_data = np.zeros(shape=out_shape, dtype=in_img_data.dtype)
    elif value == 'One':
        padded_data = np.ones(shape=out_shape, dtype=in_img_data.dtype)
    elif value == 'Random':
        random_data = np.random.rand(out_shape[0], out_shape[1])
        # padded_data = conversion(0, 1, -PI, PI, random_data)
        padded_data = random_data.astype(in_img_data.dtype)

    padded_data[0:in_shape[0], 0:in_shape[1]] = in_img_data
    return padded_data


"""
size_filter
@:param img : a numpy array representing the unfiltered image, should be of uint8
@:param min_size: an int, minimal size for an object to be kept
@:return removed_out : a numpy array of same dimension as img, representing filtered image
"""


def size_filter(img, min_size=0):
    removed = sk_remove_small_objects(img.astype(np.bool), min_size=min_size, connectivity=2)
    mask = np.logical_and(removed, img).astype(np.uint8)
    removed_out = np.multiply(img, mask)
    return removed_out


"""
match_size_req
@:param img: input image (h, w)
@:param divisible : an integer specifying that the image should have dimensions divisible by this number
@:return max image starting from top left corner that matches this divisible constraint
"""


def match_size_req(img: np.ndarray, divisible: int) -> np.ndarray:
    assert img.ndim == 2, "Expecting shape (h,w) of input"
    h, w = img.shape
    max_h = (h // divisible) * divisible
    max_w = (w // divisible) * divisible
    return img[:max_h, :max_w]


"""
IV. Utilities
"""

"""
contain_any_as_substring
@:param string : the string we are looking at
@:param group_of_candidates : a list of strings that could potentially be substring of string
@:return 0 if a string in the list is found in string, -1 otherwise (mirroring .find() method)
"""


def find_any_as_substring(string, group_of_candidates):
    for candidate in group_of_candidates:
        if string.find(candidate) != -1:
            return 0
    return -1


"""
read_json_file_wrapper: wrapper function to read a json file, returns the contents in dictionary
"""


def read_json_file_wrapper(f_name):
    with open(f_name) as f:
        data = json.load(f)
    return data


"""
write_json_string_wrapper: wrapper function to convert a dictionary to a readable string (for printing/logging)
"""


def write_json_string_wrapper(dictionary_obj):
    string = json.dumps(dictionary_obj, indent=4)
    return string


"""
write_json_file_wrapper: wrapper function to write a dictionary object into a json file
"""


def write_json_file_wrapper(dictionary_obj, f_name):
    with open(f_name, 'w') as outfile:
        json.dump(dictionary_obj, outfile, indent=4)


"""
save_image_wrapper
@:param f_name : name/path of the image to save
@:param img_arr : numpy array that represents the image
Just a wrapper function of tiff_imsave
"""


def save_image_wrapper(f_name, img_arr):
    tiff_imsave(f_name, img_arr)


"""
read_image_wrapper
@:param f_name : name/path of the image to be read
@:return : None
Just a wrapper function of tiff_imread
"""


def read_image_wrapper(f_name):
    return tiff_imread(f_name)


"""
file_and_console: function to print a string to console and write it to a txt file
@:param line : the string that will get printed/written
@:param f_name : the file to write the string to
@:return none
"""


def file_and_console(line, f_name):
    with open(f_name, 'a') as f:
        f.write(line + '\n')
    print(line)


"""
get_hostname
"""


def get_hostname():
    return socket.gethostname()


"""
index_based_on_time
@:return a string like: "20190815_154334"
"""


def index_based_on_time():
    idx = datetime.now().strftime("%Y%m%d_%H%M%S")
    return idx


def append_model_summary_to_file(filename, model):
    with open(filename, 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def prepare_log_info(
        num_epochs=-1,
        cp_period=-1,
        verbose=-1,
        l_rate=-1,
        validation_split=-1,
        input_normal=None,
        batch_size=-1,
        idx='1',
        train_count=-1, val_count=-1, test_count=-1,
        load_time='-1',
        out_base_dir='no where',
):
    line = ''
    line += 'Number of epochs is ' + str(num_epochs) + '\n'
    line += 'Verbose mode is ' + str(verbose) + '\n'
    line += 'Learning rate is ' + str(l_rate) + '\n'
    line += 'Training count ' + str(train_count) + '\n'
    line += 'Validation count ' + str(val_count) + '\n'
    line += 'Test count ' + str(test_count) + '\n'
    line += 'Check point period is ' + str(cp_period) + '\n'
    line += 'Validation split is ' + str(validation_split) + '\n'
    line += "Input Normalized? " + str(input_normal) + '\n'
    line += "Batch size is " + str(batch_size) + '\n'
    line += 'Unique identifier is ' + idx + '\n'
    line += 'Output goes to ' + out_base_dir + '\n'
    return line


"""
fractional_seconds_to_string
@:param f_sec : number of seconds elapsed, fractional (floating point number)
@:return a string in the format "xxx days xxx hours xxx minutes xxx seconds"
"""


def fractional_seconds_to_string(f_sec):
    days = f_sec // (3600 * 24)
    hours = (f_sec - days * 3600 * 24) // 3600
    minutes = (f_sec - days * 3600 * 24 - hours * 3600) // 60
    seconds = (f_sec - days * 3600 * 24 - hours * 3600 - minutes * 60)
    if days > 0:
        return str(days) + ' days ' + str(hours) + ' hours ' + str(minutes) + ' minutes ' + '%.2f' % (
            seconds) + ' seconds'
    if hours > 0:
        return str(hours) + ' hours ' + str(minutes) + ' minutes ' + '%.2f' % (seconds) + ' seconds'
    return str(minutes) + ' minutes ' + '%.2f' % (seconds) + ' seconds'


"""
assert_dir: create the directory if not existent
@:param path: path to the directory to be created
"""


def assert_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path
