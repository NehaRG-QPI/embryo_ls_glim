import model_pool as model_pool
import unet2 as model_scratch
import helper

import os
import time
import argparse
import random

import segmentation_models as sm
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback

from tensorflow.keras.models import load_model

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np




from typing import *




import glob


def main():
    data_base = r"./../data"

    train_res_dir = r"sessions/trained_model"
    model_name = r"final_model.h5"
    model_path = os.path.join(train_res_dir, model_name)
    param_path = os.path.join(train_res_dir, "trained_summary.json")
    pred_dir = os.path.join(train_res_dir, f"eval_{model_name.split('.')[0]}", "infer_on_all")
    fig_dir = os.path.join(train_res_dir, f"infer_{model_name.split('.')[0]}")
    img_paths = glob.glob(os.path.join(data_base, "*.tif"))
    tic = time.perf_counter()
    infer_on_images(img_paths=img_paths, pred_dir=pred_dir, model_path=model_path, param_path=param_path,
                    fig_dir=fig_dir)
    toc = time.perf_counter()
    print(f"Evaluating on {len(img_paths)} images took {toc - tic} seconds.")


def infer_on_images(
        img_paths: List[str],
        pred_dir: str,
        model_path: str,
        param_path: str,
        fig_dir: str
):
    # make sure input files exist
    assert os.path.isfile(model_path), f"Invalid model path {model_path}"
    assert os.path.isfile(param_path), f"Invalid param path {param_path}"
    # create output directory
    helper.assert_dir(pred_dir)
    helper.assert_dir(fig_dir)

    # Data information
    params = helper.read_json_file_wrapper(param_path)
    input_normal = params["normal_input"]
    output_normal = params["normal_output"]
    #  Scaling parameters
    #  x range
    magic_x_in_range = None
    magic_x_out_range = None
    if input_normal:
        magic_x_in_range = params["magic_x_in_range"]
        magic_x_out_range = params["magic_x_out_range"]
        assert magic_x_in_range[0] or magic_x_in_range[1], "X in range cannot be both 0"
        assert magic_x_out_range[0] or magic_x_out_range[1], "X out range cannot be both 0"
    #  y range
    magic_y_in_range = None
    magic_y_out_range = None
    if output_normal:
        magic_y_in_range = params["magic_y_in_range"]
        magic_y_out_range = params["magic_y_out_range"]
        assert magic_y_in_range[0] or magic_y_in_range[1], "Y in range cannot be both 0"
        assert magic_y_out_range[0] or magic_y_out_range[1], "Y out range cannot be both 0"

    #  y shift
    y_shift = params["y_shift"]

    num_classes = params["num_classes"]
    size_divisible = params["size_divisible"]

    if params["use_pretrained_encoder?"]:
        preprocessor = sm.get_preprocessing(params["backbone_name"])
    else:
        preprocessor = None

    # turning off compile so that we won't need to import all the modules for the loss function
    trained_model = load_model(model_path, compile=False)

    sample_amount = min(5, int(0.01 * len(img_paths)))
    visualization_paths = set(random.sample(img_paths, sample_amount))
    vs_count = 0

    for in_img_path in img_paths:
        # format the input image properly
        in_img = helper.preprocess_input(
            img_path=in_img_path,
            input_normalization=input_normal, x_in_range=magic_x_in_range, x_out_range=magic_x_out_range,
            pre_processor=preprocessor,
            size_divisible=size_divisible
        )
        pred = trained_model.predict(in_img)
        formatted_pred = helper.output_processor(pred, num_classes)
        helper.save_image_wrapper(os.path.join(pred_dir, os.path.basename(in_img_path)), formatted_pred)
        if in_img_path in visualization_paths:
            raw_in = helper.read_image_wrapper(in_img_path)
            fig, axs = plt.subplots(1, 2)
            fig.set_size_inches(8, 4)
            axs[0].imshow(raw_in, vmin=-0.3, vmax=0.9, cmap='jet')
            axs[1].imshow(formatted_pred, vmin=0, vmax=max(1, num_classes - 1), interpolation='none')
            plt.tight_layout()
            vs_count += 1
            plt.savefig(os.path.join(fig_dir, "random_visualization_{}".format(vs_count)))
            plt.clf()


if __name__ == "__main__":
    main()
