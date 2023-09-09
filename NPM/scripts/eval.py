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


def main():
    train_name = r"trained_model"
    model_name = r"final_model.h5"
    set_name = 'test'
    save_raw_pred = True

    input_cmap = None
    gt_cmap = None
    pred_cmap = gt_cmap
    input_display_range = None
    gt_display_range = None
    pred_display_range = None 

    # Training Directories
    train_res_base = r"sessions"
    train_res_dir = os.path.join(train_res_base, train_name)
    model_path = os.path.join(train_res_dir, model_name)
    assert os.path.isdir(train_res_dir), "Training directory {} does not exist!".format(train_res_dir)
    assert os.path.isfile(model_path), "Model path {} does not exist!".format(model_path)
    # Save evaluation results
    eval_dir = helper.assert_dir(os.path.join(train_res_dir, "eval_{}".format(model_name.split('.')[0])))
    # Data information
    param_file = os.path.join(train_res_dir, 'trained_summary.json')
    params = helper.read_json_file_wrapper(param_file)
    #  Path to data
    data_path = params["data_path"]
    in_dir = params["input_dir"]
    gt_dir = params["output_dir"]
    input_normal = params["normal_input"]
    output_normal = params["normal_output"]
    train_csv_path = os.path.join(data_path, params["train_csv"])
    val_csv_path = os.path.join(data_path, params["val_csv"])
    test_csv_path = os.path.join(data_path, params["test_csv"])
    data_in_dir = os.path.join(data_path, in_dir)
    data_out_dir = os.path.join(data_path, gt_dir)
    assert os.path.isdir(data_in_dir), "Data input directory {} does not exist!".format(in_dir)
    assert os.path.isdir(data_out_dir), "Data ground truth directory {} does not exist!".format(gt_dir)
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

    #  Number of images
    train_count = params["train_count"]
    validation_count = params["val_count"]
    test_count = params["test_count"]

    num_classes = params["num_classes"]
    size_divisible = params["size_divisible"]

    if num_classes == 1:
        metrics_needed = ['PSNR', 'Pearson','MSSSIM']
    else:
        metrics_needed = ['F1']

    if params["use_pretrained_encoder?"]:
        preprocessor = sm.get_preprocessing(params["backbone_name"])
    else:
        preprocessor = None

    set_pred_dir = helper.assert_dir(os.path.join(eval_dir, "pred_{}".format(set_name)))
    helper.assert_dir(os.path.join(set_pred_dir, "Input"))
    helper.assert_dir(os.path.join(set_pred_dir, "Gt"))
    helper.assert_dir(os.path.join(set_pred_dir, "Pred"))
    if save_raw_pred:
        helper.assert_dir(os.path.join(set_pred_dir, 'Pred_raw'))
    out_txt = os.path.join(eval_dir, "eval_{}_res.txt".format(set_name))
    val_names = open(os.path.join(data_path, params["{}_csv".format(set_name)])).read().strip().split('\n')
    print(len(val_names))

    # turning off compile so that we won't need to import all the modules for the loss function
    trained_model = load_model(model_path, compile=False)

    metrics = [[] for _ in metrics_needed]
    for metric_name in metrics_needed:
        with open(os.path.join(eval_dir, '{}_on_{}.csv'.format(metric_name, set_name)), 'w') as fp:
            fp.write("")

    visualize_lines = random.sample(val_names, 10)

    # Set display range
    if input_display_range is None:
        input_display_range = (-0.3, 0.9)
    if gt_display_range is None:
        if magic_y_in_range is not None:
            gt_display_range = magic_y_in_range
        else:
            gt_display_range = (0, 1) if num_classes <= 2 else (0, num_classes - 1)
    if pred_display_range is None:
        pred_display_range = (0, 1) if num_classes <= 2 else (0, num_classes - 1)

    if input_cmap is None:
        input_cmap = 'jet'
    if gt_cmap is None:
        gt_cmap = 'gray' if num_classes == 1 else 'viridis'
    if pred_cmap is None:
        pred_cmap = gt_cmap

    vs_count = 0
    for line in val_names:
        in_name, gt_name = line.split(',')
        raw_in = helper.read_image_wrapper(os.path.join(data_in_dir, in_name))
        in_img = helper.preprocess_input(
            img_path=os.path.join(data_in_dir, in_name),
            input_normalization=input_normal, x_in_range=magic_x_in_range, x_out_range=magic_x_out_range,
            pre_processor=preprocessor,
            size_divisible=size_divisible
        )
        raw_out = helper.read_image_wrapper(os.path.join(data_out_dir, gt_name))
        out_img = helper.preprocess_output(
            img_path=os.path.join(data_out_dir, gt_name),
            output_normalization=output_normal,
            y_in_range=magic_y_in_range,
            y_out_range=magic_y_out_range,
            size_divisible=size_divisible
        )

        pred = trained_model.predict(in_img)
        formatted_pred = helper.output_processor(pred, num_classes)
        for mi in range(len(metrics_needed)):
            metric_name = metrics_needed[mi]
            metric_res = helper.metric_wrapper(metric_name, out_img, formatted_pred, num_classes)
            metrics[mi].append(metric_res)
            with open(os.path.join(eval_dir, '{}_on_{}.csv'.format(metric_name, set_name)), 'a') as fp:
                if isinstance(metric_res, np.ndarray):
                    fp.write("%s,%s\n" % (in_name, ','.join([str(x) for x in metric_res])))
                else:
                    fp.write("%s,%f\n" % (in_name, metric_res))

        helper.save_image_wrapper(os.path.join(set_pred_dir, 'Input', 'x_' + in_name), raw_in)
        helper.save_image_wrapper(os.path.join(set_pred_dir, 'Gt', 'y_' + in_name), raw_out)
        helper.save_image_wrapper(os.path.join(set_pred_dir, 'Pred', 'p_' + in_name), formatted_pred)
        if save_raw_pred:
            helper.save_image_wrapper(os.path.join(set_pred_dir, "Pred_raw", 'rp_' + in_name), pred.astype(np.float))

        if line in visualize_lines:
            plt.clf()
            fig, axs = plt.subplots(1, 3)
            fig.set_size_inches(12, 4)

            axs[0].imshow(raw_in, vmin=input_display_range[0], vmax=input_display_range[1], cmap=input_cmap)
            axs[1].imshow(raw_out, vmin=gt_display_range[0], vmax=gt_display_range[1], interpolation='none', cmap=gt_cmap)
            axs[2].imshow(formatted_pred, vmin=pred_display_range[0], vmax=pred_display_range[1], interpolation='none', cmap=pred_cmap)

            plt.tight_layout()
            vs_count += 1
            plt.savefig(os.path.join(eval_dir, "random_visualizaiton_{}_{}".format(set_name, vs_count)))

    for mi in range(len(metrics_needed)):
        metric_name = metrics_needed[mi]
        if isinstance(metrics[mi][0], np.ndarray):
            this_avg = np.average(np.array(metrics[mi]), axis=0)
            this_avg = ','.join([str(x) for x in this_avg])
        else:
            this_avg = str(np.mean(metrics[mi]))
        helper.file_and_console('Average %s on %s images is %s' % (metric_name, set_name, this_avg), out_txt)


if __name__ == "__main__":
    main()
