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

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    """
    Find the parameter file from command line argument
    """
    parser = argparse.ArgumentParser(description='find json file')
    parser.add_argument('json_file', type=str, help='Name of the parameter json file')
    args = parser.parse_args()
    param_file = args.json_file

    """
    Loading Training parameter from Json
    """
    params = helper.read_json_file_wrapper(param_file)
    project_name = params["project_name"]
    num_epochs = params["num_epochs"]
    cp_period = params["checkpoint_period"]
    verbose = params["verbose"]
    l_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    input_normal = params["normal_input"]
    output_normal = params["normal_output"]
    augmenting_factor = params["augmentation_factor"]
    base_num_filters = params["base_num_filters"]
    out_goes_to = params["out_goes_to"]
    in_dir = params["input_dir"]
    out_dir = params["output_dir"]
    num_classes = params["num_classes"]
    class_weights = params["class_weights"]
    metrics_used = helper.default_metrics(num_classes=num_classes)
    loss_functions_used = helper.default_loss_functions(num_classes=num_classes)
    data_mode = "generator" if params["use_data_generator?"] else "load_all"
    network_mode = "pre-trained" if params["use_pretrained_encoder?"] else "scratch"

    sample_data = params["sample_data"]
    sample_dim = tuple(params["sample_dim"]) if params["sample_dim"] else None

    """
    Training-related tweaking from JSON file
    """
    user_specified_loss_functions = params["loss_functions"]
    user_specified_loss_ratio = params["loss_ratio"]
    if user_specified_loss_functions:
        loss_functions_used = user_specified_loss_functions
    if not user_specified_loss_ratio:
        user_specified_loss_ratio = [1 for _ in range(len(loss_functions_used))]
    else:
        assert len(user_specified_loss_ratio) == len(loss_functions_used)
    scheduler_name = params["scheduler"]

    """
    Loading Data info from Json
    """
    #  Path to data
    data_path = params["data_path"]
    train_csv_path = os.path.join(data_path, params["train_csv"])
    val_csv_path = os.path.join(data_path, params["val_csv"])
    test_csv_path = os.path.join(data_path, params["test_csv"])
    special_input_path = os.path.join(data_path, in_dir, params["special_input"])

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
    y_onehot = params["y_onehot"]
    if num_classes == 1 and y_onehot:
        raise Exception("Turn off y_onehot if doing regression")

    #  Number of images
    train_count = params["train_count"]
    validation_count = params["val_count"]
    test_count = params["test_count"]

    """
    Prepare output files
    """
    #  Generate training session specific information
    idx = helper.index_based_on_time()
    helper.assert_dir(out_goes_to)
    out_base_dir = os.path.join(out_goes_to, 'train_' + project_name + '_' + idx)
    out_process_dir = os.path.join(out_base_dir, 'evovling_after_each_epoch')
    out_fig_dir = os.path.join(out_base_dir, 'train_vis')
    out_txt = os.path.join(out_base_dir, 'training_log.txt')
    summary_file = os.path.join(out_base_dir, 'trained_summary.json')
    #  Make the root directory for all training output
    helper.assert_dir(out_base_dir)
    helper.assert_dir(out_process_dir)
    helper.assert_dir(out_fig_dir)

    #  Write info to logging file and to console
    params["identifier"] = idx
    params["training instance after augmentation"] = train_count * augmenting_factor
    info_str = helper.write_json_string_wrapper(params)
    helper.file_and_console(info_str, out_txt)

    """
    Create model and verify the information
    """
    if network_mode == "scratch":
        unet = model_scratch.unet(input_size=(None, None, 1), l_rate=l_rate, f=base_num_filters,
                                  num_classes=num_classes)
        preprocess_input = None
    else:
        unet = model_pool.network(
            network_name="U-Net",
            backbone_name=params["backbone_name"],
            input_size=(None, None, 3),
            l_rate=l_rate,
            num_classes=num_classes,
            class_weights=class_weights,
            loss_functions=loss_functions_used,
            loss_ratios=user_specified_loss_ratio
        )
        preprocess_input = sm.get_preprocessing(params["backbone_name"])
    params["model parameters"] = unet.count_params()
    #  Ensure the model's params matched the input file
    helper.file_and_console("Model has parameters " + str(unet.count_params()), out_txt)
    helper.file_and_console("Model has optimizer " + str(unet.optimizer.__class__), out_txt)
    helper.file_and_console("Model has learning rate " + str(K.eval(unet.optimizer.lr)), out_txt)
    helper.file_and_console("Model has class weights " + str(class_weights), out_txt)

    #  Set up callback functions
    filepath = os.path.join(out_base_dir, "{epoch:02d}_checkpoint.h5")
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='min', period=cp_period)
    evaluation_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: helper.evaluate_on_special(
            special_input_path, out_process_dir, unet, str(epoch + 1), num_classes,
            input_normalization=input_normal,
            x_in_range=magic_x_in_range, x_out_range=magic_x_out_range,
            pre_processor=preprocess_input, size_divisible=params["size_divisible"]
        )
    )
    callbacks_list = [checkpoint, evaluation_callback]

    with open(train_csv_path, 'r') as f_csv:
        train_names = f_csv.readlines()

    with open(test_csv_path, 'r') as f_csv:
        test_names = f_csv.readlines()

    with open(val_csv_path, 'r') as f_csv:
        val_names = f_csv.readlines()

    train_augment = (augmenting_factor > 1)

    my_valid_gen = helper.CustomDataGenerator(
        in_dir=os.path.join(data_path, in_dir), out_dir=os.path.join(data_path, out_dir),
        count=validation_count,
        normal_x=input_normal, normal_y=output_normal,
        x_in_range=magic_x_in_range,
        x_out_range=magic_x_out_range,
        y_in_range=magic_y_in_range,
        y_out_range=magic_y_out_range,
        y_shift=y_shift,
        augmentation=False, batch_size=1,
        image_names=val_names,
        network_mode=network_mode,
        preprocessor=preprocess_input,
        size_divisible=params["size_divisible"],
        y_onehot=y_onehot,
        num_classes=num_classes
    )

    if data_mode == 'generator':
        my_train_gen = helper.CustomDataGenerator(
            in_dir=os.path.join(data_path, in_dir), out_dir=os.path.join(data_path, out_dir),
            count=train_count,
            normal_x=input_normal, normal_y=output_normal,
            x_in_range=magic_x_in_range,
            x_out_range=magic_x_out_range,
            y_in_range=magic_y_in_range,
            y_out_range=magic_y_out_range,
            y_shift=y_shift,
            augmentation=train_augment, batch_size=batch_size,
            image_names=train_names,
            random_sampling=sample_data, sample_dim=sample_dim,
            network_mode=network_mode,
            preprocessor=preprocess_input,
            size_divisible=params["size_divisible"],
            y_onehot=y_onehot,
            num_classes=num_classes
        )

        unet.summary()

        helper.visualize_train_data(my_train_gen, out_fig_dir, num_classes)

        def linear_scheduler(epoch):
            start = 50
            second_start = 250
            interval = 20
            if epoch < start:
                return l_rate
            elif epoch > second_start:
                return l_rate * (0.8 ** ((epoch - start) // interval + 1))
            else:
                return l_rate * (0.88 ** ((epoch - start) // interval + 1))

        def cos_scheduler(epoch):
            total = num_epochs * 2
            warmup = 5
            if epoch < warmup:
                return (epoch * l_rate) / warmup
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / total)) * l_rate

        if scheduler_name == "cos":
            callbacks_list.append(tf.keras.callbacks.LearningRateScheduler(cos_scheduler))
        elif scheduler_name == "linear":
            callbacks_list.append(tf.keras.callbacks.LearningRateScheduler(linear_scheduler))
        elif scheduler_name is not None:
            raise Exception(f"Scheduler: {scheduler_name} not recognized")

        train_start = time.perf_counter()
        history = unet.fit_generator(
            generator=my_train_gen,
            steps_per_epoch=train_count // batch_size,
            epochs=num_epochs,
            validation_data=my_valid_gen,
            validation_steps=validation_count,
            callbacks=callbacks_list,
            verbose=verbose
        )
        train_end = time.perf_counter()
    else:

        x_train, y_train, x_val, y_val = helper.load_all_data(
            data_path, in_dir, out_dir,
            input_normal, magic_x_in_range, output_normal, magic_y_in_range,
            train_names, val_names, augmenting_factor,
            out_txt
        )
        print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

        train_start = time.perf_counter()
        history = unet.fit(
            x=x_train, y=y_train, batch_size=batch_size,
            epochs=num_epochs,
            validation_data=(x_val, y_val),
            shuffle=True,
            callbacks=callbacks_list,
            verbose=verbose
        )
        train_end = time.perf_counter()

    time_line = 'Training took ' + helper.fractional_seconds_to_string(train_end - train_start)
    helper.file_and_console(time_line, out_txt)
    params['Training Time'] = helper.fractional_seconds_to_string(train_end - train_start)

    #  Save the model
    unet.save(os.path.join(out_base_dir, 'final_model.h5'))

    #  Evaluate the model on validation data set
    count = 0
    metrics = [[] for _ in metrics_used]
    for metric_name in metrics_used:
        with open(os.path.join(out_base_dir, '%s_on_validation.csv' % metric_name), 'w') as fp:
            fp.write("")
    random_save_idxes = random.sample(range(validation_count), max(1, int(0.1 * validation_count)))
    save_count = 0
    random_save_dir = os.path.join(out_base_dir, "random_val_results")
    helper.assert_dir(random_save_dir)
    test_start = time.perf_counter()
    for ti in range(validation_count):
        test_input, test_output = my_valid_gen.__getitem__(count)
        if test_input.ndim == 3:
            test_input = np.expand_dims(test_input)
        test_pred = unet.predict(test_input)
        # format output
        if y_onehot:
            test_output = np.argmax(test_output, axis=-1)
        # format output and save matrix
        formatted_pred = helper.output_processor(test_pred, num_classes)
        for mi in range(len(metrics_used)):
            metric_name = metrics_used[mi]
            metric_res = helper.metric_wrapper(metric_name, test_output, formatted_pred, num_classes)
            metrics[mi].append(metric_res)
            with open(os.path.join(out_base_dir, '%s_on_validation.csv' % metric_name), 'a') as fp:
                if isinstance(metric_res, np.ndarray):
                    fp.write("%s\n" % (','.join([str(x) for x in metric_res])))
                else:
                    fp.write("%f\n" % metric_res)
        # save some results (for space and transfer time concern)
        if count in random_save_idxes:
            helper.save_image_wrapper(os.path.join(random_save_dir, "val_%d_input.tif" % (save_count + 1)), test_input[0, :, :, 0])
            helper.save_image_wrapper(os.path.join(random_save_dir, "val_%d_output.tif" % (save_count + 1)),
                                      test_output)
            helper.save_image_wrapper(os.path.join(random_save_dir, "val_%d_pred.tif" % (save_count + 1)),
                                      formatted_pred)
            save_count += 1
        count += 1
    test_end = time.perf_counter()
    helper.file_and_console('', out_txt)
    helper.file_and_console(
        'Running model on ' + str(count) + ' val images took ' + helper.fractional_seconds_to_string(
            test_end - test_start), out_txt)
    helper.file_and_console('', out_txt)
    for mi in range(len(metrics_used)):
        metric_name = metrics_used[mi]
        if isinstance(metrics[mi][0], np.ndarray):
            this_avg = np.average(np.array(metrics[mi]), axis=0)
            this_avg = ','.join([str(x) for x in this_avg])
        else:
            this_avg = str(np.mean(metrics[mi]))
        helper.file_and_console('Average %s on val images is %s' % (metric_name, this_avg), out_txt)
        params['Average %s on val' % (metric_name)] = this_avg

    params['Evaluation on val time'] = helper.fractional_seconds_to_string(test_end - test_start)
    helper.write_json_file_wrapper(params, summary_file)

    #  More generic way to get metrics and plot values
    for metrics_key in history.history:
        print(metrics_key)
        if metrics_key.find('val') != -1:
            continue
        val_metrics_key = 'val_' + metrics_key
        #  Plot
        plt.plot(history.history[metrics_key])
        if val_metrics_key in history.history:
            plt.plot(history.history[val_metrics_key])
        plt.title('model ' + metrics_key)
        plt.ylabel(metrics_key)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(os.path.join(out_base_dir, 'plot_' + metrics_key + '.png'))
        plt.clf()
        #  Save the loss function as csv as well
        numpy_train_history = np.array(history.history[metrics_key])
        np.savetxt(os.path.join(out_base_dir, "history_train_" + metrics_key + ".csv"), numpy_train_history,
                   delimiter=",")
        if val_metrics_key in history.history:
            numpy_val_history = np.array(history.history[val_metrics_key])
            np.savetxt(os.path.join(out_base_dir, "history_" + val_metrics_key + ".csv"), numpy_val_history,
                       delimiter=",")


if __name__ == "__main__":
    main()
