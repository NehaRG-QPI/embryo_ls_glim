import tensorflow as tf

"""
cbr_block
Creates the convolution-batch normalization-residual connection block
@:param num_featuers: the number of filters after convolution
@:param in_layer: input layer
Reference:
1.) https://stackoverflow.com/questions/49045843/why-is-relu-applied-after-residual-connection-in-resnet
2.) https://stackoverflow.com/questions/49045843/why-is-relu-applied-after-residual-connection-in-resnet
"""


def cbr_block(num_features, in_layer):
    # CBR
    x = tf.keras.layers.Conv2D(num_features, 3, padding='same', kernel_initializer='he_normal')(in_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # CBR
    x = tf.keras.layers.Conv2D(num_features, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # Residual connection
    x0 = tf.keras.layers.Conv2D(num_features, 1, padding='same', kernel_initializer='he_normal')(in_layer)
    x0 = tf.keras.layers.BatchNormalization()(x0)
    x = tf.keras.layers.Add()([x, x0])
    x = tf.keras.layers.Activation('relu')(x)

    return x


"""
unet
Creates the U-Net architecture with convolution-batch noramlization-residual as building block
@:param input_size: the size of input tensor (H, W, C)
@:param l_rate: learning rate of the model
@:param f: the number of filters in the first layer of U-Net. In the origianl paper, this number was 64.
@:param num_classes: the number of channels in the output channel.
                        1 -- sigmoid activation, regression problem (float to float)
                        2 -- sigmoid activation, binary classification problem (float to [0,1])
                        3 -- softmax activation, multi-class classification problem (float to [0,1,2])
                        ...
"""


def unet(input_size=(None, None, 1), l_rate=1e-4, f=32, num_classes=2):
    inputs = tf.keras.Input(input_size)

    # -------
    # Encoder
    # -------

    conv1 = cbr_block(1 * f, inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = cbr_block(2 * f, pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = cbr_block(4 * f, pool2)
    drop0 = tf.keras.layers.Dropout(0.5)(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop0)

    conv4 = cbr_block(8 * f, pool3)
    drop1 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop1)

    # -------
    # Bottle neck
    # -------

    mid1 = cbr_block(16 * f, pool4)
    mid2 = cbr_block(16 * f, mid1)
    drop2 = tf.keras.layers.Dropout(0.5)(mid2)

    # -------
    # decoder
    # -------
    up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(drop2)
    merge6 = tf.keras.layers.concatenate([conv4, up6], axis=3)
    conv6 = cbr_block(8 * f, merge6)

    up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = cbr_block(4 * f, merge7)

    up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = cbr_block(2 * f, merge8)

    up9 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = cbr_block(1 * f, merge9)

    if num_classes == 1 or num_classes == 2:
        conv10 = tf.keras.layers.Conv2D(1, 3, padding='same', kernel_initializer='he_normal')(conv9)
        conv11 = tf.keras.layers.Add()([conv10, inputs])
        decoded = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv11)
        model = tf.keras.Model(inputs=inputs, outputs=decoded)
        if num_classes == 1:
            model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=l_rate))
        else:
            model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=l_rate))
    else:
        decoded = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(conv9)
        model = tf.keras.Model(inputs=inputs, outputs=decoded)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=l_rate))
        # sparse_categorical_crossentropy works with integer label
        # categorical_crossentropy works with one-hot encoded label

    return model
