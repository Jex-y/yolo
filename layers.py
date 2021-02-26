import tensorflow as tf
from tensorflow.keras import layers

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

class Mish(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))

class RouteGroup(tf.keras.layers.Layer):
    def __init__(self, groups, group_id):
        super().__init__()
        self.groups = groups
        self.group_id = group_id

    def call(self, x):
        return tf.split(x, num_or_size_splits=self.groups, axis=-1)[self.group_id]

class Upsample(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        shape = tf.shape(x)
        return tf.image.resize(x, (shape[1] * 2, shape[2] * 2), method='bilinear')

class YOLOConv(tf.keras.layers.Layer):
    def __init__(self, filters_shape, downsample=False, batchNormalization=False, activation="mish", **kwargs):
        super().__init__(**kwargs)

        self.layer_stack = []
        if downsample:
            self.layer_stack.append(
                tf.keras.layers.ZeroPadding2D(
                    (
                        (1, 0), 
                        (1, 0))
                    ))
            padding = "valid"
            strides = 2
        else:
            strides = 1
            padding = "same"

        self.layer_stack.append(
            tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[:2], strides=strides, padding=padding,
                          use_bias=not batchNormalization, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                          bias_initializer=tf.constant_initializer(0.))
            )

        if batchNormalization:
            self.layer_stack.append(
                BatchNormalization()
            )
        
        if activation:
            if callable(activation):
                self.layer_stack.append(activation)
            elif activation == "leaky":
                self.layer_stack.append(
                    tf.keras.layers.LeakyReLU(alpha=0.1)
                )
            elif activation == "mish":
                self.layer_stack.append(
                    Mish()
                )
            else:
                raise ValueError("Activation should either be a function or one of \"leaky\" or \"mish\".")


    def call(self, x):
        for layer in self.layer_stack:
            x = layer(x)

        return x

class ResidialBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num1, filter_num2, activation="mish"):
        super().__init__()
        self.layer_stack = [
            YOLOConv(filters_shape=(1, 1, filter_num1), activation=activation),
            YOLOConv(filters_shape=(3, 3, filter_num2), activation=activation),
        ]

    def call(self, x):
        short_cut = x

        for layer in self.layer_stack:
            x = layer(x)

        return short_cut + x

