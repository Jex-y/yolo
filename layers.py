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

class Mish(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)
        return x * tf.math.tanh(tf.math.softplus(x))

class RouteGroup(layers.Layer):
    def __init__(self, groups, group_id):
        super().__init__()
        self.groups = groups
        self.group_id = group_id

    def call(self, x):
        return tf.split(x, num_or_size_splits=self.groups, axis=-1)[self.group_id]

class Upsample(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear')

class YOLOConv(layers.Layer):
    def __init__(self, filters_shape, downsample=False, batchNormalization=True, activation="leaky"):
        super().__init__()

        self.layer_stack = []
        if downsample:
            self.layer_stack.append(
                layers.ZeroPadding2D(
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
            layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
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
                    layers.LeakyReLU(alpha=0.1)
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

class ResidialBlock(layers.Layer):
    def __init__(self, input_channel, filter_num1, filter_num2, activation):
        super().__init__()
        self.layer_stack = [
            YOLOConv(filters_shape=(1, 1, input_channel, filter_num1), activation=activation),
            YOLOConv(filters_shape=(3, 3, filter_num1,   filter_num2), activation=activation),
        ]

    def call(self, x):
        short_cut = x

        for layer in self.layer_stack:
            x = layer(x)

        return short_cut + x

class Darknet53(layers.Layer):
    def __init__(self):
        super().__init__()

        self.layer_stack_1 = [
            YOLOConv( (3, 3,  3,  32) ),
            YOLOConv( (3, 3, 32,  64), downsample=True ),
        ]

        for i in range(1):
            self.layer_stack_1.append( ResidialBlock( 65, 32, 64 ) )

        self.layer_stack_1.append(YOLOConv( (3, 3,  64, 128), downsample=True ) )

        for i in range(2):
            self.layer_stack_1.append( ResidialBlock( 128, 64, 128 ) )

        self.layer_stack_1.append(YOLOConv( (3, 3,  128, 256), downsample=True ) )

        for i in range(8):
            self.layer_stack_1.append( ResidialBlock( 256, 128, 256 ) )


        self.layer_stack_2 = [
            YOLOConv( (3, 3,  256,  512), downsample=True),
        ]
        
        for i in range(8):
            self.layer_stack_2.append( ResidialBlock( 512, 256, 512 ) )
        

        self.layer_stack_3 = [
            YOLOConv((3, 3, 512, 1024), downsample=True),
        ]

        for i in range(4):
            self.layer_stack_3.append( ResidialBlock( 1024, 512, 1024 ) )
        
    def call(self, x):
        for layer in self.layer_stack_1:
            x = layer(x)

        route_1 = x

        for layer in self.layer_stack_2:
            x = layer(x)

        route_2 = x

        for layer in self.layer_stack_3:
            x = layer(x)

        return route_1, route_2, x

class CSPDarknet53(layers.Layer):
    def __init__(self):
        super().__init__()
    
        self.layer_stack_1 = [
            YOLOConv( (3, 3, 3, 32), activation="mish" ),
            YOLOConv( (3, 3, 32, 54), downsample=True, activation="mish" ),  
        ]

        self.layer_stack_2 = [
            YOLOConv( (1, 1, 64, 64), activation="mish" ),
        ]

        self.layer_stack_2_skip = [
            YOLOConv( (1, 1, 64, 64), activation="mish" ),
        ]

        for i in range(1):
            self.layer_stack_2_skip.append( ResidialBlock( 64, 32, 64, activation="mish" ) )

        self.layer_stack_2_skip.append( YOLOConv( (1, 1, 64, 64), activation="mish" ) )

        self.layer_stack_3 = [
            YOLOConv( (1, 1, 128, 64), activation="mish" ),
            YOLOConv( (3, 3, 64, 128), downsample=True, activation="mish" ),
        ]

        self.layer_stack_4 = [
            YOLOConv( (1, 1, 128, 64), activation="mish" ),
        ]

        for i in range(2):
            self.layer_stack_4.append( ResidialBlock( 64, 64, 64, activation="mish") )
        
        self.layer_stack_4.append( YOLOConv( (1, 1, 64, 64), activation="mish" ) )
        
        self.layer_stack_4_skip = [
            YOLOConv( (1, 1, 128, 64), activation="mish" ),
        ]

        self.layer_stack_5 = [
            YOLOConv( (1, 1, 128, 128), activation="mish" ),
            YOLOConv( (3, 3, 128, 256), downsample=True, activation="mish" ),
        ]

        self.layer_stack_6 = [
            YOLOConv( (1, 1, 256, 128), activation="mish" ),
        ]

        for i in range(8):
            self.layer_stack_6.append( ResidialBlock( 128, 128, 128, activation="mish") )

        self.layer_stack_6.append( YOLOConv( (1, 1, 128, 128), activation="mish" ) )

        self.layer_stack_6_skip = [
            YOLOConv( (1, 1, 256, 128), activation="mish" ),
        ]

        self.layer_stack_7 = [
            YOLOConv( (1, 1, 256, 256), activation="mish" ),
        ]

        self.layer_stack_8 = [
            YOLOConv( (3, 3, 256, 512), downsample=True, activation="mish" ),
        ]

        self.layer_stack_9 = [
            YOLOConv( (1, 1, 512, 256), activation="mish" ),
        ]

        for i in range(8):
            self.layer_stack_9.append( ResidialBlock( 256, 256, 256, activation="mish" ) )

        self.layer_stack_9.append( YOLOConv( (1, 1, 256, 256), activation="mish" ) )

        self.layer_stack_9_skip = [
            YOLOConv( (1, 1, 512, 256), activation="mish" ),
        ]

        self.layer_stack_10 = [
            YOLOConv( (1, 1, 512, 512), activation = "mish" ),
        ]
        
        self.layer_stack_11 = [
            YOLOConv( (3, 3, 512, 1024), downsample=True, activation="mish" ),
        ]

        self.layer_stack_12 = [
            YOLOConv( (1, 1, 1024, 512), activation="mish" ),
        ]

        for i in range(4):
            self.layer_stack_12.append( ResidialBlock( 512, 512, 512, activation="mish") )

        self.layer_stack_12.append( YOLOConv( (1, 1, 512, 512), activation="mish" ) )

        self.layer_stack_12_skip = [
            YOLOConv( (1, 1, 1024, 512), activation="mish" ),
        ]

        self.layer_stack_13 = [
            YOLOConv( (1, 1, 1024, 1024), activation="mish" ),
            YOLOConv( (1, 1, 1024, 512) ),
            YOLOConv( (3, 3, 512, 1024) ),
            YOLOConv( (1, 1, 1024, 512) ),
        ]

        self.layer_stack_14 = [
            YOLOConv( (1, 1, 2048, 512) ),
            YOLOConv( (3, 3, 512, 1024) ),
            YOLOConv( (1, 1, 1024, 512) ),
        ]

    def call(self, x):
        for layer in self.layer_stack_1:
            x = layer(x)
        
        route = x

        for layer in self.layer_stack_2:
            route = layer(route)

        for layer in self.layer_stack_2_skip:
            x = layer(x)

        x = tf.concat( [x, route], axis=-1 )

        for layer in self.layer_stack_3:
            x = layer(x)

        route = x

        for layer in self.layer_stack_4:
            x = layer(x)

        for layer in self.layer_stack_4_skip:
            route = layer(route)

        x = tf.concat( [x, route], axis=-1 )

        for layer in self.layer_stack_5:
            x = layer(x)

        route = x 

        for layer in self.layer_stack_6:
            x = layer(x)

        for layer in self.layer_stack_6_skip:
            route = layer(route)

        x = tf.concat([x, route], axis=-1)
        
        for layer in self.layer_stack_7:
            x = layer(x)

        route_1 = x

        for layer in self.layer_stack_8:
            x = layer(x)

        route = x

        for layer in self.layer_stack_9:
            x = layer(x)

        for layer in self.layer_stack_9_skip:
            route = layer(route)

        x = tf.concat( [x, route], axis=-1 )

        for layer in self.layer_stack_10:
            x = layer(x)

        route_2 = x

        for layer in self.layer_stack_11:
            x = layer(x)

        route = x

        for layer in self.layer_stack_12:
            x = layer(x)
        
        for layer in self.layer_stack_12_skip:
            route = layer(route)

        x = tf.concat( [x, route], axis=-1 )

        for layer in self.layer_stack_13:
            x = layer(x)

        x = tf.concat( [
            tf.nn.max_pool( x, ksize=13, padding='SAME', strides=1 ),
            tf.nn.max_pool(x, ksize=9, padding='SAME', strides=1),
            tf.nn.max_pool(x, ksize=5, padding='SAME', strides=1),
            x
        ], axis=-1 ) 

        for layer in self.layer_stack_14:
            x = layer(x)

        return route_1, route_2, x
