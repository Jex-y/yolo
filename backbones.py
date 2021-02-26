import tensorflow as tf
from .layers import *

# Backbones are the first stage in the YOLO model. They are used to extract features at multiple scales.

class Darknet53(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layer_stack_1 = [
            YOLOConv( (3, 3,  32) ),
            YOLOConv( (3, 3,  64), downsample=True ),
        ]

        for i in range(1):
            self.layer_stack_1.append( ResidialBlock( 32, 64 ) )

        self.layer_stack_1.append(YOLOConv( (3, 3, 128), downsample=True ) )

        for i in range(2):
            self.layer_stack_1.append( ResidialBlock( 64, 128 ) )

        self.layer_stack_1.append(YOLOConv( (3, 3, 256), downsample=True ) )

        for i in range(8):
            self.layer_stack_1.append( ResidialBlock( 128, 256 ) )


        self.layer_stack_2 = [
            YOLOConv( (3, 3,  512), downsample=True),
        ]
        
        for i in range(8):
            self.layer_stack_2.append( ResidialBlock( 256, 512 ) )
        

        self.layer_stack_3 = [
            YOLOConv((3, 3, 1024), downsample=True),
        ]

        for i in range(4):
            self.layer_stack_3.append( ResidialBlock( 512, 1024 ) )
        
    def call(self, x):
        for layer in self.layer_stack_1:
            x = layer(x)

        scale_1 = x

        for layer in self.layer_stack_2:
            x = layer(x)

        scale_2 = x

        for layer in self.layer_stack_3:
            x = layer(x)

        scale_3 = x

        return scale_1, scale_2, scale_3

class CSPDarknet53(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
        self.layer_stack_1 = [
            YOLOConv( (3, 3, 32), activation="mish" ),
            YOLOConv( (3, 3, 54), downsample=True, activation="mish" ),  
        ]

        self.layer_stack_2 = [
            YOLOConv( (1, 1, 64), activation="mish" ),
        ]

        self.layer_stack_2_skip = [
            YOLOConv( (1, 1, 64), activation="mish" ),
        ]

        for i in range(1):
            self.layer_stack_2_skip.append( ResidialBlock( 64, 32, 64, activation="mish" ) )

        self.layer_stack_2_skip.append( YOLOConv( (1, 1, 64), activation="mish" ) )

        self.layer_stack_3 = [
            YOLOConv( (1, 1, 64), activation="mish" ),
            YOLOConv( (3, 3, 128), downsample=True, activation="mish" ),
        ]

        self.layer_stack_4 = [
            YOLOConv( (1, 1, 64), activation="mish" ),
        ]

        for i in range(2):
            self.layer_stack_4.append( ResidialBlock( 64, 64, 64, activation="mish") )
        
        self.layer_stack_4.append( YOLOConv( (1, 1, 64), activation="mish" ) )
        
        self.layer_stack_4_skip = [
            YOLOConv( (1, 1, 64), activation="mish" ),
        ]

        self.layer_stack_5 = [
            YOLOConv( (1, 1, 128), activation="mish" ),
            YOLOConv( (3, 3, 256), downsample=True, activation="mish" ),
        ]

        self.layer_stack_6 = [
            YOLOConv( (1, 1, 128), activation="mish" ),
        ]

        for i in range(8):
            self.layer_stack_6.append( ResidialBlock( 128, 128, 128, activation="mish") )

        self.layer_stack_6.append( YOLOConv( (1, 1, 128), activation="mish" ) )

        self.layer_stack_6_skip = [
            YOLOConv( (1, 1, 128), activation="mish" ),
        ]

        self.layer_stack_7 = [
            YOLOConv( (1, 1, 256), activation="mish" ),
        ]

        self.layer_stack_8 = [
            YOLOConv( (3, 3, 512), downsample=True, activation="mish" ),
        ]

        self.layer_stack_9 = [
            YOLOConv( (1, 1, 256), activation="mish" ),
        ]

        for i in range(8):
            self.layer_stack_9.append( ResidialBlock( 256, 256, 256, activation="mish" ) )

        self.layer_stack_9.append( YOLOConv( (1, 1, 256), activation="mish" ) )

        self.layer_stack_9_skip = [
            YOLOConv( (1, 1, 256), activation="mish" ),
        ]

        self.layer_stack_10 = [
            YOLOConv( (1, 1, 512), activation = "mish" ),
        ]
        
        self.layer_stack_11 = [
            YOLOConv( (3, 3, 1024), downsample=True, activation="mish" ),
        ]

        self.layer_stack_12 = [
            YOLOConv( (1, 1, 512), activation="mish" ),
        ]

        for i in range(4):
            self.layer_stack_12.append( ResidialBlock( 512, 512, 512, activation="mish") )

        self.layer_stack_12.append( YOLOConv( (1, 1, 512), activation="mish" ) )

        self.layer_stack_12_skip = [
            YOLOConv( (1, 1, 512), activation="mish" ),
        ]

        self.layer_stack_13 = [
            YOLOConv( (1, 1, 1024), activation="mish" ),
            YOLOConv( (1, 1, 512) ),
            YOLOConv( (3, 3, 1024) ),
            YOLOConv( (1, 1, 512) ),
        ]

        self.layer_stack_14 = [
            YOLOConv( (1, 1, 512) ),
            YOLOConv( (3, 3, 1024) ),
            YOLOConv( (1, 1, 512) ),
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