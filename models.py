import tensorflow as tf
from tensorflow.keras import models
from . import layers

class YOLOv4(models.Model):
    def __init__(self, NUM_CLASSES):
        super().__init__()

        self.darknet = layers.CSPDarknet53()

        self.layer_stack_1 = [
            layers.YOLOConv( (1, 1, 512, 256) ),
            layers.Upsample(),
        ]

        self.layer_stack_1_skip = [
            layers.YOLOConv( (1, 1, 512, 256) ),
        ]

        self.layer_stack_2 = [
            layers.YOLOConv( (1, 1, 512, 256) ),
            layers.YOLOConv( (3, 3, 256, 512) ),
            layers.YOLOConv( (1, 1, 512, 256) ),
            layers.YOLOConv( (3, 3, 256, 512) ),
            layers.YOLOConv( (1, 1, 512, 256) ),
        ]

        self.layer_stack_3 = [
            layers.YOLOConv( (1, 1, 256, 128) ),
            layers.Upsample(),
        ]

        self.layer_stack_3_skip = [
            layers.YOLOConv( (1, 1, 256, 128) ),
        ]

        self.layer_stack_4 = [
            layers.YOLOConv( (1, 1, 256, 128) ),
            layers.YOLOConv( (3, 3, 128, 256) ),
            layers.YOLOConv( (1, 1, 256, 128) ),
            layers.YOLOConv( (3, 3, 128, 256) ),
            layers.YOLOConv( (1, 1, 256, 128) ),
        ]

        self.sbbox_layer_stack = [
            layers.YOLOConv( (3, 3, 128, 256) ),
            layers.YOLOConv( (1, 1, 256, 3 * (NUM_CLASSES + 5)), activation=None, batchNormalization=False ),
        ]

        self.sbbox_layer_stack_skip = [
            layers.YOLOConv( (3, 3, 128, 256), downsample=True )
        ]

        self.layer_stack_5 = [
            layers.YOLOConv( (1, 1, 512, 256) ),
            layers.YOLOConv( (3, 3, 256, 512) ),
            layers.YOLOConv( (1, 1, 512, 256) ),
            layers.YOLOConv( (3, 3, 256, 512) ),
            layers.YOLOConv( (1, 1, 512, 256) ),
        ]

        self.mbbox_layer_stack = [
            layers.YOLOConv( (3, 3, 256, 512) ),
            layers.YOLOConv( (1, 1, 512, 3 * (NUM_CLASSES + 5) ), activation=None, batchNormalization=False ),
        ]

        self.mbbox_layer_stack_skip = [
            layers.YOLOConv( (3, 3, 256, 512), downsample=True ),
        ]

        self.layer_stack_6 = [
            layers.YOLOConv( (1, 1, 1024, 512) ),
            layers.YOLOConv( (3, 3, 512, 1024) ),
            layers.YOLOConv( (1, 1, 1024, 512) ),
            layers.YOLOConv( (3, 3, 512, 1024) ),
            layers.YOLOConv( (1, 1, 1024, 512) ),
        ]

        self.lbbox_layer_stack = [
            layers.YOLOConv( (3, 3, 512, 1024) ),
            layers.YOLOConv( (1, 1, 1024, 3 * (NUM_CLASSES + 5)), activation=None, batchNormalization=False )
        ]

    def call(self, x):
        assert x.shape[1] == x.shape[2]
        
        route_1, route_2, x = self.darknet(x)

        route = x 

        for layer in self.layer_stack_1:
            x = layer(x)

        for layer in self.layer_stack_1_skip:
            route_2 = layer(route_2)

        x = tf.concat( [route_2, x], axis=-1 )

        for layer in self.layer_stack_2:
            x = layer(x)

        route_2 = x

        for layer in self.layer_stack_3:
            x = layer(x)

        for layer in self.layer_stack_3_skip:
            route_1 = layer(route_1)

        x = tf.concat( [route_1, x], axis=-1 )
        
        for layer in self.layer_stack_4:
            x = layer(x)

        route_1 = x

        for layer in self.sbbox_layer_stack:
            x = layer(x)

        x_sbbox = x
        x = route_1

        for layer in self.sbbox_layer_stack_skip:
            x = layer(x)
        
        x = tf.concat( [x, route_2], axis=-1 )

        for layer in self.layer_stack_5:
            x = layer(x)

        route_2 = x

        for layer in self.mbbox_layer_stack:
            x = layer(x)

        x_mbbox = x
        x = route_2

        for layer in self.mbbox_layer_stack_skip:
            x = layer(x)

        x = tf.concat( [x, route], axis=-1 )

        for layer in self.layer_stack_6:
            x = layer(x)

        for layer in self.lbbox_layer_stack:
            x = layer(x)

        x_lbbox = x

        return [x_sbbox, x_mbbox, x_lbbox]