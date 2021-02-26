import tensorflow as tf
from . import layers

class YoloV3(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes

        self.layer_stack_1 = [
            layers.YOLOConv( (1, 1, 512) ),
            layers.YOLOConv( (3, 3, 1024) ),
            layers.YOLOConv( (1, 1, 512) ),
            layers.YOLOConv( (3, 3, 1024) ),
            layers.YOLOConv( (1, 1, 512) ),
        ]

        self.lbbox_layer_stack = [
            layers.YOLOConv( (3, 3, 1024) ),
            layers.YOLOConv( (1, 1, 3*(self.num_classes + 5)), activation=None, batchNormalization=False, name="lbbox" ),
        ]

        self.layer_stack_2 = [
            layers.YOLOConv( (1, 1, 512) ),
            layers.YOLOConv( (3, 3, 1024) ),
            layers.YOLOConv( (1, 1, 512) ),
            layers.YOLOConv( (3, 3, 1024) ),
            layers.YOLOConv( (1, 1, 512) ),
        ]

        self.mbbox_layer_stack = [
            layers.YOLOConv( (3, 3, 1024) ),
            layers.YOLOConv( (1, 1, 3*(self.num_classes + 5)), activation=None, batchNormalization=False, name="mbbox" ),
        ]

        self.sbbox_layer_stack = [
            layers.YOLOConv( (1, 1, 512) ),
            layers.YOLOConv( (3, 3, 1024) ),
            layers.YOLOConv( (1, 1, 512) ),
            layers.YOLOConv( (3, 3, 1024) ),
            layers.YOLOConv( (1, 1, 512) ),
            layers.YOLOConv( (3, 3, 1024) ),
            layers.YOLOConv( (1, 1, 3*(self.num_classes + 5)), activation=None, batchNormalization=False, name="sbbox"),
        ]

    def call(self, scale_1, scale_2, scale_3, training=False) :
        x = scale_3

        for layer in self.layer_stack_1:
            x = layer(x)

        lbbox_output = x
        
        for layer in self.lbbox_layer_stack:
            lbbox_output = layer(lbbox_output)

        x = layers.Upsample()(x)

        x = tf.concat( [scale_2, x], axis=-1 )


        for layer in self.layer_stack_2:
            x = layer(x)

        mbbox_output = x

        for layer in self.mbbox_layer_stack:
            mbbox_output = layer(mbbox_output)

        x = layers.Upsample()(x)

        sbbox_output = tf.concat( [scale_1, x], axis=-1 )

        for layer in self.sbbox_layer_stack:
            sbbox_output = layer(sbbox_output)

        return sbbox_output, mbbox_output, lbbox_output
        