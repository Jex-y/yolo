import tensorflow as tf
from . import layers, config, utils
import numpy as np

def compute_loss(pred, conv, label, bboxes, num_classes, scale=0):
    """Computes the loss of a YOLO model. 

    Args:
        pred (np.ndarray): [description]
        conv (np.ndarray): [description]
        label (np.ndarray): [description]
        bboxes (np.ndarray): [description]
        num_classes (int): The number of classes being used.
        scale (int, optional): The scale to calculate the loss for. Defaults to 0.

    Returns:
        (float, float, float): giou_loss, conf_loss, prob_loss
    """
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = config["yolo"]["strides"][scale] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + num_classes))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(utils.bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    iou = utils.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss

class YOLOv4(tf.keras.models.Model):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        self.num_classes = num_classes

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
            layers.YOLOConv( (1, 1, 256, 3 * (self.num_classes + 5)), activation=None, batchNormalization=False ),
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
            layers.YOLOConv( (1, 1, 512, 3 * (self.num_classes + 5) ), activation=None, batchNormalization=False ),
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
            layers.YOLOConv( (1, 1, 1024, 3 * (self.num_classes + 5)), activation=None, batchNormalization=False )
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