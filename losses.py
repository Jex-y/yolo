import tensorflow as tf
from . import utils
import numpy as np

# strides = self.strides[scale]
# num_classes = self.num_classes
# iou_loss_threshold = self.iou_loss_threshold

# @tf.function
def yolo_loss(pred, conv, label, bboxes, input_size, iou_loss_threshold):
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

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < iou_loss_threshold, tf.float32 )

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

# @tf.function
def decode_train(conv_output, scale, output_size, num_classes, strides, xyscale, anchors):
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, num_classes),
                                                                        axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * xyscale[scale]) - 0.5 * (xyscale[scale] - 1) + xy_grid) * \
            strides[scale]
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors[scale])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def get_yolo_loss_at_scales(scales, num_classes, strides, iou_loss_threshold, xyscale, anchors):
    losses = []
    for scale in scales[::-1]:
        # @tf.function
        def loss(target, output):
            output_shape = tf.shape(output)
            batch_size = output_shape[0]
            output_size = output_shape[1]
            input_size = output_size * strides[scale]
            output_shape = (batch_size, output_size, output_size, 3, 5+num_classes)
            target_split = np.prod(output_shape[1:])

            output = tf.reshape(output, output_shape)
            labels = tf.reshape(target[:,:target_split], output_shape)
            bboxes = tf.reshape(target[:,target_split:], (batch_size, 128, 4))

            decoded = decode_train(output, scale, output_size, num_classes, strides, xyscale, anchors)
            giou_loss, conf_loss, prob_loss = yolo_loss(decoded, output, labels, bboxes, input_size, iou_loss_threshold)
            return giou_loss + conf_loss + prob_loss
        
        losses.append(loss)

    return losses



