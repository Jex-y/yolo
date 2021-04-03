import numpy as np
import cv2
import tensorflow as tf

def image_preprocess(image, bboxes, train_input_size):
    """Rescales images to the correct size whilst also adjusting the bounding boxes accordingly. 

    Args:
        image (np.ndarray): The image to be resized.
        bboxes (np.ndarray): The bounding boxes related to the image. 

    Returns:
        np.ndarray: The adjusted image.
        np.ndarray: The adjusted bboxes.
    """
    target_height = target_width = train_input_size
    height,  width, _  = image.shape

    scale = min(target_width/width, target_height/height)
    new_height, new_width  = int(scale * height), int(scale * width)
    image_resized = cv2.resize(image, (new_height, new_width))

    image_paded = np.full(shape=[target_height, target_width, 3], fill_value=128.0)
    pad_height, pad_width = (target_height-new_height) // 2, (target_width - new_width) // 2
    image_paded[pad_width:new_width+pad_width, pad_height:new_height+pad_height, :] = image_resized
    image_paded = image_paded / 256.
    # 256 used as it is an exact power of 2 so only the exponent is changed, useful for storing in half precision

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + pad_width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + pad_height
    return image_paded, bboxes

@tf.function
def bbox_iou(bboxes1, bboxes2):
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou

@tf.function
def bbox_giou(bboxes1, bboxes2):
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

    return giou

@tf.function
def decode(conv_output, scale, output_size, num_classes, strides, xyscale, anchors):
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

    return pred_xywh, pred_conf, pred_prob

@tf.function
def decode_train(conv_output, scale, output_size, num_classes, strides, xyscale, anchors):
    pred_xywh, pred_conf, pred_prob = decode(conv_output, scale, output_size, num_classes, strides, xyscale, anchors)
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

@tf.function
def decode_predict(conv_output, scale, output_size, num_classes, strides, xyscale, anchors):
    pred_xywh, pred_conf, pred_prob = decode(conv_output, scale, output_size, num_classes, strides, xyscale, anchors)
    batch_size = tf.shape(conv_output)[0]
    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, num_classes))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))
    return pred_xywh, pred_prob

@tf.function
def filter_bboxes(bboxes, scores, score_threshold, image_size):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(bboxes, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    # image_size = tf.cast(image_size, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / image_size
    box_maxes = (box_yx + (box_hw / 2.)) / image_size
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    return boxes, pred_conf

@tf.function
def bbox_nms(model_output, image_size, max_outputs_per_class=32, max_outputs=32, iou_threshold=0.45, score_threshold=0.25):
    pred_bboxes, pred_prob = [],[]
    for i in range(3):
        bbox, prob = decode_predict(model_output, i, image_size)
        pred_bboxes.append(bbox)
        pred_prob.append(prob)

    boxes, conf = filter_bboxes(
        tf.concat(pred_bboxes, axis=-1),
        tf.concat(pred_prob, axis=-1),
        score_threshold,
        image_size
    )

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            conf, (tf.shape(conf)[0], -1, tf.shape(conf)[-1])),
        max_output_size_per_class=max_outputs_per_class,
        max_total_size=max_outputs,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )

    return boxes, scores, classes, valid_detections


def draw_bbox(images, model_output, image_size, num_classes, classes=None, show_label=True):
    import colorsys
    output_images = []

    all_boxes, all_scores, all_classes, all_num_boxes = bbox_nms(
            model_output, 
            image_size
        )

    if not classes:
        classes = [str(i) for i in range(num_classes)]

    height, width, _ = images[0].shape

    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))


    for image, boxes, scores, classes, num_boxes in zip(images, all_boxes, all_scores, all_classes, all_num_boxes):
        for j in range(num_boxes[0]):
            if int(classes[0][j]) < 0 or int(classes[0][j]) > num_classes: continue
            coor = boxes[0][j]
            coor[0] = int(coor[0] * height)
            coor[2] = int(coor[2] * height)
            coor[1] = int(coor[1] * width)
            coor[3] = int(coor[3] * width)

            fontScale = 0.5
            score = scores[0][j]
            class_ind = int(classes[0][j])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (height + width) / 600)
            c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if show_label:
                bbox_mess = '%s: %.2f' % (classes[class_ind], score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

                cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
        output_images.append(image)
    return output_images

    