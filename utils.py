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