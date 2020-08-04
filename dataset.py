import tensorflow as tf
import numpy as np
from . import config

class YOLODataset(tf.data.TFRecordDataset):
    MAX_BBOX_PER_SCALE=128
    def __init__(self, filenames, num_classes, image_size=416, augment=False, compression_type=None, buffer_size=None, num_parallel_reads=None):
        super().__init__(filenames, compression_type, buffer_size, num_parallel_reads)
        self.num_classes = num_classes
        self.image_size = image_size
        self.augment = augment

    def _parse(self, feature_dict={
                            "image/height": tf.io.FixedLenFeature([], tf.int64),
                            "image/width": tf.io.FixedLenFeature([], tf.int64),
                            "image/filename": tf.io.FixedLenFeature([], tf.string),
                            "image/source_id": tf.io.FixedLenFeature([], tf.string),
                            "image/encoded": tf.io.FixedLenFeature([], tf.string),
                            "image/format": tf.io.FixedLenFeature([], tf.string),
                            "image/object/bbox/xmin": tf.io.RaggedFeature(tf.float32),
                            "image/object/bbox/xmax": tf.io.RaggedFeature(tf.float32),
                            "image/object/bbox/ymin": tf.io.RaggedFeature(tf.float32),
                            "image/object/bbox/ymax": tf.io.RaggedFeature(tf.float32),
                            "image/object/class/text": tf.io.FixedLenFeature([], tf.string),
                            "image/object/class/label": tf.io.FixedLenFeature([], tf.string),
                            }):
        return self.map(
            lambda example: tf.io.parse_single_example(example, feature_dict))

    def preprocess(self, num_parallel_calls=4):
        self._parse()
        return self.map(
            lambda example: self._preprocess(example),
            num_parallel_calls=num_parallel_calls)

    def _augment(self, image, bboxes):
        return image, bboxes


    def _preprocess(self, example):
        
        image = tf.io.decode_jpeg(example["image/encoded"], channels=3)
        image = tf.image.resize(image, (self.image_size, self.image_size))
        image = tf.cast(image, tf.float32)/255

        
        anchor_per_scale = config["yolo"]["anchor_per_scale"]
        height = example["image/height"]
        width = example["image/width"]
        bboxes = (np.array([
            example['image/object/bbox/xmin'],
            example['image/object/bbox/ymin'],
            example['image/object/bbox/xmax'],
            example['image/object/bbox/ymax'],
            1
            ]) * np.array([width, height, width, height, 1])).astype(np.int64)

        bboxes[4] = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])

        if augment:
            # Multiply dataset length by some amount.
            self._augment(np.copy(image), np.copy(bboxes))

        (
            label_sbbox,
            label_mbbox,
            label_lbbox,
            sbboxes,
            mbboxes,
            lbboxes,
        ) = self._preprocess_true_boxes(bboxes)

        smaller_target = label_sbbox, sbboxes
        medium_target = label_mbbox, mbboxes
        large_target = label_lbbox, label_lbbox
        return image, (
            smaller_target, 
            medium_target, 
            large_target
            )



    def _preprocess_true_boxes(self, bboxes):
        anchor_per_scale = config["yolo"]["anchor_per_scale"]
        strides = config["yolo"]["strides"]
        anchors = config["yolo"]["anchors"]

        label = [
            np.zeros(
                (
                    self.image_size/strides[i],
                    self.image_size/strides[i],
                    anchor_per_scale,
                    5 + self.num_classes,
                )
            )
            for i in range(3)
        ]
        bboxes_xywh = [np.zeros((self.MAX_BBOX_PER_SCALE, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes
            )
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )
            bbox_xywh_scaled = (
                1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
            )

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = utils.bbox_iou(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        np.int32
                    )

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.MAX_BBOX_PER_SCALE)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchor_per_scale)
                best_anchor = int(best_anchor_ind % anchor_per_scale)
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(
                    bbox_count[best_detect] % self.MAX_BBOX_PER_SCALE
                )
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

