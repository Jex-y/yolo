import numpy as np
import yaml
import cv2
import tensorflow as tf
import os
from . import config
from . import utils

class Dataset(object):
    MAX_BBOX_PER_SCALE = 128

    def __init__(self, path, train_or_test, image_size=416, batch_size = 1, augment = False):
        """Created a dataset to be used to train or test a YOLO model

        Args:
            path (string): The path of the dataset file.
            train_or_test (string): One of train or test
            image_size (int, optional): The size that images will be resized to. Defaults to 416.
            batch_size (int, optional): The number of examples per batch. Defaults to 2.
            augment (bool, optional): Apply random augmentations to the images each batch.

        Raises:
            FileNotFoundError: Raised when dataset file cannot be found. 
        """
        try:
            with open(path, "r") as f:
                dataset_dict = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {path}")

        self.working_path = os.path.dirname(os.path.realpath(path))
        
        self.train_or_test = train_or_test.lower()
        assert train_or_test in ("train", "test")

        self.augment_images = augment

        self.batch_size = batch_size
        
        self.num_classes = int(dataset_dict["classes"]["num_classes"])

        self.class_dict = dataset_dict["classes"]["class_names"]

        self.examples = dataset_dict["examples"][train_or_test]
        self.num_examples = len(self.examples)
        self.num_batches = int(np.ceil(self.num_examples / self.batch_size))
        self.batch_count = 0

        self.input_size = image_size
        self.strides = np.array(config["yolo"]["strides"])
        self.train_output_sizes = self.input_size // self.strides

        self.anchor_per_scale = config["yolo"]["anchor_per_scale"]
        self.anchors = np.array(config["yolo"]["anchors"])

    def __iter__(self):
        return self

    def __next__(self):
        """Return the next batch of the datset until it runs out of data
        """
        with tf.device("/cpu:0"):
            batch_image = np.zeros(
                (
                    self.batch_size,
                    self.input_size,
                    self.input_size,
                    3,
                ),
                dtype=np.float32,
            )

            batch_label_sbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[0],
                    self.train_output_sizes[0],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )
            batch_label_mbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[1],
                    self.train_output_sizes[1],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )
            batch_label_lbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[2],
                    self.train_output_sizes[2],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )

            batch_sbboxes = np.zeros(
                (self.batch_size, self.MAX_BBOX_PER_SCALE, 4), dtype=np.float32
            )
            batch_mbboxes = np.zeros(
                (self.batch_size, self.MAX_BBOX_PER_SCALE, 4), dtype=np.float32
            )
            batch_lbboxes = np.zeros(
                (self.batch_size, self.MAX_BBOX_PER_SCALE, 4), dtype=np.float32
            )


            num = 0
            if self.batch_count < self.num_batches:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_examples:
                        index -= self.num_examples
                    image, bboxes = self.parse_example(self.examples[index])
                    (
                        label_sbbox,
                        label_mbbox,
                        label_lbbox,
                        sbboxes,
                        mbboxes,
                        lbboxes,
                    ) = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target = batch_label_mbbox, batch_mbboxes
                batch_larger_target = batch_label_lbbox, batch_lbboxes

                return (
                    batch_image,
                    (
                        batch_smaller_target,
                        batch_medium_target,
                        batch_larger_target,
                    ),
                )
            else:
                self.batch_count = 0
                np.random.shuffle(self.examples)
                raise StopIteration

    def __len__(self):
        return self.num_batches

    def parse_example(self, example):
        image = cv2.imread(os.path.join(self.working_path, example["image/filename"]))
        if type(image) == type(None):
            print(image)
            raise FileNotFoundError(f"Image {example['image/filename']} not found")

        height, width = example["image/height"], example["image/width"]

        labels = example["image/object/class/label"]

        bboxes = np.zeros((len(labels),5), dtype=np.int64)

        bboxes[:, 0] = [int(x*width) for x in example["image/object/bbox/xmin"]]
        bboxes[:, 1] = [int(y*height) for y in example["image/object/bbox/ymin"]]
        bboxes[:, 2] = [int(x*width) for x in example["image/object/bbox/xmax"]]
        bboxes[:, 3] = [int(y*height) for y in example["image/object/bbox/ymax"]]
        bboxes[:, 4] = [int(x) for x in labels]

        if self.augment_images:
            image, bboxes = self.random_horizontal_flip(
                np.copy(image), np.copy(bboxes)
            )
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(
                np.copy(image), np.copy(bboxes)
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = utils.image_preprocess(
            np.copy(image),
            np.copy(bboxes),
            self.input_size,
        )
        return image, bboxes

    def random_horizontal_flip(self, image, bboxes):
        """Randomly flips images and adjusts bboxes. 

        Args:
            image (np.ndarray): The image to be resized.
            bboxes (np.ndarray): The bounding boxes related to the image. 

        Returns:
            np.ndarray: The adjusted image.
            np.ndarray: The adjusted bboxes.
        """
        if np.random.rand() < 0.5:
            _, width, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = width - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        """Randomly translated images without cropping out any bboxes and adjusts bboxes. 

        Args:
            image (np.ndarray): The image to be resized.
            bboxes (np.ndarray): The bounding boxes related to the image. 

        Returns:
            np.ndarray: The adjusted image.
            np.ndarray: The adjusted bboxes.
        """
        if np.random.rand() < 0.5:
            height, width, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_left = max_bbox[0]
            max_up = max_bbox[1]
            max_right = width - max_bbox[2]
            max_down = height - max_bbox[3]

            crop_xmin = max(
                0, int(max_bbox[0] - np.random.uniform(0, max_left))
            )
            crop_ymin = max(
                0, int(max_bbox[1] - np.random.uniform(0, max_up))
            )
            crop_xmax = max(
                width, int(max_bbox[2] + np.random.uniform(0, max_right))
            )
            crop_ymax = max(
                height, int(max_bbox[3] + np.random.uniform(0, max_down))
            )

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes
        
    def random_translate(self, image, bboxes):
        """Randomly translated images without removing any bboxes and adjusts bboxes.

        Args:
            image (np.ndarray): The image to be resized.
            bboxes (np.ndarray): The bounding boxes related to the image. 

        Returns:
            np.ndarray: The adjusted image.
            np.ndarray: The adjusted bboxes.
        """
        if np.random.rand() < 0.5:
            height, width, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_left = max_bbox[0]
            max_up = max_bbox[1]
            max_right = width - max_bbox[2]
            max_down = height - max_bbox[3]

            translation_x = np.random.uniform(-(max_left - 1), (max_right - 1))
            translation_y = np.random.uniform(-(max_up - 1), (max_down - 1))

            M = np.array([[1, 0, translation_x], [0, 1, translation_y]])
            image = cv2.warpAffine(image, M, (width, height))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + translation_x
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + translation_y

        return image, bboxes

    def preprocess_true_boxes(self, bboxes):
        """Preprocesses bounding boxes from the dataset so that they can be used to train on. 

        Args:
            bboxes (np.nparray): Unprocessed bboxes. 

        Returns:
            np.ndarray * 6: label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
        """
        label = [
            np.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                    self.anchor_per_scale,
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
                1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            )

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
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
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

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
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
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