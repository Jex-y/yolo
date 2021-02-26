import numpy as np
import cv2
import tensorflow as tf
import os
from . import config
from . import utils

class WIDERDataset(object):
    MAX_BBOX_PER_SCALE = 128

    def __init__(self, path, image_size=416, batch_size = 1, augment = False):
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
            dataset_file = open(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {path}")
        
        self.working_path = os.path.realpath(dataset_file.readline().strip())

        self.augment_images = augment

        self.batch_size = batch_size
        
        self.num_classes = 1

        self.classes = ["Face"]

        self.examples = self.split_examples(dataset_file)
        dataset_file.close()

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

    def _bytes_feature(self, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _floats_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def to_tfrecord(self, filename, shards, use_sparse=False):
        self.batch_size = 1
        shard_size = len(self) // shards
        shard_index = 0
        examples = []

        print(f"Serialising {shard_size * shards} examples into {shards} shards")

        try:
            for i, (image, targets) in enumerate(self):
                if use_sparse:
                    small_targets = tf.sparse.from_dense(targets[0][0])
                    medium_targets = tf.sparse.from_dense(targets[1][0])
                    large_targets = tf.sparse.from_dense(targets[2][0])

                    feature = {
                        'image'             : self._bytes_feature(tf.io.serialize_tensor(image[0])),
                        'small_targets'     : self._bytes_feature(tf.io.serialize_sparse(small_targets)[0]),
                        'medium_targets'    : self._bytes_feature(tf.io.serialize_sparse(medium_targets)[0]),
                        'large_targets'     : self._bytes_feature(tf.io.serialize_sparse(large_targets)[0]),
                    }
                else:
                    small_targets = targets[0][0]
                    medium_targets = targets[1][0]
                    large_targets = targets[2][0]

                    feature = {
                        'image'             : self._bytes_feature(tf.io.serialize_tensor(image[0])),
                        'small_targets'     : self._floats_feature(small_targets),
                        'medium_targets'    : self._floats_feature(medium_targets),
                        'large_targets'     : self._floats_feature(large_targets),
                    }


                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                examples.append(example_proto.SerializeToString())

                shard_count = (i+1) % shard_size
                progress = (shard_count / shard_size) * 100
                print(f"\rShard {shard_index} {progress:.2f}% Complete \r", end="")

                if shard_count == 0:

                    write_path = filename + f".shard{shard_index+1}of{shards}"

                    print(f"\nWriting shard {shard_index} to {write_path}")

                    with tf.io.TFRecordWriter(write_path) as writer:
                        for example in examples:
                            writer.write(example)

                    print("Completed writing shard")
                    
                    examples = []

                    shard_index += 1

                    if shard_index == shards:
                        break
        except StopIteration:
            pass

        print("All shards created and written")


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
                dtype=np.float16,
            )

            batch_label_sbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[0],
                    self.train_output_sizes[0],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float16,
            )
            batch_label_mbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[1],
                    self.train_output_sizes[1],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float16,
            )
            batch_label_lbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[2],
                    self.train_output_sizes[2],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float16,
            )

            batch_sbboxes = np.zeros(
                (self.batch_size, self.MAX_BBOX_PER_SCALE, 4), dtype=np.float16
            )
            batch_mbboxes = np.zeros(
                (self.batch_size, self.MAX_BBOX_PER_SCALE, 4), dtype=np.float16
            )
            batch_lbboxes = np.zeros(
                (self.batch_size, self.MAX_BBOX_PER_SCALE, 4), dtype=np.float16
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
                batch_smaller_target = [ np.reshape(batch_label_sbbox, (self.batch_size, -1)), np.reshape(batch_sbboxes, (self.batch_size, -1)) ]
                batch_medium_target = [ np.reshape(batch_label_mbbox, (self.batch_size, -1)), np.reshape(batch_mbboxes, (self.batch_size, -1)) ]
                batch_larger_target = [ np.reshape(batch_label_lbbox, (self.batch_size, -1)) , np.reshape(batch_lbboxes, (self.batch_size, -1)) ]
                return (
                    batch_image,
                    (
                        np.concatenate(batch_smaller_target, -1),
                        np.concatenate(batch_medium_target, -1),
                        np.concatenate(batch_larger_target, -1),
                    ),
                )
            else:
                self.batch_count = 0
                np.random.shuffle(self.examples)
                raise StopIteration

    def __len__(self):
        return self.num_batches

    def parse_example(self, example):
        path, num_detections, detections = example

        image = cv2.imread(os.path.join(self.working_path, path))
        if type(image) == type(None):
            raise FileNotFoundError(f"Image {path} not found")

        labels = [0 for i in range(num_detections)]

        height, width, _ = image.shape
        
        bboxes = np.zeros((num_detections,5), dtype=np.int64)

        for i in range(num_detections):
            bbox = [int(x) for x in detections[i].split(" ")[:4]]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            for j in range(4):
                if j%2 == 0:
                    bbox[j] = bbox[j] if bbox[j] < width else width - 1
                else:
                    bbox[j] = bbox[j] if bbox[j] < height else height - 1
            bbox.append(labels[i])
            bboxes[i] = bbox

        if self.augment_images:
            image, bboxes = self.random_horizontal_flip(
                np.copy(image), np.copy(bboxes)
            )
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(
                np.copy(image), np.copy(bboxes)
            )
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return utils.image_preprocess(
            np.copy(image),
            np.copy(bboxes),
            self.input_size,
        )

    def split_examples(self, dataset_file):
        examples = []
        while True:
            path = dataset_file.readline().strip()
            if not path:
                break
            num_detections = int(dataset_file.readline().strip())
            if num_detections > 0:
                detections = [dataset_file.readline().strip() for i in range(num_detections)]
                examples.append((path, num_detections, detections))
            else:
                dataset_file.readline()
        return examples


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