import tensorflow as tf
from . import layers, config, utils
import numpy as np
from time import time_ns as ns
import datetime
import os


class YOLOModel(tf.keras.Model):
    def __init__(self, num_classes, *args, **kwargs):
        super(YOLOModel, self).__init__(*args, **kwargs)

        self.num_classes = num_classes
        self.frozen = False
        self.optimizer = tf.keras.optimizers.Adam()

        self.strides = None
        self.xyscale = None
        self.anchors = None
        self.iou_loss_threshold = None
        
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def fit(self, dataset, epochs=1, first_stage_epochs=None, second_stage_epochs=None, val_dataset=None, start_lr = 1e-3, end_lr = 1e-6, checkpoint_dir = "./checkpoints", steps_per_metric_update=1000):
        steps_per_epoch = len(dataset)

        total_steps = 0
        if first_stage_epochs and second_stage_epochs:
            epochs = first_stage_epochs + second_stage_epochs
            self.freeze(True)

        training_steps = steps_per_epoch * epochs

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if val_dataset:
            test_log_dir = 'logs/' + current_time + '/test'
            test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        if checkpoint_dir:
            checkpoint_dir = checkpoint_dir + "/" + current_time

        for epoch in range(epochs):
            tf.print(f"Epoch {epoch+1}/{epochs}")
            start = ns()
            epoch_history = {
                "total_loss":[],
                "giou_loss":[],
                "conf_loss":[],
                "prob_loss":[],
                "val_total_loss":[],
                "val_giou_loss":[],
                "val_conf_loss":[],
                "val_prob_loss":[],
            }
            for i, (image, target) in enumerate(dataset):
                giou_loss, conf_loss, prob_loss = self.train_step(image, target)
                total_loss = giou_loss + conf_loss + prob_loss

                epoch_history["total_loss"].append(total_loss)
                epoch_history["giou_loss"].append(giou_loss)
                epoch_history["conf_loss"].append(conf_loss)
                epoch_history["prob_loss"].append(prob_loss)

                lr = start_lr + 0.5 * (start_lr - end_lr) * (
                    (1 + tf.cos((total_steps)/(training_steps)) * np.pi)
                )

                self.optimizer.lr.assign(lr.numpy())
                
                done = f"[{i+1:05d}|{steps_per_epoch:05d}]"
                time_per_step = (ns()-start)/(i+1)
                time_left = self._ns_to_string((steps_per_epoch - i+1) * time_per_step)
                bar = self._progress_bar((i+1)/steps_per_epoch)

                tf.print(
                    f"\r%s %s %s - total loss: %4.2f - giou loss: %4.2f - conf loss: %4.2f - prob loss %4.2f - lr %f {' '*10}" 
                    % ( done, 
                        bar,
                        time_left,
                        total_loss, 
                        giou_loss, 
                        conf_loss, 
                        prob_loss, 
                        lr,
                    ),end="")

                total_steps += 1   

                if i % steps_per_metric_update == 0:
                    total_loss = sum(epoch_history["total_loss"])/i
                    giou_loss = sum(epoch_history["giou_loss"])/i
                    conf_loss = sum(epoch_history["conf_loss"])/i
                    prob_loss = sum(epoch_history["prob_loss"])/i

                    with train_summary_writer.as_default():
                        tf.summary.scalar("loss/total_loss", total_loss, total_steps)
                        tf.summary.scalar("loss/giou_loss", giou_loss, total_steps)
                        tf.summary.scalar("loss/conf_loss", conf_loss, total_steps)
                        tf.summary.scalar("loss/prob_loss", prob_loss, total_steps)
                        tf.summary.scalar("lr", lr, total_steps)

            time_taken = self._ns_to_string(ns()-start)

            total_loss = sum(epoch_history["total_loss"])/steps_per_epoch
            giou_loss = sum(epoch_history["giou_loss"])/steps_per_epoch
            conf_loss = sum(epoch_history["conf_loss"])/steps_per_epoch
            prob_loss = sum(epoch_history["prob_loss"])/steps_per_epoch

            tf.print(
                f"\r%s %s %s - total loss: %4.2f - giou loss: %4.2f - conf loss: %4.2f - prob loss %4.2f - lr %f{' '*10}"
                % ( done, 
                    bar, 
                    time_taken,
                    total_loss,
                    giou_loss,
                    conf_loss,
                    prob_loss,
                    lr,
                ))

            with train_summary_writer.as_default():
                tf.summary.scalar("loss/total_loss", total_loss, total_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, total_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, total_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, total_steps)
                tf.summary.scalar("lr", lr, total_steps)
            
            if val_dataset:
                start = ns()
                step_per_val_epoch = len(val_dataset)
                for i, (image, target) in enumerate(val_dataset):
                    giou_loss, conf_loss, prob_loss = self.val_step(image, target)
                    total_loss = giou_loss + conf_loss + prob_loss

                    epoch_history["val_total_loss"].append(total_loss)
                    epoch_history["val_giou_loss"].append(giou_loss)
                    epoch_history["val_conf_loss"].append(conf_loss)
                    epoch_history["val_prob_loss"].append(prob_loss)

                    done = f"[{i+1:05d}|{step_per_val_epoch:05d}]"
                    time_per_step = (ns()-start)/(i+1)
                    time_left = self._ns_to_string((step_per_val_epoch - i+1) * time_per_step)
                    bar = self._progress_bar((i+1)/step_per_val_epoch)

                    tf.print(
                    f"\r%s %s %s - val total loss: %4.2f - val giou loss: %4.2f - val conf loss: %4.2f - val prob loss %4.2f {' '*10}" 
                    % ( done,
                        bar,
                        time_left, 
                        total_loss, 
                        giou_loss, 
                        conf_loss, 
                        prob_loss
                    ),end="")

                    if i % steps_per_metric_update == 0:
                        total_loss = sum(epoch_history["val_total_loss"])/i
                        giou_loss = sum(epoch_history["val_giou_loss"])/i
                        conf_loss = sum(epoch_history["val_conf_loss"])/i
                        prob_loss = sum(epoch_history["val_prob_loss"])/i

                        example_bboxes = self.draw_bboxes(image[0])

                        with test_summary_writer.as_default():
                            steps = epoch*step_per_val_epoch + i
                            tf.summary.scalar("loss/total_loss", total_loss, steps)
                            tf.summary.scalar("loss/giou_loss", giou_loss, steps)
                            tf.summary.scalar("loss/conf_loss", conf_loss, steps)
                            tf.summary.scalar("loss/prob_loss", prob_loss, steps)
                            tf.summary.image("example_bboxes", example_bboxes, step=steps)

                time_taken = self._ns_to_string(ns()-start)

                total_loss = sum(epoch_history["val_total_loss"])/step_per_val_epoch
                giou_loss = sum(epoch_history["val_giou_loss"])/step_per_val_epoch
                conf_loss = sum(epoch_history["val_conf_loss"])/step_per_val_epoch
                prob_loss = sum(epoch_history["val_prob_loss"])/step_per_val_epoch

                tf.print(
                f"\r%s %s %s - val total loss: %4.2f - val giou loss: %4.2f - val conf loss: %4.2f - val prob loss %4.2f {' '*10}"
                % ( done, 
                    bar, 
                    time_taken,
                    total_loss,
                    giou_loss,
                    conf_loss,
                    prob_loss,
                ))

                example_bboxes = self.draw_bboxes(image[0])

                with test_summary_writer.as_default():
                    tf.summary.scalar("loss/total_loss", total_loss, epoch)
                    tf.summary.scalar("loss/giou_loss", giou_loss, epoch)
                    tf.summary.scalar("loss/conf_loss", conf_loss, epoch)
                    tf.summary.scalar("loss/prob_loss", prob_loss, epoch)
                    tf.summary.image("example_bboxes", example_bboxes, step=epoch)
            
            if checkpoint_dir:
                file_name = f"epoch{epoch+1:04d}of{epochs:04d}.ckpt"
                path = os.path.join(checkpoint_dir, file_name)
                self.save_weights(path)
                tf.print(f"Checkpoint saved to {file_name}")

            if first_stage_epochs and second_stage_epochs:
                if epoch+1== first_stage_epochs:
                    tf.print("Entering second training stage - Unfreezing output layers")
                    self.freeze(False)

    @tf.function
    def __call__(self, x, **kwargs):
        assert x.shape[1] == x.shape[2]
        return super(YOLOModel, self).__call__(x, **kwargs)

    def freeze(self, freeze=True):
        for layer in self.freeze_layers:
            layer.trainable = not freeze

    def _ns_to_string(self, time):
        small_units = ("ns", "µs", "ms")
        time = float(time)
        for unit in small_units:
            if time >= 1000:
                time /= 1000
            else:
                output = f"[{time:3.0f} {unit}]"
                break
        else:
            secs = time%60
            mins = (time//60)
            hours = (mins//60)
            mins = mins % 60
            output = f"[{hours:02.0f}:{mins:02.0f}:{secs:02.0f}]"
        return output

    def _progress_bar(
        self, progress, length=32, 
        start="[", end="]", head=">", fill="=", empty = " "):
        len_fill = int(progress * length)
        len_empty = length - len_fill
        if progress >= 1.0:
            head = ""
        elif head != "":
            len_fill -= 1
        return start + (fill * len_fill) + head + (empty * len_empty) + end

    @tf.function
    def train_step(self, image, target):
        with tf.GradientTape() as tape:
            raw_result = self(image, training=True)
            pred_result = []
            for i in range(3):
                pred_result.append(raw_result[i])
                pred_result.append(self._decode_train(raw_result[i], i, image.shape[1]))
            giou_loss = conf_loss = prob_loss = 0

            for scale in range(3): # Len freeze layers, maybe freeze layers are the output ones for each scale??
                conv, pred = pred_result[scale * 2], pred_result[scale * 2 + 1]
                losses = self._compute_loss(pred, conv, target[scale][0], target[scale][1], scale)
                giou_loss += losses[0]
                conf_loss += losses[1]
                prob_loss += losses[2]

            loss = giou_loss + conf_loss + prob_loss

            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return giou_loss, conf_loss, prob_loss

    @tf.function
    def val_step(self, image, target):
        raw_result = self(image, training=True)
        pred_result = []
        for i in range(3):
            pred_result.append(raw_result[i])
            pred_result.append(self._decode_train(raw_result[i], i, image.shape[1]))
        giou_loss = conf_loss = prob_loss = 0

        for scale in range(3): # Len freeze layers, maybe freeze layers are the output ones for each scale??
            conv, pred = pred_result[scale * 2], pred_result[scale * 2 + 1]
            losses = self._compute_loss(pred, conv, target[scale][0], target[scale][1], scale)
            giou_loss += losses[0]
            conf_loss += losses[1]
            prob_loss += losses[2]

        return giou_loss, conf_loss, prob_loss
        
    @tf.function
    def _compute_loss(self, pred, conv, label, bboxes, scale=0):
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
        input_size  = self.strides[scale] * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + self.num_classes))

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

        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < self.iou_loss_threshold, tf.float32 )

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

    @tf.function
    def _decode_train(self, conv_output, scale, image_size):
        output_size = image_size // self.strides[scale]
        batch_size = tf.shape(conv_output)[0]
        conv_output = tf.reshape(conv_output,
                                (batch_size, output_size, output_size, 3, 5 + self.num_classes))

        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, self.num_classes),
                                                                            axis=-1)

        xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = ((tf.sigmoid(conv_raw_dxdy) * self.xyscale[scale]) - 0.5 * (self.xyscale[scale] - 1) + xy_grid) * \
                self.strides[scale]
        pred_wh = (tf.exp(conv_raw_dwdh) * self.anchors[scale])
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    @tf.function
    def _decode(self, conv_output, scale, image_size):
        batch_size = tf.shape(conv_output)[0]

        decode_output = self._decode_train(conv_output, scale, image_size)
        pred_xywh, pred_conf, pred_prob = decode_output[:, :, :, :, 0:4], decode_output[:, :, :, :, 4], decode_output[:, :, :, :, 5] 
        pred_prob = pred_conf * pred_prob
        pred_prob = tf.reshape(pred_prob, (batch_size, -1, self.num_classes))
        pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))
        return pred_xywh, pred_prob

    @tf.function
    def _filter_bboxes(self, bboxes, scores, score_threshold, image_size):
        scores_max = tf.math.reduce_max(scores, axis=-1)

        mask = scores_max >= score_threshold
        class_boxes = tf.boolean_mask(bboxes, mask)
        pred_conf = tf.boolean_mask(scores, mask)
        class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
        pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

        box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

        image_size = tf.cast(image_size, dtype=tf.float32)

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
        return (boxes, pred_conf)

    @tf.function
    def predict_bboxes_nms(self, image_batch, max_outputs_per_class=32, max_outputs=32, iou_threshold=0.45, score_threshold=0.25):
        image_size = image_batch.shape[1]
        raw_pred = self(image_batch)
        pred_bboxes, pred_prob = [],[]
        for i in range(3):
            bbox, prob = self._decode(raw_pred[i], i, image_size)
            pred_bboxes.append(bbox)
            pred_prob.append(prob)

        boxes, conf = self._filter_bboxes(
            tf.concat(pred_bboxes, axis=1),
            tf.concat(pred_prob, axis=1),
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


    def draw_bboxes(self, image_batch, clasess=[], show_label=True):
        import colorsys, cv2
        if len(image_batch.shape) == 3:
            image_batch = np.expand_dims(image_batch, 0)

        boxes, scores, classes, num_boxes = [ x.numpy() for x in self.predict_bboxes_nms(
            image_batch
        )]

        if len(classes) == 0:
            classes = list(self.dataset.classes_dict.values())

        images = []
        for image in image_batch:

            height, width, _ = image.shape
            hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

            boxes, scores, classes, num_boxes
            for i in range(num_boxes[0]):
                if int(classes[0][i]) < 0 or int(classes[0][i]) > self.num_classes: continue
                coor = boxes[0][i]
                coor[0] = int(coor[0] * height)
                coor[2] = int(coor[2] * height)
                coor[1] = int(coor[1] * width)
                coor[3] = int(coor[3] * width)

                fontScale = 0.5
                score = scores[0][i]
                class_ind = int(classes[0][i])
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
            images.append(image)    
        return images if len(image) > 1 else images[0]

class YOLOv4(YOLOModel):
    def __init__(self, num_classes, image_size=416, *args, **kwargs):
        super(YOLOv4, self).__init__(num_classes, image_size, *args, **kwargs)

        self.xyscale = np.array(config.config["yolo"]["xyscale"])
        self.strides = np.array(config.config["yolo"]["strides"])
        self.anchors = np.array(config.config["yolo"]["anchors"])
        self.iou_loss_threshold = config.config["yolo"]["IOU_loss_threshold"]


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

        self.freeze_layers = [
            self.lbbox_layer_stack[-1],
            self.mbbox_layer_stack[-1],
            self.lbbox_layer_stack[-1],
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