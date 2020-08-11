import tensorflow as tf
from . import layers, config, utils
import numpy as np
from time import time_ns as ns
import datetime
import os


class YOLOModel(tf.keras.Model):
    def __init__(self, num_classes, image_size=416, *args, **kwargs):
        super(YOLOModel, self).__init__(*args, **kwargs)

        self.num_classes = num_classes
        self.image_size = image_size
        self.frozen = False
        self.optimizer = tf.keras.optimizers.Adam()

        self.strides = None
        self.xyscale = None
        self.anchors = None
        self.iou_loss_threshold = None
        
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)


    def fit(self, dataset, first_stage_epochs=1, second_stage_epochs=0, val_dataset=None, start_lr = 1e-3, end_lr = 1e-6, checkpoint_dir = "./checkpoints"):
        steps_per_epoch = len(dataset)

        total_steps = 0
        epochs = first_stage_epochs + second_stage_epochs
        training_steps = steps_per_epoch * epochs

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if val_dataset:
            test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
            test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        if checkpoint_dir:
            checkpoint_dir = checkpoint_dir + "/" + current_time

        for epoch in range(epochs):
            tf.print(f"Epoch {epoch+1}/{epochs}")
            start = ns()
            for i, (image, target) in enumerate(dataset):
                giou_loss, conf_loss, prob_loss = self.train_step(image, target)
                total_loss = giou_loss + conf_loss + prob_loss

                lr = start_lr + 0.5 * (start_lr - end_lr) * (
                    (1 + tf.cos((total_steps)/(training_steps)) * np.pi)
                )

                self.optimizer.lr.assign(lr.numpy())
                
                done = f"[{i+1:04d}|{steps_per_epoch:04d}]"
                time_per_step = (ns()-start)/(i+1)
                time_left = self._ns_to_string((steps_per_epoch - i+1) * time_per_step)
                bar = self._progress_bar((i+1)/steps_per_epoch)
                tf.print(
                    f"\r{done} {bar} {time_left} - total loss: %4.2f - giou loss: %4.2f - conf loss: %4.2f - prob loss %4.2f - lr %f {' '*10}" 
                    % ( total_loss, giou_loss, conf_loss, prob_loss, lr),
                    end="")

                total_steps += 1   

            time_taken = self._ns_to_string(ns()-start)
            tf.print(
                f"\r{done} {bar} {time_taken} - total loss: %4.2f - giou loss: %4.2f - conf loss: %4.2f - prob loss %4.2f - lr %f{' '*10}"
                % (total_loss, giou_loss, conf_loss, prob_loss, lr))

            with train_summary_writer.as_default():
                tf.summary.scalar("lr", lr, epoch)
                tf.summary.scalar("loss/total_loss", total_loss, epoch)
                tf.summary.scalar("loss/giou_loss", giou_loss, epoch)
                tf.summary.scalar("loss/conf_loss", conf_loss, epoch)
                tf.summary.scalar("loss/prob_loss", prob_loss, epoch)
            
            if val_dataset:
                start = ns()
                step_per_val_epoch = len(val_dataset)
                for i, (image, target) in enumerate(val_dataset):
                    giou_loss, conf_loss, prob_loss = self.val_step(image, target)
                    total_loss = giou_loss + conf_loss + prob_loss

                    done = f"[{i+1:04d}|{step_per_val_epoch:04d}]"
                    time_per_step = (ns()-start)/(i+1)
                    time_left = self._ns_to_string((step_per_val_epoch - i+1) * time_per_step)
                    bar = self._progress_bar((i+1)/step_per_val_epoch)
                    tf.print(
                    f"\r{done} {bar} {time_left} - val total loss: %4.2f - val giou loss: %4.2f - val conf loss: %4.2f - val prob loss %4.2f {' '*10}" 
                    % ( total_loss, giou_loss, conf_loss, prob_loss),
                    end="")

                time_taken = self._ns_to_string(ns()-start)
                tf.print(
                f"\r{done} {bar} {time_taken} - val total loss: %4.2f - val giou loss: %4.2f - val conf loss: %4.2f - val prob loss %4.2f {' '*10}"
                % (total_loss, giou_loss, conf_loss, prob_loss))

                with test_summary_writer.as_default():
                    tf.summary.scalar("loss/total_loss", total_loss, epoch)
                    tf.summary.scalar("loss/giou_loss", giou_loss, epoch)
                    tf.summary.scalar("loss/conf_loss", conf_loss, epoch)
                    tf.summary.scalar("loss/prob_loss", prob_loss, epoch)

            
            if checkpoint_dir:
                file_name = f"epoch{epoch+1:04d}of{epochs:04d}.ckpt"
                path = os.path.join(checkpoint_dir, file_name)
                self.save_weights(path)
                tf.print(f"Checkpoint saved to {file_name}")

            if epoch+1== first_stage_epochs:
                tf.print("Entering second training stage - Freezing output layers")
                self.freeze(True)

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
                pred_result.append(self._decode_train(raw_result[i], i))
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
            pred_result.append(self._decode_train(raw_result[i], i))
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

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4])) * 10
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4])) * 1
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4])) * 1

        return giou_loss, conf_loss, prob_loss

    @tf.function
    def _decode_train(self, conv_output, scale):
        output_size = self.image_size // self.strides[scale]
        conv_output = tf.reshape(conv_output,
                                (tf.shape(conv_output)[0], output_size, output_size, 3, 5 + self.num_classes))

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


class YOLOv4(YOLOModel):
    def __init__(self, num_classes, image_size=416, *args, **kwargs):
        super(YOLOv4, self).__init__(num_classes, image_size, *args, **kwargs)

        self.xyscale = config.config["yolo"]["xyscale"]
        self.strides = config.config["yolo"]["strides"]
        self.anchors = config.config["yolo"]["anchors"]
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