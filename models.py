import tensorflow as tf
from tensorflow.keras import models
from . import layers

class YOLOv4(models.Model):
    def __init__(self, NUM_CLASSES):
        super().__init__()

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
            layers.YOLOConv( 1, 1, 256, 128 ),
            layers.YOLOConv( 3, 3, 128, 256 ),
            layers.YOLOConv( 1, 1, 256, 128 ),
            layers.YOLOConv( 3, 3, 128, 256 ),
            layers.YOLOConv( 1, 1, 256, 128 ),
        ]

        self.sbbox_layer_stack = [
            layers.YOLOConv( (3, 3, 128, 256) ),
            layers.YOLOConv( (1, 1, 256, 3 * (NUM_CLASSES + 5)), activation=None, batchNormalization=False ),
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
            layers.YOLOConv( (1, 1, 512, 3 * (NUM_CLASSES + 5) ), activation=None, batchNormalization=False ),
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
            layers.YOLOConv( (1, 1, 1024, 3 * (NUM_CLASSES + 5)), activation=None, batchNormalization=False )
        ]

    def call(self, x):
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

        s_mbbox = x
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


def setup():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    trainset = Dataset(FLAGS, is_training=True)
    testset = Dataset(FLAGS, is_training=False)
    logdir = "./data/log"
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()

    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)


    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(len(freeze_layers)):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                    "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),
                                                            giou_loss, conf_loss,
                                                            prob_loss, total_loss))
        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()

def test_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(len(freeze_layers)):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                    "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                            prob_loss, total_loss))
