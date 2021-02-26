import tensorflow as tf

class ImageNet(tf.keras.layers.Layer):
    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)

        self.layer_stack = [
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(classes),
            tf.keras.activations.softmax(),
        ]

    def call(self, x):
        for layer in self.layer_stack:
            x = layer(x)
        
        return x
