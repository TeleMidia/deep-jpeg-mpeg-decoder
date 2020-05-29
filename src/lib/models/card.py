import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Layer
from tensorflow.keras.activations import relu


class card():
    def __init__(self):
        self.wb_count = 0
        #super(card, self).__init__()
        input = tf.keras.Input(shape=(12,12,192), name="inputs")

        x = input
        shortcut = x

        for i in range(20):            
            y = self.warb()(x)
            x = tf.keras.layers.Add()([y, x]) 

        x = tf.keras.layers.Add()([shortcut, x])

        output = x

        self.model = tf.keras.Model(inputs=[input], outputs=[output])

    def warb(self):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        warb_layer = tf.keras.Sequential(name=f"WideActivationResidualBlock_{self.wb_count}")
        warb_layer.add(Conv2D(384, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer))
        warb_layer.add(Conv2D(192, 3, activation=None, strides=1, padding='same', kernel_initializer=initializer))
        self.wb_count += 1
        return warb_layer


