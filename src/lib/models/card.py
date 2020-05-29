import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Conv2DTranspose, Layer
from tensorflow.keras.activations import relu

def one_8_8(shape, dtype=None):
    ker = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                ker[i][j][k][i*8+j*3+k]=1
    return ker
     

class card():
    def __init__(self):
        self.wb_count = 0
        #super(card, self).__init__()
        input = tf.keras.Input(shape=(96,96,3), name="inputs")

        x = self.dct2c_layer(name = 'DCT2C')(input)
        shortcut = x

        for i in range(20):            
            y = self.warb()(x)
            x = tf.keras.layers.Add()([y, x]) 

        x = tf.keras.layers.Add()([shortcut, x])

        output = self.c2dct_layer(name = 'C2DCT')(x)

        self.model = tf.keras.Model(inputs=[input], outputs=[output])

    def warb(self):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        warb_layer = tf.keras.Sequential(name=f"WideActivationResidualBlock_{self.wb_count}")
        warb_layer.add(Conv2D(384, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer))
        warb_layer.add(Conv2D(192, 3, activation=None, strides=1, padding='same', kernel_initializer=initializer))
        self.wb_count += 1
        return warb_layer

    def dct2c_layer(self, name):
        dct2c = tf.keras.Sequential(name = name)
        #dct2c.add((tf.keras.Input(shape=(96,96,3))))
        dct2c.add(Conv2D(192,8,strides=8, kernel_initializer=one_8_8, trainable = False))

        return dct2c

    def c2dct_layer(self, name):
        c2dtc = tf.keras.Sequential(name = name)
        #c2dtc.add((tf.keras.Input(shape=(12,12,64))))
        c2dtc.add(Conv2DTranspose(3,8,strides=8, kernel_initializer=one_8_8, trainable = False))

        return c2dtc
