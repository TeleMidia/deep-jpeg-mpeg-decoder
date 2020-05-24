import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Conv2DTranspose
from tensorflow.keras.activations import relu

def one_8_8(shape, dtype=None):
        print(shape)
        ker = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    ker[i][j][k][i*8+j]=1
        return ker

class dct_dnCNN(Model):
    def __init__(self):
        super(dct_dnCNN, self).__init__()

        self.layer_0 = self.dct2c_layer()
        self.layer_1 = self.dnCNN_layer(64,3)
        self.layer_2 = self.dnCNN_layer(64,3)
        self.layer_3 = self.dnCNN_layer(64,3)
        self.layer_4 = self.dnCNN_layer(64,3)
        self.layer_5 = self.dnCNN_layer(64,3)
        self.layer_6 = self.dnCNN_layer(64,3)
        self.layer_7 = self.dnCNN_layer(64,3)
        self.layer_8 = self.dnCNN_layer(64,3)
        self.layer_9 = self.dnCNN_layer(64,3)
        self.layer_10 = self.dnCNN_layer(64,3)
        self.layer_11 = self.dnCNN_layer(64,3)
        self.layer_12 = self.dnCNN_layer(64,3)
        self.layer_13 = self.dnCNN_layer(64,3)
        self.layer_14 = self.dnCNN_layer(64,3)
        self.layer_15 = self.dnCNN_layer(64,3)
        self.layer_16 = self.dnCNN_layer(64,3)
        self.layer_17 = self.dnCNN_layer(64,3)
        self.layer_18 = self.dnCNN_layer(64,3)
        self.layer_19 = self.dnCNN_layer(64,3)
        self.layer_20 = self.dnCNN_layer(64,3)
        #initializer = tf.keras.initializers.GlorotNormal(seed=0)
        #self.last_conv = Conv2D(3, 3, strides=1, padding='same', kernel_initializer=initializer)
        self.last_layer = self.c2dct_layer()
       
        
    def dnCNN_layer(self, filters, kernel_size):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        return result

    def dct2c_layer(self):
        dct2c = tf.keras.Sequential(name = "DCTtoChannel")
        dct2c.add((tf.keras.Input(shape=(96,96,3))))
        dct2c.add(Conv2D(64,8,strides=8, kernel_initializer=one_8_8, trainable = True))

        return dct2c

    def c2dct_layer(self):
        c2dtc = tf.keras.Sequential(name = "Channel2DCT")
        c2dtc.add((tf.keras.Input(shape=(12,12,64))))
        c2dtc.add(Conv2DTranspose(3,8,strides=8, kernel_initializer=one_8_8, trainable = True))

        return c2dtc

    def call(self, input):
        x = self.layer_0(input)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        x = self.layer_10(x)
        x = self.layer_11(x)
        x = self.layer_12(x)
        x = self.layer_13(x)  
        x = self.layer_14(x) 
        x = self.layer_15(x)
        x = self.layer_16(x)
        x = self.layer_17(x)
        x = self.layer_18(x)
        x = self.layer_19(x)
        x = self.layer_20(x)                                                                      
        x = self.last_layer(x)
        return tf.keras.layers.Subtract()([input, x])   