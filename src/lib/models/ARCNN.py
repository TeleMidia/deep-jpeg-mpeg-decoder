import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, PReLU
from tensorflow.keras.activations import relu

class ARCNN(Model):
    def __init__(self):
        super(ARCNN, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        
        self.layer_1 = Conv2D(64, 9, activation=None, strides=1, padding='same', kernel_initializer=initializer)
       
        self.layer_2 = Conv2D(32, 7, activation=None, strides=1, padding='same', kernel_initializer=initializer)
        self.layer_3 = Conv2D(16, 1, activation=None, strides=1, padding='same', kernel_initializer=initializer)
        self.layer_4 = Conv2D(3, 5, activation=None, strides=1, padding='same', kernel_initializer=initializer)
        
   

    def call(self, input):
        
        x = self.layer_1(input)
        x = PReLU()(x)
        x = self.layer_2(x)
        x = PReLU()(x)
        x = self.layer_3(x)
        x = PReLU()(x)
        x = self.layer_4(x)
      
        return x  