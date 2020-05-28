import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.activations import relu

class DeeperSRCNN(Model):
    def __init__(self):
        super(DeeperSRCNN, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        
        self.layer_1 = Conv2D(32, 5, activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.layer_2_4 = self.DeeperSRCNN_block(32, 5, initializer)
        self.layer_5_7 = self.DeeperSRCNN_block(32, 5, initializer)
        self.layer_8_10 = self.DeeperSRCNN_block(32, 5, initializer)
        self.layer_11_13 = self.DeeperSRCNN_block(32, 5, initializer)
        self.layer_14_16 = self.DeeperSRCNN_block(32, 5, initializer)
        self.layer_17_19 = self.DeeperSRCNN_block(32, 5, initializer) 
        self.layer_20 = Conv2D(3, 5, activation=None, strides=1, padding='same', kernel_initializer=initializer)
        
    def DeeperSRCNN_block(self, filters, kernel_size, initializer):

        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, activation='relu', strides=1, padding='same', kernel_initializer=initializer))
        result.add(Conv2D(filters, kernel_size, activation='relu', strides=1, padding='same', kernel_initializer=initializer))
        result.add(Conv2D(filters, kernel_size, activation='relu', strides=1, padding='same', kernel_initializer=initializer))
        return result

    def call(self, input):
        
        x = self.layer_1(input)
        x = self.layer_2_4(x)
        
        skip = x
        x = self.layer_5_7(x)
        x = tf.keras.layers.Add()([skip, x])
        
        skip = x  
        x = self.layer_8_10(x)
        x = tf.keras.layers.Add()([skip, x])
        
        skip = x  
        x = self.layer_11_13(x)
        x = tf.keras.layers.Add()([skip, x])

        skip = x  
        x = self.layer_14_16(x)
        x = tf.keras.layers.Add()([skip, x])

        skip = x  
        x = self.layer_17_19(x)
        x = tf.keras.layers.Add()([skip, x])

        x = self.layer_20(x)


        return tf.keras.layers.Add()([input, x])   