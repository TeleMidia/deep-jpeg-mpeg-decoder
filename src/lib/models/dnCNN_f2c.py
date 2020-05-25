import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.activations import relu

class dnCNN_f2c(Model):
    def __init__(self, first_kernel):
        super(dnCNN_f2c, self).__init__()

        self.layer_1 = self.dnCNN_layer(16,first_kernel)
        self.layer_2 = self.dnCNN_layer(16,3)
        self.layer_3 = self.dnCNN_layer(16,3)
        self.layer_4 = self.dnCNN_layer(16,3)
        self.layer_5 = self.dnCNN_layer(16,3)
        self.layer_6 = self.dnCNN_layer(16,3)
        self.layer_7 = self.dnCNN_layer(16,3)
        self.layer_8 = self.dnCNN_layer(16,3)
        self.layer_9 = self.dnCNN_layer(16,3)
        self.layer_10 = self.dnCNN_layer(16,3)
        self.layer_11 = self.dnCNN_layer(16,3)
        self.layer_12 = self.dnCNN_layer(16,3)
        self.layer_13 = self.dnCNN_layer(16,3)
        self.layer_14 = self.dnCNN_layer(16,3)
        self.layer_15 = self.dnCNN_layer(16,3)
        self.layer_16 = self.dnCNN_layer(16,3)
        self.layer_17 = self.dnCNN_layer(16,3)
        self.layer_18 = self.dnCNN_layer(16,3)
        self.layer_19 = self.dnCNN_layer(16,3)
        self.layer_20 = self.dnCNN_layer(16,3)
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        self.last_conv = Conv2D(192, 1, strides=1, padding='same', kernel_initializer=initializer)
        
    def dnCNN_layer(self, filters, kernel_size):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        return result

    def call(self, input):
        
        x = self.layer_1(input)
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
        x = self.last_conv(x)
        return tf.keras.layers.Subtract()([input, x])   