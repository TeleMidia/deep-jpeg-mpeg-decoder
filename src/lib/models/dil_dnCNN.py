import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.activations import relu

class dil_dnCNN(Model):
    def __init__(self):
        super(dil_dnCNN, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        self.layer_1 = Conv2D(64, 3, dilation_rate=(8, 8), activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.layer_2 = self.dnCNN_layer(64,3,initializer)
        self.layer_3 = self.dnCNN_layer(64,3,initializer)
        self.layer_4 = self.dnCNN_layer(64,3,initializer)
        self.layer_5 = self.dnCNN_layer(64,3,initializer)
        self.layer_6 = self.dnCNN_layer(64,3,initializer)
        self.layer_7 = self.dnCNN_layer(64,3,initializer)
        self.layer_8 = self.dnCNN_layer(64,3,initializer)
        self.layer_9 = self.dnCNN_layer(64,3,initializer)
        self.layer_10 = self.dnCNN_layer(64,3,initializer)
        self.layer_11 = self.dnCNN_layer(64,3,initializer)
        self.layer_12 = self.dnCNN_layer(64,3,initializer)
        self.layer_13 = self.dnCNN_layer(64,3,initializer)
        self.layer_14 = self.dnCNN_layer(64,3,initializer)
        self.layer_15 = self.dnCNN_layer(64,3,initializer)
        self.layer_16 = self.dnCNN_layer(64,3,initializer)
        self.layer_17 = self.dnCNN_layer(64,3,initializer)
        self.layer_18 = self.dnCNN_layer(64,3,initializer)
        self.layer_19 = self.dnCNN_layer(64,3,initializer)
        self.layer_20 = Conv2D(3, 3,dilation_rate=(8, 8), activation=None, strides=1, padding='same', kernel_initializer=initializer)
        
    def dnCNN_layer(self, filters, kernel_size, initializer):
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, dilation_rate=(8, 8), activation=None, strides=1, padding='same', kernel_initializer=initializer))
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
        return tf.keras.layers.Subtract()([input, x])   