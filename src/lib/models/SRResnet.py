import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU

class SRResnet(Model):
    def __init__(self):
        super(SRResnet, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        
        self.layer_1 = Conv2D(64, 3, activation='relu', strides=1, padding='same', kernel_initializer=initializer)
        self.layer_2_3 = self.SRResnet_block(64, 3, initializer)
        self.layer_4_5 = self.SRResnet_block(64, 3, initializer)
        self.layer_6_7 = self.SRResnet_block(64, 3, initializer)
        self.layer_8_9 = self.SRResnet_block(64, 3, initializer)
        self.layer_10_11 = self.SRResnet_block(64, 3, initializer)
        self.layer_12_13 = self.SRResnet_block(64, 3, initializer)
        self.layer_14_15 = self.SRResnet_block(64, 3, initializer)
        self.layer_16_17 = self.SRResnet_block(64, 3, initializer)
        self.layer_18_19 = self.SRResnet_block(64, 3, initializer)
        self.layer_20 = Conv2D(3, 3, activation=None, strides=1, padding='same', kernel_initializer=initializer)
        
    def SRResnet_block(self, filters, kernel_size, initializer):

        result = tf.keras.Sequential() 
    
        result.add(Conv2D(filters, kernel_size, activation=None, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())

        result.add(Conv2D(filters, kernel_size, activation=None, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
       
        return result

    def call(self, input):
        
        x = self.layer_1(input)
        
        skip = x
        x = self.layer_2_3(x)
        x = tf.keras.layers.Add()([skip, x])
     
        skip = x
        x = self.layer_4_5(x)
        x = tf.keras.layers.Add()([skip, x])
   
        skip = x
        x = self.layer_6_7(x)
        x = tf.keras.layers.Add()([skip, x])
      
        skip = x
        x = self.layer_8_9(x)
        x = tf.keras.layers.Add()([skip, x])
     
        skip = x
        x = self.layer_10_11(x)
        x = tf.keras.layers.Add()([skip, x])
       
        skip = x
        x = self.layer_12_13(x)
        x = tf.keras.layers.Add()([skip, x])
     
        skip = x
        x = self.layer_14_15(x)
        x = tf.keras.layers.Add()([skip, x])
      
        skip = x
        x = self.layer_16_17(x)
        x = tf.keras.layers.Add()([skip, x])
     
        skip = x
        x = self.layer_18_19(x)
        x = tf.keras.layers.Add()([skip, x])       

        x = self.layer_20(x)


        return tf.keras.layers.Add()([input, x])   