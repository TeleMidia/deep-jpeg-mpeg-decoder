import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU

class CARD(Model):
    def __init__(self):
        super(CARD, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        
        self.layer_1_2 = self.WARB_block(384, 192, 3, initializer)
        self.layer_3_4 = self.WARB_block(384, 192, 3, initializer)
        self.layer_5_6 = self.WARB_block(384, 192, 3, initializer)
        self.layer_7_8 = self.WARB_block(384, 192, 3, initializer)
        self.layer_9_10 = self.WARB_block(384, 192, 3, initializer)
        self.layer_11_12 = self.WARB_block(384, 192, 3, initializer)
        self.layer_13_14 = self.WARB_block(384, 192, 3, initializer)
        self.layer_15_16 = self.WARB_block(384, 192, 3, initializer)
        self.layer_17_18 = self.WARB_block(384, 192, 3, initializer)
        self.layer_19_20 = self.WARB_block(384, 192, 3, initializer)
       
        
    def WARB_block(self, filters_1, filters_2, kernel_size, initializer):

        result = tf.keras.Sequential() 
        result.add(Conv2D(filters_1, kernel_size, activation='relu', strides=1, padding='same', kernel_initializer=initializer))
        result.add(Conv2D(filters_2, kernel_size, activation=None, strides=1, padding='same', kernel_initializer=initializer))
        return result

    def call(self, input):
                
        x = self.layer_1_2(input)
        x = tf.keras.layers.Add()([input, x])
     
        skip = x
        x = self.layer_3_4(x)
        x = tf.keras.layers.Add()([skip, x])
   
        skip = x
        x = self.layer_5_6(x)
        x = tf.keras.layers.Add()([skip, x])
      
        skip = x
        x = self.layer_7_8(x)
        x = tf.keras.layers.Add()([skip, x])
     
        skip = x
        x = self.layer_9_10(x)
        x = tf.keras.layers.Add()([skip, x])
       
        skip = x
        x = self.layer_11_12(x)
        x = tf.keras.layers.Add()([skip, x])
     
        skip = x
        x = self.layer_13_14(x)
        x = tf.keras.layers.Add()([skip, x])
      
        skip = x
        x = self.layer_15_16(x)
        x = tf.keras.layers.Add()([skip, x])
     
        skip = x
        x = self.layer_17_18(x)
        x = tf.keras.layers.Add()([skip, x])       

        skip = x
        x = self.layer_19_20(x)
        x = tf.keras.layers.Add()([skip, x])

        return tf.keras.layers.Add()([input, x])   