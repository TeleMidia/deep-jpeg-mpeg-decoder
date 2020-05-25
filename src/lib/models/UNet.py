import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Conv2DTranspose, MaxPool2D
from tensorflow.keras.activations import relu

class UNet(Model):
    def __init__(self, first_kernel):
        super(UNet, self).__init__()

        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.conv_1 = self.Conv2dBatchLayer(32,first_kernel)
        self.conv_2 = self.Conv2dBatchLayer(64,3)
        self.conv_3 = self.Conv2dBatchLayer(128,3)
        self.conv_4 = self.Conv2dBatchLayer(256,3)
        self.conv_5 = self.Conv2dBatchLayer(512,3)
        self.conv_6 = self.Conv2dBatchLayer(1024,3)
        self.up_1 = Conv2DTranspose(512, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_7 = self.Conv2dBatchLayer(512,3)
        self.up_2 = Conv2DTranspose(256, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_8 = self.Conv2dBatchLayer(256,3)
        self.up_3 = Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_9 = self.Conv2dBatchLayer(128,3)
        self.up_4 = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_10 = self.Conv2dBatchLayer(64,3)
        self.up_5 = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same', kernel_initializer=initializer)
        self.conv_11 = self.Conv2dBatchLayer(32, 3)
        self.last_conv = Conv2D(3, 1, strides=1, padding='same', kernel_initializer=initializer)

    def Conv2dBatchLayer(self, filters, kernel_size):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, kernel_size, strides=1, padding='same', kernel_initializer=initializer))
        result.add(BatchNormalization())
        result.add(ReLU())
        return result
    

    def call(self, input):
        
        #96x96x3 -> 48x48x32
        conv1 = self.conv_1(input)
        down1 = self.max_pool(conv1)

        #48x48x32 -> 24x24x64
        conv2 = self.conv_2(down1)
        down2 = self.max_pool(conv2)

        #24x24x64 -> 12x12x128
        conv3 = self.conv_3(down2)
        down3 = self.max_pool(conv3)

        #12x12x128 -> 6x6x256
        conv4 = self.conv_4(down3)
        down4 = self.max_pool(conv4)

        #6x6x256 -> 3x3x512
        conv5 = self.conv_5(down4)
        down5 = self.max_pool(conv5)

        #3x3x512 -> 3x3x1024
        conv6 = self.conv_6(down5)

        #3x3x1024 -> 6x6x512
        up1 =  self.up_1(conv6)
        concat1 = tf.keras.layers.concatenate([conv5, up1], axis=3)
        conv7 = self.conv_7(concat1)

        #6x6x512 -> 12x12x246
        up2 =  self.up_2(conv7)
        concat2 = tf.keras.layers.concatenate([conv4, up2], axis=3)
        conv8 = self.conv_8(concat2)

        #12x12x246 -> 24x24x128
        up3 =  self.up_3(conv8)
        concat3 = tf.keras.layers.concatenate([conv3, up3], axis=3)
        conv9 = self.conv_9(concat3)

        #24x24x128 -> 48x48x64
        up4 =  self.up_4(conv9)
        concat4 = tf.keras.layers.concatenate([conv2, up4], axis=3)
        conv10 = self.conv_10(concat4)

        #48x48x64 -> 96x96x32
        up5 =  self.up_5(conv10)
        concat5 = tf.keras.layers.concatenate([conv1, up5], axis=3)
        conv11 = self.conv_11(concat5)

        #96x96x32 -> 96x96x3
        return self.last_conv(conv11)  