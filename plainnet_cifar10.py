import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, Sequential
from    keras.initializers import VarianceScaling, Orthogonal

class BasicBlock(layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3,3), strides=stride,
                                   kernel_initializer=VarianceScaling(),
                                   kernel_regularizer=keras.regularizers.l2(
                                       0.0005),
                                   padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1,
                                   kernel_initializer=VarianceScaling(),
                                   kernel_regularizer=keras.regularizers.l2(
                                       0.0005),
                                   padding='same')
        self.bn2 = layers.BatchNormalization()


    def call(self, inputs, training=None):

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        output = tf.nn.relu(out)

        return  output


class PlainNet(keras.Model):
    
    def __init__(self, layer_dims, num_classes=10): #[[0], [1], [2]]
        super(PlainNet, self).__init__()

        self.stem = Sequential([layers.Conv2D(16, (3, 3), strides=2,
                                              padding='same',
                                              kernel_initializer=VarianceScaling(),
                                              kernel_regularizer=
                                              keras.regularizers.l2(0.0005)),
                                layers.BatchNormalization(),
                                layers.Activation('relu')])

        self.layer1 = self.build_plablock(16, layer_dims[0])
        self.layer2 = self.build_plablock(32, layer_dims[1], stride=2)
        self.layer3 = self.build_plablock(64, layer_dims[2], stride=2)

        #output: [b, 64, h, w] =>  [b, 512]
        self.avgpool = layers.GlobalAveragePooling2D()
        # [b, 64] => [b, num_classes]
        self.fc = layers.Dense(num_classes,
                               kernel_regularizer=
                               keras.regularizers.l2(0.0005))


    def call(self, inputs, training=None):
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #[b,c]
        x = self.avgpool(x)
        #[b,num_classes]
        x = self.fc(x)

        return x

    def build_plablock(self, filter_num, blocks, stride=1):

        pla_blocks = Sequential()

        pla_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            pla_blocks.add(BasicBlock(filter_num, stride=1))

        return  pla_blocks

def plainnet20():

    return PlainNet([3, 3, 3])

def plainnet32():

    return PlainNet([5, 5, 5])

def plainnet44():

    return PlainNet([7, 7, 7])

def plainnet56():

    return PlainNet([9, 9, 9])

def plainnet110():

    return PlainNet([18, 18, 18])