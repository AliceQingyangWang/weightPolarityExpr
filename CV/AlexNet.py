# AlexNet Model 

"""
AlexNet, Krizhevsky, Alex, Ilya Sutskever and Geoffrey E. Hinton, 2012
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Michael Guerzhoy and Davi Frossard, 2016
AlexNet implementation in TensorFlow, with weights Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

The script is the realizatio of object oriented style of AlexNet. The construction method of AlexNet 
includes three parameters, including self, input_shape, num_classes, of which, input_shape works as a 
placeholder. 
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np


# Define the AlexNet Model 
class AlexNet(Sequential):
    def __init__(self, input_shape, num_classes, pre_trained=False): #will still keep the pre-trained functionality. as it is different from loading the polarity alone!
        super().__init__()

        # No.1 Convolutional Layer: (227- 11 + 2 * 0) / 4 + 1 = 55
        self.add(Conv2D(filters=96, kernel_size=(11,11), strides=4,
                        padding='valid', activation='relu',
                        input_shape=input_shape, kernel_initializer='GlorotNormal', name = 'conv1'))
        # Max Pooling: (55- 3 + 2 * 0) / 2 + 1 = 27
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2),
                              padding='valid', data_format=None))
        
        # No.2 Conv Layer: (27- 5 + 2 * 2) / 1 + 1 = 27
        self.add(Conv2D(filters=256, kernel_size=(5,5), strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='GlorotNormal', name='conv2'))
        # Max Pooling: (27-  3 + 2 * 0) / 2 + 1 = 13
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2),
                              padding='valid', data_format=None))
        
        # No.3 Conv Layer: (13 - 3 + 2 * 1) / 1 + 1 = 13
        self.add(Conv2D(filters=384, kernel_size=(3,3), strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='GlorotNormal', name='conv3'))

        # No.4 Conv Layer: (13 - 3 + 2 * 1) / 1 + 1 = 13
        self.add(Conv2D(filters=384, kernel_size=(3,3), strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='GlorotNormal', name='conv4'))

        # No.5 Conv Layer: (13 - 3 + 2 * 1) / 1 + 1 = 13
        self.add(Conv2D(filters=256, kernel_size=(3,3), strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='GlorotNormal', name='conv5')) 
        # Max Pooling: (13 - 3 + 2 * 0) / 2 + 1 =  6
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2),
                              padding='valid', data_format=None))

        # # Flatten the three dimensions of 256 x 6 x 6 into one dimension of 9216.
        self.add(Flatten())

        # No.6 FC Layer 
        self.add(Dense(4096, activation='relu', name='fc6'))
        # Add Dropout
        self.add(Dropout(0.5))
        
        # No.7 FC Layer         
        self.add(Dense(4096, activation='relu', name='fc7'))
        # Add Dropout 
        self.add(Dropout(0.5))

        # No.8 FC Layer
        self.add(Dense(1000, activation='relu', name='fc8'))
        self.add(Dense(num_classes, activation='softmax', name='dense', kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0,stddev=0.25)))

        if pre_trained:
            self.load_weights_pickle()

    def fix_split_gpu(self, in_mat):
        sz = np.array(in_mat.shape)
        sz[2] = sz[2]*2
        tmp = np.zeros(sz)
        assert(sz[3]%2 == 0)
        tmp[:, :, :int(sz[2]/2), :int(sz[3]/2)] = in_mat[:, :, :, :int(sz[3]/2)]
        tmp[:, :, int(sz[2]/2):, int(sz[3]/2):] = in_mat[:, :, :, int(sz[3]/2):]
        return tmp    

    def load_weights_pickle(self, path = 'bvlc_alexnet.npy'):
        wt_data = np.load(path, encoding = 'bytes', allow_pickle=True).item()
        wt_data['conv2'][0] = self.fix_split_gpu(wt_data['conv2'][0])
        wt_data['conv4'][0] = self.fix_split_gpu(wt_data['conv4'][0])
        wt_data['conv5'][0] = self.fix_split_gpu(wt_data['conv5'][0])
        for layer in self.layers:
            if isinstance(layer, (Dense, Conv2D)):
                if layer.name in wt_data.keys():
                    layer.set_weights([wt_data[layer.name][0], wt_data[layer.name][1]])

    def clone(self):
        # will follow the tf convention that the weight is not copied. simpy created a new model. 
        return AlexNet(self.input_shape[1:], self.output_shape[1]) #Theoretically should have created a new one. 
        
        