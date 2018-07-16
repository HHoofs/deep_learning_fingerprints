import os

import keras
import keras.backend as K
import numpy as np
from PIL import Image
import glob
from keras import models
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Input, MaxPooling2D, Conv2D, Activation, BatchNormalization, AveragePooling2D, concatenate, \
    GlobalAveragePooling2D, Dense
from skimage.exposure import equalize_adapthist, exposure
from skimage.transform import resize
from scipy import misc as scipy_misc
from skimage import util as skimage_util


import random

from augment import image_augment_array


class Neural_Net():
    def __init__(self, img_size):
        self.img_size = img_size

    def set_net(self, relative_size):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        raw = Input(shape=(*self.img_size, 1), name='raw')
        x = conv2d_bn_alt(raw, relative_size * 32, 3, 3, strides=(2, 2), padding='valid')
        x = conv2d_bn_alt(x, relative_size * 32, 3, 3, padding='valid')
        x = conv2d_bn_alt(x, relative_size * 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv2d_bn_alt(x, relative_size * 80, 1, 1, padding='valid')
        x = conv2d_bn_alt(x, relative_size * 100, 3, 3, padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0:
        branch1x1 = conv2d_bn_alt(x, relative_size * 64, 1, 1)

        branch5x5 = conv2d_bn_alt(x, relative_size * 48, 1, 1)
        branch5x5 = conv2d_bn_alt(branch5x5, relative_size * 64, 5, 5)

        branch3x3dbl = conv2d_bn_alt(x, relative_size * 64, 1, 1)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_alt(branch_pool, 32, 1, 1)
        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis)

        # mixed 1:
        branch1x1 = conv2d_bn_alt(x, relative_size * 64, 1, 1)

        branch5x5 = conv2d_bn_alt(x, relative_size * 48, 1, 1)
        branch5x5 = conv2d_bn_alt(branch5x5, relative_size * 64, 5, 5)

        branch3x3dbl = conv2d_bn_alt(x, relative_size * 64, 1, 1)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_alt(branch_pool, relative_size * 64, 1, 1)
        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis)

        # mixed 2:
        branch1x1 = conv2d_bn_alt(x, relative_size * 64, 1, 1)

        branch5x5 = conv2d_bn_alt(x, relative_size * 48, 1, 1)
        branch5x5 = conv2d_bn_alt(branch5x5, relative_size * 64, 5, 5)

        branch3x3dbl = conv2d_bn_alt(x, relative_size * 64, 1, 1)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_alt(branch_pool, relative_size * 64, 1, 1)

        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis)

        # mixed 3:
        branch3x3 = conv2d_bn_alt(x, relative_size * 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = conv2d_bn_alt(x, relative_size * 64, 1, 1)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([branch3x3, branch3x3dbl, branch_pool], axis=channel_axis)

        # mixed 4:
        branch1x1 = conv2d_bn_alt(x, relative_size * 192, 1, 1)

        branch7x7 = conv2d_bn_alt(x, relative_size * 128, 1, 1)
        branch7x7 = conv2d_bn_alt(branch7x7, relative_size * 128, 1, 7)
        branch7x7 = conv2d_bn_alt(branch7x7, relative_size * 192, 7, 1)

        branch7x7dbl = conv2d_bn_alt(x, relative_size * 128, 1, 1)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 128, 7, 1)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 128, 1, 7)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 128, 7, 1)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_alt(branch_pool, relative_size * 192, 1, 1)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis)

        # mixed 5:
        branch1x1 = conv2d_bn_alt(x, relative_size * 192, 1, 1)

        branch7x7 = conv2d_bn_alt(x, relative_size * 192, 1, 1)
        branch7x7 = conv2d_bn_alt(branch7x7, relative_size * 192, 1, 7)
        branch7x7 = conv2d_bn_alt(branch7x7, relative_size * 192, 7, 1)

        branch7x7dbl = conv2d_bn_alt(x, relative_size * 192, 1, 1)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 192, 7, 1)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 192, 1, 7)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 192, 7, 1)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_alt(branch_pool, relative_size * 192, 1, 1)

        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis)

        # mixed 6:
        branch3x3 = conv2d_bn_alt(x, relative_size * 192, 1, 1)
        branch3x3 = conv2d_bn_alt(branch3x3, relative_size * 320, 3, 3, strides=(2, 2), padding='valid')

        branch7x7x3 = conv2d_bn_alt(x, relative_size * 192, 1, 1)
        branch7x7x3 = conv2d_bn_alt(branch7x7x3, relative_size * 192, 1, 7)
        branch7x7x3 = conv2d_bn_alt(branch7x7x3, relative_size * 192, 7, 1)
        branch7x7x3 = conv2d_bn_alt(branch7x7x3, relative_size * 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis)

        # end:
        x = GlobalAveragePooling2D()(x)

        pre = Input(shape=(299,299,3), name='pre')
        x_pre = InceptionV3(include_top=False, input_tensor=pre, pooling='avg')

        x_combined = concatenate([x,x_pre.output])

        y = Dense(8, activation='softmax')(x_combined)

        model = models.Model(inputs=[raw, pre], outputs=y)

        self.neural_net = model
        self.inception_v3_layers = x_pre.layers

    def freeze_inception_layers(self):
        for layer2freeze in self.inception_v3_layers:
            self.neural_net.get_layer(layer2freeze.name).trainable = False

    def compile(self, loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(), metrics=['acc']):
        self.neural_net.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, training_generator, validation_generator=None, epochs=None, callbacks=None):
        self.neural_net.fit_generator(generator=training_generator,
                                      validation_data=validation_generator,
                                      epochs=epochs,
                                      callbacks=callbacks)



def conv2d_bn_alt(x,
                  filters,
                  num_row,
                  num_col,
                  padding='same',
                  strides=(1, 1),
                  name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    filters = int(round(filters))

    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def random_string():
    return str(np.random.random())[2:]


class DataGenerator(keras.utils.Sequence):
    #Generates data for Keras
    def __init__(self, list_ids, path, batch_size=8, dim=(512, 512), n_channels=1, shuffle=True, mode='L',
                 prop_image=.5, prop_array=.5):
        #Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        if path:
            self.path = path
        else:
            self.path = ''
        self.prop_image = prop_image
        self.prop_array = prop_array
        self.mode = mode
        self.on_epoch_end()
        self.indexes = np.arange(len(self.list_ids))
        self.mapping = {'A':0, 'D':1, 'L':2, 'P':3, 'R':4, 'T':5, 'U':6, 'W':7}

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        # Generates data containing batch_size samples X : (n_samples, *dim, n_channels)
        x_image_raw = np.zeros((self.batch_size, *self.dim, self.n_channels))
        x_image_pre = np.zeros((self.batch_size, 299, 299, 3))

        x_images = {'raw': x_image_raw, 'pre': x_image_pre}

        y = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, sample in enumerate(list_ids_temp):

            with Image.open(os.path.join(self.path, sample[:-4] + '.png')) as x_img:
                x_img = x_img.convert(mode=self.mode)

                if self.prop_array == 0 and self.prop_image == 0:
                    x_arr = np.array(x_img)
                else:
                    x_arr = image_augment_array(x_img, self.prop_image, self.prop_array)

                x_arr_raw = resize(x_arr, output_shape=(self.dim[0], self.dim[1]))

                # normaliz here because preprocess for inception V3 should be clean
                _array_x_raw = x_arr_raw
                _array_x_raw *= 255.0 / _array_x_raw.max()

                x_images['raw'][i,] = np.expand_dims(_array_x_raw, 2)

                x_arr_pre = resize(x_arr, output_shape=(299, 299))
                if self.mode == 'L':
                    x_arr_pre = to_rgb(x_arr_pre)

                x_images['pre'][i,] = preprocess_input(x_arr_pre)

            with open(sample) as y_txt:
                for line in y_txt.readlines():
                    if line.startswith('Class'):
                        y[i] = self.mapping.get(line[7])

        return x_images, keras.utils.to_categorical(y, num_classes=len(self.mapping))




def to_rgb(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.zeros((w, h, 3))
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

if __name__ == '__main__':
    model = Neural_Net(img_size=(512,512))
    model.set_net(relative_size=1)
    model.freeze_inception_layers()
    print(model.neural_net.summary())
    quit(0)
    model.compile()

    list_ids = glob.glob('sd04/png_txt/figs_0/*.txt')
    training_gen = DataGenerator(list_ids=list_ids, path=None)
    callback_tb = keras.callbacks.TensorBoard()

    model.fit(training_generator=training_gen, epochs=2, callbacks=[callback_tb])
