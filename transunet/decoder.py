import tensorflow as tf

from typing import List
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, UpSampling2D
from tensorflow.keras.regularizers import L2


class Conv2DNormBlock(Layer):

    def __init__(self, filters, kernel_size, padding='same', strides=1, **kwargs):
        super(Conv2DNormBlock, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

        self.conv = self.batch_norm = None

    def build(self, input_shape):
        self.conv = Conv2D(filters=self.filters,
                           kernel_size=self.kernel_size,
                           strides=self.strides,
                           padding=self.padding,
                           use_bias=False,
                           kernel_regularizer=L2(1e-4),
                           kernel_initializer='lecun_normal')
        self.batch_norm = BatchNormalization(momentum=0.9, epsilon=1e-5)

    def call(self, inputs, *args, **kwargs):
        return tf.nn.relu(self.batch_norm(self.conv(inputs)))

    def get_config(self):
        config = super(Conv2DNormBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'strides': self.strides
        })
        return config


class DecoderBlock(Layer):

    def __init__(self, filters, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.filters = filters
        self.conv1 = self.conv2 = self.upsampling = None

    def build(self, _):
        self.conv1 = Conv2DNormBlock(filters=self.filters, kernel_size=3)
        self.conv2 = Conv2DNormBlock(filters=self.filters, kernel_size=3)
        self.upsampling = UpSampling2D(size=2, interpolation='bilinear')

    def call(self, inputs, *args, **kwargs):
        inputs, skip = inputs['inputs'], inputs['skip']

        x = self.upsampling(inputs)
        if skip is not None:
            x = tf.concat([x, skip], axis=-1)
        return self.conv2(self.conv1(x))

    def get_config(self):
        config = super(DecoderBlock, self).get_config()
        config.update({
            'filters': self.filters
        })
        return config


class DecoderCascadedUpSampling(Layer):

    def __init__(self, decoder_channels: List[int], num_skips=3, **kwargs):
        super(DecoderCascadedUpSampling, self).__init__(**kwargs)

        self.decoder_channels = decoder_channels
        self.num_skips = num_skips
        self.conv = self.decoder_blocks = None

    def build(self, _):
        self.conv = Conv2DNormBlock(filters=512, kernel_size=3)
        self.decoder_blocks = [DecoderBlock(filters=channels_out) for channels_out in self.decoder_channels]

    def call(self, inputs, *args, **kwargs):
        hidden_states, skips = inputs['hidden_states'], inputs['skips']
        x = self.conv(hidden_states)
        for i, decoder_block in enumerate(self.decoder_blocks):
            if skips is not None:
                skip = skips[i] if i < self.num_skips else None
            else:
                skip = None

            x = decoder_block({
                'inputs': x,
                'skip': skip
            })
        return x

    def get_config(self):
        config = super(DecoderCascadedUpSampling, self).get_config()
        config.update({
            'decoder_channels': self.decoder_channels,
            'num_skips': self.num_skips
        })
        return config


class DecoderOutput(Layer):

    def __init__(self, filters=1, kernel_size=1, scale_factor=1, name='decoder_outputs', **kwargs):
        super(DecoderOutput, self).__init__(name=name, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.conv = self.upsampling = None

    def build(self, _):
        self.conv = Conv2D(filters=self.filters,
                           kernel_size=self.kernel_size,
                           padding='same',
                           kernel_regularizer=L2(1e-4),
                           kernel_initializer='lecun_normal')
        self.upsampling = UpSampling2D(size=self.scale_factor, interpolation='bilinear')

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        if self.scale_factor > 1:
            x = self.upsampling(x)
        return x

    def get_config(self):
        config = super(DecoderOutput, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'scale_factor': self.scale_factor
        })
        return config
