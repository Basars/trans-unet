import tensorflow as tf
import numpy as np

from scipy import ndimage
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant as ConstantInit


class PositionEmbedding(Layer):

    def __init__(self, w=None, trainable=True, **kwargs):
        super(PositionEmbedding, self).__init__(trainable=trainable, **kwargs)

        self.w = w
        self.embedding = None

    def build(self, input_shape):
        assert len(input_shape) == 3, 'Expected ndim = 3, but ndim = {}'.format(len(input_shape))

        shape = (1, input_shape[1], input_shape[2])
        if self.w is None:
            initializer = tf.random_normal_initializer(stddev=0.06)
        else:
            key_name = '{}/pos_embedding'.format(self.name)
            weights = self.w[key_name]
            if shape == weights.shape:
                initializer = ConstantInit(weights)
            elif shape[1] == weights.shape[1] - 1:
                initializer = ConstantInit(weights[:, 1:])
            else:
                values = weights[0, 1:]
                shape_in = int(np.sqrt(values.shape[0]))
                shape_out = int(np.sqrt(shape[1]))
                zoom = (shape_out / shape_in, shape_out / shape_in, 1)
                zoom = ndimage.zoom(values.reshape((shape_in, shape_in, -1)), zoom, order=1)
                zoom = zoom.reshape((1, shape_out * shape_out, -1))
                initializer = ConstantInit(zoom)

        self.embedding = self.add_weight('pos_embedding',
                                         shape=shape,
                                         dtype=tf.float32,
                                         initializer=initializer,
                                         trainable=self.trainable)

    def call(self, inputs, *args, **kwargs):
        return inputs + tf.cast(self.embedding, dtype=inputs.dtype)

    def get_config(self):
        return super(PositionEmbedding, self).get_config()
