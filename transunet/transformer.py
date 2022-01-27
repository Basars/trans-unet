import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Lambda
from tensorflow.keras.initializers import Constant as ConstantInit


def scaled_dot_product_attention(query, key, values):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    attention_weights = tf.nn.softmax(logits, axis=-1)
    outputs = tf.matmul(attention_weights, values)
    return outputs, attention_weights


class MultiHeadAttention(Layer):

    def __init__(self,
                 num_heads: int,
                 w=None,
                 trainable: bool = True,
                 *args, **kwargs):

        super(MultiHeadAttention, self).__init__(trainable=trainable, *args, **kwargs)

        self.num_heads = num_heads
        self.w = w
        self.hidden_size = self.depth = None
        self.query_dense = self.key_dense = self.values_dense = self.dense = None

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        assert hidden_size % self.num_heads == 0

        self.hidden_size = hidden_size
        self.depth = hidden_size // self.num_heads
        self.query_dense = Dense(hidden_size, name='query',
                                 kernel_initializer=ConstantInit(self.w['{}/query/kernel'.format(self.name)]) if self.w is not None else 'glorot_uniform',
                                 bias_initializer=ConstantInit(self.w['{}/query/bias'.format(self.name)]) if self.w is not None else 'zeros')
        self.key_dense = Dense(hidden_size, name='key',
                               kernel_initializer=ConstantInit(self.w['{}/key/kernel'.format(self.name)]) if self.w is not None else 'glorot_uniform',
                               bias_initializer=ConstantInit(self.w['{}/key/bias'.format(self.name)]) if self.w is not None else 'zeros')
        self.values_dense = Dense(hidden_size, name='values',
                                  kernel_initializer=ConstantInit(self.w['{}/value/kernel'.format(self.name)]) if self.w is not None else 'glorot_uniform',
                                  bias_initializer=ConstantInit(self.w['{}/value/bias'.format(self.name)]) if self.w is not None else 'zeros')
        self.dense = Dense(hidden_size, name='combine',
                           kernel_initializer=ConstantInit(self.w['{}/out/kernel'.format(self.name)]) if self.w is not None else 'glorot_uniform',
                           bias_initializer=ConstantInit(self.w['{}/out/bias'.format(self.name)]) if self.w is not None else 'zeros')

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def call(self, inputs, *args, **kwargs):
        batch_size = tf.shape(inputs)[0]

        query = self.split_heads(self.query_dense(inputs), batch_size)
        key = self.split_heads(self.key_dense(inputs), batch_size)
        values = self.split_heads(self.values_dense(inputs), batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(query, key, values)
        scaled_attention = tf.transpose(scaled_attention, perm=(0, 2, 1, 3))

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.hidden_size))
        combined = self.dense(concat_attention)
        return combined, attention_weights

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads
        })
        return config


class TransformerBlock(Layer):

    def __init__(self,
                 num_heads: int, units: int, dropout_rate: float,
                 w=None,
                 trainable: bool = True,
                 *args, **kwargs):

        super(TransformerBlock, self).__init__(trainable=trainable, *args, **kwargs)

        self.num_heads = num_heads
        self.units = units
        self.dropout_rate = dropout_rate
        self.w = w

        self.attention = self.mlp = self.layer_norm1 = self.layer_norm2 = self.dropout = None

    def build(self, input_shape):
        self.attention = MultiHeadAttention(num_heads=self.num_heads, w=self.w, name='{}/MultiHeadDotProductAttention_1'.format(self.name))

        mlp_name = 'MlpBlock_3'
        dense0_name = '{}/{}/Dense_0'.format(self.name, mlp_name)
        dense1_name = '{}/{}/Dense_1'.format(self.name, mlp_name)
        self.mlp = Sequential(name=mlp_name, layers=[
            Dense(self.units, name=dense0_name, activation='linear',
                  kernel_initializer=ConstantInit(self.w['{}/kernel'.format(dense0_name)]) if self.w is not None else 'glorot_uniform',
                  bias_initializer=ConstantInit(self.w['{}/bias'.format(dense0_name)]) if self.w is not None else 'zeros'),
            Lambda(lambda x: gelu(x, approximate=False)),
            Dropout(self.dropout_rate),
            Dense(input_shape[-1], name=dense1_name,
                  kernel_initializer=ConstantInit(self.w['{}/kernel'.format(dense1_name)]) if self.w is not None else 'glorot_uniform',
                  bias_initializer=ConstantInit(self.w['{}/bias'.format(dense1_name)]) if self.w is not None else 'zeros'),
            Dropout(self.dropout_rate)
        ])

        layer_norm_name = '{}/LayerNorm_0'.format(self.name)
        self.layer_norm1 = LayerNormalization(epsilon=1e-6, name=layer_norm_name,
                                              gamma_initializer=ConstantInit(self.w['{}/scale'.format(layer_norm_name)]) if self.w is not None else 'ones',
                                              beta_initializer=ConstantInit(self.w['{}/bias'.format(layer_norm_name)]) if self.w is not None else 'zeros')

        layer_norm_name = '{}/LayerNorm_2'.format(self.name)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6, name=layer_norm_name,
                                              gamma_initializer=ConstantInit(self.w['{}/scale'.format(layer_norm_name)]) if self.w is not None else 'ones',
                                              beta_initializer=ConstantInit(self.w['{}/bias'.format(layer_norm_name)]) if self.w is not None else 'zeros')
        self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs, *args, **kwargs):
        training = kwargs['training'] if hasattr(kwargs, 'training') else None

        x = self.layer_norm1(inputs)
        x, weights = self.attention(x)
        x = self.dropout(x, training=training)
        x = x + inputs
        y = self.layer_norm2(x)
        y = self.mlp(y)
        return x + y, weights

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'units': self.units,
            'dropout_rate': self.dropout_rate
        })
        return config