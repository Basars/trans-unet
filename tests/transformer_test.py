import pytest
import numpy as np

from tensorflow.keras.initializers import Constant as ConstantInit, GlorotUniform, Zeros
from transunet.transformer import MultiHeadAttention


def perform_attention_weights_test(expected_inits, hidden_size, w=None):
    assert len(expected_inits) == 2, 'Number of expected inits must be 2. One for kernel, the others for bias'

    attention = MultiHeadAttention(num_heads=6, w=w, name='attention')
    attention.build(input_shape=(None, hidden_size))

    keys = ['query', 'key', 'value', 'out']
    for key, layer in zip(keys, [attention.query_dense, attention.key_dense, attention.values_dense, attention.dense]):
        layer.build(input_shape=(None, hidden_size))

        kernel_initializer = layer.kernel_initializer
        bias_initializer = layer.bias_initializer

        kernel_weights_key = 'attention/{}/kernel'.format(key)
        bias_weights_key = 'attention/{}/bias'.format(key)

        assert isinstance(kernel_initializer, expected_inits[0]), \
            'Expected weights key: {}; attention name: {}'.format(kernel_weights_key, attention.name)
        assert isinstance(bias_initializer, expected_inits[1]), \
            'Expected weights key: {}, attention name: {}'.format(bias_weights_key, attention.name)

        if w is not None:
            layer_weights = layer.get_weights()
            assert np.array_equal(layer_weights[0], w[kernel_weights_key])
            assert np.array_equal(layer_weights[1], w[bias_weights_key])

    return True


def test_attention_initializers():
    hidden_size = 1024 * 3
    weights = {
        'attention/query/kernel': np.ones((hidden_size, hidden_size)),
        'attention/query/bias': np.zeros((hidden_size,)),
        'attention/key/kernel': np.ones((hidden_size, hidden_size)),
        'attention/key/bias': np.zeros((hidden_size,)),
        'attention/value/kernel': np.ones((hidden_size, hidden_size)),
        'attention/value/bias': np.zeros((hidden_size,)),
        'attention/out/kernel': np.ones((hidden_size, hidden_size)),
        'attention/out/bias': np.zeros((hidden_size,))
    }
    assert perform_attention_weights_test([ConstantInit, ConstantInit], hidden_size, w=weights)
    assert perform_attention_weights_test([GlorotUniform, Zeros], hidden_size)


if __name__ == '__main__':
    pytest.main([__file__])
