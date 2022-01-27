import pytest
import numpy as np

from tensorflow.keras.initializers import Constant as ConstantInit
from transunet.utils import default_initializers, default_initializer


def test_default_initializer_usages():
    weights = {
        'block1_conv1': np.identity(5)
    }
    assert default_initializer('block1_conv1') is None
    assert default_initializer('block1_conv1', default='glorot_uniform') == 'glorot_uniform'
    assert default_initializer('block1_conv1') or 'glorot_uniform' == 'glorot_uniform'

    assert isinstance(default_initializer('block1_conv1', w=weights), ConstantInit)
    assert np.array_equal(default_initializer('block1_conv1', w=weights).value,
                          ConstantInit(weights['block1_conv1']).value)

    assert isinstance(default_initializer('block1_conv1', default='glorot_uniform', w=weights), ConstantInit)
    assert default_initializer('block1_conv2',
                               default='glorot_uniform',
                               w=weights) == 'glorot_uniform'


def test_default_initializers_usages():
    scope_name = 'block1_conv1'
    weights = {
        f'{scope_name}/kernel': np.identity(5),
        f'{scope_name}/bias': np.identity(5)
    }
    assert default_initializers(scope_name, ['kernel', 'bias']) == [None, None]
    assert default_initializers(scope_name, ['kernel', 'bias'], defaults=['ones', 'zeros']) == ['ones', 'zeros']

    inits = default_initializers(scope_name, ['kernel', 'bias'])
    assert len(inits) == 2
    assert inits[0] is None and inits[1] is None
    assert inits[1] or 'zeros' == 'zeros'

    inits = default_initializers(scope_name, ['kernel', 'bias'], w=weights)
    assert len(inits) == 2
    assert inits[0] is not None and inits[1] is not None
    assert inits[1] or 'zeros' == weights['{}/bias'.format(scope_name)]


if __name__ == '__main__':
    pytest.main([__file__])
