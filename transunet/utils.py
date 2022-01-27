from typing import Union, Callable, Sequence, Optional
from tensorflow.keras.initializers import Constant as ConstantInit


def default_initializer(key: str, default: Optional[Union[str, Callable]] = None, w=None):
    if w is not None and key in w:
        return ConstantInit(w[key])
    else:
        return default


def default_initializers(scope_name: str,
                         keys: Sequence[str],
                         defaults: Sequence[Optional[Union[str, Callable]]] = None,
                         delimiter='/',
                         w=None):

    if defaults is not None:
        assert len(keys) == len(defaults), 'Number of keys and default values must be equivalent'

    initializers = []
    for index, key in enumerate(keys):
        key = '{}{}{}'.format(scope_name, delimiter, key)
        if defaults is not None:
            default = defaults[index]
        else:
            default = None
        initializers.append(default_initializer(key, w=w, default=default))
    return initializers
