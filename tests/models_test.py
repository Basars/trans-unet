import pytest
import numpy as np
import tensorflow as tf

from transunet import VisionTransformer


IMG_SIZE = 224
BATCH_SIZE = 24
NUM_CLASSES = 1


def test_vision_transformer_architecture():
    model = VisionTransformer(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES)

    assert tuple(model.input.shape) == (None, IMG_SIZE, IMG_SIZE, 3)
    assert tuple(model.output.shape) == (None, IMG_SIZE, IMG_SIZE, NUM_CLASSES)


def test_vision_transformer_feed_forward():
    inputs = np.random.normal(size=IMG_SIZE * IMG_SIZE * 3 * BATCH_SIZE)
    inputs = np.reshape(inputs, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))

    model = VisionTransformer(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    predicted = model(inputs)

    assert len(predicted.shape) == 4
    assert tuple(predicted.shape) == (BATCH_SIZE, IMG_SIZE, IMG_SIZE, NUM_CLASSES)


if __name__ == '__main__':
    pytest.main([__file__])
