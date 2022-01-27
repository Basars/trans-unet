# TransUNet

An ML model with U-shaped architecture with ResNet50V2 and Vision Transformer based encoders

#### Usage:
```python
import numpy as np

from transunet import VisionTransformer

# Encoder weights from Google
weights = np.load('R50+ViT-B_16.npz', allow_pickle=True)

model = VisionTransformer(input_shape=(224, 224, 3), 
                          num_classes=1, 
                          w=weights, 
                          encoder_trainable=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(...)
```