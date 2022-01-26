# TransUNet

An ML model with U-shaped architecture with ResNet50V2 and Vision Transformer based encoders

#### Usage:
```python
from transunet import VisionTransformer

model = VisionTransformer(input_shape=(224, 224, 3), num_classes=1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(...)
```

#### Pre-trained weights

The model have been tested with `R50+ViT-B/16`