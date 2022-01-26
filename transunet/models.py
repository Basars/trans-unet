import math

from typing import List
from tensorflow.keras import Model, applications
from tensorflow.keras.initializers import Constant as ConstantInit
from tensorflow.keras.layers import Input, Conv2D, Reshape, Dropout, LayerNormalization
from transunet import PositionEmbedding, TransformerBlock, DecoderCascadedUpSampling, DecoderOutput


RESNET_SKIP_LAYER_NAMES = ['conv3_block4_preact_relu', 'conv2_block3_preact_relu', 'conv1_conv']
RESNET_BOTTLENECK_LAYER_NAME = 'conv4_block6_preact_relu'


class VisionTransformer(Model):

    def __init__(self,
                 input_shape=(224, 224, 3),
                 patch_size=16,
                 grid_size=14,
                 transformer_num_blocks=12,
                 transformer_hidden_size=16 * 16 * 3,
                 transformer_num_heads=12,
                 transformer_mlp_units=1024 * 3,
                 transformer_dropout_rate=0.1,
                 output_kernel_size=1,
                 num_classes=1,
                 hybrid=True,
                 num_skips=3,
                 upsample_channels: List[int] = (256, 128, 64, 16),
                 w=None,
                 encoder_trainable=True,
                 name='VisionTransformer'):
        assert input_shape[0] % patch_size == 0

        inputs = Input(shape=input_shape)
        inputs, outputs, skips, patch_size = self._build_hybrid_model(inputs=inputs,
                                                                      embedding_info=(patch_size, grid_size),
                                                                      hybrid_model_info=(hybrid, num_skips),
                                                                      trainable=encoder_trainable)
        outputs = self._build_transformer_encoder(outputs,
                                                  patch_size,
                                                  num_blocks=transformer_num_blocks,
                                                  hidden_size=transformer_hidden_size,
                                                  num_heads=transformer_num_heads,
                                                  mlp_units=transformer_mlp_units,
                                                  dropout_rate=transformer_dropout_rate,
                                                  trainable=encoder_trainable,
                                                  w=w)
        outputs = self._build_decoder(outputs, skips, upsample_channels, num_skips, num_classes, output_kernel_size)

        super(VisionTransformer, self).__init__(name=name, inputs=inputs, outputs=outputs)

    def _build_hybrid_model(self, inputs, embedding_info, hybrid_model_info, trainable):
        patch_size, grid_size = embedding_info
        hybrid, num_skips = hybrid_model_info

        input_shape = inputs.shape
        if len(input_shape) > 3:
            input_shape = input_shape[1:]

        if hybrid:
            patch_size = input_shape[0] // 16 // grid_size
            patch_size = max(1, patch_size)

            hybrid_model = applications.ResNet50V2(input_shape=input_shape, include_top=False)
            hybrid_model.trainable = trainable

            hybrid_model(inputs)  # evaluate

            skips = []
            if num_skips > 0:
                skips = [hybrid_model.get_layer(name).output for name in RESNET_SKIP_LAYER_NAMES]
            outputs = hybrid_model.get_layer(RESNET_BOTTLENECK_LAYER_NAME).output
            inputs = hybrid_model.input
        else:
            outputs = inputs
            skips = None

        return inputs, outputs, skips, patch_size

    def _build_transformer_encoder(self,
                                   inputs,
                                   patch_size,
                                   num_blocks, hidden_size, num_heads, mlp_units, dropout_rate,
                                   trainable, w):
        outputs = inputs
        outputs = Conv2D(filters=hidden_size,
                         kernel_size=patch_size,
                         strides=patch_size,
                         padding='valid',
                         name='embedding',
                         trainable=trainable,
                         kernel_initializer=ConstantInit(w['embedding/kernel']) if w is not None else 'glorot_uniform',
                         bias_initializer=ConstantInit(w['embedding/bias']) if w is not None else 'zeros')(outputs)
        outputs = Reshape((outputs.shape[1] * outputs.shape[2], hidden_size))(outputs)
        outputs = PositionEmbedding(name='Transformer/posembed_input', w=w, trainable=trainable)(outputs)
        outputs = Dropout(0.1)(outputs)

        for n in range(num_blocks):
            outputs, _ = TransformerBlock(num_heads, mlp_units, dropout_rate,
                                          name='Transformer/encoderblock_{}'.format(n),
                                          w=w,
                                          trainable=trainable)(outputs)

        layer_norm_name = 'Transformer/encoder_norm'
        outputs = LayerNormalization(epsilon=1e-6, name=layer_norm_name,
                                     gamma_initializer=ConstantInit(w['{}/scale'.format(layer_norm_name)]) if w is not None else 'ones',
                                     beta_initializer=ConstantInit(w['{}/bias'.format(layer_norm_name)]) if w is not None else 'zeros')(outputs)
        num_patch_sqrt = int(math.sqrt(outputs.shape[1]))
        outputs = Reshape((num_patch_sqrt, num_patch_sqrt, hidden_size))(outputs)
        return outputs

    def _build_decoder(self,
                       inputs,
                       skips,
                       decoder_channels,
                       num_skips,
                       num_classes,
                       output_kernel_size):
        outputs = inputs
        if decoder_channels is not None:
            outputs = DecoderCascadedUpSampling(decoder_channels, num_skips)({
                'hidden_states': outputs,
                'skips': skips
            })
        outputs = DecoderOutput(num_classes, output_kernel_size)(outputs)
        return outputs

    def call(self, inputs, training=None, mask=None):
        return super(VisionTransformer, self).call(inputs, training, mask)

    def get_config(self):
        return super(VisionTransformer, self).get_config()
