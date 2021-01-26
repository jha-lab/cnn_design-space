# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base operations used by the modules in this search space."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf

# Currently, only channels_last is well supported.
VALID_DATA_FORMATS = frozenset(['channels_last', 'channels_first'])
MIN_FILTERS = 8
BN_MOMENTUM = 0.997
BN_EPSILON = 1e-5
DROPOUT_RATE = 0.25
LEAKY_RELU_ALPHA = 0.3


def conv_bn_relu(inputs, conv_size, conv_filters, strides, is_training, data_format):
  """Convolution followed by batch norm and ReLU."""
  if data_format == 'channels_last':
    axis = 3
  elif data_format == 'channels_first':
    axis = 1
  else:
    raise ValueError('invalid data_format')

  net = tf.keras.layers.Conv2D(
      filters=conv_filters, 
      kernel_size=conv_size,
      strides=strides,
      use_bias=False, # bias already taken into account by batch-norm
      kernel_initializer='uniform',
      padding='same',
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.ReLU()(net)

  return net


def conv_bn_leaky_relu(inputs, conv_size, conv_filters, strides, is_training, data_format):
  """Convolution followed by batch norm and ReLU."""
  if data_format == 'channels_last':
    axis = 3
  elif data_format == 'channels_first':
    axis = 1
  else:
    raise ValueError('invalid data_format')

  net = tf.keras.layers.Conv2D(
      filters=conv_filters, 
      kernel_size=conv_size,
      strides=strides,
      use_bias=False, # bias already taken into account by batch-norm 
      kernel_initializer='uniform',
      padding='same',
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(net)

  return net


def conv_dil_bn_relu(inputs, conv_size, conv_filters, strides, dilation_rate, is_training, data_format):
  """Dilated Convolution followed by batch norm and ReLU."""
  if data_format == 'channels_last':
    axis = 3
  elif data_format == 'channels_first':
    axis = 1
  else:
    raise ValueError('invalid data_format')

  assert dilation_rate >= 2, 'Dilation rate must be greater than 1. Else use conv_bn_relu()'

  net = tf.keras.layers.Conv2D(
      filters=conv_filters, 
      kernel_size=conv_size,
      strides=strides,
      dilation_rate=dilation_rate,
      use_bias=False, # bias already taken into account by batch-norm 
      kernel_initializer='uniform',
      padding='same',
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.ReLU()(net)

  return net  


def conv_dep_bn_relu(inputs, conv_size, strides, is_training, data_format):
  """Depthwise Convolution followed by batch norm and ReLU."""
  if data_format == 'channels_last':
    axis = 3
  elif data_format == 'channels_first':
    axis = 1
  else:
    raise ValueError('invalid data_format')

  net = tf.keras.layers.DepthwiseConv2D(
      kernel_size=conv_size,
      strides=strides,
      use_bias=False, # bias already taken into account by batch-norm 
      depthwise_initializer='uniform',
      padding='same',
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.ReLU()(net)

  return net  


def conv_gr_bn_relu(inputs, conv_size, conv_filters, strides, groups, is_training, data_format):
  """Grouped Convolution followed by batch norm and ReLU."""
  if data_format == 'channels_last':
    axis = 3
  elif data_format == 'channels_first':
    axis = 1
  else:
    raise ValueError('invalid data_format')

  net = tf.keras.layers.Conv2D(
      filters=conv_filters, 
      kernel_size=conv_size,
      groups=groups,
      strides=strides,
      use_bias=False, # bias already taken into account by batch-norm 
      kernel_initializer='uniform',
      padding='same',
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.ReLU()(net)

  return net


def conv_3D_bn_relu(inputs, conv_size, conv_filters, strides, is_training, data_format):
  """3D Convolution followed by batch norm and ReLU."""
  # TODO: add support for separate channel and depth in input

  if data_format == 'channels_last':
    axis = 3
  elif data_format == 'channels_first':
    axis = 1
  else:
    raise ValueError('invalid data_format')

  net = tf.keras.layers.Conv3D(
      filters=conv_filters, 
      kernel_size=conv_size, # an integer or a tuple/list: (depth, height, width)
      strides=strides,
      use_bias=False, # bias already taken into account by batch-norm
      kernel_initializer='uniform',
      padding='valid',
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.ReLU()(net)

  return net


def conv_tr_bn_relu(inputs, conv_size, conv_filters, strides, is_training, data_format):
  """Transposed Convolution followed by batch norm and ReLU."""
  if data_format == 'channels_last':
    axis = 3
  elif data_format == 'channels_first':
    axis = 1
  else:
    raise ValueError('invalid data_format')

  net = tf.keras.layers.Conv2DTranspose(
      filters=conv_filters, 
      kernel_size=conv_size,
      strides=strides,
      use_bias=False, # bias already taken into account by batch-norm 
      kernel_initializer='uniform',
      padding='valid', # TODO: check for padding in popular CNNs, might need to use padding before this layer
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.ReLU()(net)

  return net


def channel_shuffle(x, groups, data_format='channels_last'):  
    """Channel shuffle based on groups"""
    if data_format == 'channels_last':
      _, height, width, channels = x.shape
      group_ch = channels // groups
      x = tf.keras.layers.Reshape([height, width, group_ch, groups])(x)
      x = tf.keras.layers.Permute([1, 2, 4, 3])(x)
      x = tf.keras.layers.Reshape([height, width, channels])(x)
    else:
      # TODO: check code for 'channels_first'
      _, channels, height, width = x.shape
      group_ch = channels // groups
      x = tf.keras.layers.Reshape([group_ch, groups, height, width])(x)
      x = tf.keras.layers.Permute([2, 1, 3, 4])(x)
      x = tf.keras.layers.Reshape([channels, height, width])(x)

    return x


class BaseOp(object):
  """Abstract base operation class."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, is_training, data_format='channels_last'):
    self.is_training = is_training
    if data_format.lower() not in VALID_DATA_FORMATS:
      raise ValueError('invalid data_format')
    self.data_format = data_format.lower()

  @abc.abstractmethod
  def build(self, inputs, channels): # TODO: implement without channels
    """Builds the operation with input tensors and returns an output tensor.
    Args:
      inputs: a 4-D Tensor.
      channels: int number of output channels of operation. The operation may
        choose to ignore this parameter.
    Returns:
      a 4-D Tensor with the same data format.
    """
    pass


class Identity(BaseOp):
  """Identity operation (ignores channels)."""

  def build(self, inputs, channels):
    del channels    # Unused
    return tf.identity(inputs, name='identity')


class Conv11x11BnRelu(BaseOp):
  """11x11 convolution with batch norm and ReLU activation."""
  # TODO: add support for variable number of filters

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 11, channels, (1, 1), self.is_training, self.data_format)

    return net


class Conv7x7BnRelu(BaseOp):
  """7x7 convolution with batch norm and ReLU activation."""
  # TODO: add support for variable number of filters

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 7, channels, (1, 1), self.is_training, self.data_format)

    return net


class Conv5x5BnRelu(BaseOp):
  """5x5 convolution with batch norm and ReLU activation."""
  # TODO: add support for variable number of filters

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 5, channels, (1, 1), self.is_training, self.data_format)

    return net


class Conv3x3BnRelu(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""
  # TODO: add support for variable number of filters

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, channels, (1, 1), self.is_training, self.data_format)

    return net


class Conv1x1BnRelu(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""
  # TODO: add support for variable number of filters

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, channels, (1, 1), self.is_training, self.data_format)

    return net


class MaxPool3x3(BaseOp):
  """3x3 max pool with no subsampling."""

  def build(self, inputs, channels):
    del channels    # Unused
    net = tf.keras.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same',
        data_format=self.data_format)(inputs)

    return net


class AvgPool3x3(BaseOp):
  """3x3 average pool with no subsampling."""

  def build(self, inputs, channels):
    del channels    # Unused
    net = tf.keras.layers.AveragePooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same',
        data_format=self.data_format)(inputs)

    return net


class GlobalAvgPool(BaseOp):
  """global average pool with no subsampling."""

  def build(self, inputs, channels):
    del channels    # Unused
    net = tf.keras.layers.GlobalAveragePooling2D(
        data_format=self.data_format)(inputs)

    return net


class Flatten(BaseOp):
  """simple flattening layer"""

  def build(self, inputs, channels):
    del channels    # Unused
    net = tf.keras.layers.Flatten(
        data_format=self.data_format)(inputs)

    return net


class ZeroPadding(BaseOp):
  """3x3 max pool with no subsampling."""

  def build(self, inputs, channels):
    del channels    # Unused
    net = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), # TODO: add support for multiple paddings
        data_format=self.data_format)(inputs)

    return net


class BottleneckConv3x3(BaseOp):
  """[1x1(/4)]+3x3+[1x1(*4)] conv. Uses BN + ReLU post-activation."""
  # TODO: verify this block can reproduce results of ResNet-50.

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, channels // 4, self.is_training, self.data_format)
    net = conv_bn_relu(
        net, 3, channels // 4, self.is_training, self.data_format)
    net = conv_bn_relu(
        net, 1, channels, self.is_training, self.data_format)

    return net


class BottleneckConv5x5(BaseOp):
  """[1x1(/4)]+5x5+[1x1(*4)] conv. Uses BN + ReLU post-activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, channels // 4, self.is_training, self.data_format)
    net = conv_bn_relu(
        net, 5, channels // 4, self.is_training, self.data_format)
    net = conv_bn_relu(
        net, 1, channels, self.is_training, self.data_format)

    return net


class MaxPool3x3Conv1x1(BaseOp):
  """3x3 max pool with no subsampling followed by 1x1 for rescaling."""

  def build(self, inputs, channels):
    net = tf.keras.layers.MaxPool3D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same',
        data_format=self.data_format)(inputs)

    net = conv_bn_relu(net, 1, channels, self.is_training, self.data_format)

    return net


## Specialized base operations with channels pre-defined

## Base operations for AlexNet
class Conv11x11BnRelu_F96_S4(BaseOp):
  """11x11 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 11, 96, (4, 4), self.is_training, self.data_format)

    return net

class MaxPool3x3_S2(BaseOp):
  """3x3 max pool with no subsampling."""

  def build(self, inputs, channels):
    net = tf.keras.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same',
        data_format=self.data_format)(inputs)

    return net

class Conv5x5BnRelu_F256(BaseOp):
  """5x5 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 5, 256, (1, 1), self.is_training, self.data_format)

    return net

class Conv3x3BnRelu_F384(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 384, (1, 1), self.is_training, self.data_format)

    return net

class Conv3x3BnRelu_F256(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 256, (1, 1), self.is_training, self.data_format)

    return net


## Base operations for ResNet-50
class Conv7x7BnRelu_F64_S2(BaseOp):
  """7x7 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 7, 64, (2, 2), self.is_training, self.data_format)

    return net

# conv2_x block
class Conv1x1BnRelu_F64(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 64, (1, 1), self.is_training, self.data_format)

    return net

class Conv3x3BnRelu_F64(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 64, (1, 1), self.is_training, self.data_format)

    return net

class Conv1x1BnRelu_F256(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 256, (1, 1), self.is_training, self.data_format)

    return net

# conv3_x block
# First downsampling CONV layer (left stream)
class Conv1x1BnRelu_F128_S2(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 128, (2, 2), self.is_training, self.data_format)

    return net

class Conv3x3BnRelu_F128(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 128, (1, 1), self.is_training, self.data_format)

    return net

class Conv1x1BnRelu_F512(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 512, (1, 1), self.is_training, self.data_format)

    return net

# 512-dimentional projection shortcut (right stream)
class Conv1x1BnRelu_F512_S2(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 512, (2, 2), self.is_training, self.data_format)

    return net

# conv4_x block
# First downsampling CONV layer (left stream)
class Conv1x1BnRelu_F256_S2(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 256, (2, 2), self.is_training, self.data_format)

    return net

class Conv3x3BnRelu_F256(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 256, (1, 1), self.is_training, self.data_format)

    return net

class Conv1x1BnRelu_F1024(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 1024, (1, 1), self.is_training, self.data_format)

    return net

# 1024-dimentional projection shortcut (right stream)
class Conv1x1BnRelu_F1024_S2(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 1024, (2, 2), self.is_training, self.data_format)

    return net

# conv5_x block
# First downsampling CONV layer (left stream)
# class Conv1x1BnRelu_F512_S2(BaseOp)

class Conv3x3BnRelu_F512(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 512, (1, 1), self.is_training, self.data_format)

    return net

class Conv1x1BnRelu_F2048(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 2048, (1, 1), self.is_training, self.data_format)

    return net

# 2048-dimentional projection shortcut (right stream)
class Conv1x1BnRelu_F2048_S2(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 2048, (2, 2), self.is_training, self.data_format)

    return net

# class Flatten(BaseOp)

class Dense4096_ReLU(BaseOp):
  """dense layer with ReLU activation."""

  def build(self, inputs, channels):
    del channels
    net = tf.keras.layers.Dense(
        units=4096, activation='relu', kernel_initializer='uniform')(inputs)

    return net

class Dropout_p5(BaseOp):
  """dropout layer."""

  def build(self, inputs, channels):
    del channels    # Unused
    net = tf.keras.layers.Dropout(
        rate=0.5)(inputs)

    return net


## Base operations for DenseNet-121
# class Conv7x7BnRelu_F64_S2(BaseOp)

# class MaxPool3x3_S2(BaseOp)

# dense block
class Conv1x1BnRelu_F128(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 128, (1, 1), self.is_training, self.data_format)

    return net

class Conv3x3BnRelu_F32(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 512, (1, 1), self.is_training, self.data_format)

    return net

# transition block (80 filters for k=32 and theta=0.5)
class Conv1x1BnRelu_F80(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 80, (1, 1), self.is_training, self.data_format)

    return net

class AvgPool2x2_S2(BaseOp):
  """2x2 average pool with no subsampling."""

  def build(self, inputs, channels):
    del channels    # Unused
    net = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        data_format=self.data_format)(inputs)

    return net

# class GlobalAvgPool(BaseOp)


## Base operations for GoogLeNet
# class Conv7x7BnRelu_F64_S2(BaseOp)

# class MaxPool3x3_S2(BaseOp)

# class Conv1x1BnRelu_F64(BaseOp)

class Conv3x3BnRelu_F192(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 192, (1, 1), self.is_training, self.data_format)

    return net

# inception (3a)
# class Conv1x1BnRelu_F64(BaseOp)

class Conv1x1BnRelu_F96(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 96, (1, 1), self.is_training, self.data_format)

    return net

# class Conv3x3BnRelu_F128(BaseOp)

class Conv1x1BnRelu_F16(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 16, (1, 1), self.is_training, self.data_format)

    return net

class Conv5x5BnRelu_F32(BaseOp):
  """5x5 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 5, 32, (1, 1), self.is_training, self.data_format)

    return net

# MaxPool3x3(BaseOp)

class Conv1x1BnRelu_F32(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 32, (1, 1), self.is_training, self.data_format)

    return net

# inception (3b)
# class Conv1x1BnRelu_F128(BaseOp)

# class Conv1x1BnRelu_F128(BaseOp)

class Conv3x3BnRelu_F192(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 192, (1, 1), self.is_training, self.data_format)

    return net

# class Conv1x1BnRelu_F32(BaseOp)

class Conv5x5BnRelu_F96(BaseOp):
  """5x5 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 5, 96, (1, 1), self.is_training, self.data_format)

    return net

# MaxPool3x3(BaseOp)

# class Conv1x1BnRelu_F64(BaseOp)

# inception (4a)
class Conv1x1BnRelu_F192(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 192, (1, 1), self.is_training, self.data_format)

    return net

# class Conv1x1BnRelu_F96(BaseOp)

class Conv3x3BnRelu_F208(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 208, (1, 1), self.is_training, self.data_format)

    return net

# class Conv1x1BnRelu_F16(BaseOp)

class Conv5x5BnRelu_F48(BaseOp):
  """5x5 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 5, 48, (1, 1), self.is_training, self.data_format)

    return net

# MaxPool3x3(BaseOp)

# class Conv1x1BnRelu_F64(BaseOp)

# inception (4b)
class Conv1x1BnRelu_F160(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 160, (1, 1), self.is_training, self.data_format)

    return net

class Conv1x1BnRelu_F112(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 112, (1, 1), self.is_training, self.data_format)

    return net

class Conv3x3BnRelu_F224(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 224, (1, 1), self.is_training, self.data_format)

    return net

class Conv1x1BnRelu_F24(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 24, (1, 1), self.is_training, self.data_format)

    return net

class Conv5x5BnRelu_F64(BaseOp):
  """5x5 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 5, 64, (1, 1), self.is_training, self.data_format)

    return net

# MaxPool3x3(BaseOp)

# class Conv1x1BnRelu_F64(BaseOp)

# inception (4c)
# class Conv1x1BnRelu_F128(BaseOp)

# class Conv1x1BnRelu_F128(BaseOp)

# class Conv3x3BnRelu_F256(BaseOp)

# class Conv1x1BnRelu_F24(BaseOp)

# class Conv5x5BnRelu_F64(BaseOp)

# MaxPool3x3(BaseOp)

# class Conv1x1BnRelu_F64(BaseOp)

# inception (4d)
class Conv1x1BnRelu_F112(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 112, (1, 1), self.is_training, self.data_format)

    return net

class Conv1x1BnRelu_F144(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 144, (1, 1), self.is_training, self.data_format)

    return net

class Conv3x3BnRelu_F288(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 288, (1, 1), self.is_training, self.data_format)

    return net

# class Conv1x1BnRelu_F32(BaseOp)

# class Conv5x5BnRelu_F64(BaseOp)

# MaxPool3x3(BaseOp)

# class Conv1x1BnRelu_F64(BaseOp)

# inception (4e) and (5a)
# class Conv1x1BnRelu_F256(BaseOp)

# class Conv1x1BnRelu_F160(BaseOp)

class Conv3x3BnRelu_F320(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 320, (1, 1), self.is_training, self.data_format)

    return net

# class Conv1x1BnRelu_F32(BaseOp)

class Conv5x5BnRelu_F128(BaseOp):
  """5x5 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 5, 128, (1, 1), self.is_training, self.data_format)

    return net

# MaxPool3x3(BaseOp)

# class Conv1x1BnRelu_F128(BaseOp)


# inception (5b)
class Conv1x1BnRelu_F384(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 384, (1, 1), self.is_training, self.data_format)

    return net

# class Conv1x1BnRelu_F192(BaseOp)

# class Conv3x3BnRelu_F384(BaseOp)

class Conv1x1BnRelu_F48(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 49, (1, 1), self.is_training, self.data_format)

    return net

# class Conv5x5BnRelu_F128(BaseOp)

# MaxPool3x3(BaseOp)

# class Conv1x1BnRelu_F128(BaseOp)

class AvgPool7x7(BaseOp):
  """7x7 average pool with no subsampling."""

  def build(self, inputs, channels):
    del channels    # Unused
    net = tf.keras.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='same',
        data_format=self.data_format)(inputs)

    return net

class Dropout_p4(BaseOp):
  """dropout layer"""

  def build(self, inputs, channels):
    del channels    # Unused
    net = tf.keras.layers.Dropout(
        rate=0.4)(inputs)

    return net

# class GlobalAvgPool(BaseOp)


## Base operations for MobileNet
class Conv3x3BnRelu_F32_S2(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 32, (2, 2), self.is_training, self.data_format)

    return net

class DepthwiseConv3x3BnRelu(BaseOp):
  """3x3 depthwise-convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_dep_bn_relu(
        inputs, 3, (1, 1), self.is_training, self.data_format)

    return net

# class Conv1x1BnRelu_F64(BaseOp)

class DepthwiseConv3x3BnRelu_S2(BaseOp):
  """3x3 depthwise-convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_dep_bn_relu(
        inputs, 3, (2, 2), self.is_training, self.data_format)

    return net

# class Conv1x1BnRelu_F128(BaseOp)

# class DepthwiseConv3x3BnRelu(BaseOp)

# class Conv1x1BnRelu_F128(BaseOp)

# class DepthwiseConv3x3BnRelu_S2(BaseOp)

# class Conv1x1BnRelu_256(BaseOp)

# class DepthwiseConv3x3BnRelu(BaseOp)

# class Conv1x1BnRelu_F256(BaseOp)

# class DepthwiseConv3x3BnRelu_S2(BaseOp)

# class Conv1x1BnRelu_F512(BaseOp)

# class DepthwiseConv3x3BnRelu(BaseOp)

# class Conv1x1BnRelu_F512(BaseOp)

# class DepthwiseConv3x3BnRelu_S2(BaseOp)

# class Conv1x1BnRelu_F1024(BaseOp)

# class DepthwiseConv3x3BnRelu(BaseOp)

# class Conv1x1BnRelu_F1024(BaseOp)

# class GlobalAvgPool(BaseOp)


## Base operations for ShuffleNet (g=8)
class Conv3x3BnRelu_F24_S2(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 24, (2, 2), self.is_training, self.data_format)

    return net

class GroupedConv1x1BnRelu_G8_F96(BaseOp):
  """1x1 grouped-convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_gr_bn_relu(
        inputs, 1, 96, (1, 1), 8, self.is_training, self.data_format)

    return net

class ChannelShuffle_G8(BaseOp):
  """1x1 grouped-convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = channel_shuffle(
        inputs, 8, self.data_format)

    return net

# class DepthwiseConv3x3BnRelu_S2(BaseOp)

class GroupedConv1x1BnRelu_G8_F360(BaseOp):
  """1x1 grouped-convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_gr_bn_relu(
        inputs, 1, 360, (1, 1), 8, self.is_training, self.data_format)

    return net

class GroupedConv1x1BnRelu_G8_F384(BaseOp):
  """1x1 grouped-convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_gr_bn_relu(
        inputs, 1, 384, (1, 1), 8, self.is_training, self.data_format)

    return net

class AvgPool3x3_S2(BaseOp):
  """3x3 average pool with no subsampling."""

  def build(self, inputs, channels):
    del channels    # Unused
    net = tf.keras.layers.AveragePooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same',
        data_format=self.data_format)(inputs)

    return net

class GroupedConv1x1BnRelu_G8_F192(BaseOp):
  """1x1 grouped-convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_gr_bn_relu(
        inputs, 1, 192, (1, 1), 8, self.is_training, self.data_format)

    return net

# class ChannelShuffle_G8(BaseOp)

# class DepthwiseConv3x3BnRelu_S2(BaseOp)

class GroupedConv1x1BnRelu_G8_F384(BaseOp):
  """1x1 grouped-convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_gr_bn_relu(
        inputs, 1, 384, (1, 1), 8, self.is_training, self.data_format)

    return net

class GroupedConv1x1BnRelu_G8_F762(BaseOp):
  """1x1 grouped-convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_gr_bn_relu(
        inputs, 1, 762, (1, 1), 8, self.is_training, self.data_format)

    return net

# class AvgPool3x3_S2(BaseOp)

# class GroupedConv1x1BnRelu_G8_F384(BaseOp)

# class ChannelShuffle_G8(BaseOp)

# class DepthwiseConv3x3BnRelu_S2(BaseOp)

# class GroupedConv1x1BnRelu_G8_F762(BaseOp)

class GroupedConv1x1BnRelu_G8_F1536(BaseOp):
  """1x1 grouped-convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_gr_bn_relu(
        inputs, 1, 1536, (1, 1), 8, self.is_training, self.data_format)

    return net

# class AvgPool3x3_S2(BaseOp)

# class GlobalAvgPool(BaseOp)


## Base operations for Xception
"""We use combination of DepthwiseConvBnRelu and Conv1x1BnRelu
as a substitute to the SeparableConv used in the paper"""
# class Conv3x3BnRelu_F32_S2(BaseOp)

# class Conv3x3BnRelu_F64(BaseOp)

# class DepthwiseConv3x3BnRelu(BaseOp)

# class Conv1x1BnRelu_F128(BaseOp)

# class MaxPool3x3_S2(BaseOp)

# class Conv1x1BnRelu_F128_S2(BaseOp)

# class DepthwiseConv3x3BnRelu(BaseOp)

# class Conv1x1BnRelu_F256(BaseOp)

# class DepthwiseConv3x3BnRelu(BaseOp)

# class Conv1x1BnRelu_F256(BaseOp)

# class MaxPool3x3_S2(BaseOp)

# class Conv1x1BnRelu_F256_S2(BaseOp)

# class DepthwiseConv3x3BnRelu(BaseOp)

class Conv1x1BnRelu_F728(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 728, (1, 1), self.is_training, self.data_format)

    return net

# class DepthwiseConv3x3BnRelu(BaseOp)

# class Conv1x1BnRelu_F728(BaseOp)

# class MaxPool3x3_S2(BaseOp)

# class Conv1x1BnRelu_F728(BaseOp)

# middle_flow uses SeparableConv with 728 filters and 3x3 kernels

class Conv1x1BnRelu_F1536(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, 1536, (1, 1), self.is_training, self.data_format)

    return net

# class Conv1x1BnRelu_F2048(BaseOp)

# class GlobalAvgPool(BaseOp)


## Base operations for SqueezeNet
class Conv7x7BnRelu_F96(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 7, 96, (1, 1), self.is_training, self.data_format)

    return net

# class MaxPool3x3_S2(BaseOp)

# class Conv1x1BnRelu_F16(BaseOp)

# class Conv1x1BnRelu_F64(BaseOp)

# class Conv3x3BnRelu_F64(BaseOp)

# class Conv1x1BnRelu_F32(BaseOp)

# class Conv1x1BnRelu_F128(BaseOp)

# class Conv3x3BnRelu_F128(BaseOp)

# class MaxPool3x3_S2(BaseOp)

# class Conv1x1BnRelu_F48(BaseOp)

# class Conv1x1BnRelu_F192(BaseOp)

# class Conv3x3BnRelu_F192(BaseOp)

# class Conv1x1BnRelu_F64(BaseOp)

# class Conv1x1BnRelu_F256(BaseOp)

# class Conv3x3BnRelu_F256(BaseOp)

# class MaxPool3x3_S2(BaseOp)

# class GlobalAvgPool(BaseOp)


# Base operations for VGG
# class Conv3x3BnRelu_F64(BaseOp)

class MaxPool2x2_S2(BaseOp):
  """2x2 max pool with no subsampling."""

  def build(self, inputs, channels):
    net = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        data_format=self.data_format)(inputs)

    return net

# class Conv3x3BnRelu_F256(BaseOp)

# class MaxPool2x2_S2(BaseOp)

# class Conv3x3BnRelu_F512(BaseOp)

# class Flatten(BaseOp)

# class Dense4096_ReLU(BaseOp)


## Base operations for LeNet
class Conv3x3BnRelu_F6(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 6, (1, 1), self.is_training, self.data_format)

    return net

class AvgPool2x2(BaseOp):
  """3x3 average pool with no subsampling."""

  def build(self, inputs, channels):
    del channels    # Unused
    net = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(1, 1),
        padding='same',
        data_format=self.data_format)(inputs)

    return net

class Conv3x3BnRelu_F16(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, 16, (1, 1), self.is_training, self.data_format)

    return net

# class AvgPool2x2(BaseOp)

class Dense120_ReLU(BaseOp):
  """dense layer with ReLU activation."""

  def build(self, inputs, channels):
    del channels
    net = tf.keras.layers.Dense(
        units=120, activation='relu', kernel_initializer='uniform')(inputs)

    return net

class Dense84_ReLU(BaseOp):
  """dense layer with ReLU activation."""

  def build(self, inputs, channels):
    del channels
    net = tf.keras.layers.Dense(
        units=84, activation='relu', kernel_initializer='uniform')(inputs)

    return net

# Commas should not be used in op names
CONV_MAP = {
    'identity': Identity,
    'conv11x11-bn-relu': Conv11x11BnRelu,
    'conv7x7-bn-relu': Conv7x7BnRelu,
    'conv5x5-bn-relu': Conv5x5BnRelu,
    'conv3x3-bn-relu': Conv3x3BnRelu,
    'conv1x1-bn-relu': Conv1x1BnRelu,
    'maxpool3x3': MaxPool3x3,
    'bottleneck3x3': BottleneckConv3x3,
    'bottleneck5x5': BottleneckConv5x5,
    'maxpool3x3-conv1x1': MaxPool3x3Conv1x1,
}

DENSE_MAP = {
    'global-avg-pool': GlobalAvgPool,
    'flatten': Flatten,
    'dense4096-relu': Dense4096_ReLU,
    'dense120-relu': Dense120_ReLU,
    'dense84-relu': Dense84_ReLU,
    'dropout-p5': Dropout_p5,
    'dropout-p4': Dropout_p4,
}