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


def conv_bn_relu(inputs, conv_size, conv_filters, is_training, data_format):
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
      strides=(1, 1),
      use_bias=False, # TODO: check for bias in popular CNNs
      kernel_initializer='glorot_uniform',
      padding='same',
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.ReLU()(net)

  return net


def conv_bn_leaky_relu(inputs, conv_size, conv_filters, is_training, data_format):
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
      strides=(1, 1),
      use_bias=False, # TODO: check for bias in popular CNNs
      kernel_initializer='glorot_uniform',
      padding='same',
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(net)

  return net


def conv_dil_bn_relu(inputs, conv_size, conv_filters, dilation_rate=2, is_training, data_format):
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
      strides=(1, 1),
      dilation_rate=dilation_rate,
      use_bias=False, # TODO: check for bias in popular CNNs
      kernel_initializer='glorot_uniform',
      padding='same',
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.ReLU()(net)

  return net  


def conv_dep_bn_relu(inputs, conv_size, conv_filters, is_training, data_format):
  """Depthwise Convolution followed by batch norm and ReLU."""
  if data_format == 'channels_last':
    axis = 3
  elif data_format == 'channels_first':
    axis = 1
  else:
    raise ValueError('invalid data_format')

  assert dilation_rate >= 2, 'Dilation rate must be greater than 1. Else use conv_bn_relu()'

  net = tf.keras.layers.SeparableConv2D(
      filters=conv_filters, 
      kernel_size=conv_size,
      strides=(1, 1),
      dilation_rate=dilation_rate,
      use_bias=False, # TODO: check for bias in popular CNNs
      depthwise_initializer='glorot_uniform',
      pointwise_initializer='glorot_uniform',
      depth_multiplier=1, # TODO: check for depth_multiplier in popular CNNs
      padding='same',
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.ReLU()(net)

  return net  


def conv_gr_bn_relu(inputs, conv_size, conv_filters, groups, is_training, data_format):
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
      strides=(1, 1),
      use_bias=False, # TODO: check for bias in popular CNNs
      kernel_initializer='glorot_uniform',
      padding='same',
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.ReLU()(net)

  return net


def conv_3D_bn_relu(inputs, conv_size, conv_filters, is_training, data_format):
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
      strides=(1, 1, 1),
      use_bias=False, # TODO: check for bias in popular CNNs
      kernel_initializer='glorot_uniform',
      padding='valid',
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.ReLU()(net)

  return net


def conv_tr_bn_relu(inputs, conv_size, conv_filters, is_training, data_format):
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
      strides=(1, 1),
      use_bias=False, # TODO: check for bias in popular CNNs
      kernel_initializer='glorot_uniform',
      padding='valid', # TODO: check for padding in popular CNNs, might need to use padding before this layer
      data_format=data_format)(inputs)

  net = tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      trainable=is_training)(net)

  net = tf.keras.layers.ReLU()(net)

  return net


def channel_shuffle(x, groups):  
    _, width, height, channels = x.shape
    group_ch = channels // groups

    x = tf.rkeras.layers.Reshape([width, height, group_ch, groups])(x)
    x = tf.keras.layers.Permute([1, 2, 4, 3])(x)
    x = tf.keras.layers.Reshape([width, height, channels])(x)
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
  def build(self, inputs, channels):
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
        inputs, 11, channels, self.is_training, self.data_format)

    return net


class Conv7x7BnRelu(BaseOp):
  """7x7 convolution with batch norm and ReLU activation."""
  # TODO: add support for variable number of filters

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 7, channels, self.is_training, self.data_format)

    return net


class Conv5x5BnRelu(BaseOp):
  """5x5 convolution with batch norm and ReLU activation."""
  # TODO: add support for variable number of filters

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 5, channels, self.is_training, self.data_format)

    return net


class Conv3x3BnRelu(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""
  # TODO: add support for variable number of filters

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 3, channels, self.is_training, self.data_format)

    return net


class Conv1x1BnRelu(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""
  # TODO: add support for variable number of filters

  def build(self, inputs, channels):
    net = conv_bn_relu(
        inputs, 1, channels, self.is_training, self.data_format)

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


class GobalAvgPool(BaseOp):
  """global average pool with no subsampling."""

  def build(self, inputs, channels):
    del channels    # Unused
    net = tf.keras.layers.GlobalAveragePooling2D(
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
        data_format=self.data_format)(ipnuts)

    net = conv_bn_relu(net, 1, channels, self.is_training, self.data_format)

    return net


class Dropout(BaseOp):
  """3x3 max pool with no subsampling."""

  def build(self, inputs, channels):
    del channels    # Unused
    net = tf.keras.layers.Dropout(
        rate=DROPOUT_RATE)(inputs)

    return net


# Commas should not be used in op names
OP_MAP = {
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
