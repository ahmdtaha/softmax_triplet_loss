# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition of the Inception V4 architecture.

As described in http://arxiv.org/abs/1602.07261.

  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow as tf
import constants as const
# import configuration as config
import nets.batch_augment as batch_augment
import utils.os_utils as os_utils
import os

from nets import inception_utils

slim = tf.contrib.slim


def block_inception_a(inputs, scope=None, reuse=None):
  """Builds Inception-A block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockInceptionA', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 96, [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(branch_3, 96, [1, 1], scope='Conv2d_0b_1x1')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])


def block_reduction_a(inputs, scope=None, reuse=None):
  """Builds Reduction-A block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockReductionA', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 384, [3, 3], stride=2, padding='VALID',
                               scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
        branch_1 = slim.conv2d(branch_1, 256, [3, 3], stride=2,
                               padding='VALID', scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                   scope='MaxPool_1a_3x3')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])


def block_inception_b(inputs, scope=None, reuse=None):
  """Builds Inception-B block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockInceptionB', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 224, [1, 7], scope='Conv2d_0b_1x7')
        branch_1 = slim.conv2d(branch_1, 256, [7, 1], scope='Conv2d_0c_7x1')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
        branch_2 = slim.conv2d(branch_2, 224, [1, 7], scope='Conv2d_0c_1x7')
        branch_2 = slim.conv2d(branch_2, 224, [7, 1], scope='Conv2d_0d_7x1')
        branch_2 = slim.conv2d(branch_2, 256, [1, 7], scope='Conv2d_0e_1x7')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])


def block_reduction_b(inputs, scope=None, reuse=None):
  """Builds Reduction-B block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockReductionB', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
        branch_0 = slim.conv2d(branch_0, 192, [3, 3], stride=2,
                               padding='VALID', scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 256, [1, 7], scope='Conv2d_0b_1x7')
        branch_1 = slim.conv2d(branch_1, 320, [7, 1], scope='Conv2d_0c_7x1')
        branch_1 = slim.conv2d(branch_1, 320, [3, 3], stride=2,
                               padding='VALID', scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                   scope='MaxPool_1a_3x3')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])


def block_inception_c(inputs, scope=None, reuse=None):
  """Builds Inception-C block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockInceptionC', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = tf.concat(axis=3, values=[
            slim.conv2d(branch_1, 256, [1, 3], scope='Conv2d_0b_1x3'),
            slim.conv2d(branch_1, 256, [3, 1], scope='Conv2d_0c_3x1')])
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, 448, [3, 1], scope='Conv2d_0b_3x1')
        branch_2 = slim.conv2d(branch_2, 512, [1, 3], scope='Conv2d_0c_1x3')
        branch_2 = tf.concat(axis=3, values=[
            slim.conv2d(branch_2, 256, [1, 3], scope='Conv2d_0d_1x3'),
            slim.conv2d(branch_2, 256, [3, 1], scope='Conv2d_0e_3x1')])
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(branch_3, 256, [1, 1], scope='Conv2d_0b_1x1')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])


def inception_v4_base(inputs, final_endpoint='Mixed_7d', scope=None):
  """Creates the Inception V4 network up to the given final endpoint.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    final_endpoint: specifies the endpoint to construct the network up to.
      It can be one of [ 'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'Mixed_3a', 'Mixed_4a', 'Mixed_5a', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
      'Mixed_5e', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
      'Mixed_6f', 'Mixed_6g', 'Mixed_6h', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c',
      'Mixed_7d']
    scope: Optional variable_scope.

  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
  """
  end_points = {}

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.variable_scope(scope, 'InceptionV4', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # 299 x 299 x 3
      net = slim.conv2d(inputs, 32, [3, 3], stride=2,
                        padding='VALID', scope='Conv2d_1a_3x3')
      if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points
      # 149 x 149 x 32
      net = slim.conv2d(net, 32, [3, 3], padding='VALID',
                        scope='Conv2d_2a_3x3')
      if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
      # 147 x 147 x 32
      net = slim.conv2d(net, 64, [3, 3], scope='Conv2d_2b_3x3')
      if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
      # 147 x 147 x 64
      with tf.variable_scope('Mixed_3a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_0a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 96, [3, 3], stride=2, padding='VALID',
                                 scope='Conv2d_0a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1])
        if add_and_check_final('Mixed_3a', net): return net, end_points

      # 73 x 73 x 160
      with tf.variable_scope('Mixed_4a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, 96, [3, 3], padding='VALID',
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 64, [7, 1], scope='Conv2d_0c_7x1')
          branch_1 = slim.conv2d(branch_1, 96, [3, 3], padding='VALID',
                                 scope='Conv2d_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1])
        if add_and_check_final('Mixed_4a', net): return net, end_points

      # 71 x 71 x 192
      with tf.variable_scope('Mixed_5a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [3, 3], stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1])
        if add_and_check_final('Mixed_5a', net): return net, end_points

      # 35 x 35 x 384
      # 4 x Inception-A blocks
      for idx in range(4):
        block_scope = 'Mixed_5' + chr(ord('b') + idx)
        net = block_inception_a(net, block_scope)
        if add_and_check_final(block_scope, net): return net, end_points

      # 35 x 35 x 384
      # Reduction-A block
      net = block_reduction_a(net, 'Mixed_6a')
      if add_and_check_final('Mixed_6a', net): return net, end_points

      # 17 x 17 x 1024
      # 7 x Inception-B blocks
      for idx in range(7):
        block_scope = 'Mixed_6' + chr(ord('b') + idx)
        net = block_inception_b(net, block_scope)
        if add_and_check_final(block_scope, net): return net, end_points

      # 17 x 17 x 1024
      # Reduction-B block
      net = block_reduction_b(net, 'Mixed_7a')
      if add_and_check_final('Mixed_7a', net): return net, end_points

      # 8 x 8 x 1536
      # 3 x Inception-C blocks
      for idx in range(3):
        block_scope = 'Mixed_7' + chr(ord('b') + idx)
        net = block_inception_c(net, block_scope)
        if add_and_check_final(block_scope, net): return net, end_points
  raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v4(inputs, num_classes=1001, is_training=True,
                 dropout_keep_prob=0.8,
                 reuse=None,
                 scope='InceptionV4',
                 create_aux_logits=True):
  """Creates the Inception V4 model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    create_aux_logits: Whether to include the auxiliary logits.

  Returns:
    net: a Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped input to the logits layer
      if num_classes is 0 or None.
    end_points: the set of end_points from the inception model.
  """
  end_points = {}
  with tf.variable_scope(scope, 'InceptionV4', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v4_base(inputs, scope=scope)

      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'):
        # Auxiliary Head logits
        if create_aux_logits and num_classes:
          with tf.variable_scope('AuxLogits'):
            # 17 x 17 x 1024
            aux_logits = end_points['Mixed_6h']
            aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3,
                                         padding='VALID',
                                         scope='AvgPool_1a_5x5')
            aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                     scope='Conv2d_1b_1x1')
            aux_logits = slim.conv2d(aux_logits, 768,
                                     aux_logits.get_shape()[1:3],
                                     padding='VALID', scope='Conv2d_2a')
            aux_logits = slim.flatten(aux_logits)
            aux_logits = slim.fully_connected(aux_logits, num_classes,
                                              activation_fn=None,
                                              scope='Aux_logits')
            end_points['AuxLogits'] = aux_logits

        # Final pooling and prediction
        # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
        # can be set to False to disable pooling here (as in resnet_*()).
        with tf.variable_scope('Logits'):
          # 8 x 8 x 1536
          kernel_size = net.get_shape()[1:3]
          if kernel_size.is_fully_defined():
            net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                  scope='AvgPool_1a')
          else:
            net = tf.reduce_mean(net, [1, 2], keep_dims=True,
                                 name='global_pool')
          end_points['global_pool'] = net
          if not num_classes:
            return net, end_points
          # 1 x 1 x 1536
          net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
          net = slim.flatten(net, scope='PreLogitsFlatten')
          end_points['PreLogitsFlatten'] = net
          # 1536
          logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                        scope='Logits')
          end_points['Logits'] = logits
          end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
    return logits, end_points
inception_v4.default_image_size = 299


inception_v4_arg_scope = inception_utils.inception_arg_scope


class InceptionV4:

    def var_2_train(self):
        scopes = [scope.strip() for scope in 'InceptionV4/Logits'.split(',')]
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        print(variables_to_train)
        return variables_to_train;

    def resume_model(self,save_model_dir,ckpt_file,sess,saver):
        variables_to_restore = []
        exclusions = [scope.strip() for scope in '**'.split(',')]
        for var in tf.contrib.slim.get_model_variables():
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    break
            else:
                variables_to_restore.append(var)

        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(ckpt_file,
                                                                 variables_to_restore, ignore_missing_vars=False)
        init_fn(sess)

    def load_model(self,save_model_dir,ckpt_file,sess,saver,load_logits=False):
        if (os.path.exists(save_model_dir) and os_utils.chkpt_exists(save_model_dir)):
            # Try to restore everything if possible
            saver.restore(sess, ckpt_file)
            print('Model Loaded Normally');
            return 'Model Loaded Normally';
        else:
            print('Failed to Model Loaded Normally from ',ckpt_file);
            if (load_logits):
                exclusions = [scope.strip() for scope in '**'.split(',')]
            else:
                exclusions = [scope.strip() for scope in 'InceptionV4/Logits,InceptionV4/AuxLogits'.split(',')]
            # exclusions = [scope.strip() for scope in '**'.split(',')]
            variables_to_restore = []
            for var in tf.contrib.slim.get_model_variables():
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                        break
                else:
                    variables_to_restore.append(var)
            # print(variables_to_restore)
            init_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.cfg.imagenet__weights_filepath, variables_to_restore,ignore_missing_vars=False)
            # init_fn = tf.contrib.framework.assign_from_checkpoint_fn(config.imagenet__weights_filepath)
            init_fn(sess)
            print('Some variables loaded from imagenet')
            return 'Failed to Model Loaded Normally from '+ckpt_file


    def __init__(self, cfg=None, is_training=True,
                 dropout_keep_prob=0.8,
                 reuse=None,
                 scope='InceptionV4',
                 create_aux_logits=True,
                 images_ph=None,
                 lbls_ph=None
                 ):
        self.cfg = cfg
        batch_size = None
        num_classes = cfg.num_classes
        if lbls_ph is not None:
            self.gt_lbls = tf.reshape(lbls_ph, [-1, num_classes])
        else:
            self.gt_lbls = tf.placeholder(tf.int32, shape=(batch_size, num_classes), name='class_lbls')

        self.do_augmentation = tf.placeholder(tf.bool, name='do_augmentation')
        self.loss_class_weight = tf.placeholder(tf.float32, shape=(num_classes, num_classes), name='weights')
        if cfg.db_name == 'honda':
            self.input = tf.placeholder(tf.float32, shape=(batch_size, const.frame_height, const.frame_width,
                                                           const.context_channels), name='context_input')
        else:
            self.input = tf.placeholder(tf.float32, shape=(batch_size, const.max_frame_size, const.max_frame_size,
                                                           const.frame_channels), name='context_input')

        # if is_training:
        if images_ph is not None:
            self.input = images_ph
            _, w, h, c = self.input.shape
            aug_imgs = tf.reshape(self.input, [-1, w, h, 3])
            print('No nnutils Augmentation')
        else:
            if cfg.db_name == 'honda':
                aug_imgs = self.input
            else:
                aug_imgs = tf.cond(self.do_augmentation,
                                   lambda: batch_augment.augment(self.input,cfg.preprocess_func, horizontal_flip=True, vertical_flip=False,
                                                            rotate=0, crop_probability=0, color_aug_probability=0)
                                   , lambda: batch_augment.center_crop(self.input,cfg.preprocess_func))



        with slim.arg_scope(inception_v4_arg_scope()):
            _, train_end_points = inception_v4(aug_imgs, num_classes,
                                               dropout_keep_prob=dropout_keep_prob,
                                               create_aux_logits=create_aux_logits,
                                               is_training=True,reuse=reuse, scope=scope)

            _, val_end_points = inception_v4(aug_imgs, num_classes,
                                             dropout_keep_prob=dropout_keep_prob,
                                             create_aux_logits=create_aux_logits,
                                             is_training=False,reuse=True, scope=scope)


        def  cal_metrics(end_points):
            gt = tf.argmax(self.gt_lbls, 1);
            logits = tf.reshape(end_points['Logits'], [-1, num_classes])
            pre_logits = end_points['Mixed_6h']

            center_supervised_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.gt_lbls,
                                                                                         logits=logits,
                                                                                         name='xentropy_center')
            loss = tf.reduce_mean(center_supervised_cross_entropy, name='xentropy_mean')
            predictions = tf.reshape(end_points['Predictions'], [-1, num_classes])
            class_prediction = tf.argmax(predictions, 1)
            supervised_correct_prediction = tf.equal(gt, class_prediction)
            supervised_correct_prediction_cast = tf.cast(supervised_correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(supervised_correct_prediction_cast)
            confusion_mat = tf.confusion_matrix(gt, class_prediction, num_classes=num_classes)
            _, accumulated_accuracy = tf.metrics.accuracy(gt, class_prediction)
            _, per_class_acc_acc = tf.metrics.mean_per_class_accuracy(gt, class_prediction,num_classes=num_classes)
            per_class_acc_acc = tf.reduce_mean(per_class_acc_acc)
            return loss,pre_logits,accuracy,confusion_mat,accumulated_accuracy,per_class_acc_acc,class_prediction

        self.train_loss,self.train_pre_logits,self.train_accuracy,self.train_confusion_mat,\
                        self.train_accumulated_accuracy,self.train_per_class_acc_acc ,self.train_class_prediction = cal_metrics(train_end_points);


        self.val_loss,self.val_pre_logits,self.val_accuracy, self.val_confusion_mat,\
                        self.val_accumulated_accuracy,self.val_per_class_acc_acc ,self.val_class_prediction = cal_metrics(val_end_points);