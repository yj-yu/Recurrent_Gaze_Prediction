#-*- coding: utf-8 -*-

"""
model_grcn.py

Implement a simple recurrent gaze prediction model based on GRU+RCN.
"""

import numpy as np
import os
import sys
import time

from PIL import Image
import tensorflow as tf
rnn_cell = tf.nn.rnn_cell

from tensorflow.contrib.layers.python.layers import (
    initializers,
    convolution2d, fully_connected
)

from collections import OrderedDict
import cPickle as pkl
# import crc_input_data_seq not used anymore
# append parent directory to pathPp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import log, override
from models.base import ModelBase, BaseModelConfig

from models.model_util import tf_normalize_map, normalize_probability_map
from models.model_util import tf_softmax_2d, tf_softmax_cross_entropy_with_logits_2d
from evaluation_metrics import saliency_score, AVAILABLE_METRICS

from models.gaze_rnn import GazePredictionGRU, GRUModelConfig

from easydict import EasyDict as E

CONSTANTS = E()
CONSTANTS.image_width = 98
CONSTANTS.image_height = 98
CONSTANTS.gazemap_width = 49
CONSTANTS.gazemap_height = 49
CONSTANTS.saliencymap_width = 49
CONSTANTS.saliencymap_height = 49


class GRU_RCN_Cell(rnn_cell.RNNCell):
    """Gated Recurrent Unit RCN cell. http://arxiv.org/abs/1511.06432  """
    """N.Ballas et al. DELVING DEEPER INTO CONVOLUTIONAL NETWORKS """

    def __init__(self,
                 num_units,
                 dim_feature,
                 spatial_shape=[7, 7],
                 kernel_spatial_shape=[3, 3],
                 ):
        self.spatial_H, self.spatial_W = spatial_shape
        assert self.spatial_H > 0 and self.spatial_W > 0
        kernel_H, kernel_W = kernel_spatial_shape

        self._num_units = num_units
        self.dim_feature = dim_feature
        self.W_z = tf.Variable(tf.truncated_normal([kernel_H, kernel_W, dim_feature, self._num_units],
                                                   dtype=tf.float32, stddev=1e-4),
                               name='GRU_Conv_Wz')
        self.U_z = tf.Variable(tf.truncated_normal([kernel_H, kernel_W, self._num_units, self._num_units],
                                                   dtype=tf.float32, stddev=1e-4),
                               name='GRU_Conv_Uz')
        self.W_r = tf.Variable(tf.truncated_normal([kernel_H, kernel_W, dim_feature, self._num_units],
                                                   dtype=tf.float32, stddev=1e-4),
                               name='GRU_Conv_Wr')
        self.U_r = tf.Variable(tf.truncated_normal([kernel_H, kernel_W, self._num_units, self._num_units],
                                                   dtype=tf.float32, stddev=1e-4),
                               name='GRU_Conv_Ur')
        self.W = tf.Variable(tf.truncated_normal([kernel_H, kernel_W, dim_feature, self._num_units],
                                                 dtype=tf.float32, stddev=1e-4),
                             name='GRU_Conv_W')
        self.U = tf.Variable(tf.truncated_normal([kernel_H, kernel_W, self._num_units, self._num_units],
                                                 dtype=tf.float32, stddev=1e-4),
                             name='GRU_Conv_U')

    @property
    def input_size(self):
        return self._num_units  # ?????????????????????????????????????????

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        B, H, W, C = inputs.get_shape().as_list()
        assert H == self.spatial_H and W == self.spatial_W, \
            ("input tensor dimension mismatch : inputs [%s, %s] != expected [%s, %s]" % (
                H, W, self.spatial_H, self.spatial_W)
             )

        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not udpate.
                # k1 x k2 x Ox x Oh where k1 x k2 is the convolutional kernel spatial
                # size.
                Wzconv = tf.nn.conv2d(
                    inputs, self.W_z, [1, 1, 1, 1], padding='SAME')
                Uzconv = tf.nn.conv2d(
                    state, self.U_z, [1, 1, 1, 1], padding='SAME')
                Wrconv = tf.nn.conv2d(
                    inputs, self.W_r, [1, 1, 1, 1], padding='SAME')
                Urconv = tf.nn.conv2d(
                    state, self.U_r, [1, 1, 1, 1], padding='SAME')
                # sig(W_r * x_t + U_r * h_t-1 )

                u = tf.sigmoid(Wzconv + Uzconv)
                r = tf.sigmoid(Wrconv + Urconv)
                with tf.variable_scope("Candidate"):
                    # tanh(W * x_t + U * (r_t dot h_t-1) not confident yet.
                    Wconv = tf.nn.conv2d(
                        inputs, self.W, [1, 1, 1, 1], padding='SAME')
                    Uconv = tf.nn.conv2d(
                        r * state, self.U, [1, 1, 1, 1], padding='SAME')
                    c = tf.tanh(tf.add(Wconv, Uconv))
                    new_h = u * state + (1 - u) * c
                    # output, state size, H=7, W=7, num_units)
                    return new_h, new_h

    # this might be really useful for the idenity rnn...
    def zero_state(self, batch_size, dtype):
        """Return state tensor (shape [batch_size x 7 x 7 x state_size]) filled with 0.

        Args:
            batch_size: int, float, or unit Tensor representing the batch size.
            dtype: the data type to use for the state.

        Returns:
            A 4D Tensor of shape [batch_size x state_size] filled with zeros.
        """
        zeros = tf.zeros(tf.stack([batch_size, self.spatial_H, self.spatial_W,
                                   self.state_size]), dtype=dtype)  # tf.pack converts list to numpy matrix
        zeros.set_shape(
            [None, self.spatial_H, self.spatial_W, self.state_size])
        return zeros


##########################################
##########################################

class GazePredictionGRCN(GazePredictionGRU):

    """
    Just simply extend GazePredictionGRU.
    Only differs build_model() part as of now.
    """

    def __init__(self,
                 session,
                 data_sets,
                 config=GRUModelConfig()
                 ):

        super(GazePredictionGRCN, self).__init__(session,  # python 3 super().methodname
                                                 data_sets,
                                                 config=config)

        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                [],
                                                name='placeholder_dropout_keep_prob')

    @staticmethod
    def create_gazeprediction_network(frame_images, c3d_input,
                                      dropout_keep_prob=1.0,
                                      net=None):
        '''
        Args:d
            frame_images: a [B x T x IH x IW x 3] tensor (frame images)
            c3d_input : a [B x T x 1024 x 7 x 7] tensor for C3D convmap features
            gt_gazemap : a [B x T x GH x GW] tensor of ground truth per-frame gaze maps
            dropout_keep_prob : float tensor
            (optional) net : a dictionary to get intra-layer activations or tensors.

        Outputs:
            predicted_gazemaps : a [B x T x GH x GW] tensor,
                predicted gaze maps per frame
        '''

        if net is None:
            net = {}
        else:
            assert isinstance(net, dict)

        vars = E()

        # (0) input sanity check
        GH, GW = CONSTANTS.gazemap_height, CONSTANTS.gazemap_width
        IH, IW = CONSTANTS.image_height, CONSTANTS.image_width
        B, T = frame_images.get_shape().as_list()[:2]

        assert B > 0 and T > 0
        frame_images.get_shape().assert_is_compatible_with([B, T, IH, IW, 3])
        c3d_input.get_shape().assert_is_compatible_with([B, T, 1024, 7, 7])

        dim_cnn_proj = 512  # XXX FIXME (see __init__ in GazePredictionGRU)

        # some variables
        # --------------
        # not a proper name, it should be rnn_state_feature_size in # GRCN????????? FIXME
        rnn_state_size = 128  # dim_cnn_proj # filter size is more correct name

        ''' The RGP (Recurrent Gaze Prediction) model. '''

        with tf.variable_scope("RGP"):

            # (2) C3D
            # -------
            # a. reduce filter size [7 x 7 x 1024] -> [7 x 7 x 32] via FC or CONV
            # b. apply RCN, and get the [7 x 7 x 32] outputs from RNN

            # c3d input.
            net['c3d_input'] = c3d_input   # [B x T x 1024 x 7 x 7]
            # change axis and reshape to [B x T x 7 x 7 x 1024]
            net['c3d_input_reshape'] = tf.transpose(net['c3d_input'],
                                                    perm=[0, 1, 3, 4, 2],
                                                    name='c3d_input_reshape')
            log.info('c3d_input_reshape shape : %s',
                     net['c3d_input_reshape'].get_shape().as_list())
            net['c3d_input_reshape'].get_shape().assert_is_compatible_with([
                B, T, 7, 7, 1024])

            # c3d_embedded: project each 1024 feature (per 7x7 c3d conv-feature map) into 12
            vars.proj_c3d_W = tf.Variable(tf.random_uniform(
                [1024, dim_cnn_proj], -0.1, 0.1), name="proj_c3d_W")
            vars.proj_c3d_b = tf.Variable(tf.random_uniform(
                [dim_cnn_proj], -0.1, 0.1), name="proj_c3d_b")

            net['c3d_embedded'] = tf.nn.xw_plus_b(
                tf.reshape(net['c3d_input_reshape'], [-1, 1024]),
                vars.proj_c3d_W, vars.proj_c3d_b
            )  # [(B*T*7*7) x 1024] --> [(B*T*7*7) x 12] by appling W:1024->12

            if dropout_keep_prob != 1.0:
                net['c3d_embedded'] = tf.nn.dropout(
                    net['c3d_embedded'], dropout_keep_prob)

            # --> [B x T x 7 x 7 x 12]
            net['c3d_embedded'] = tf.reshape(
                net['c3d_embedded'], [B, T, 7, 7, dim_cnn_proj])
            log.info('c3d_embedded shape : %s',
                     net['c3d_embedded'].get_shape().as_list())
            net['c3d_embedded'].get_shape().assert_is_compatible_with([
                B, T, 7, 7, dim_cnn_proj])

            # The RNN Part.
            # -------------

            with tf.variable_scope('RCNBottom') as scope:
                vars.lstm_u = GRU_RCN_Cell(rnn_state_size, dim_cnn_proj)

                state_u = vars.lstm_u.zero_state(B, tf.float32)
                log.info('RNN state shape : %s', state_u.get_shape().as_list())

                predicted_gazemaps = []
                net['rcn_outputs'] = rcn_outputs = []

                # n_lstm_step for example, 35. -> 42 has highest performance
                for i in range(T):  # T = number of timesteps
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()

                    # We use cnn embedding + ... as RNN input (as a flatted/concatenated vector)
                    rnn_input = tf.concat(values=[  # 0     1  2  3
                        # (i) C3D map (embedded into 7x7x12)
                        net['c3d_embedded'][:, i, :, :, :],
                    ],
                        axis=3,  # [:, i, 7, 7, HERE]
                        name='rnn_input' + str(i))
                
                    
                    # with tf.variable_scope("RNN"):
                    output_u, state_u = vars.lstm_u(rnn_input, state_u)

                    # at time t
                    output_u.get_shape().assert_is_compatible_with(
                        [B, 7, 7, rnn_state_size])  # Bx{time}x7x7x32
                    rcn_outputs.append(output_u)

            # (3) RCN output unpooling to 49x49 size
            # each of (7x7x32) maps are up-sampled to (49x49x8)
            vars.upsampling_filter1 = tf.get_variable('Upsampling/weight1',
                                                      [5, 5,
                                                       64, rnn_state_size],  # rnn_state_size bad name (indeed a channel size)
                                                      initializer=initializers.xavier_initializer_conv2d(
                                                          uniform=True)
                                                      )
            vars.upsampling_filter2 = tf.get_variable('Upsampling/weight2',
                                                      [5, 5,
                                                       32, 64],  # rnn_state_size bad name (indeed a channel size)
                                                      initializer=initializers.xavier_initializer_conv2d(
                                                          uniform=True)
                                                      )

            vars.upsampling_filter3 = tf.get_variable('Upsampling/weight3',
                                                      [7, 7,
                                                       12, 32],  # rnn_state_size bad name (indeed a channel size)
                                                      initializer=initializers.xavier_initializer_conv2d(
                                                          uniform=True)
                                                      )
            vars.out_W = tf.Variable(tf.random_uniform(
                [12, 1], -0.1, 0.1), name="out_W")
            vars.out_b = tf.Variable(
                tf.random_uniform([1], -0.1, 0.1), name="out_b")

            predicted_gazemaps = []
            # Batch normalization assumption (if wrong fix): apply before eac convolutional layer
            for i in range(T):
                rcn_output_map = rcn_outputs[i]  # [B x 7 x 7 x 128]

                # for now in here - later will add to base:

                # batch_mean, batch_var = tf.nn.moments(rcn_output_map, axes = [0,1,2]) #global normalization for conv_filters
                # what to do with offset and scale?
                rcn_output_map = tf.layers.batch_normalization(rcn_output_map)
                rcn_upsampled_output = tf.nn.conv2d_transpose(rcn_output_map,
                                                              vars.upsampling_filter1,
                                                              output_shape=[
                                                                  B, 23, 23, 64],
                                                              strides=[
                                                                  1, 3, 3, 1],
                                                              padding='VALID',
                                                              name='upsampled_rcn_output_' + str(i))
               
                #rcn_upsampled_output.get_shape().assert_is_compatible_with([B, GH, GW, upsampling_output_channel])
                rcn_upsampled_output = tf.nn.conv2d_transpose(rcn_upsampled_output,
                                                              vars.upsampling_filter2,
                                                              output_shape=[
                                                                  B, 49, 49, 32],
                                                              strides=[
                                                                  1, 2, 2, 1],
                                                              padding='VALID',
                                                              name='upsampled_rcn_output_' + str(i))
               
                input_concat = tf.concat(axis=3,  # the last dimension
                                         values=[
                                             # [B x 49 x 49 x 8]
                                             rcn_upsampled_output,
                                             #                                            net['frm_sal_cubic'][:, i, :, :, :],  # [B x 49 x 49 x 1]
                                             # last_output_gazemap                   # [B x 49 x 49 x 1]
                                         ])

                output = tf.nn.conv2d_transpose(input_concat,
                                                vars.upsampling_filter3,
                                                output_shape=[B, 49, 49, 12],
                                                strides=[1, 1, 1, 1],
                                                padding='SAME',
                                                    name='upsampled_rcn_output_' + str(i))

                output = tf.nn.xw_plus_b(tf.reshape(
                    output, [-1, 12]), vars.out_W, vars.out_b)
                output = tf.nn.dropout(output, dropout_keep_prob)

                # [B x 49 x 49 x 1] -> [B x 49 x 49] squeeze
                predicted_gazemap = tf.reshape(output, [B, GH, GW])
                predicted_gazemaps.append(predicted_gazemap)
                # TODO should we normalize predicted_gazemap ????????????????????????????

            # pack as a tensor
            # T-list of [B x 49 x 49] --> [B x 49 x 49]
            net['predicted_gazemaps'] = tf.transpose(tf.stack(predicted_gazemaps), [
                                                     1, 0, 2, 3], name='predicted_gazemaps')
            net['predicted_gazemaps'].get_shape(
            ).assert_is_compatible_with([B, T, GH, GW])

        return net['predicted_gazemaps']
