#-*- coding: utf-8 -*-

"""
C3D + Conv
"""

import numpy as np
import os
import sys
import time

from PIL import Image
import tensorflow as tf

from tensorflow.contrib.layers.python.layers import (
    initializers,
    convolution2d, fully_connected
)

from collections import OrderedDict
import cPickle as pkl
import crc_input_data_seq

from util import log, override
from models.base import ModelBase, BaseModelConfig
from models.saliency_shallownet import SaliencyModel

from models.model_util import tf_normalize_map, normalize_probability_map
from models.model_util import tf_softmax_2d, tf_softmax_cross_entropy_with_logits_2d
from evaluation_metrics import saliency_score, AVAILABLE_METRICS

from easydict import EasyDict as E

CONSTANTS = E()
CONSTANTS.image_width = 98
CONSTANTS.image_height = 98
CONSTANTS.gazemap_width = 49
CONSTANTS.gazemap_height = 49
CONSTANTS.saliencymap_width = 49
CONSTANTS.saliencymap_height = 49

from models.gaze_rnn import GRUModelConfig, GazePredictionGRU


class GazePredictionConv(GazePredictionGRU):

    def __init__(self,
                 session,
                 data_sets,
                 config=GRUModelConfig()
                 ):

        super(self.__class__, self).__init__(session,
                                                 data_sets,
                                                 config=config)



        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                [],
                                                name='placeholder_dropout_keep_prob')


    @staticmethod
    def create_gazeprediction_network(frame_images, c3d_input,
                                      dropout_keep_prob = 1.0,
                                      net=None):
        '''
        Args:
            frame_images: a [B x T x IH x IW x 3] tensor (frame images)
            c3d_input : a [B x T x 1024 x 7 x 7] tensor for C3D convmap features
            gt_gazemap : a [B x T x GH x GW] tensor of ground truth per-frame gaze maps
            dropout_keep_prob : float tensor
            (optional) net : a dictionary to get intra-layer activations or tensors.

        Outputs:
            predicted_gazemaps : a [B x T x GH x GW] tensor,
                predicted gaze maps per frame
        '''

        if net is None: net = {}
        else: assert isinstance(net, dict)

        vars = E()

        # (0) input sanity check
        GH, GW = CONSTANTS.gazemap_height, CONSTANTS.gazemap_width
        IH, IW = CONSTANTS.image_height, CONSTANTS.image_width
        B, T = frame_images.get_shape().as_list()[:2]

        assert B > 0 and T > 0
        frame_images.get_shape().assert_is_compatible_with([B, T, IH, IW, 3])
        c3d_input.get_shape().assert_is_compatible_with([B, T, 1024, 7, 7])


        dim_cnn_proj = 512 # XXX FIXME (see __init__ in GazePredictionGRU)

        # some variables
        # --------------
        # not a proper name, it should be rnn_state_feature_size in # GRCN????????? FIXME
        rnn_state_size = 128 #dim_cnn_proj # filter size is more correct name

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
                                                    perm=[0,1,3,4,2],
                                                    name='c3d_input_reshape')
            log.info('c3d_input_reshape shape : %s', net['c3d_input_reshape'].get_shape().as_list())
            net['c3d_input_reshape'].get_shape().assert_is_compatible_with([B, T, 7, 7, 1024])


            # c3d_embedded: project each 1024 feature (per 7x7 c3d conv-feature map) into 12
            vars.proj_c3d_W = tf.Variable(tf.random_uniform([1024, dim_cnn_proj], -0.1, 0.1), name="proj_c3d_W")
            vars.proj_c3d_b = tf.Variable(tf.random_uniform([dim_cnn_proj], -0.1, 0.1), name="proj_c3d_b")

            net['c3d_embedded'] = tf.nn.xw_plus_b(
                tf.reshape(net['c3d_input_reshape'], [-1, 1024]),
                vars.proj_c3d_W, vars.proj_c3d_b
            ) # [(B*T*7*7) x 1024] --> [(B*T*7*7) x 12] by appling W:1024->12

            if dropout_keep_prob != 1.0:
                net['c3d_embedded'] = tf.nn.dropout(net['c3d_embedded'], dropout_keep_prob)

            # --> [B x T x 7 x 7 x 12]
            net['c3d_embedded'] = tf.reshape(net['c3d_embedded'], [B, T, 7, 7, dim_cnn_proj])
            log.info('c3d_embedded shape : %s', net['c3d_embedded'].get_shape().as_list())
            net['c3d_embedded'].get_shape().assert_is_compatible_with([B, T, 7, 7, dim_cnn_proj])


            # Instead of RNN part, we have deconvolution
            # -------------


            rcn_outputs = [None] * T
            for i in range(T):
                rcn_outputs[i] = net['c3d_embedded'][:, i, :, :, :]
                # B x 7 x 7 x 512(dim_cnn_proj)


            # (3) RCN output unpooling to 49x49 size
            # each of (7x7x32) maps are up-sampled to (49x49x8)
            vars.upsampling_filter1 = tf.get_variable('Upsampling/weight1',
                                                    [5, 5,
                                                    64,
                                                    dim_cnn_proj, # directly project 512->64
                                                    #rnn_state_size
                                                    ], # rnn_state_size bad name (indeed a channel size)
                                                    initializer=initializers.xavier_initializer_conv2d(uniform=True)
                                                    )
            vars.upsampling_filter2 = tf.get_variable('Upsampling/weight2',
                                                    [5, 5,
                                                    32, 64], # rnn_state_size bad name (indeed a channel size)
                                                    initializer=initializers.xavier_initializer_conv2d(uniform=True)
                                                    )

            vars.upsampling_filter3 = tf.get_variable('Upsampling/weight3',
                                                    [7, 7,
                                                    12, 32], # rnn_state_size bad name (indeed a channel size)
                                                    initializer=initializers.xavier_initializer_conv2d(uniform=True)
                                                    )
            vars.out_W = tf.Variable(tf.random_uniform([12, 1], -0.1, 0.1), name="out_W")
            vars.out_b = tf.Variable(tf.random_uniform([1], -0.1, 0.1), name="out_b")

            predicted_gazemaps = []
            for i in range(T):
                rcn_output_map = rcn_outputs[i]  # [B x 7 x 7 x 128]

                rcn_upsampled_output = tf.nn.conv2d_transpose(rcn_output_map,
                                                            vars.upsampling_filter1,
                                                            output_shape=[B, 23, 23, 64],
                                                            strides=[1, 3, 3, 1],
                                                            padding='VALID',
                                                            name='upsampled_rcn_output_' + str(i))
                #rcn_upsampled_output.get_shape().assert_is_compatible_with([B, GH, GW, upsampling_output_channel])
                rcn_upsampled_output = tf.nn.conv2d_transpose(rcn_upsampled_output,
                                                            vars.upsampling_filter2,
                                                            output_shape=[B, 49, 49, 32],
                                                            strides=[1, 2, 2, 1],
                                                            padding='VALID',
                                                            name='upsampled_rcn_output_' + str(i))
                input_concat = tf.concat(concat_dim=3, # the last dimension
                                            values=[
                                                rcn_upsampled_output,             # [B x 49 x 49 x 8]
#                                            net['frm_sal_cubic'][:, i, :, :, :],  # [B x 49 x 49 x 1]
                                                # last_output_gazemap                   # [B x 49 x 49 x 1]
                                            ])

                output = tf.nn.conv2d_transpose(input_concat,
                                                            vars.upsampling_filter3,
                                                            output_shape=[B, 49, 49, 12],
                                                            strides=[1, 1, 1, 1],
                                                            padding='SAME',
                                                            name='upsampled_rcn_output_' + str(i))

                output = tf.nn.xw_plus_b(tf.reshape(output,[-1,12]),vars.out_W,vars.out_b)
                output = tf.nn.dropout(output,dropout_keep_prob)

                predicted_gazemap = tf.reshape(output, [B, GH, GW])  # [B x 49 x 49 x 1] -> [B x 49 x 49] squeeze
                predicted_gazemaps.append(predicted_gazemap)
                # TODO should we normalize predicted_gazemap ????????????????????????????

            # pack as a tensor
            # T-list of [B x 49 x 49] --> [B x 49 x 49]
            net['predicted_gazemaps'] = tf.transpose(tf.pack(predicted_gazemaps), [1, 0, 2, 3], name='predicted_gazemaps')
            net['predicted_gazemaps'].get_shape().assert_is_compatible_with([B, T, GH, GW])

        return net['predicted_gazemaps']
