#-*- coding: utf-8 -*-

"""
model_grcn.py

Implement a simple recurrent gaze prediction model based on GRU+RCN.
THIS IS A 7X7 VERSION
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

from models.gaze_rnn import GazePredictionGRU, GRUModelConfig

from easydict import EasyDict as E

CONSTANTS = E()
CONSTANTS.image_width = 98
CONSTANTS.image_height = 98
CONSTANTS.gazemap_width = 7
CONSTANTS.gazemap_height = 7


from gaze_grcn import GRU_RCN_Cell


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

        super(GazePredictionGRCN, self).__init__(session,
                                                 data_sets,
                                                 config=config,
                                                 gazemap_height=CONSTANTS.gazemap_height,
                                                 gazemap_width=CONSTANTS.gazemap_width
                                                 )

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

        # (1) Input frame saliency
        # ------------------------

        # Input.
        net['frame_images'] = frame_images  # [B x T x IH x IW x 3]

        #net['frm_sal'] = SaliencyModel.create_shallownet(
        #    tf.reshape(net['frame_images'], [-1, IH, IW, 3]),
        #    scope='ShallowNet',
        #    dropout=False
        #) # [-1, 49, 49]
        #net['frm_sal'] = tf.reshape(net['frm_sal'], [B, T, GH, GW]) # [B x T x 49 x 49]

        # # [B x T x 49 x 49] --> [B x T x 49 x 49 x 1]
        # net['frm_sal_cubic'] = tf.reshape(net['frm_sal'], [B, T, GH, GW, 1],
                                          # name='frame_saliency_cubic')


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


        # The RNN Part.
        # -------------



        with tf.variable_scope('RCNBottom') as scope:
            vars.lstm_u = GRU_RCN_Cell(rnn_state_size, dim_cnn_proj)

            state_u = vars.lstm_u.zero_state(B, tf.float32)
            log.info('RNN state shape : %s', state_u.get_shape().as_list())

            predicted_gazemaps = []
            net['rcn_outputs'] = rcn_outputs = []

            vars.out_W = tf.Variable(tf.random_uniform([rnn_state_size, 1], -0.1, 0.1), name="out_W")
            vars.out_b = tf.Variable(tf.random_uniform([1], -0.1, 0.1), name="out_b")

            # n_lstm_step for example, 35.
            for i in range(T):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                # We use cnn embedding + ... as RNN input (as a flatted/concatenated vector)
                rnn_input = tf.concat(concat_dim=3,  # [:, i, 7, 7, HERE]
                                      values=[       #  0     1  2  3
                                          net['c3d_embedded'][:, i, :, :, :],      # (i) C3D map (embedded into 7x7x12)
                                      ],
                                      name='rnn_input' + str(i))

                #with tf.variable_scope("RNN"):
                output_u, state_u = vars.lstm_u(rnn_input, state_u)

                # at time t
                output_u.get_shape().assert_is_compatible_with([B, 7, 7, rnn_state_size]) # Bx{time}x7x7x32
                rcn_outputs.append(output_u)

                # a FC layer follows
                output = tf.nn.xw_plus_b(
                    tf.reshape(output_u, [-1, rnn_state_size]),
                    vars.out_W, vars.out_b)
                output = tf.nn.dropout(output, dropout_keep_prob)

                predicted_gazemap = tf.reshape(output, [B, GH, GW])  # 7x7 softmax logit
                predicted_gazemaps.append(predicted_gazemap)

        # pack as a tensor
        # T-list of [B x 49 x 49] --> [B x 49 x 49]
        net['predicted_gazemaps'] = tf.transpose(tf.pack(predicted_gazemaps), [1, 0, 2, 3], name='predicted_gazemaps')
        net['predicted_gazemaps'].get_shape().assert_is_compatible_with([B, T, GH, GW])
        return net['predicted_gazemaps']


