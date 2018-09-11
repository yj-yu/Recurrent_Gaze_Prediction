#-*- coding: utf-8 -*-

"""
Just a baseline algorithm:
emit shallownet result framewise!!!!!!!!!!!!

(Code is COPIED ... SORRY)
"""

# TODO separate pupil from gazemaps, AWFUL design

import numpy as np
import os
import sys
import time

from PIL import Image
import tensorflow as tf

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


class GRUModelConfig(BaseModelConfig):

    def __init__(self):
        super(GRUModelConfig, self).__init__()

        self.n_lstm_steps = 35
        self.batch_size = 5 # XXX XXX XXX XXX

        self.dim_feature = 1024 #1024
        self.dim_sal = 1024*49  #196
        self.dim_sal_proj = 1024

        # use adam by default
        self.optimization_method = 'adam'
        self.loss_type = 'l2'


from gaze_rnn import GazePredictionGRU

class FramewiseShallowNet(GazePredictionGRU):

    # Does not compatible model.py yet. mount that to this model. TODO"
    def __init__(self,
                 session,
                 data_sets,
                 config=GRUModelConfig()
                 ):

        super(FramewiseShallowNet, self).__init__(session, data_sets, config)



    @staticmethod
    def create_gazeprediction_network(frame_images, c3d_input,
                                      dropout_keep_prob = 1.0,
                                      net=None):

        B, T, IH, IW, _ = frame_images.get_shape().as_list()
        GH, GW = CONSTANTS.gazemap_height, CONSTANTS.gazemap_width

        predicted_gazemaps = SaliencyModel.create_shallownet(
            tf.reshape(frame_images, [-1, IH, IW, 3]),
            scope='ShallowNet',
            dropout=False
        ) # [ -1, GH, GW ]

        predicted_gazemaps = tf.reshape(predicted_gazemaps, [B, T, GH, GW])  # [B x T x 49 x 49]
        return predicted_gazemaps



    @override
    def build_train_op(self):
        """
        build learning_rate, global_step variables
        and optimizer and related summary.
        """

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.learning_rate = self._build_learning_rate()
        #assert isinstance(self.learning_rate, tf.Variable)
        tf.scalar_summary('learning_rate', self.learning_rate)

        self.train_op = self.create_train_op(self.loss,
                                             tf.trainable_variables(),
                                             self.learning_rate,
                                             self.global_step)
        return self.train_op
