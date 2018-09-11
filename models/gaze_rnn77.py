#-*- coding: utf-8 -*-

"""
gaze_rnn7.py

Implement a simple recurrent gaze prediction model based on RNN(GRU).
In this version, the gaze DIM is REDUCED to 7x7 dimension.
"""

# TODO separate pupil from gazemaps, AWFUL design

import numpy as np
import os
import sys
import time

from PIL import Image
import tensorflow as tf
rnn_cell = tf.nn.rnn_cell

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
CONSTANTS.gazemap_width = 7
CONSTANTS.gazemap_height = 7
CONSTANTS.saliencymap_width = 49
CONSTANTS.saliencymap_height = 49


# config : changed as paramter later
class GRUModelConfig(BaseModelConfig):

    def __init__(self):
        super(GRUModelConfig, self).__init__()

        self.n_lstm_steps = 35
        self.batch_size = 7 # XXX XXX XXX XXX

        self.dim_feature = 1024 #1024
        self.dim_sal = 1024*49  #196
        self.dim_sal_proj = 1024

        # use adam by default
        self.optimization_method = 'adam'

        self.loss_type = 'l2'


from gaze_rnn import GazePredictionGRU as GazePredictionGRU4949

class GazePredictionGRU(GazePredictionGRU4949):

    # Does not compatible model.py yet. mount that to this model. TODO"
    def __init__(self,
                 session,
                 data_sets,
                 config=GRUModelConfig()
                 ):
        self.session = session
        self.data_sets = data_sets
        self.config = config
        #assert isinstance(self.config, GRUModelConfig)

        super(GazePredictionGRU, self).__init__(session, data_sets, config)

        # other configuration
        self.batch_size = config.batch_size
        self.n_lstm_steps = config.n_lstm_steps
        self.dim_feature = config.dim_feature
        self.dim_sal = config.dim_sal # 49 * 49
        self.dim_sal_proj = config.dim_sal_proj # 14 * 14 = 196
        self.dim_cnn_proj = 32

        self.initial_learning_rate = config.initial_learning_rate
        self.learning_rate_decay = config.learning_rate_decay
        self.max_grad_norm = config.max_grad_norm

        self.gazemap_height = CONSTANTS.gazemap_height # 7
        self.gazemap_width = CONSTANTS.gazemap_width # 7
        self.image_height = CONSTANTS.image_height
        self.image_width = CONSTANTS.image_width

        # Finally, build the model and optimizer
        self.build_model()
        #self.build_generator()
        self.build_train_op()
        self.session.run(tf.initialize_all_variables())


