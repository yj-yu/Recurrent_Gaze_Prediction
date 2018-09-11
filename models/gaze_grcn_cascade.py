#-*- coding: utf-8 -*-

"""
model_grcn.py

Implement a simple recurrent gaze prediction model based on GRU+RCN.
"""

# TODO separate pupil from gazemaps, AWFUL design

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
import crc_input_data_seq

from util import log, override
from models.base import ModelBase, BaseModelConfig
from models.saliency_shallownet import SaliencyModel

from models.model_util import tf_normalize_map
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
                 spatial_shape = [7, 7],
                 kernel_spatial_shape = [3, 3],
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
        self.W   = tf.Variable(tf.truncated_normal([kernel_H, kernel_W,dim_feature, self._num_units],
                                                   dtype=tf.float32, stddev=1e-4),
                               name='GRU_Conv_W')
        self.U   = tf.Variable(tf.truncated_normal([kernel_H, kernel_W, self._num_units, self._num_units],
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
                Wzconv = tf.nn.conv2d(inputs, self.W_z, [1,1,1,1], padding='SAME')
                Uzconv = tf.nn.conv2d(state, self.U_z, [1,1,1,1], padding='SAME')
                Wrconv = tf.nn.conv2d(inputs, self.W_r, [1,1,1,1], padding='SAME')
                Urconv = tf.nn.conv2d(state, self.U_r, [1,1,1,1], padding='SAME')
                # sig(W_r * x_t + U_r * h_t-1 )
                u = tf.sigmoid(Wzconv + Uzconv)
                r = tf.sigmoid(Wrconv + Urconv)
                with tf.variable_scope("Candidate"):
                    # tanh(W * x_t + U * (r_t dot h_t-1) not confident yet.
                    Wconv = tf.nn.conv2d(inputs, self.W, [1,1,1,1], padding='SAME')
                    Uconv = tf.nn.conv2d(r*state, self.U, [1,1,1,1], padding='SAME')
                    c = tf.tanh(tf.add(Wconv,Uconv))
                    new_h = u * state + (1 - u) * c
                    # output, state is (batch_size, H=7, W=7, num_units)
                    return new_h, new_h

    def zero_state(self, batch_size, dtype): #this might be really useful for the idenity rnn...
        """Return state tensor (shape [batch_size x 7 x 7 x state_size]) filled with 0.

        Args:
            batch_size: int, float, or unit Tensor representing the batch size.
            dtype: the data type to use for the state.

        Returns:
            A 4D Tensor of shape [batch_size x state_size] filled with zeros.
        """
        zeros = tf.zeros(tf.pack([batch_size, self.spatial_H, self.spatial_W, self.state_size]), dtype=dtype) #tf.pack converts list to numpy matrix
        zeros.set_shape([None, self.spatial_H, self.spatial_W, self.state_size])
        return zeros


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
                                                 config=config)


    def build_model(self):
        B, T = self.batch_size, self.n_lstm_steps
        GH, GW = CONSTANTS.gazemap_height, CONSTANTS.gazemap_width
        IH, IW = CONSTANTS.image_height, CONSTANTS.image_width

        # Network Inputs.
        self.frame_images = tf.placeholder(tf.float32, [B, T, IH, IW, 3],
                                           name='placeholder_frame_images')
        self.c3d_input = tf.placeholder(tf.float32,
                                        [self.batch_size, self.n_lstm_steps, 1024, 7, 7],
                                        name='placeholder_frame_c3d'
                                        ) # [B x T x 1024 x 7 x 7]

        self.gt_gazemap = tf.placeholder(tf.float32,
                                         [B, T, GH, GW],
                                         name='placeholder_gazemap')

        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                [],
                                                name='placeholder_dropout_keep_prob')

        # Build the "RGP" Network
        self.predicted_gazemaps, self.loss, self.image_summaries = GazePredictionGRCN.create_gazeprediction_network(
            frame_images = self.frame_images,
            c3d_input = self.c3d_input,
            gt_gazemap = self.gt_gazemap,
            dropout_keep_prob = self.dropout_keep_prob
        )


    # TODO TODO extract out loss and image summaries separately!!!
    @staticmethod
    def create_gazeprediction_network(frame_images, c3d_input, gt_gazemap,
                                      dropout_keep_prob, net=None):
        '''
        Args:
            frame_images: a [B x T x IH x IW x 3] tensor (frame images)
            c3d_input : a [B x T x 1024 x 7 x 7] tensor for C3D convmap features
            gt_gazemap : a [B x T x GH x GW] tensor of ground truth per-frame gaze maps
            dropout_keep_prob : float tensor
            (optional) net : a dictionary to get intra-layer activations or tensors.

        Outputs:
            [predicted_gazemaps, loss, image_summary] where

            predicted_gazemaps : a [B x T x GH x GW] tensor,
                predicted gaze maps per frame
            loss: a scalar (float) tensor of RNN supervision loss.
            image_summary
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
        gt_gazemap.get_shape().assert_is_compatible_with([B, T, GH, GW])


        dim_cnn_proj = 512 # XXX FIXME (see __init__ in GazePredictionGRU)

        # some variables
        # --------------
        # not a proper name, it should be rnn_state_feature_size in # GRCN????????? FIXME
        rnn_state_size = 256 #dim_cnn_proj # filter size is more correct name

        ''' The RGP (Recurrent Gaze Prediction) model. '''

        # (1) Input frame saliency
        # ------------------------

        # Input.
        net['frame_images'] = frame_images  # [B x T x IH x IW x 3]

        net['frm_sal'] = SaliencyModel.create_shallownet(
            tf.reshape(net['frame_images'], [-1, IH, IW, 3]),
            scope='ShallowNet',
            dropout=False
        ) # [-1, 49, 49]
        net['frm_sal'] = tf.reshape(net['frm_sal'], [B, T, GH, GW]) # [B x T x 49 x 49]

        # [B x T x 49 x 49] --> [B x T x 49 x 49 x 1]
        net['frm_sal_cubic'] = tf.reshape(net['frm_sal'], [B, T, GH, GW, 1],
                                          name='frame_saliency_cubic')


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

        # --> [B x T x 7 x 7 x 12]
        net['c3d_embedded'] = tf.reshape(net['c3d_embedded'], [B, T, 7, 7, dim_cnn_proj])
        log.info('c3d_embedded shape : %s', net['c3d_embedded'].get_shape().as_list())
        net['c3d_embedded'].get_shape().assert_is_compatible_with([B, T, 7, 7, dim_cnn_proj])


        # The RNN Part.
        # -------------

        # Batch size x (gaze map size), per frame
        net['gt_gazemap'] = gt_gazemap   # [B x T x GH, GW]
        log.info('gt_gazemap shape : %s', net['gt_gazemap'].get_shape().as_list())


        with tf.variable_scope('RCNBottom') as scope:
            vars.lstm_u = GRU_RCN_Cell(rnn_state_size, dim_cnn_proj)

            state_u = vars.lstm_u.zero_state(B, tf.float32)
            log.info('RNN state shape : %s', state_u.get_shape().as_list())

            # n_lstm_step for example, 35.
            net['rcn_outputs'] = rcn_outputs = []
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

        # (3) RCN output unpooling to 49x49 size
        # each of (7x7x32) maps are up-sampled to (49x49x8)
        upsampling_filter_size = 11
        upsampling_output_channel = 64
        vars.upsampling_filter = tf.get_variable('Upsampling/weight',
                                                 [upsampling_filter_size, upsampling_filter_size,
                                                  upsampling_output_channel, rnn_state_size], # rnn_state_size bad name (indeed a channel size)
                                                 initializer=initializers.xavier_initializer_conv2d(uniform=True)
                                                 )

        net['rcn_upsampled_outputs'] = rcn_upsampled_outputs = []
        for i in range(T):
            rcn_output_map = rcn_outputs[i]  # [B x 7 x 7 x 128]

            rcn_upsampled_output = tf.nn.conv2d_transpose(rcn_output_map,
                                                          vars.upsampling_filter,
                                                          output_shape=[B, GH, GW, upsampling_output_channel],
                                                          strides=[1, 7, 7, 1],
                                                          padding='SAME',
                                                          name='upsampled_rcn_output_' + str(i))
            rcn_upsampled_output.get_shape().assert_is_compatible_with([B, GH, GW, upsampling_output_channel])
            rcn_upsampled_outputs.append(rcn_upsampled_output)

            if i == 0:
                log.info('RCN input map size : %s', rcn_output_map.get_shape().as_list())
                log.info('RCN upsampled size : %s', rcn_upsampled_output.get_shape().as_list())

        # (4) The upper layer of GRCN to emit gaze map
        # --------------------------------------------
        with tf.variable_scope('RCNGaze') as scope:

            vars.lstm_g = GRU_RCN_Cell(num_units=3,
#                                       dim_feature=upsampling_output_channel + 1 + 1, # 10?
                                       dim_feature=upsampling_output_channel + 1, # 10?
                                       spatial_shape = [GH, GW],
                                       kernel_spatial_shape = [5, 5]
                                       )

            state_g = vars.lstm_g.zero_state(B, tf.float32)
#            last_output_gazemap = tf.zeros([B, GH, GW, 1])


            predicted_gazemaps = []
            for i in range(T):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                # try RNN supervision with GT gazemap.
                # FIXME decoder should be spin off here
                #if i > 0:
                #    last_output_gazemap = tf.expand_dims(gt_gazemap[:, i - 1, :, :], 3)


                # now, combine image saliency, rcn map from the bottom layer,
                # and the previous input
                '''
                rcn_input_concat = tf.concat(concat_dim=3, # the last dimension
                                            values=[
                                                rcn_upsampled_outputs[i],             # [B x 49 x 49 x 8]
                                                net['frm_sal_cubic'][:, i, :, :, :],  # [B x 49 x 49 x 1]
#                                                last_output_gazemap                   # [B x 49 x 49 x 1]
                                            ])
                '''
                #with tf.variable_scope("RNN"):
                output_g, state_g = vars.lstm_g(rcn_upsampled_outputs[i], state_g)

                output_g.get_shape().assert_is_compatible_with([B, GH, GW, 3])
                rcn_outputs.append(rcn_outputs)
                output_g = tf.reshape(output_g,[B,-1])

                # apply another convolutional layer (== fc in fact) to gaze map
                # [B x 49 x 49 x 3] -> # [B x 49 x 49 x 1]

                with tf.variable_scope('LastProjection') as scope_proj:
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()

                    fc1 = fully_connected(output_g, 4802,
                                            activation_fn=None,#tf.nn.relu,
                                            weight_init=initializers.xavier_initializer(uniform=True),
                                            bias_init=tf.constant_initializer(0.0),
                                            weight_collections=['MODEL_VARS'], bias_collections=['MODEL_VARS'],
                                            name='fc1')
                    #net['fc1'] = tflearn.layers.batch_normalization(net['fc1'])
                    fc1 = tf.nn.relu(fc1)

                    if dropout_keep_prob is not None:
                        fc1 = tf.nn.dropout( fc1, dropout_keep_prob )

                    fc1_slice1, fc1_slice2 = tf.split(1, 2, fc1, name='fc1_slice')
                    max_out = tf.maximum(fc1_slice1, fc1_slice2, name='fc1_maxout')

                    fc2 = fully_connected(max_out, 4802 ,
                                        activation_fn=None, # no relu here
                                        weight_init=initializers.xavier_initializer(uniform=True),
                                        bias_init=tf.constant_initializer(0.0),
                                        weight_collections=['MODEL_VARS'], bias_collections=['MODEL_VARS'],
                                        name='fc2')
                    #net['fc2'] = tflearn.layers.batch_normalization(net['fc2'])
                    fc2 = tf.nn.relu(fc2)

                    #if dropout:
                    #    net['fc2'] = tf.nn.dropout( net['fc2'], net['dropout_keep_prob'] )

                    fc2_slice1, fc2_slice2 = tf.split(1, 2, fc2, name='fc2_slice')
                    max_out2 = tf.maximum(fc2_slice1, fc2_slice2, name='fc2_maxout')

                predicted_gazemap = tf.reshape(max_out2, [B, GH, GW])  # [B x 49 x 49 x 1] -> [B x 49 x 49] squeeze
                predicted_gazemaps.append(predicted_gazemap)
                # TODO should we normalize predicted_gazemap ????????????????????????????


        # (4) Finally, calculate the loss
        loss = 0.0

        for i in range(T):
            predicted_gazemap = predicted_gazemaps[i]

            # Cross entropy and softmax??
            l2loss = tf.nn.l2_loss(predicted_gazemap - gt_gazemap[:,i,:,:])  # on Bx49x49
            current_gaze_loss = tf.reduce_sum(l2loss)

            current_loss = current_gaze_loss
            loss += current_loss

        # loss: take average
        loss = tf.div(loss, float(B * T), name='loss_avg')

        # FIXME may be duplicates?
        tf.scalar_summary('loss/train', loss)
        tf.scalar_summary('loss/val', loss, collections=['TEST_SUMMARIES'])

        # pack as a tensor
        # T-list of [B x 49 x 49] --> [B x 49 x 49]
        net['predicted_gazemaps'] = tf.transpose(tf.pack(predicted_gazemaps), [1, 0, 2, 3], name='predicted_gazemaps')
        net['predicted_gazemaps'].get_shape().assert_is_compatible_with([B, T, GH, GW])


        # Debugging Informations
        # ----------------------


        # OPTIONAL: for debugging and visualization
        # XXX only last predicted_gazemap is shown as of now :( T^T
        # XXX rename saliency -> gaze (to avoid confusion)
        def _add_image_summary(tag, tensor):
            return tf.image_summary(tag, tensor, max_images=2, collections=['IMAGE_SUMMARIES'])

        _input_image = frame_images[:, i, :, :, :] # last rnn step
        _saliency_output = tf.reshape(predicted_gazemap, [-1, GH, GW, 1])
        _saliency_gt = tf.reshape(gt_gazemap[:, i, :, :], [-1, GH, GW, 1])
        _saliency_shallow = tf.reshape(net['frm_sal'][:, i, :, :], [-1, GH, GW, 1])

        _add_image_summary('inputimage', _input_image)
        _add_image_summary('saliency_maps_gt', _saliency_gt)
        _add_image_summary('saliency_maps_pred_original', _saliency_output)
        _add_image_summary('saliency_maps_pred_norm', tf_normalize_map(_saliency_output))
        #_add_image_summary('saliency_zimgframe_shallow77', _saliency_shallow77)
        _add_image_summary('saliency_zshallownet', _saliency_shallow)

        image_summaries = tf.merge_summary(
            inputs = tf.get_collection('IMAGE_SUMMARIES'),
            collections = [],
            name = 'merged_image_summary',
        )

        return net['predicted_gazemaps'], loss, image_summaries





def self_test(args):
    global model, config

    assert 0.0 < args.gpu_fraction <= 1.0
    session = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction),
        device_count={'GPU': True},        # self-testing: NO GPU, USE CPU
    ))

    log.warn('Building Model ...')
    # default configuration as of now
    config = GRUModelConfig()

    # CRC likes 28 :)
    config.batch_size = 28

    config.train_dir = args.train_dir
    if args.train_tag:
        config.train_tag = args.train_tag
    config.initial_learning_rate = 0.0001
    config.max_grad_norm = 1.0

    if args.learning_rate is not None:
        config.initial_learning_rate = float(args.learning_rate)
    if args.learning_rate_decay is not None:
        config.learning_rate_decay = float(args.learning_rate_decay)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)

    config.steps_per_evaluation = 500
    config.steps_per_validation = 50  #?!?!
    config.steps_per_checkpoint = 500

    if args.max_steps:
        config.max_steps = int(args.max_steps)
    if args.dataset == 'crc':
        config.steps_per_evaluation = 100

    config.dump(sys.stdout)

    log.warn('Dataset (%s) Loading ...', args.dataset)
    assert args.dataset in ('crc', 'hollywood2')
    data_sets = crc_input_data_seq.read_crc_data_sets(CONSTANTS.image_height,
                                                    CONSTANTS.image_width,
                                                    CONSTANTS.gazemap_height,
                                                    CONSTANTS.gazemap_width,
                                                    np.float32,
                                                    use_cache=True,
                                                    dataset=args.dataset)
    log.warn('Dataset Loading Finished ! (%d instances)',
             len(data_sets))

    log.warn('Start Fitting Model ...')
    model = GazePredictionGRCN(session, data_sets, config)
    print model

    if args.shallownet_pretrain is not None:
        log.warn('Loading ShallowNet weights from checkpoint %s ...', args.shallownet_pretrain)
        model.initialize_pretrained_shallownet(args.shallownet_pretrain)

    model.fit()

    log.warn('Fitting Done. Evaluating!')
    model.evaluate(data_sets.test)
    #session.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max_steps', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--gpu_fraction', default=0.98, type=float)
    parser.add_argument('--train_dir', default=None, type=str)
    parser.add_argument('--train_tag', '--tag', default=None, type=str)
    parser.add_argument('--learning_rate', default=None, type=float)
    parser.add_argument('--learning_rate_decay', default=None, type=float)
    parser.add_argument('--dataset', default='crc',
                        help='[crc, hollywood2]')
    parser.add_argument('--shallownet_pretrain', default=None, type=str)
    args = parser.parse_args()

    self_test(args)
