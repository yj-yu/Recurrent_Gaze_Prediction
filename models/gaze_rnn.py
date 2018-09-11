#-*- coding: utf-8 -*-

"""
model_rnn.py

Implement a simple recurrent gaze prediction model based on RNN(GRU).
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
#import crc_input_data_seq-  unused model - also doesn't exist anymore
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #append parent directory to path
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


# config : changed as paramter later
class GRUModelConfig(BaseModelConfig):

    def __init__(self):
        super(GRUModelConfig, self).__init__()  #config calling basemodelconfig
        #adddiotional configurations added to base class config 

        self.n_lstm_steps = 42

        self.batch_size = 7 # XXX XXX XXX XXX 

        self.dim_feature = 1024 #1024
        self.dim_sal = 1024*49  #196
        self.dim_sal_proj = 1024

        # use adam by default
        self.optimization_method = 'adam'
        self.loss_type = 'xentropy'
        self.use_flip_batch = True

##########################################
##########################################

class GazePredictionGRU(ModelBase):

    # Does not compatible model.py yet. mount that to this model. TODO"
    def __init__(self,
                 session,
                 data_sets,
                 config=GRUModelConfig(),
                 gazemap_height=CONSTANTS.gazemap_height,
                 gazemap_width=CONSTANTS.gazemap_width,
                 ):
        self.session = session
        self.data_sets = data_sets
        self.config = config
        #assert isinstance(self.config, GRUModelConfig)

        super(GazePredictionGRU, self).__init__(config)

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

        self.gazemap_height = gazemap_height
        self.gazemap_width = gazemap_width
        self.image_height = CONSTANTS.image_height
        self.image_width = CONSTANTS.image_width

        # Finally, build the model and optimizer
        self.build_model()
        #self.build_generator()
        self.build_train_op()
        self.session.run(tf.initialize_all_variables())


    def build_model(self):
        B, T = self.batch_size, self.n_lstm_steps
        GH, GW = self.gazemap_height, self.gazemap_width
        IH, IW = self.image_height, self.image_width

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


        self.dropout_keep_prob = tf.placeholder_with_default( np.float32(1.0), [],
                                                             name='placeholder_dropout')


        # Build Network
        net = {}
        self.predicted_gazemaps = \
            self.create_gazeprediction_network(
                frame_images = self.frame_images,
                c3d_input = self.c3d_input,
                dropout_keep_prob = self.dropout_keep_prob,
                net = net
            )

        self.loss = \
            self.create_loss_and_summary(
                self.predicted_gazemaps,
                self.gt_gazemap,
                loss_type = self.config.loss_type,
            )

        if self.config.loss_type == 'xentropy' or self.config.loss_type == 'KLD':
            # if xentropy loss, apply softmax to each frame to obtain a
            # valid probability map.
            predicted_gazemaps_asprob = []
            for i in range(T):
                predicted_gazemaps_asprob.append( tf_softmax_2d(self.predicted_gazemaps[:, i, :, :]) )

            self.predicted_gazemaps_logit = self.predicted_gazemaps
            self.predicted_gazemaps = tf.transpose(
                tf.stack(predicted_gazemaps_asprob),
                [1, 0, 2, 3], name='predicted_gazemaps_asprob')

        # FIXME may be duplicates?
        tf.summary.scalar('loss/train', self.loss)
        tf.summary.scalar('loss/val', self.loss, collections=['TEST_SUMMARIES'])


        # Debugging Informations
        # ----------------------

        # OPTIONAL: for debugging and visualization
        # XXX only last predicted_gazemap is shown as of now :( T^T
        # XXX rename saliency -> gaze (to avoid confusion)
        def _add_image_summary(tag, tensor):
            return tf.summary.image(tag, tensor, max_outputs=2, collections=['IMAGE_SUMMARIES'])

        _input_image = self.frame_images[:, T-1, :, :, :] # last rnn step
        _saliency_output = tf.reshape(self.predicted_gazemaps[:, T-1, :, :], [-1, GH, GW, 1])
        _saliency_gt = tf.reshape(self.gt_gazemap[:, T-1, :, :], [-1, GH, GW, 1])

        _add_image_summary('inputimage', _input_image)
        _add_image_summary('saliency_maps_gt', _saliency_gt)

        if self.config.loss_type == 'l2':
            # 0-1 normalize
            _add_image_summary('saliency_maps_pred_original', _saliency_output)
            _add_image_summary('saliency_maps_pred_norm', tf_normalize_map(_saliency_output))
        elif self.config.loss_type == 'xentropy':
            # softmax normalize
            _saliency_output_logit = tf.reshape(self.predicted_gazemaps_logit[:, T-1, :, :], [-1, GH, GW, 1])
            _add_image_summary('saliency_maps_pred_original', _saliency_output_logit)
            _add_image_summary('saliency_maps_pred_norm', _saliency_output)

        elif self.config.loss_type == "KLD":
            _saliency_output_logit = tf.reshape(self.predicted_gazemaps_logit[:, T-1, :, :], [-1, GH, GW, 1])
            _add_image_summary('saliency_maps_pred_original', _saliency_output_logit)
            _add_image_summary('saliency_maps_pred_norm', _saliency_output)
            
        else:
            raise NotImplementedError()

        if 'frm_sal' in net:
            _saliency_shallow = tf.reshape(net['frm_sal'][:, T-1, :, :], [-1, GH, GW, 1])
            _add_image_summary('saliency_zshallownet', _saliency_shallow)

        self.image_summaries = tf.summary.merge(
            inputs = tf.get_collection('IMAGE_SUMMARIES'),
            collections = [],
            name = 'merged_image_summary',
        )


    @staticmethod  #RPG model being build here
    def create_gazeprediction_network(frame_images, c3d_input,
                                      dropout_keep_prob = 1.0,
                                      net=None):
        '''
        Args:
            frame_images: a [B x T x IH x IW x 3] tensor (frame images)
            c3d_input : a [B x T x 1024 x 7 x 7] tensor for C3D convmap features

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


        dim_cnn_proj = 32 # XXX FIXME (see __init__ in GazePredictionGRU)

        # some variables
        # --------------
        rnn_state_size = 7 * 7 * dim_cnn_proj     # flatten with this dimension order
        rnn_state_size += 7 * 7 * 1          # C3D projected PLUS saliency map (49) # FIXME

        ''' C3D + Image Saliency + Vanila RNN '''

        # (1) Input frame saliency
        # ------------------------

        # Input.
        net['frame_images'] = frame_images  # [B x T x IH x IW x 3]

        net['frm_sal'] = SaliencyModel.create_shallownet(
            tf.reshape(net['frame_images'], [-1, IH, IW, 3]),
            scope='ShallowNet',
            dropout=False
        ) # [-1, 49, 49]

        if (GH, GW) == (7, 7):
            log.warn('Downsampling 49x49 saliency to 7x7 ...')
            # downsampling 49,49 -> 7,7
            net['frm_sal'] = tf.nn.avg_pool(
                tf.expand_dims(net['frm_sal'], 3), # [B, 49, 49, '1']
                [1, 7, 7, 1], [1, 7, 7, 1],
                padding='VALID'
            )

        net['frm_sal'] = tf.reshape(net['frm_sal'], [B, T, GH, GW]) # [B x T x 49 x 49]

        # [B x T x 49 x 49] --> [B x T x 49 x 49 x 1]
        net['frm_sal_cubic'] = tf.reshape(net['frm_sal'], [B, T, GH, GW, 1],
                                          name='frame_saliency_cubic')


        # (2) C3D
        # -------
        # a. reduce filter size [7 x 7 x 1024] -> [7 x 7 x 32] via FC or CONV
        # b. apply RCN, and unpool the [7 x 7 x 32] outputs to [49, 49, 8]

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

        with tf.variable_scope("RNN") as scope:
            vars.lstm_u = rnn_cell.GRUCell(rnn_state_size, kernel_initializer="orthogonal")

            state_u = vars.lstm_u.zero_state(B, tf.float32)

            vars.proj_out_W = tf.Variable( tf.random_uniform( [rnn_state_size, GH*GW], -0.1, 0.1), name = "proj_out_W")
            vars.proj_out_b = tf.Variable( tf.zeros( [GH*GW], name = "proj_out_b"))

            log.info('RNN state shape : %s', state_u.get_shape().as_list())

            predicted_gazemaps = []
            # n_lstm_step for example, 35.
            for i in range(T):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                # We use cnn embedding + ... as RNN input (as a flatted/concatenated vector)
                rnn_input = tf.concat(axis=3,  # [:, i, 7, 7, HERE]
                                    values=[       #  0     1  2  3
                                        net['c3d_embedded'][:, i, :, :, :],      # (i) C3D map (embedded into 7x7x12)
#                                      self.frm_sal_771[:, i, :, :, :],  # (ii) frame saliency
                                    ],
                                    name='rnn_input')

                # flatten rnn_input[i] into rank 2 (e.g. [B x -1]
                # TODO this part should be seamlessly done in RNN cell
                rnn_input_flatten = tf.reshape(rnn_input, [B, -1],
                                            name='rnn_input_flatten')

                output_u, state_u = vars.lstm_u(rnn_input_flatten, state_u)

                # at time t
                predicted_gazemap = tf.nn.xw_plus_b(output_u,
                                                    vars.proj_out_W, vars.proj_out_b)
                predicted_gazemap = tf.reshape(predicted_gazemap, [-1, GH, GW])
                predicted_gazemap.get_shape().assert_is_compatible_with([B, GH, GW])

                predicted_gazemaps.append(predicted_gazemap)


        # pack as a tensor
        # T-list of [B x 49 x 49] --> [B x 49 x 49]
        net['predicted_gazemaps'] = tf.transpose(tf.stack(predicted_gazemaps), [1, 0, 2, 3], name='predicted_gazemaps')
        net['predicted_gazemaps'].get_shape().assert_is_compatible_with([B, T, GH, GW])
        

        return net['predicted_gazemaps']


    @staticmethod
    def create_loss_and_summary(predicted_gazemaps,
                                gt_gazemap,
                                loss_type):
        '''
        Args:
            gt_gazemap : a [B x T x GH x GW] tensor of ground truth per-frame gaze maps
            loss_type : ['l2' or 'xentropy']
            (optional) net : a dictionary to get intra-layer activations or tensors.

        Output: [loss, image_summary] where
            loss: a scalar (float) tensor of RNN supervision loss.
            image_summary
        '''
        B, T, GH, GW = predicted_gazemaps.get_shape().as_list()
        gt_gazemap.get_shape().assert_is_compatible_with([B, T, GH, GW])
        predicted_gazemaps.get_shape().assert_is_compatible_with([B, T, GH, GW])

        loss = 0.0
        for i in range(T):
            predicted_gazemap = predicted_gazemaps[:, i, :, :]
            
            
            # Cross entropy and softmax??
            if loss_type == 'l2':
                l2loss = tf.nn.l2_loss(predicted_gazemap - gt_gazemap[:,i,:,:])  # on Bx49x49
                current_gaze_loss = tf.reduce_sum(l2loss)
            elif loss_type == 'xentropy':
                xloss = tf_softmax_cross_entropy_with_logits_2d(logits=predicted_gazemap,
                                                                labels=gt_gazemap[:,i,:,:])
                current_gaze_loss = tf.reduce_sum(xloss)

            elif loss_type == 'KLD':

                ## remove kld loss agian :(
                
                kld_loss = tf.contrib.distributions.kl_divergence(predicted_gazemaps, gt_gazemap[:,i,:,:])
            else:
                raise NotImplementedError(str(loss_type))

            current_loss = current_gaze_loss
            loss += current_loss

        # loss: take average
        loss = tf.div(loss, float(B * T), name='loss_avg')
        return loss



    def initialize_pretrained_shallownet(self, checkpoint_path):
        """
        Initialize (assign) the variable's weight.
        """
        assert os.path.exists(checkpoint_path)
        ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)

        # hmm.. really?
        tf.get_variable_scope().reuse_variables()

        tensors = ckpt_reader.get_variable_to_shape_map()
        for var_name, var_shape in tensors.iteritems():
            if not var_name.startswith('ShallowNet'): continue
            if 'Adam' in var_name: continue

            # tflearn BN sucks. https://github.com/tflearn/tflearn/issues/7
            if 'is_training' in var_name: continue

            var_value = ckpt_reader.get_tensor(var_name)
            var = tf.get_variable(var_name)
            self.session.run(var.assign(var_value))
            log.info('Using pretrained value for %s : %s', var_name, str(var_shape))

    @override
    def _build_learning_rate(self):
        lr = self.initial_learning_rate
        # Use exponential decay
        var_lr_decay = tf.train.exponential_decay(lr, self.global_step,
                                                  decay_steps=500,
                                                  decay_rate=self.learning_rate_decay,
                                                  staircase=True,
                                                  name="var_lr_decay")
        return var_lr_decay


    @override
    def build_train_op(self):
        """
        Gaze RNN Module: build optimizer
        """

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.learning_rate = self._build_learning_rate()
        tf.summary.scalar('learning_rate', self.learning_rate)

        # train op for ShallowNet Part
        shallownet_learning_rate = 0 #self.learning_rate * 0.1  # DO NOT LEARN??????????
        shallownet_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope='ShallowNet')
        shallownet_train_op = self.create_train_op(self.loss,
                                                   shallownet_params,
                                                   shallownet_learning_rate,
                                                   gradient_summary_tag_name='gradient_norm/shallownet'
                                                   )
        # train op for the rest part
        rest_params = [v for v in tf.trainable_variables()
                       if not v in shallownet_params]
        rest_train_op = self.create_train_op(self.loss,
                                             rest_params,
                                             self.learning_rate,
                                             self.global_step)

        # join!
        train_op = tf.group(shallownet_train_op, rest_train_op)
        self.train_op = train_op
        return self.train_op




    def single_step(self, train_mode=True, dataset=None):
        _start_time = time.time()
        GH, GW = self.gazemap_height, self.gazemap_width

        if dataset is None:
            if train_mode:
                dataset = self.data_sets.train
            else:
                dataset = self.data_sets.valid

        """ prepare input """
        # batch_c3d : (batch_size, 35, 512, 2, 7, 7)
        batch_images, batch_maps, batch_fixmaps, batch_c3d, batch_pupil, batch_clipnames = dataset.next_batch(self.batch_size)
        assert batch_images.dtype == np.float32
        batch_c3d = np.reshape(batch_c3d, [self.batch_size, -1, 1024, 7, 7])

        if self.config.loss_type == 'xentropy' or self.config.loss_type == 'KLD':
            batch_maps = normalize_probability_map(batch_maps)

        """ data augmentation """
        # Flip half of the images in this batch at random:
        if train_mode and self.config.use_flip_batch:
            batch_size = len(batch_images)
            indices = np.random.choice(batch_size, batch_size / 2, replace=False)
            batch_images[indices, :] = batch_images[indices, :, :, ::-1, :] # B x T x 98 x 98 x 3
            batch_maps[indices, :] = batch_maps[indices, :, :, ::-1] # B x T x 49 x 49
            batch_c3d[indices, :] = batch_c3d[indices, :, :, :, ::-1] # B x T x 1024 x 7 x 7
            batch_fixmaps[indices, :] = batch_fixmaps[indices, :, :, ::-1] # B x T x 49 x 49?

        """ run """
        _merged_summary = {True: self.merged_summary_train,
                           False: self.merged_summary_val}[train_mode]

        eval_targets = [self.loss, _merged_summary]
        if train_mode: eval_targets += [self.train_op]

        if not train_mode:
            eval_targets += [self.image_summaries]
            #eval_targets += [self.model_var_summaries]

        eval_result = dict(zip(eval_targets, self.session.run(
            eval_targets,
            feed_dict={
                self.c3d_input: batch_c3d,   # [B x T x 1024 x 7 x 7] (watch out for the order of 1024 and 49)
                self.frame_images: batch_images, # [B x T x IH x IW x 3]
                self.gt_gazemap: batch_maps, #[B x T x GH x GW]
                self.dropout_keep_prob : 0.5 if train_mode else 1.0
            }
        )))
        loss    = eval_result[self.loss]
        summary = eval_result.get(_merged_summary)

        step  = self.current_step
        epoch = float(step * self.batch_size) / len(self.data_sets.train) # estimated training epoch

        if step >= 30 and summary is not None:
            self.writer.add_summary(summary, step)

        if not train_mode:
            image_summary = eval_result[self.image_summaries]
            self.writer.add_summary(image_summary, step)
            #var_summary = eval_result[self.model_var_summaries]
            #self.writer.add_summary(var_summary, step)

        _end_time = time.time()

        if (not train_mode) or np.mod(step, self.config.steps_per_logprint) == 0:
            log_fn = (train_mode and log.info or log.infov)
            log_fn((" [{split_mode:5} epoch {epoch:.1f} / step {step:4d}] {tag} " +
                    "batch total-loss: {total_loss:.5f}, target-loss: {target_loss:.5f} " +
                    "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) " +
                    "(lr={learning_rate:.3g})"
                    ).format(split_mode=(train_mode and 'train' or 'val'),
                             epoch=epoch, step=step,
                             tag=(self.config.train_tag + ' |' if self.config.train_tag else ''),
                             total_loss=loss, target_loss=loss,
                             sec_per_batch=(_end_time - _start_time),
                             instance_per_sec=self.batch_size / (_end_time - _start_time),
                             learning_rate=self.current_learning_rate
                            )
                   )

        return step


    def generate(self, dataset, max_instances=50):
        GH, GW = self.gazemap_height, self.gazemap_width

        pred_gazemap_list = []         # pred gaze map
        pred_pupil_list = []
        gt_gazemap_list = []            # gt gaze map
        gt_pupil_list = []
        fixationmap_list = []
        images_list = []
        filename_list = []
        c3d_list = []

        # TODO remove duplicates.

            
        # the number of steps to iterate a single epoch of validation or dataset
        n_instances = len(dataset)

        # XXXXXXX because of slowness
        if max_instances is not None:
            n_instances = min(n_instances, max_instances)

        step_num = int(np.ceil(n_instances / float(self.batch_size)))
        assert step_num > 0

        for v in range(step_num):
            batch_images, batch_maps, batch_fixmaps, batch_c3d, batch_pupil, batch_filename = dataset.next_batch(self.batch_size)
            batch_images = np.asarray(list(batch_images))

            assert batch_images.dtype == np.float32
            batch_c3d = np.reshape(batch_c3d, [self.batch_size, -1, 1024, 7, 7])

            if self.config.loss_type == 'xentropy':
                batch_maps = normalize_probability_map(batch_maps)

            [gazes,] = self.session.run(
                [self.predicted_gazemaps],
                feed_dict={
                    self.c3d_input: batch_c3d,
                    self.frame_images: batch_images, # [B x T x IH x IW x 3]
                    self.gt_gazemap: batch_maps,      #[B x T x GH x GW]
                    self.dropout_keep_prob : 1.0,
                }
            )
            pupils = np.zeros([self.batch_size, self.n_lstm_steps, 1])        # not implemented

            # each shape is [B x T x 49].
            # we need evaluate frame-wise mean :)
            c3d_list.extend(batch_c3d)
            pred_gazemap_list.extend(gazes)
            pred_pupil_list.extend(pupils)
            gt_gazemap_list.extend(batch_maps)
            gt_pupil_list.extend(batch_pupil)
            fixationmap_list.extend(batch_fixmaps)
            images_list.extend(np.concatenate(batch_images)) # TODO why fold in (BxT)?
            filename_list.extend(batch_filename)
            
        # flatten the time dimension so that we have one map per single frame.
        # Moreover, all the maps are unflattened (e.g. 49 -> 7x7)
        pred_gazemap_list = np.vstack(pred_gazemap_list).reshape([-1, GH, GW])
        pred_pupil_list = np.vstack(pred_pupil_list)
        gt_gazemap_list = np.vstack(gt_gazemap_list).reshape([-1, GH, GW])
        gt_pupil_list = np.vstack(gt_pupil_list)
        c3d_list = np.vstack(c3d_list).reshape([-1, 1024,7,7])

        try:
            fixationmap_list = np.vstack(fixationmap_list)
        except:
            # TODO image resolution might differ if it comes to the original scale
            fixationmap_folded = []
            for fixationmap in fixationmap_list: # (TxH'xW')
                for t in xrange(len(fixationmap)):
                    fixationmap_folded.append(fixationmap[t, :, :])
            fixationmap_list = fixationmap_folded
            assert len(fixationmap_list) == len(pred_gazemap_list)

        return {'pred_gazemap_list' : pred_gazemap_list,
                'gt_gazemap_list' : gt_gazemap_list,
                'images_list' : images_list,
                'fixationmap_list' : fixationmap_list,
                'clipname_list' : filename_list,
                'c3d_list' : c3d_list
                }


    def evaluate(self,
                 pred_gazemap_list,
                 gt_gazemap_list,
                 fixationmap_list,
                 images_list,
                 ):

        assert len(pred_gazemap_list) == len(gt_gazemap_list) == len(fixationmap_list) == len(images_list), \
            "Length mismatch: %d %d %d %d" % (
                len(pred_gazemap_list), len(gt_gazemap_list),
                len(fixationmap_list), len(images_list)
            )

        # Evaluate list.
        batch_scores = {}
        log.infov('Evaluation on %d images', len(gt_gazemap_list))
        for metric in AVAILABLE_METRICS:
            batch_scores[metric] = saliency_score(metric, pred_gazemap_list, gt_gazemap_list, fixationmap_list)
            log.infov('Saliency %s : %f', metric, batch_scores[metric])

        self.report_evaluate_summary(batch_scores)
        return batch_scores


    def generate_and_evaluate(self, dataset, max_instances=50):
        ret = self.generate(dataset, max_instances)
        scores = self.evaluate(**ret)
        return ret, scores
