#-*- coding: utf-8 -*-

"""
saliency_model.py

This class implements a shallow convnet saliency prediction model [1].
The input is a 96x96 image, and the output is a 48*48 saliency map.

[1] Pan, J., McGuinness, K., Sayrol, E., O'Connor, N. and Giro-i-Nieto, X.
    Shallow and Deep Convolutional Networks for Saliency Prediction. In CVPR 2016.
"""

from models.base import ModelBase, BaseModelConfig

import numpy as np
from util import log
import os, os.path
import sys

import time
import tensorflow as tf
from itertools import chain

from tensorflow.contrib.layers.python.layers import (
    initializers,
    convolution2d, fully_connected
)
import tflearn.layers
import salicon_input_data
import crc_input_data_seq as crc_input_data

from models.model_util import tf_normalize_map
import evaluation_metrics


class SaliencyModel(ModelBase):

    def __init__(self,
                 session,
                 data_sets,
                 config=BaseModelConfig()
                 ):
        self.session = session
        self.data_sets = data_sets
        self.config = config

        super(SaliencyModel, self).__init__(config)

        # other configuration
        self.batch_size = config.batch_size
        self.initial_learning_rate = config.initial_learning_rate
        self.max_grad_norm = config.max_grad_norm

        # Finally, build the model and optimizer
        self.build_model()

        self.build_train_op()
        self.prepare_data()

        self.session.run(tf.initialize_all_variables())


    # learning rate decay
    def _build_learning_rate(self):
        #return tf.train.exponential_decay(
        #    self.initial_learning_rate,
        #    global_step = self.global_step,
        #    decay_steps = len(self.data_sets.train.images) / self.batch_size,
        #    decay_rate = 0.995, # per one epoch
        #    staircase = True,
        #    name="var_lr"
        #)
        return tf.Variable(self.initial_learning_rate,
                           name="var_lr", trainable=False)

    @staticmethod
    def create_shallownet(images, scope=None, net=None, dropout=True):
        """
        Args:
            images: a tensor of shape [B x H x W x C]
            net: An optional dict object
            scope: The variable scope for the subgraph, defaults to ShallowNet

        Returns:
            saliency_output: a tensor of shape [B x 48 x 48]
        """
        assert len(images.get_shape()) == 4     # [B, H, W, C]

        if net is None: net = {}
        else: assert isinstance(net, dict)

        net['dropout_keep_prob'] = tf.placeholder(tf.float32, name='dropout_keep_prob')

        with tf.variable_scope(scope or 'ShallowNet'):
            # CONV
            net['conv1'] = convolution2d(images, 64,
                                        kernel_size=(5, 5), stride=(1, 1), padding='VALID',
                                        activation_fn=None,#tf.nn.relu,
                                        weight_init=initializers.xavier_initializer_conv2d(uniform=True),
                                        bias_init=tf.constant_initializer(0.0),
                                        weight_collections=['MODEL_VARS'], bias_collections=['MODEL_VARS'],
                                        name='conv1')
            net['conv1'] = tflearn.layers.batch_normalization(net['conv1'])
            net['conv1'] = tf.nn.relu(net['conv1'])

            net['pool1'] = tf.nn.max_pool(net['conv1'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                          padding='SAME', name='pool1')
            log.info('Conv1 size : %s', net['conv1'].get_shape().as_list())
            log.info('Pool1 size : %s', net['pool1'].get_shape().as_list())

            net['conv2'] = convolution2d(net['pool1'], 128,
                                        kernel_size=(3, 3), stride=(1, 1), padding='VALID',
                                        activation_fn=None,#tf.nn.relu,
                                        weight_init=initializers.xavier_initializer_conv2d(uniform=True),
                                        bias_init=tf.constant_initializer(0.0),
                                        weight_collections=['MODEL_VARS'], bias_collections=['MODEL_VARS'],
                                        name='conv2')
            net['conv2'] = tflearn.layers.batch_normalization(net['conv2'])
            net['conv2'] = tf.nn.relu(net['conv2'])

            net['pool2'] = tf.nn.max_pool(net['conv2'], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                        padding='SAME', name='pool2')
            log.info('Conv2 size : %s', net['conv2'].get_shape().as_list())
            log.info('Pool2 size : %s', net['pool2'].get_shape().as_list())

            net['conv3'] = convolution2d(net['pool2'], 128,
                                        kernel_size=(3, 3), stride=(1, 1), padding='VALID',
                                        activation_fn=None,#tf.nn.relu,
                                        weight_init=initializers.xavier_initializer_conv2d(uniform=True),
                                        bias_init=tf.constant_initializer(0.0),
                                        weight_collections=['MODEL_VARS'], bias_collections=['MODEL_VARS'],
                                        name='conv3')
            net['conv3'] = tflearn.layers.batch_normalization(net['conv3'])
            net['conv3'] = tf.nn.relu(net['conv3'])


            net['pool3'] = tf.nn.max_pool(net['conv3'], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                        padding='SAME', name='pool3')
            log.info('Conv3 size : %s', net['conv3'].get_shape().as_list())
            log.info('Pool3 size : %s', net['pool3'].get_shape().as_list())

            # FC layer
            n_inputs = int(np.prod(net['pool3'].get_shape().as_list()[1:]))
            pool3_flat = tf.reshape(net['pool3'], [-1, n_inputs])
            net['fc1'] = fully_connected(pool3_flat, 98,
                                    activation_fn=None,#tf.nn.relu,
                                    weight_init=initializers.xavier_initializer(uniform=True),
                                    bias_init=tf.constant_initializer(0.0),
                                    weight_collections=['MODEL_VARS'], bias_collections=['MODEL_VARS'],
                                    name='fc1')
            log.info('fc1 size : %s', net['fc1'].get_shape().as_list())

            net['fc1'] = tflearn.layers.batch_normalization(net['fc1'])
            net['fc1'] = tf.nn.relu(net['fc1'])

            if dropout:
                net['fc1'] = tf.nn.dropout( net['fc1'], net['dropout_keep_prob'] )

            fc1_slice1, fc1_slice2 = tf.split(1, 2, net['fc1'], name='fc1_slice')
            net['max_out'] = tf.maximum(fc1_slice1, fc1_slice2, name='fc1_maxout')


            log.info('maxout size : %s', net['max_out'].get_shape().as_list())

            net['fc2'] = fully_connected(net['max_out'], 98 ,
                                        activation_fn=None, # no relu here
                                        weight_init=initializers.xavier_initializer(uniform=True),
                                        bias_init=tf.constant_initializer(0.0),
                                        weight_collections=['MODEL_VARS'], bias_collections=['MODEL_VARS'],
                                        name='fc2')

            net['fc2'] = tflearn.layers.batch_normalization(net['fc2'])
            net['fc2'] = tf.nn.relu(net['fc2'])

            #if dropout:
            #    net['fc2'] = tf.nn.dropout( net['fc2'], net['dropout_keep_prob'] )

            log.info('fc2 size : %s', net['fc2'].get_shape().as_list())

            fc2_slice1, fc2_slice2 = tf.split(1, 2, net['fc2'], name='fc2_slice')
            net['max_out2'] = tf.maximum(fc2_slice1, fc2_slice2, name='fc2_maxout')


            # debug and summary
            #net['fc1'].get_shape().assert_is_compatible_with([None, 4802])
            #net['fc2'].get_shape().assert_is_compatible_with([None, 4802])
            #net['fc3'].get_shape().assert_is_compatible_with([None, 4802])
            #for t in [self.conv1, self.conv2, self.conv3,
            #          self.pool1, self.pool2, self.pool3,
            #          self.fc1, self.max_out, self.fc2]:
            #    _add_activation_histogram_summary(t)

            net['saliency'] = tf.reshape(net['max_out2'], [-1, 7, 7],
                                        name='saliency')

        return net['saliency']


    def build_model(self):

        self.images = tf.placeholder(tf.float32, shape=(None, 98, 98, 3))
        log.info('images : %s', self.images.get_shape().as_list())

        # saliency maps (GT)
        self.saliencymaps_gt = tf.placeholder(tf.float32, shape=(None, 7, 7))
        log.info('gt_saliencymaps : %s', self.saliencymaps_gt.get_shape().as_list())

        # shallow net (inference)
        net = {}
        self.saliency_output = SaliencyModel.create_shallownet(
            self.images,
            net=net
        )
        self.dropout_keep_prob = net['dropout_keep_prob']
        log.info('saliency output: %s', self.saliency_output.get_shape().as_list())


        def _add_activation_histogram_summary(tensor):
            # WARNING: This summary WILL MAKE LEARNING EXTREMELY SLOW
            tf.histogram_summary(tensor.name + '/activation', tensor)
            tf.histogram_summary(tensor.name + '/sparsity', tf.nn.zero_fraction(tensor))
            if hasattr(tensor, 'W'): tf.histogram_summary(tensor.name + '/W', tensor.W)
            if hasattr(tensor, 'b'): tf.histogram_summary(tensor.name + '/b', tensor.b)


        # build euclidean loss
        self.reg_loss = 1e-7 * sum([tf.nn.l2_loss(t) for t in tf.get_collection('MODEL_VARS')])
        self.target_loss = 2.0 * tf.nn.l2_loss(self.saliency_output - self.saliencymaps_gt) / (7 * 7)
        self.target_loss = tf.div(self.target_loss, self.batch_size, name='loss_normalized')
        self.loss = self.reg_loss + self.target_loss

        tf.scalar_summary('loss/total/train', self.loss)
        tf.scalar_summary('loss/total/val', self.loss, collections=['TEST_SUMMARIES'])
        tf.scalar_summary('loss/target/train', self.target_loss)
        tf.scalar_summary('loss/target/val', self.target_loss, collections=['TEST_SUMMARIES'])

        # Debugging Informations
        # ----------------------


        # OPTIONAL: for debugging and visualization
        def _add_image_summary(tag, tensor):
            return tf.image_summary(tag, tensor, max_images=2, collections=['IMAGE_SUMMARIES'])

        _add_image_summary('inputimage', self.images)
        _add_image_summary('saliency_maps_gt', tf.expand_dims(self.saliencymaps_gt, 3))
        _add_image_summary('saliency_maps_pred_original',
                           tf.reshape(self.saliency_output, [-1, 7, 7, 1]))
        _add_image_summary('saliency_maps_pred_norm',
                           tf.reshape(tf_normalize_map(self.saliency_output), [-1, 7, 7, 1]))
                            # normalize_map -> tf_normalize_map

        self.image_summaries = tf.merge_summary(
            inputs = tf.get_collection('IMAGE_SUMMARIES'),
            collections = [],
            name = 'merged_image_summary',
        )

        # activations
        self.model_var_summaries = tf.merge_summary([
            tf.histogram_summary(var.name, var, collections=[]) \
            for var in tf.get_collection('MODEL_VARS')
        ])


    def prepare_data(self):
        self.n_train_instances = len(self.data_sets.train.images)



    def single_step(self, train_mode=True):
        _start_time = time.time()

        """ prepare the input (get batch-style tensor) """
        _dataset = train_mode and self.data_sets.train \
                               or self.data_sets.valid
        batch_images, batch_saliencymaps = _dataset.next_batch(self.batch_size)[:2]

        if len(batch_images[0].shape) == 4:
            # maybe, temporal axis is given [B x T x 96 x 96 x 3]
            # in the plain saliency model, we concatenate all of them
            # to learn/evaluate accuracy across frame independently.
            batch_images = np.concatenate(batch_images)
            batch_saliencymaps = np.concatenate(batch_saliencymaps)

        # Flip half of the images in this batch at random:
        if train_mode and self.config.use_flip_batch:
            batch_size = len(batch_images)
            indices = np.random.choice(batch_size, batch_size / 2, replace=False)
            batch_images[indices, :] = batch_images[indices, :, ::-1, :]
            batch_saliencymaps[indices, :] = batch_saliencymaps[indices, :, ::-1]


        """ run the optimization step """
        _merged_summary = {True: self.merged_summary_train,
                           False: self.merged_summary_val}[train_mode]

        eval_targets = [self.loss, self.target_loss, self.reg_loss, _merged_summary]
        if train_mode: eval_targets += [self.train_op]

        if not train_mode:
            eval_targets += [self.image_summaries]
            eval_targets += [self.model_var_summaries]

        eval_result = dict(zip(eval_targets, self.session.run(
            eval_targets,
            feed_dict = {
                self.images : batch_images,
                self.saliencymaps_gt : batch_saliencymaps,
                self.dropout_keep_prob : 0.4 if train_mode else 1.0
            }
        )))

        loss    = eval_result[self.loss]
        target_loss = eval_result[self.target_loss]
        reg_loss    = eval_result[self.reg_loss]
        summary = eval_result[_merged_summary]

        step  = self.current_step
        epoch = float(step * self.batch_size) / self.n_train_instances # estimated epoch

        if step >= 20:
            self.writer.add_summary(summary, step)

        if not train_mode:
            image_summary = eval_result[self.image_summaries]
            self.writer.add_summary(image_summary, step)
            var_summary = eval_result[self.model_var_summaries]
            self.writer.add_summary(var_summary, step)

        _end_time = time.time()

        if (not train_mode) or np.mod(step, self.config.steps_per_logprint) == 0:
            log_fn = (train_mode and log.info or log.infov)
            log_fn((" [{split_mode:5} epoch {epoch:.1f} / step {step:4d}]  " +
                    "batch total-loss: {total_loss:.5f}, target-loss: {target_loss:.5f} " +
                    "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec)"
                    ).format(split_mode=(train_mode and 'train' or 'val'),
                            epoch=epoch, step=step,
                            total_loss=loss, target_loss=target_loss,
                            sec_per_batch=(_end_time - _start_time),
                            instance_per_sec=self.batch_size / (_end_time - _start_time)
                            )
                   )

        return step


    def evaluate(self, dataset):
        num_steps = len(dataset) / self.batch_size

        gt_maps = []        # GT saliency maps (each 48x48)
        pred_maps = []      # predicted saliency maps (each 48x48)
        fixation_maps = []

        for v in range(num_steps):
            if v % 10 == 0: log.info('Evaluating step %d ...', v)
            batch_images, batch_saliencymaps, batch_fixationmaps = \
                dataset.next_batch(self.batch_size)[:3]

            if len(batch_images[0].shape) == 4:
                # maybe, temporal axis is given [B x T x 96 x 96 x 3]
                # in the plain saliency model, we concatenate all of them
                # to learn/evaluate accuracy across frame independently.
                batch_images = np.concatenate(batch_images)
                batch_saliencymaps = np.concatenate(batch_saliencymaps)
                batch_fixationmaps = chain(*batch_fixationmaps)

            [saliency_output, ] = self.session.run(
                [self.saliency_output, ],
                feed_dict = {
                    self.images : batch_images,
                    self.saliencymaps_gt : batch_saliencymaps,
                    self.dropout_keep_prob : 1.0
                })

            saliency_output = saliency_output.reshape(-1, 7, 7)
            assert len(saliency_output) == len(batch_saliencymaps)

            gt_maps.extend(batch_saliencymaps)
            pred_maps.extend(saliency_output)
            fixation_maps.extend(batch_fixationmaps)

        # Evaluate.
        batch_scores = {}
        log.infov('Validation on total %d images', len(pred_maps))
        for metric in evaluation_metrics.AVAILABLE_METRICS:
            batch_scores[metric] = evaluation_metrics.saliency_score(metric, pred_maps, gt_maps, fixation_maps)
            log.infov('Saliency %s : %f', metric, batch_scores[metric])

        self.report_evaluate_summary(batch_scores)



def self_test(args):
    global model, data_sets

    session = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
        device_count={'GPU': True},        # self-testing: NO GPU, USE CPU
    ))


    log.warn('Loading %s input data ...', args.dataset)
    if args.dataset == 'salicon':
        data_sets = salicon_input_data.read_salicon_data_sets(
            98, 98, 7, 7, np.float32,
            use_example=False, # only tens
            use_val_split=True,
        ) # self test small only

    elif args.dataset == 'crc':
        data_sets = crc_input_data.read_crc_data_sets(
            98, 98, 7, 7, np.float32,
            use_cache=True
        )
    else:
        raise ValueError('Unknown dataset : %s' % args.dataset)

    print 'Train', data_sets.train
    print 'Validation', data_sets.valid

    log.warn('Building Model ...')
    # default configuration as of now
    config = BaseModelConfig()
    config.train_dir = args.train_dir
    if args.train_tag:
        config.train_tag = args.train_tag

    config.batch_size = 200
    config.use_flip_batch = True
    #config.initial_learning_rate = 0.03
    config.initial_learning_rate = 0.00003
    config.optimization_method = 'adam'
    config.steps_per_evaluation = 7000  # for debugging

    if args.learning_rate is not None:
        config.initial_learning_rate = float(args.learning_rate)
    if args.learning_rate_decay is not None:
        config.learning_rate_decay = float(args.learning_rate_decay)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)

    if args.max_steps:
        config.max_steps = int(args.max_steps)

    if args.dataset == 'crc':
        config.batch_size = 2     # because of T~=35
        config.steps_per_evaluation = 200

    config.dump(sys.stdout)

    log.warn('Start Fitting Model ...')
    model = SaliencyModel(session, data_sets, config)
    print model

    model.fit()

    log.warn('Fitting Done. Evaluating!')
    model.evaluate(data_sets.test)
    #session.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='salicon', help='[salicon, crc]')
    parser.add_argument('--max_steps', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--train_dir', default=None, type=str)
    parser.add_argument('--train_tag', '--tag', default=None, type=str)
    parser.add_argument('--learning_rate', default=None, type=float)
    parser.add_argument('--learning_rate_decay', default=None, type=float)
    args = parser.parse_args()

    self_test(args)
