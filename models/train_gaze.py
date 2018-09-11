#!/usr/bin/env python

import numpy as np
import os
import sys
import time

from PIL import Image
import tensorflow as tf

from collections import OrderedDict
import cPickle as pkl
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #is in parent directory 
import crc_input_data_seq

from util import log, override
from models.base import ModelBase, BaseModelConfig
#from models.saliency_shallownet import SaliencyModel

from models.model_util import tf_normalize_map
from evaluation_metrics import saliency_score, AVAILABLE_METRICS #not used??

from easydict import EasyDict as E



def train(args):
    global model, config

    assert 0.0 < args.gpu_fraction <= 1.0
    session = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction,
                                  allow_growth=True),
        device_count={'GPU': True},        # self-testing: NO GPU, USE CPU
    ))

    log.warn('Building Model ...')

    log.infov('MODEL   : %s', args.model)
    log.infov('DATASET : %s', args.dataset)
    if args.model == 'gaze_grcn':
        from gaze_grcn import GazePredictionGRCN as TheModel
        from gaze_grcn import CONSTANTS, GRUModelConfig
    elif args.model == 'gaze_lstm':
        from gaze_lstm import GazePredictionLSTM as TheModel
        from gaze_lstm import CONSTANTS, GRUModelConfig
    elif args.model == 'gaze_grcn77':
        from gaze_grcn77 import GazePredictionGRCN as TheModel
        from gaze_grcn77 import CONSTANTS, GRUModelConfig
    elif args.model == 'gaze_rnn77':
        from gaze_rnn77 import GazePredictionGRU as TheModel
        from gaze_rcn77 import CONSTANTS, GRUModelConfig
    elif args.model == 'gaze_rnn':
        from gaze_rnn import GazePredictionGRU as TheModel
        from gaze_rnn import CONSTANTS, GRUModelConfig
    elif args.model == 'gaze_c3d_conv':
        from gaze_c3d_conv import GazePredictionConv as TheModel
        from gaze_c3d_conv import CONSTANTS, GRUModelConfig
    elif args.model == 'gaze_shallownet_rnn':
        from gaze_shallownet_rnn import GazePredictionGRU as TheModel
        from gaze_shallownet_rnn import CONSTANTS, GRUModelConfig
    elif args.model == 'gaze_framewise_shallownet':
        from gaze_framewise_shallownet import FramewiseShallowNet as TheModel
        from gaze_framewise_shallownet import CONSTANTS, GRUModelConfig
    elif args.model == 'gaze_deeprnn':
        from gaze_rnn_deep import DEEPRNN as TheModel
        from gaze_rnn_deep import CONSTANTS, GRUModelConfig
    else:
        raise NotImplementedError(args.model)

    # default configuration as of now
    config = GRUModelConfig()

    # CRC likes 28 :)
    config.batch_size = 28

    config.train_dir = args.train_dir
    if args.train_tag:
        config.train_tag = args.train_tag
    config.initial_learning_rate = 0.0001
    config.max_grad_norm = 10.0
    config.use_flip_batch = True

    if args.max_grad_norm is not None:
        config.max_grad_norm = args.max_grad_norm
    if args.learning_rate is not None:
        config.initial_learning_rate = float(args.learning_rate)
    if args.learning_rate_decay is not None:
        config.learning_rate_decay = float(args.learning_rate_decay)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.loss_type is not None:
        config.loss_type = args.loss_type

    config.steps_per_evaluation = 100
    config.steps_per_validation = 20
    config.steps_per_checkpoint = 100


    if args.max_steps:
        config.max_steps = int(args.max_steps)

    config.dump(sys.stdout)

    log.warn('Dataset (%s) Loading ...', args.dataset)
    assert args.dataset in ('crc', 'hollywood2', 'crcxh2')
    data_sets = crc_input_data_seq.read_crc_data_sets(CONSTANTS.image_height,
                                                    CONSTANTS.image_width,
                                                    CONSTANTS.gazemap_height,
                                                    CONSTANTS.gazemap_width,
                                                    np.float32,
                                                    use_cache=True,
                                                    batch_norm = args.batch_norm,
                                                    dataset=args.dataset)


    
    log.warn('Dataset Loading Finished ! (%d instances)',
             len(data_sets))


    log.warn('Start Fitting Model ...')
    model = TheModel(session, data_sets, config)
    
    print model

    if args.shallownet_pretrain is not None:
        log.warn('Loading ShallowNet weights from checkpoint %s ...', args.shallownet_pretrain)
        model.initialize_pretrained_shallownet(args.shallownet_pretrain)

    model.fit()

    log.warn('Fitting Done. Evaluating!')
    model.generate_and_evaluate(data_sets.test, max_instances=None) #WHERE IS THIS FUNCTION I CANNOT FIND IT
    #session.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=str,
                        help='e.g. gaze_grcn gaze_grcn77 gaze_rnn77 gaze_rnn gaze_framewise_shallownet')
    parser.add_argument('--max_steps', default=None, type=int)
    parser.add_argument('--batch_norm', action= 'store_true')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--gpu_fraction', default=0.48, type=float)
    parser.add_argument('--train_dir', default=None, type=str)
    parser.add_argument('--train_tag', '--tag', default=None, type=str)
    parser.add_argument('--max_grad_norm', default=None, type=float)
    parser.add_argument('--learning_rate', default=None, type=float)
    parser.add_argument('--learning_rate_decay', default=None, type=float)
    parser.add_argument('--loss_type', type=str, choices=['l2', 'xentropy', 'KLD'])
    parser.add_argument('--dataset', default='crc',
                        help='[crc, hollywood2]')
    parser.add_argument('--shallownet_pretrain', default=None, type=str)
    args = parser.parse_args()

    train(args)
