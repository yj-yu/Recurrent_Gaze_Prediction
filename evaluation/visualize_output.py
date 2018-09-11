# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import cPickle as pkl

import matplotlib.pyplot as plt

from util import log, override

from models.base import BaseModelConfig

import crc_input_data_seq
import tensorflow as tf



# in a courtesy of Caffe's filter visualization example
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

def imshow_grid(data, height=None, width=None, normalize=False, padsize=1, padval=0):
    '''
    Take an array of shape (N, H, W) or (N, H, W, C)
    and visualize each (H, W) image in a grid style (height x width).
    '''
    if normalize:
        data -= data.min()
        data /= data.max()

    N = data.shape[0]
    if height is None:
        if width is None:
            height = int(np.ceil(np.sqrt(N)))
        else:
            height = int(np.ceil( N / float(width) ))

    if width is None:
        width = int(np.ceil( N / float(height) ))

    assert height * width >= N

    # append padding
    padding = ((0, (width*height) - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((height, width) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((height * data.shape[1], width * data.shape[3]) + data.shape[4:])

    return plt.imshow(data)


def visualize_outputs(images, gt_gazemap, pred_gazemap):
    print 'Not implemented Yet :('
    #imshow_grid(images, width=35)

def visualize_outputs_of(model, dataset, max_instances=100):
    """
    Args:
        model: The Model (gaze-rnn or gaze-grcn).
        dataset: a dataset split (e.g. train set or validation set)

    Returns:
        a dictionary that contains the model output (e.g.
        images, gt_gazemaps, etc.)
    """

    # (i) run feed forward
    global ret
    ret = model.generate(dataset, max_instances=max_instances)

    pred_gazemaps = ret['pred_gazemaps'] = np.asarray(ret['pred_gazemap_list'])
    gt_gazemaps = ret['gt_gazemaps'] = np.asarray(ret['gt_gazemap_list'])
    images = ret['images'] = np.asarray(ret['images_list'])

    visualize_outputs(images,
                      gt_gazemaps,
                      pred_gazemaps)

    return ret


from models.gaze_grcn import GazePredictionGRCN
from models.gaze_rnn77 import GazePredictionGRU, GRUModelConfig

def visualize_outputs_wrapper(checkpoint_path, session=None,
                              split_mode='valid',
                              dataset='hollywood2',
                              model_class=GazePredictionGRU,
                              max_instances=100):

    if session is None:
        session = tf.InteractiveSession()

    # WTF model persistence

    if checkpoint_path is not None:
        # load config and data loader
        #config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.pkl')
        #with open(config_path, 'rb') as fp:
        #    config = pkl.load(fp)
        #    log.info('Loaded config from %s', config_path)
        #    config.dump(sys.stdout)
        config_path = os.path.join(os.path.dirname(checkpoint_path), '../config.json')
        config = BaseModelConfig.load(config_path)
        log.info('Loaded config from %s', config_path)
    else:
        # default config!?
        config = GRUModelConfig()

    # do not noise original train dirs.
    config.train_dir = None
    config.dump(sys.stdout)

    log.warn('Dataset (%s) Loading ...', dataset)
    assert dataset in ('crc', 'hollywood2')

    from models.gaze_rnn import CONSTANTS # XXX Dirty here
    data_sets = crc_input_data_seq.read_crc_data_sets(CONSTANTS.image_height,
                                                      CONSTANTS.image_width,
                                                      CONSTANTS.gazemap_height,
                                                      CONSTANTS.gazemap_width,
                                                      np.float32,
                                                      use_cache=True,
                                                      dataset=dataset,
                                                      split_modes=[split_mode])

    # resurrect model
    # XXX remove hard-codes

    # TODO assuming there can be only one graph in the process?
    # TODO should any of our model should contain a graph context ????
    #tf.reset_default_graph()
    model = model_class(session, data_sets, config)

    # load checkpoint
    if checkpoint_path is not None:
        assert os.path.isfile(checkpoint_path)
        model.load_model_from_checkpoint_file(checkpoint_path)

    # Run!
    if split_mode == 'valid':
        model_outputs = visualize_outputs_of(model, data_sets.valid, max_instances)
    elif split_mode == 'train':
        model_outputs = visualize_outputs_of(model, data_sets.train, max_instances)
    else:
        raise ValueError(split_mode)

    return model, model_outputs
