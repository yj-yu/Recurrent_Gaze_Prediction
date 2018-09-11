#!/usr/bin/env python

import numpy as np
import os
import sys
import time

from PIL import Image
import scipy.misc
import tensorflow as tf

from collections import OrderedDict
import cPickle as pkl
import crc_input_data_seq

from util import log, override
from models.base import ModelBase, BaseModelConfig
from models.saliency_shallownet import SaliencyModel

from models.model_util import tf_normalize_map
from evaluation_metrics import saliency_score, AVAILABLE_METRICS

from easydict import EasyDict as E

import errno
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def avg_pool(a):
    T = a.shape[0]
    p = np.zeros((T, 7, 7))
    for i in range(T):
        p[i, :, :] = scipy.misc.imresize(a[i], (7, 7))
        p[i, :, :] /= p[i, :, :].sum()
    return p

def load_model(args):
    global model, config, session

    assert 0.0 < args.gpu_fraction <= 1.0
    session = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction),
        device_count={'GPU': True},        # self-testing: NO GPU, USE CPU
    ))
    log.warn('Reloading Model ...')

    log.infov('MODEL   : %s', args.model)

    if args.model == 'gaze_grcn':
        from gaze_grcn import GazePredictionGRCN as TheModel
        from gaze_grcn import CONSTANTS, GRUModelConfig
    else:
        raise NotImplementedError(args.model)

    # default configuration as of now
    config = GRUModelConfig()
    config.batch_size = args.batch_size or 14
    config.train_dir = None  #important
    config.n_lstm_steps = 105

    if args.loss_type is not None: #important
        config.loss_type = args.loss_type
    else: config.loss_type = 'xentropy'

    #print(type(config))
    config.dump(sys.stdout)# causes error at second call for some reason

    # dummy
    data_sets = crc_input_data_seq.CRCDataSplits()
    data_sets.train = data_sets.test = data_sets.valid = []
    log.warn('Dataset Loading Finished ! (%d instances)', len(data_sets))

    model = TheModel(session, data_sets, config)
    print model

    # load checkpoint
    checkpoint_path = os.path.abspath(args.checkpoint_path)
    

    #assert os.path.isfile(checkpoint_path)  #assertion fails but model shouldn't be loaded with filename extension so this needs to go!
    model.load_model_from_checkpoint_file(checkpoint_path)

    log.warn('Model Loading Done!!')

    return model


def main(args):

    # load LSMDC (hard-code as of now)
    from glob import glob
    VID_C3D_FOLDER = '/data1/common_datasets/LSMDC/vid_c3d/'
    VID_FRM_FOLDER = '/data1/common_datasets/LSMDC/vid_frm_98/'
    GAZEMAP_FOLDER = '/data2/amelie/LSMDC/pred_gazemaps/'
    assert os.path.isdir(VID_C3D_FOLDER), "%s directory doesn't exist" % (VID_C3D_FOLDER)

    video_folders = sorted([os.path.basename(f) for f in glob(VID_C3D_FOLDER + '*')])

    ignored = []
    for i, folder in enumerate(video_folders):
        if not os.path.exists(VID_FRM_FOLDER + folder):
            log.error('not exists : %d, %s', i, folder)
            ignored.append(folder)
        elif len(glob(VID_FRM_FOLDER + folder + "/*.jpg")) == 0:
            log.error('not exists : %d, %s', i, folder)
            ignored.append(folder)
    ignored = set(ignored)

    video_folders = [f for f in video_folders if not f in ignored]
    log.infov('Target : %s folders ...' % len(video_folders))

    if args.reverse:
        video_folders = list(reversed(video_folders))

    def load_c3d(folder_name):
        with open(os.path.join(VID_C3D_FOLDER, folder_name, '%s.c3d' % folder_name)) as f:
            feat = pkl.load(f)

        feat = np.reshape(feat, (-1, 1024, 7, 7))
        return feat

    def ls_img_files(folder_name):
        return sorted(glob(os.path.join(VID_FRM_FOLDER, folder_name, "*")))

    def load_img(f):
        im = scipy.misc.imread(f).astype(np.float32)
        if im.max() > 1:
            im /= 255.0
        assert im.shape == (98, 98, 3)
        return im

    # build model
    global model


    model = load_model(args)
    B = model.batch_size
    T = model.n_lstm_steps  # T^T test generator should be different length..
    log.infov('Batch Size : %d', B)

    # GO!!!!!!
    def generate_batches():
        batch_c3d = np.zeros((B, T, 1024, 7, 7))
        batch_frames = np.zeros((B, T, 98, 98, 3))
        batch_folders = [None] * B
        batch_length = [0] * B
        b = 0


        assert len(video_folders)>0, "video folder is empty "
        for i, folder in enumerate(video_folders):
            # skip?
            gaze_folder = os.path.join(GAZEMAP_FOLDER, folder)
            if os.path.exists(gaze_folder):
                log.warn('Skipped - already exists %s', gaze_folder)
                continue

            batch_folders[b] = folder
            clip = False

            # c3d
            log.info('%d | processing %s ...', i, folder)
            c3d = load_c3d(folder)

            c3d_len = c3d.shape[0]
            if c3d_len < T:
                c3d = np.lib.pad(c3d, [(0, T - c3d_len), (0, 0), (0, 0), (0, 0)],
                                 'constant', constant_values=np.float32(0.0))
            elif c3d_len == T:
                pass
            else:
                clip = True
                log.warn('%d %s : Too long. c3d_len = %d, rnn steps = %d' % (
                    i, folder, c3d_len, T))

            # images
            image_list = ls_img_files(folder)[15::5]
            if len(image_list) != c3d_len:
                import pudb; pudb.set_trace()  # XXX BREAKPOINT
                assert False, "%d : %s length differs!!! (%d != %d)" % \
                    (i, folder, len(image_list), c3d_len)

            frames = np.array([load_img(f) for f in image_list])

            if len(frames) < T:
                frames = np.lib.pad(frames, [(0, T - len(frames)), (0, 0), (0, 0), (0, 0)],
                                    'constant', constant_values=np.float32(0.0))

            if clip:
                frames = frames[:T]
                c3d = c3d[:T]
                c3d_len = T

            batch_c3d[b, :, :, :, :] = c3d
            batch_frames[b, :, :, :] = frames
            batch_length[b] = c3d_len

            # yield.
            b += 1
            
            if b >= B:
                b = 0
                yield batch_folders, batch_c3d, batch_frames, batch_length

        if b > 0:
            yield batch_folders, batch_c3d, batch_frames, batch_length

    
    

    for batch_folders, batch_c3d, batch_frames, batch_length in generate_batches():
        
        
        # TODO INSTEAD, MAKE INFERENCE() function in the model!!!!!!!
        [gazes,] = model.session.run(
            [model.predicted_gazemaps],
            feed_dict={
                model.c3d_input: batch_c3d,
                model.frame_images: batch_frames,
            }
        )
        for folder, length, gaze in zip(batch_folders, batch_length, gazes):
            #print folder, length, gaze.shape (100, 49, 49)
            mkdir_p(os.path.join(GAZEMAP_FOLDER, folder))
            gaze_file_77 = os.path.join(GAZEMAP_FOLDER, folder, '%s.gazemap.npy' % folder)
            gaze_file_49 = os.path.join(GAZEMAP_FOLDER, folder, '%s.gazemap.49.npy' % folder)

            gaze_t_49 = gaze[:length, :, :]
            gaze_t_77 = avg_pool(gaze[:length, :, :])
            np.save(gaze_file_49, gaze_t_49)
            np.save(gaze_file_77, gaze_t_77)
            log.info('%s : saved length = %d', folder, length)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=str,
                        help='e.g. gaze_grcn gaze_rnn77 gaze_rnn gaze_framewise_shallownet')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--gpu_fraction', default=0.48, type=float)
    parser.add_argument('--checkpoint_path', required=True, type=str)
    parser.add_argument('--loss_type', default='l2', choices=['l2', 'xentropy'])
    args = parser.parse_args()

    main(args)
