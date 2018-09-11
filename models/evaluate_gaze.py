#!/usr/bin/env python

import numpy as np
import os
import sys
import time

from PIL import Image
import scipy.misc
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

import random
from collections import OrderedDict
from collections import defaultdict
import cPickle as pkl
import crc_input_data_seq

from util import log, override
from models.base import ModelBase, BaseModelConfig
from models.saliency_shallownet import SaliencyModel

from models.model_util import tf_normalize_map
from evaluation_metrics import saliency_score, AVAILABLE_METRICS, resize_onehot_tensor_sparse

from joblib import Parallel, delayed
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
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction,
                                  allow_growth=True),
        device_count={'GPU': True},        # self-testing: NO GPU, USE CPU
    ))
    log.warn('Reloading Model ...')

    log.infov('MODEL   : %s', args.model)

    if args.model == 'gaze_grcn':
        from models.gaze_grcn import GazePredictionGRCN as TheModel
        from models.gaze_grcn import CONSTANTS, GRUModelConfig
    elif args.model == 'gaze_deeprnn':
        from gaze_rnn_deep import DEEPRNN as TheModel
        from gaze_rnn_deep import CONSTANTS, GRUModelConfig
    elif args.model == 'gaze_rnn':
        from models.gaze_rnn import GazePredictionGRU as TheModel
        from models.gaze_rnn import CONSTANTS, GRUModelConfig
    elif args.model == 'gaze_c3d_conv':
        from models.gaze_c3d_conv import GazePredictionConv as TheModel
        from models.gaze_c3d_conv import CONSTANTS, GRUModelConfig
    elif args.model == 'gaze_shallownet_rnn':
        from models.gaze_shallownet_rnn import GazePredictionGRU as TheModel
        from models.gaze_shallownet_rnn import CONSTANTS, GRUModelConfig
    elif args.model == 'framesaliency':
        from models.gaze_framewise_shallownet import FramewiseShallowNet as TheModel
        from models.gaze_framewise_shallownet import CONSTANTS, GRUModelConfig
    else:
        raise NotImplementedError(args.model)

    # default configuration as of now
    config = GRUModelConfig()
    config.batch_size = args.batch_size or 14

    if args.loss_type is not None:  # important
        config.loss_type = args.loss_type
    else:
        config.loss_type = 'xentropy'

    config.dump(sys.stdout)

    # dummy
    data_sets = crc_input_data_seq.CRCDataSplits()
    data_sets.train = data_sets.test = data_sets.valid = []
    import pdb; pdb.set_trace()
    
    log.warn('Dataset Loading Finished ! (%d instances)', len(data_sets))

    model = TheModel(session, data_sets, config)
    print model

    return model


from evaluation_metrics import saliency_score, saliency_score_single, AVAILABLE_METRICS

# XXX dirty global
fixationmaps_all = None


def handle_frame(i, n_images, image, pred_gazemap, gt_gazemap, fixationmap,
                 out_dir):
    global fixationmaps_all

    #log.info("[%d/%d] fixationmap shape = %s", i, n_images, fixationmap.shape)

    # compute other map union only once per original fixationmap shape..?
    M = 10
    other_map_union = scipy.sparse.coo_matrix(
        fixationmap.shape, dtype=np.uint8)
    for oth in np.random.choice(range(len(fixationmaps_all)), M, replace=False):
        other_map = fixationmaps_all[oth]
        if other_map.shape != fixationmap.shape:
            other_map = resize_onehot_tensor_sparse(
                other_map, fixationmap.shape)
        other_map_union += (other_map > 0).astype(np.uint8)

    # evaluate metric
    scores = {}
    for metric in ('sim', 'cc', 'AUC_Borji', 'AUC_Judd', 'AUC_shuffled'):
        score = saliency_score_single(metric,
                                      pred_map=pred_gazemap,
                                      gt_map=gt_gazemap,
                                      fixation_map=fixationmap,
                                      other_map_union=other_map_union
                                      )
        scores[metric] = score

    log.debug(' '.join('[%d/%d] %s : %.4f' % (i, n_images, k, v)
                       for k, v in scores.iteritems()))

    # dump images and prediction results
    scipy.misc.imsave(os.path.join(out_dir, '%05d.frame.jpg' % i), image)
    scipy.misc.imsave(os.path.join(
        out_dir, '%05d.gaze_pred.jpg' % i), pred_gazemap)
    scipy.misc.imsave(os.path.join(
        out_dir, '%05d.gaze_gt.jpg' % i), gt_gazemap)
    with open(os.path.join(out_dir, '%05d.scores.txt' % i), 'w') as fp:
        fp.write('%d / %d\n' % (i, n_images))
        for k, v in scores.iteritems():
            fp.write('%s : %.4f\n' % (k, v))

    return scores


def reload_checkpoint(model, checkpoint_path):
    # load checkpoint
    

    checkpoint_path = os.path.abspath(checkpoint_path)
    #assert os.path.isfile(checkpoint_path)
    model.load_model_from_checkpoint_file(checkpoint_path)

    log.warn('Model Loading Done!!')


def run_evaluation(model, args, data_sets, out_dir):
    assert out_dir is not None

    # feed forward all outputs in the datset
    T = model.n_lstm_steps  # T^T test generator should be different length..
    print 'Test length : %d' % len(data_sets.valid)  # XXX valid

    ret = model.generate(data_sets.valid,  # XXX valid
                         max_instances=int(
                             np.divide(args.num_frames, T, dtype=float) + 1)
                         )
    # e.g. max_instances 50 x batch_size (35) == 1750 frames are sampled

    pred_gazemaps = ret['pred_gazemaps'] = np.asarray(ret['pred_gazemap_list'])
    gt_gazemaps = ret['gt_gazemaps'] = np.asarray(ret['gt_gazemap_list'])
    images = ret['images'] = np.asarray(ret['images_list'])
    fixationmaps = ret['fixationmap_list']  # list, not ndarray
    import pdb; pdb.set_trace()

    # dirty!! (IPC overhead slow)
    global fixationmaps_all
    fixationmaps_all = fixationmaps

    # DUMP ALL IMAGES
    n_images = len(pred_gazemaps)
    assert n_images == len(pred_gazemaps) == len(
        gt_gazemaps) == len(images) == len(fixationmaps)

    aggreagted_scores = defaultdict(list)
    with Parallel(n_jobs=args.jobs, verbose=10) as parallel:
        # each item in scores_aggregated is a {metric->float} dict
        scores_aggregated = parallel(delayed(handle_frame)(
            i, n_images, images[i], pred_gazemaps[i], gt_gazemaps[i], fixationmaps[i],
            out_dir
        ) for i in xrange(n_images))

        # synchronization barrier.
        scores_aggregated = list(scores_aggregated)

    for scores in scores_aggregated:
        # metric -> float map
        for metric, score in scores.iteritems():
            aggreagted_scores[metric].append(score)

    # report aggregated score
    overall_txt_path = os.path.join(out_dir, 'overall.txt')
    with open(overall_txt_path, 'w') as fp:
        for metric, score_list in aggreagted_scores.iteritems():
            mean_score = np.mean(score_list)
            log.infov("Average %s : %.4f" % (metric, mean_score))

            fp.write("Average %s : %.4f\n" % (metric, mean_score))
            for score in score_list:
                fp.write('%.3f ' % score)
            fp.write('\n')
    log.warn("Dumped at %s", overall_txt_path)


def get_out_dir(dataset, checkpoint_path):
    # out_dir
    try:
        checkpoint_step = os.path.basename(checkpoint_path).split('-')[-1]
    except:
        checkpoint_step = 'latest'
    out_dir = os.path.abspath(
        os.path.join(checkpoint_path,
                     '../../generated-{}-{}'.format(dataset, checkpoint_step))
    )
    mkdir_p(out_dir)
    log.info("Out directory : %s", out_dir)
    return out_dir


def main(args):
    if not args.embed and not args.checkpoint_path:
        raise ValueError('checkpoint_path needed')

    # build model
    global model
    model = load_model(args)
    B = model.batch_size
    log.infov('Batch Size : %d', B)

    # load datasets (validation only)
    log.info("Data loading start !")
    data_sets = crc_input_data_seq.read_crc_data_sets(
        98, 98, 49, 49, np.float32,
        use_cache=False,
        parallel_jobs=10,
        dataset=args.dataset,
        split_modes=['valid'],  # XXX valid
        fixation_original_scale=True,  # WTF
        max_folders=500  # XXX FOR FAST DEBUGGING AND TESTING
    )
    log.info("Data loading done")

    if args.embed:
        def go(checkpoint_path):
            reload_checkpoint(model, checkpoint_path)
            out_dir = get_out_dir(args.dataset, checkpoint_path)
            print 'go at %s' % checkpoint_path
            run_evaluation(model, args, data_sets, out_dir)
            import gc
            gc.collect()

        log.infov('Usage: >>> go(checkpoint_path)')
        from IPython import embed
        embed()
    else:
        reload_checkpoint(model, args.checkpoint_path)

        out_dir = get_out_dir(args.dataset, args.checkpoint_path)
        run_evaluation(model, args, data_sets, out_dir)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=str,
                        help='e.g. gaze_grcn gaze_rnn77 gaze_rnn gaze_framewise_shallownet')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--dataset', default='crc',
                        choices=['hollywood2', 'crc'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu_fraction', default=0.48, type=float)
    parser.add_argument('--checkpoint_path', required=False, type=str)
    parser.add_argument('--loss_type', type=str, choices=['l2', 'xentropy'])
    parser.add_argument('--num_frames', type=int, default=1500)
    parser.add_argument('-j', '--jobs', type=int, default=10)
    parser.add_argument('--embed', action='store_true')
    args = parser.parse_args()

    main(args)
