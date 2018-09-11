"""
Most of codes are "COPIED" from saliconeval.auc, etc.
just because, the salicon interfaces sucks :-(

and https://github.com/herrlich10/saliency/blob/master/benchmark/metrics.py
"""

import numpy as np
from skimage.transform import resize
import numpy.random as random
from functools import partial
import scipy.sparse


def normalize_range(x):
    res = (x - np.min(x)) / (np.max(x) - np.min(x))
    return res

def resize_onehot_tensor_sparse(x, target_shape):
    assert len(target_shape) == 2

    H1, W1 = x.shape[-2:]
    H2, W2 = target_shape

    if len(x.shape) == 2:
        ret = np.zeros((H2, W2), dtype=np.bool)
        for y, x in zip(*np.where(x > 0)):
            y_ = y * (H2 - 1.0) / (H1 - 1.0)
            x_ = x * (W2 - 1.0) / (W1 - 1.0)
            y_ = int(np.round(y_) + 1e-9)
            x_ = int(np.round(x_) + 1e-9)
            #print t, y, x, '=>', y_, x_
            ret[y_, x_] = 1

    else:
        raise ValueError('x.shape : %s' % x.shape)


    return ret


def AUC_Judd(fixation_map, saliency_map, jitter=True):
    '''
    AUC stands for Area Under ROC Curve.
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
    ROC curve is created by sweeping through threshold values
    determined by range of saliency map values at fixation locations.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at all other locations to the total number of possible other locations (non-fixated image pixels).
    AUC=0.5 is chance level.
    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
        Human fixation map.
    jitter : boolean, optional
        If True (default), a small random number would be added to each pixel of the saliency map.
        Jitter saliency maps that come from saliency models that have a lot of zero values.
        If the saliency map is made with a Gaussian then it does not need to be jittered
        as the values vary and there is not a large patch of the same value.
        In fact, jittering breaks the ordering in the small values!
    Returns
    -------
    AUC : float, between [0,1]
    '''
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='nearest')
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize_range(saliency_map)

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
        tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
    return np.trapz(tp, fp) # y, x


def AUC_Borji(fixation_map, saliency_map, n_rep=100, step_size=0.1, rand_sampler=None):
    '''
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
    ROC curve created by sweeping through threshold values at fixed step size
    until the maximum saliency map value.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at random locations to the total number of random locations
    (as many random locations as fixations, sampled uniformly from fixation_map ALL IMAGE PIXELS),
    averaging over n_rep number of selections of random locations.
    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
        Human fixation map.
    n_rep : int, optional
        Number of repeats for random sampling of non-fixated locations.
    step_size : int, optional
        Step size for sweeping through saliency map.
    rand_sampler : callable
        S_rand = rand_sampler(S, F, n_rep, n_fix)
        Sample the saliency map at random locations to estimate false positive.
        Return the sampled saliency values, S_rand.shape=(n_fix,n_rep)
    Returns
    -------
    AUC : float, between [0,1]
    '''
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='nearest')
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize_range(saliency_map)

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # For each fixation, sample n_rep values from anywhere on the saliency map
    if rand_sampler is None:
        r = random.randint(0, n_pixels, [n_fix, n_rep])
        S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
    else:
        S_rand = rand_sampler(S, F, n_rep, n_fix)
    # Calculate AUC per random split (set of random locations)
    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds)+2)
        fp = np.zeros(len(thresholds)+2)
        tp[0] = 0; tp[-1] = 1
        fp[0] = 0; fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc) # Average across random splits


def AUC_shuffled(fixation_map, saliency_map, other_map, n_rep=100, step_size=0.1):
    '''
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
    ROC curve created by sweeping through threshold values at fixed step size
    until the maximum saliency map value.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at random locations to the total number of random locations
    (as many random locations as fixations, sampled uniformly from fixation_map ON OTHER IMAGES),
    averaging over n_rep number of selections of random locations.
    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
        Human fixation map.
    other_map : binary matrix, same shape as fixation_map
        A binary fixation map (like fixation_map) by taking the union of fixations from M other random images
        (Borji uses M=10).
    n_rep : int, optional
        Number of repeats for random sampling of non-fixated locations.
    step_size : int, optional
        Step size for sweeping through saliency map.
    Returns
    -------
    AUC : float, between [0,1]
    '''
    other_map = np.array(other_map, copy=False) > 0.5
    if other_map.shape != fixation_map.shape:
        raise ValueError('other_map.shape != fixation_map.shape')
    # For each fixation, sample n_rep values (from fixated locations on other_map) on the saliency map
    def sample_other(other, S, F, n_rep, n_fix):
        fixated = np.nonzero(other)[0]
        indexer = map(lambda x: random.permutation(x)[:n_fix], np.tile(range(len(fixated)), [n_rep, 1]))
        r = fixated[np.transpose(indexer)]
        S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
        return S_rand
    return AUC_Borji(fixation_map, saliency_map, n_rep, step_size, partial(sample_other, other_map.ravel()))


def similarity(gtsAnn, resAnn):
    """
    Compute Sim score.
    For detailed explanation, refer to the DeepFix (2015) paper.
    """

    # normalize
    gtsAnnNorm = gtsAnn / gtsAnn.sum()
    resAnnNorm = resAnn / resAnn.sum()

    simMap = np.minimum(gtsAnnNorm, resAnnNorm)
    return simMap.sum()


def cc(gtsAnn, resAnn):
    """
    Compute CC score. A simple implementation
    :param gtsAnn: ground-truth fixation map (X by X)
    :param resAnn: predicted saliency map (X by X)
    :return score: float : score
    """

    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]


def saliency_score_single(metric, pred_map, gt_map, fixation_map, other_map_union=None):
    if scipy.sparse.issparse(fixation_map):
        fixation_map = fixation_map.toarray()

    # normalize responses to [0, 1]
    # which makes all of the evaluation metric invariant to scale transformation.
    pred_map = normalize_range(pred_map)

    # here, all the maps are input-image scale (e.g. H=360xW=480)
    pred_map_orig = resize(pred_map, fixation_map.shape, order=3)
    # TODO resize is performed at AUC

    # re-building fixation map in the original scale HERE is very inefficient
    # FIXME it would different from the original fixmation map
    # (obtained from gaussian filter) in pre-processing script
    # TODO if metric is wierd, we may need to draw this gt_map_orig
    gt_map_orig = resize(gt_map, fixation_map.shape, order=3)

    if metric == 'cc':
        score = cc(gt_map_orig, pred_map_orig)
    elif metric == 'sim':
        score = similarity(gt_map_orig, pred_map_orig)
    elif metric == 'AUC_Judd':
        score = AUC_Judd(fixation_map, pred_map_orig)
    elif metric == 'AUC_Borji':
        score = AUC_Borji(fixation_map, pred_map_orig)
    elif metric == 'AUC_shuffled':
        if other_map_union is None:
            raise ValueError('other_map_union required')
        score = AUC_shuffled(fixation_map, pred_map_orig, other_map_union)
    else:
        raise ValueError(metric)

    return score


def saliency_score(metric, pred_maps, gt_maps, fixation_maps):
    '''
    Args:
        metric: sim, cc, AUC_shuffled, AUC_Borji, AUC_Judd
    '''
    assert len(gt_maps) == len(pred_maps) == len(fixation_maps)

    # M is the number of maps used in _other_map_union
    M = 10
    assert len(fixation_maps) >= M
    other_map_union = np.zeros(fixation_maps[0].shape)
    for i in random.choice(range(len(fixation_maps)), M, replace=False):
        other_map_union += (fixation_maps[i] > 0).astype(np.int)

    scores = []
    for gt_map, pred_map, fixation_map in zip(gt_maps, pred_maps, fixation_maps):
        score = saliency_score_single(metric, pred_map, gt_map, fixation_map,
                                      other_map_union)
        scores.append(score)

    return np.mean(scores)

AVAILABLE_METRICS = ('sim', 'cc', 'AUC_shuffled', 'AUC_Borji',) #, 'AUC_Judd')
