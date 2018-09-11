import numpy as np
import tensorflow as tf


def tf_normalize_map(t):
    """
    Normalize each HxW entry to [0, 1] scale, for tensorflow tensors.
    """
    if len(t.get_shape()) == 3: # [B x H x W]
        t = t - tf.reduce_min(t, [1, 2], keep_dims=True)
        t = t / tf.reduce_max(t, [1, 2], keep_dims=True)
    elif len(t.get_shape()) == 4: # [B x H x W x 1]
        assert t.get_shape().as_list()[3] == 1
        t = t - tf.reduce_min(t, [1, 2, 3], keep_dims=True)
        t = t / tf.reduce_max(t, [1, 2, 3], keep_dims=True)
    else:
        raise ValueError("Unsupported shape : {}".format(t.get_shape()))
    return t

def normalize_map(t):
    """
    Normalize each HxW entry to [0, 1] scale, for numpy arrays
    """
    t = np.array(t, copy=True)
    if len(t.shape) == 3: # [B x H x W]
        pass
    elif len(t.shape) == 4: # [B x H x W x 1]
        pass
    else:
        raise ValueError("Unsupported shape : {}".format(t.get_shape()))

    assert t.dtype == np.float32 or t.dtype == float
    for i in range(len(t)):
        t[i, :] -= t[i, :].min()
        if t[i, :].max() > 0:
            t[i, :] /= t[i, :].max()

    return t

def normalize_probability_map(t):
    """
    Normalize each to probability map, for numpy arrays
    """
    #assert (t >= 0.0).all()
    assert t.dtype == np.float32 or t.dtype == float
    t = np.array(t, copy=True)

    if len(t.shape) == 3: # [B x H x W]
        for i in range(len(t)):
            t[i, :] /= t[i, :].sum()
    elif len(t.shape) == 4: # [B x T x H x W]
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                t[i, j, :] /= t[i, j, :].sum()
    else:
        raise ValueError("Unsupported shape : {}".format(t.shape))

    return t


def tf_softmax_2d(logits, name=None):
    [B, H, W] = logits.get_shape().as_list()
    softmaxed = tf.nn.softmax(tf.reshape(logits, [B, H*W]))
    return tf.reshape(softmaxed, [B, H, W], name)

def tf_softmax_cross_entropy_with_logits_2d(logits, labels, name=None):
    [B, H, W] = logits.get_shape().as_list()
    assert [B, H, W] == labels.get_shape().as_list()

    logits_flat = tf.reshape(logits, [B, -1])
    labels_flat = tf.reshape(labels, [B, -1])
    return tf.nn.softmax_cross_entropy_with_logits(logits = logits_flat, labels = labels_flat, name=name)
