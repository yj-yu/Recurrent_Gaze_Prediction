import numpy as np
import tensorflow as tf

__all__ = ["load_data"]

def _parse_function(serialized):

    mode = "train"
    #filename_queue = tf.train.string_input_producer([tfrecord], num_epochs=1)

    reader = tf.TFRecordReader()
    #_,serialized_example.read(filename_queue)
    
    features = { '/input/gazemaps_pred' : tf.FixedLenFeature([], tf.string),
                        '/input/gazemaps_gt' :  tf.FixedLenFeature([], tf.string),
                        '/input/frame' : tf.FixedLenFeature([], tf.string),
                        '/label/label' : tf.FixedLenFeature([], tf.string),
                        '/input/c3d' : tf.FixedLenFeature([], tf.string)

    }

    
    
    
    parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)

    frames_raw = parsed_example['/input/frame']
    c3d_raw = parsed_example['/input/c3d']
    labels_raw = parsed_example['/label/label']
    gazemaps_gt_raw = parsed_example['/input/gazemaps_gt']
    gazemaps_pred_raw = parsed_example['/input/gazemaps_pred']

    c3d_shape = [1024,7,7]
    frames_shape = [98,98,3]
    labels_shape = [13]
    gazemaps_gt_shape = [49,49]
    gazemaps_pred_shape = [49,49]

    frames = tf.decode_raw(frames_raw,tf.float32)
    frames = tf.reshape(frames,frames_shape)
    c3d = tf.decode_raw(c3d_raw,tf.float32)
    c3d = tf.reshape(c3d, c3d_shape)
    labels = tf.decode_raw(labels_raw, tf.uint8)
    labels = tf.reshape(labels, labels_shape)
    gazemaps_gt = tf.decode_raw(gazemaps_gt_raw, tf.float32)
    gazemaps_gt = tf.reshape(gazemaps_gt, gazemaps_gt_shape)
    gazemaps_pred = tf.decode_raw(gazemaps_pred_raw, tf.float32)
    gazemaps_pred = tf.reshape(gazemaps_pred, gazemaps_pred_shape)

    return frames, c3d,labels,gazemaps_gt, gazemaps_pred
    


def load_data(dataset,batch_size, filename, validation = False):

    #filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filename)

    dataset = dataset.map(_parse_function)
    #dataset = dataset.map(...)  # Parse the record into tensors.
    #dataset = dataset.repeat()  # Repeat the input indefinitely.
    #dataset = dataset.shuffle()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()

    #if dataset == "valid":
    #if validation:
        
    #    iterator = dataset.make_one_shot_iterator()
        
    next_element = iterator.get_next()

   # filename = "{}.tfrecord".format(dataset)

    
    return  iterator,dataset



    
if __name__ == "__main__":
    #filenames = tf.placeholder(tf.string, shape=[None])
    train_file = ["train.tfrecord"]
    valid_file = ["valid.tfrecord"]
    dataset = tf.data.TFRecordDataset(train_file)

    dataset = dataset.map(_parse_function)
    #dataset = dataset.map(...)  # Parse the record into tensors.
    #dataset = dataset.repeat()  # Repeat the input indefinitely.

    dataset = dataset.batch(10)
    
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        i = 0
        for _ in range(8000):  # fails because crc is small
            frames, c3d, labels, gaze_gt,gaze_pred= sess.run(next_element)
            i = i +1
            print(frames.shape, c3d.shape, gaze_gt.shape, gaze_pred.shape)
            print(i)
