import numpy as np
import sys
import os
import tensorflow as tf
import glob
import cPickle as pkl
from basic_graphs import NN_base
from collections import OrderedDict, defaultdict
from sklearn.metrics import hamming_loss, zero_one_loss, average_precision_score
from tensorflow.contrib import learn
# append parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gaze_grcn import GazePredictionGRCN as TheModel
from models.gaze_grcn import CONSTANTS, GRUModelConfig
import crc_input_data_seq
from read_tfrecord import load_data
from util import log, override
from joblib import Parallel, delayed

#__all__ = [Classifier]


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    # deal with mangled names
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


def create_standard_hparams():
    return tf.contrib.training.HParams(
        # Data
        feat_dimensions=[1024, 7, 7],
        label_dimensions=[13],
        batch_size=10,
        num_classes=13,
        #c3d_shape = [512,2,7,7 ]

        max_iter=2001,
        num_epochs=3,
        frame_width=98,
        frame_height=98,
        channels=3,
        gazemap_width=49,
        gazemap_height=49,
        saliencymap_width=49,
        saliencymap_height=49,
        learning_rate=0.002,
        dataset='h2',  # [crc, h2 crcxh2]
        use_gazemap=False
    )


def load_model():  # arg.gpu_fraction args vairable missing NOONONONONONO
    global model, config, session  # do we need it to be global, not so neat
    gpu_fraction = 0.48

    session = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                  allow_growth=True),
        device_count={'GPU': True},        # self-testing: NO GPU, USE CPU
    ))
    log.warn('Reloading Model ...')

    # default configuration as of now
    config = GRUModelConfig()
    config.batch_size = 14

    config.loss_type = 'xentropy'

    config.dump(sys.stdout)

    # dummy
    data_sets = crc_input_data_seq.CRCDataSplits()
    data_sets.train = data_sets.test = data_sets.valid = []

    model = TheModel(session, data_sets, config)
    print model

    return model


def load_labels(folders_paths, data_set):
    """
    args:
     folders_paths: (string) path to clipsets 
     data_set: (string) train or test
    returns:
     Returns dictionary containing thea list of touples of label and clips for each category
     labels['kiss'] = [(actionclip, label)]
    """

    # Loading test files
    if data_set == 'train':
        data_set = '*_' + data_set + '*'
    elif data_set == 'test':
        data_set = '*' + data_set + '*'
    else:
        return NameError

    labels_dict = OrderedDict()
    labels_records = OrderedDict()
    i = 0
    files = glob.glob(folders_paths + data_set)
    files.sort()

    for text_file in files:
        class_label = i
        # "AnswerPhone_test" - > [AnswerPhonw, test]
        class_name = text_file.split('_')[0]
        class_name = class_name.split('/')[-1]
        labels_records[class_name] = i
        with open(text_file) as f:
            for line in f:
                foldername, _, label = line.split(' ')
                if label[0] == '1':  # labels  == [-1\n]
                    #labels[class_name].append((foldername, class_label))
                    try:
                        labels_dict[foldername].append(class_label)
                    except:
                        labels_dict[foldername] = [class_label]
        i += 1
        label_list = []
        for name, labels in labels_dict.iteritems():
            label_list += (name, labels)

    return labels_dict  # , label_list


class Classifier(NN_base):

    def __init__(self, hparams, kernel=None):
        # add dataset filepath as iput to init,!!!!""
        super(Classifier, self).__init__()
        self.session = 0
        self.batch_size = hparams.batch_size
        self.labels = 0
        self.loss = 0
        self.dim_feature = hparams.feat_dimensions
        self.batch_size = hparams.batch_size
        self.num_classes = hparams.num_classes
        self.weights_initializer = 'glorot'
        self.bias_initializer = 'normal'
        self.learning_rate = 0
        self.input_x = 0
        self.input_dim = 0
        self.dim_c3d = hparams.feat_dimensions
        self.predictions = 0
        self.train_op = 0
        self.c3d = 0
        self.gazemap = 0
        self.valid_iterator = 0
        self.valid_handle = 0
        self.num_epochs = hparams.num_epochs
        self.gazemap_height = hparams.gazemap_height
        self.gazemap_width = hparams.gazemap_width
        self.batch_score = {}
        self.max_iter = hparams.max_iter
        self.use_gazemap = hparams.use_gazemap
        with tf.variable_scope("to_be_initialised"):
            self.global_step = tf.Variable(0,trainable = False, name = 'global_step')

    # split data into test and train set

    # declare batch size
    def build_model(self, model):
        
       
        B = self.batch_size
       
        
        if model == 'SVM':
            graph_input = self.projection(use_gazemap=self.use_gazemap)
            self.predictions = self.classification_graph_svm(
                graph_input, self.labels)
        elif model == 'NN':
            graph_input = self.projection(use_gazemap=self.use_gazemap)
            self.predictions = self.classification_graph_nn(
                graph_input, self.labels)

        elif model == 'LinearRegression':
            pass

        elif model == 'ConvNet':
            pass

        else:
            return NotImplementedError

    def projection(self, use_gazemap=False):
        with tf.variable_scope("projection"):
            B = self.batch_size
            # self.c3d = tf.placeholder(shape=tuple(
            #    [B] + self.dim_c3d), dtype=tf.float32, name = 'c3d_feat')
            # self.labels = tf.placeholder(
            #   shape=(B, self.num_classes), dtype=tf.float32, name = 'labels')
            # print(self.c3d)
            c3d_projection = tf.reshape(
                self.c3d, (B, 1024, 49))  # [B, 1024, 49]

            # self.gazemap = tf.placeholder(
            # shape=(B, self.gazemap_height, self.gazemap_width), dtype=tf.float32, name = 'gazemap')

            if self.use_gazemap:
                print "[Info] Using gazemap as attention."
                # [B, 49,49] -> [B, 2401]
                gazemap = tf.reshape(self.gazemap, (B, -1))
                gaze_projection_weights = self.get_weights(
                    (2401, 49), 'normal', 'gazemp_projection_weihts')

                gazemap_projected = tf.matmul(
                    gazemap, gaze_projection_weights)  # [B, 49]
                gazemap_projected = tf.expand_dims(gazemap_projected, 1)
                gazemap_projected = tf.tile(gazemap_projected, (1, 1024, 1))

                c3d_x_gaze = tf.multiply(c3d_projection, gazemap_projected)
                c3d_x_gaze_flattened = tf.reshape(c3d_x_gaze, (B, 1024 * 49))
                return c3d_x_gaze_flattened
            else:
                return tf.reshape(c3d_projection, (B, 1024 * 49))

    def classification_graph_svm(self, graph_input, labels):
        with tf.variable_scope("SVM"):
            W = tf.Variable(
                tf.zeros([50176, self.num_classes]), name="weights")
            b = tf.Variable(tf.zeros([13]), name="bias")
            y_pred = tf.matmul(graph_input, W) + b

            # Optimiazation
            svmC = 50
            regularization_loss = 0.5 * tf.reduce_sum(tf.square(W))
            hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([self.batch_size, 1]),
                                                  1 - labels * y_pred))
            svm_loss = regularization_loss + svmC * hinge_loss
            self.learning_rate = tf.constant(0.01)

            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.loss = svm_loss
            self.train_op = optimizer.minimize(
                self.loss, global_step=self.global_step)
            # use constant learning rate or decay?
            # tf.global_variables_initializer()
            return y_pred

    def classification_graph_nn(self, graph_input, labels):

        # Need to project data
        with tf.variable_scope('NN'):
            n_hidden1 = 256
            n_hidden2 = 256
            num_classes = self.num_classes

            h1_out = self.fc_layer(graph_input, 1024 * 49, n_hidden1,
                                   initializer='glorot', use_relu=False, batch_norm=False, name='h1')
            h2_out = self.fc_layer(h1_out, n_hidden1, n_hidden2,
                                   initializer='glorot', use_relu=False, batch_norm=False, name='h2')
            output = self.fc_layer(h2_out, n_hidden2, num_classes, initializer='glorot',
                                   use_relu=False, batch_norm=False, name='output')
            logits = output
            y_pred = tf.nn.sigmoid(output)  # prob for each class,
            y_pred_class = tf.round(y_pred)
            self.learning_rate = tf.train.exponential_decay(
                hparams.learning_rate, self.global_step, 10, 0.96)  # consider changing the hyper parameters
            # LOSS
            self.loss = loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels))
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(
                loss_op, global_step=self.global_step, name = "minimize")

            return y_pred  # y_pred_class

    def generate_and_evaluate(self, model, attention, checkpoint_path, n_jobs=9):
        count = checkpoint_path.split('-')[-1].split('.')[0]
        valid_size = 40572
        handle2 = tf.placeholder(
            tf.string, shape=[], name='dataset_handle2')
        if attention:
            valid_file = "valid_attention.tfrecord"

        else:
            valid_file = "valid.tfrecord"

        valid_iterator, valid_dataset = load_data("valid",
                                                  self.batch_size, valid_file, validation=True)

        valid_handle = self.session.run(
            valid_iterator.string_handle(name='valid_handle'))


        self.session.run(valid_iterator.initializer)
        iterator = tf.data.Iterator.from_string_handle(handle2,
                                                       valid_dataset.output_types, valid_dataset.output_shapes)

        next_element = valid_iterator.get_next()
        frames, c3d, labels, gaze_gt, gaze_pred = next_element

        self.c3d = c3d
        self.gazemap = gaze_gt
        self.labels = tf.cast(labels, dtype='float32')

        # self.session.run(self.valid_iterator.initializer)
        # graph_input = self.projection(use_gazemap=self.use_gazemap)
        # self.predictions = self.classification_graph_svm(graph_input)
        #import pdb
        #pdb.set_trace()

        # (not needed when running from programm)
        self.build_model(model=model)
          
        init  =tf.global_variables_initializer(), tf.local_variables_initializer()
        self.session.run(init)
        self.session.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='to_be_initialised')))
        log.infov("[Loading Checkpoint] %s " % checkpoint_path)

        self.load_model_from_checkpoint_file(checkpoint_path)
        frames_list = []
        c3d_list = []
        gaze_gt_list = []
        gaze_pred_list = []
        predictions_list = []
        labels_list = []
        log.info('Preparing validation data..')

        for i in xrange(valid_size / self.batch_size):
            #import pdb; pdb.set_trace()
            next_el, predictions = self.session.run(
                # [next_element, self.predictions])
                [next_element, self.predictions], feed_dict={handle2: valid_handle})
        #predictions = self.session.run(self.predictions, feed_dict= {handle : valid_handle})
            frames, c3d, labels, gaze_gt, gaze_pred = next_el
            predictions_list.append(predictions)
            frames_list.append(frames)
            c3d_list.append(c3d)
            labels_list.append(labels)
            gaze_gt_list.append(gaze_gt)
            gaze_pred_list.append(gaze_pred)
            #log.info('Done.')

        return evaluate(predictions_list, labels_list, 'NN', self.use_gazemap, count)

    def evaluate(self, pred_class, true_class):  # need to figure out the dimensions

        batch_score = {}
        #import pdb; pdb.set_trace()

        batch_score['Hamming'] = hamming_loss(true_class, np.sign(pred_class))
        batch_score['zero-one'] = zero_one_loss(
            true_class, np.sign(pred_class))
        batch_score['average-pecision'] = average_precision_score(
            true_class.reshape(-1), pred_class.reshape(-1))
        return batch_score

    def eval_helper(self, valid_handle):
        frames, c3d, labels, gaze_gt, gaze_pred = self.session.run(
            next_element, feed_dict={handle: valid_handle})
        feed_dict = {self.c3d: c3d, self.gazemap: gaze_gt, self.labels: labels}
        #self.session.run(self.predictions, feed_dict=feed_dict, options = run_options)
        batch_score = self.evaluate(self.predictions.eval(
            feed_dict=feed_dict), self.labels.eval(feed_dict=feed_dict))
        return batch_score

    def run(self, model="NN", attention=True, use_gazemap=False):
       # self.global_step = tf.Variable(0,name = 'global_step', trainable = False)
        if attention:
            valid_file = "valid_attention.tfrecord"
            train_file = "train_attention.tfrecord"
        else:
            valid_file = "valid.tfrecord"
            train_file = "train.tfrecord"

        handle = tf.placeholder(tf.string, shape=[], name='dataset_handle')
        with tf.variable_scope("valid_iterator"):

            # Valid Dataset
            valid_size = 40572
            valid_iterator, valid_dataset = load_data("valid",
                                                      self.batch_size, valid_file)
            valid_handle = self.session.run(
                valid_iterator.string_handle(name='valid_handle'))
            #self.valid_handle = valid_handle
            self.session.run(valid_iterator.initializer)

        with tf.variable_scope("train_iterator"):
            # Train Dataseta
            train_size = 37632

            train_iterator, train_dataset = load_data("train",
                                                      self.batch_size, train_file)
            training_handle = self.session.run(
                train_iterator.string_handle(name='training_handle'))
            self.session.run(train_iterator.initializer)  # check if necessary

        # Iterator
        with tf.variable_scope("Iterator"):
            iterator = tf.data.Iterator.from_string_handle(handle,
                                                           train_dataset.output_types,
                                                           train_dataset.output_shapes)

        with tf.variable_scope("Data"):
            next_element = iterator.get_next()

            frames, c3d, labels, gaze_gt, gaze_pred = next_element
            #self.frames = frames
            self.c3d = c3d
            self.gazemap = gaze_gt
            self.labels = tf.cast(labels, dtype='float32')

            #graph_input = self.projection(use_gazemap=self.use_gazemap)

        self.build_model(model)
        #graph_input = self.projection(use_gazemap=self.use_gazemap)
        #self.predictions = self.classification_graph_nn(graph_input)

        #run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        # Checkpoint paths
        if self.use_gazemap:
            gazemap_name = '_gazemap'
        elif attention:
            gazemap_name = '_c3d_attention'
        else:
            gazemap_name = ''
        checkpoint_path = "/home/amelie/gazecap/models/checkpoints/" + \
            model + gazemap_name

        # Summary
        summary_dir = './logs/' + model
        if not os.path.exists(summary_dir):
            os.mkdir(summary_dir)

        with tf.variable_scope('logging'):
            loss_summary = tf.summary.scalar('current_cost', self.loss)
            tf.summary.scalar('learning_rate', self.learning_rate)
            summary = tf.summary.merge_all()

        if not os.path.exists(os.path.join(summary_dir, 'training')):
            os.mkdir(os.path.join(summary_dir, 'training'))

        if not os.path.exists(os.path.join(summary_dir, 'testing')):
            os.mkdir(os.path.join(summary_dir, 'testing'))
        training_writer = tf.summary.FileWriter(
            os.path.join(summary_dir, 'training'), self.session.graph)
        testing_writer = tf.summary.FileWriter(
            os.path.join(summary_dir, 'testing'))

        
        init  = tf.global_variables_initializer(), tf.local_variables_initializer()
        self.session.run(init)

        self.session.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='to_be_initialised')))

        
        # Training model
        valid_it = 0
        for epoch in range(hparams.num_epochs):
            self.session.run(train_iterator.initializer)
            #import pdb; pdb.set_trace()

            for it in range(train_size / hparams.batch_size):

                        # Training
                loss, _, global_step, learning_rate, training_summary = self.session.run(
                    [self.loss, self.train_op, self.global_step, self.learning_rate, summary], feed_dict={handle: training_handle})

                # Testing
                if valid_it == valid_size / hparams.batch_size:
                    self.session.run(valid_iterator.initializer)
                    valid_it = 0
                test_loss,  testing_summary = self.session.run(
                    [self.loss, loss_summary], feed_dict={handle: valid_handle})
                valid_it += 1

                # Logging
                if global_step % self.steps_per_logprint == 0:
                    log.info('Epoch: %d - Iteration %d: learning_rate - %.20f, loss: %.10f' %
                             (epoch, global_step, learning_rate, loss))
                if global_step % self.steps_per_evaluation == 0:
                    predictions, labels = self.session.run(
                        [self.predictions, self.labels], feed_dict={handle: valid_handle})
                    valid_it += 1
                    batch_score = self.evaluate(predictions, labels)
                    for metric, score in batch_score.iteritems():
                        log.info('%s loss : %f', metric, score)
                if global_step % self.steps_per_checkpoint == 0:
                    log.infov('Saving checkpoint at %d', self.global_step)
                    #import pdb; pdb.set_trace()

                    self.create_model_checkpoint(
                        checkpoint_path, model + gazemap_name, self.session, self.global_step)
                if global_step >= self.max_iter:
                    break

                training_writer.add_summary(training_summary, global_step)
                #import pdb; pdb.set_trace()

                testing_writer.add_summary(testing_summary, global_step)

            self.session.run(train_iterator.initializer)
        log.infov('Done.')

        


def evaluate(pred_list, labels_list,  model, use_gazemap,attention, count='',  n_jobs=9,):
    log.info('Evaluation with %d parallel jobs' % (n_jobs))
    aggregated_scores = defaultdict(list)
    len_data = len(pred_list)
    with Parallel(n_jobs=n_jobs, verbose=10) as parallel:
        scores_aggregated = parallel(delayed(evaluate_helper)(
            pred_list[i], labels_list[i])
            for i in xrange(len_data))

        scores_aggregated = list(scores_aggregated)

    for scores in scores_aggregated:
        for metric, score in scores.iteritems():
            aggregated_scores[metric].append(score)
    # To do -
    out_dir = "/home/amelie/gazecap/models/scores/"
    model_out_dir = os.path.join(out_dir, model)
    if not os.path.exists(model_out_dir):
        os.mkdir(model_out_dir)
    if use_gazemap:
        use_gazemap = 'gazemap'
    elif attention:
        use_gazemap = 'c3d_attention'
    else:
        use_gazemap = ''
    overall_txt_paths = os.path.join(
        model_out_dir, 'overall_' + model + '_' + use_gazemap + '%s.txt' % count)
    overall_avg_paths = os.path.join(
        model_out_dir, 'overall_avg_' + model + '_' + use_gazemap + '%s.txt' % count)
    with open(overall_avg_paths, 'w') as fp1:
        with open(overall_txt_paths, 'w') as fp:
            for metric, score_list in aggregated_scores.iteritems():
                #metric = 'average-precision'
                mean_score = np.mean(score_list)
                log.infov("Average %s : %.4f\n" % (metric, mean_score))
                #fp.write("Average %s : %.4f\n" % (metric, mean_score))
                fp1.write("Average %s : %.4f\n" % (metric, mean_score))
                for scroe in score_list:
                    #import pdb; pdb.set_trace()
                    fp.write('%.3f' % scroe)
                    fp.write('\n')
        log.warn("Dumped at %s", model_out_dir)
    return aggregated_scores


def evaluate_helper(pred_class, true_class):  # need to figure out the dimensions

    batch_score = {}

    batch_score['Hamming'] = hamming_loss(true_class, np.sign(pred_class))
    batch_score['zero-one'] = zero_one_loss(true_class, np.sign(pred_class))
    batch_score['average-pecision'] = average_precision_score(
        true_class.reshape(-1), pred_class.reshape(-1))
    return batch_score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=str, help='eg. NN, SVM')
    parser.add_argument('--attention', required=False,
                        action='store_true', default=False)
    args = parser.parse_args()

    hparams = create_standard_hparams()
    model = Classifier(hparams)
    with tf.Session() as sess:
        model.session = sess
        model.run(model=args.model, attention=args.attention)

        
    tf.reset_default_graph()
    model = Classifier(hparams)
    if args.attention:
        att = '_c3d_attention'
    else:
        att = ''
    with tf.Session() as sess:
        
        model.session = sess
        attention = args.attention
        model.generate_and_evaluate(args.model, attention, './checkpoints/SVM' + att + '/model/SVM' + att+ '-2000')
