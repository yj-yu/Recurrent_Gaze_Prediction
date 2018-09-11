"""
Base Model and Configuation
"""

import json
import numpy as np

import os.path
import time
import scipy.misc
import re

import tensorflow as tf

from util import log
import cPickle as pkl


class BaseModelConfig(object):

    def __init__(self,
                 train_dir = None,
                 max_steps = 100000,
                 steps_per_checkpoint = 1000,
                 steps_per_validation = 100,
                 steps_per_evaluation = 2000,
                 steps_per_logprint = 5,
                 verbose_level = 2,

                 learning_rate_decay = 0.80,
                 optimization_method = 'adam',
                 initial_learning_rate = 0.003,
                 ):

        # model loading and etc.
        self.train_dir = train_dir
        self.train_tag = ''
        self.max_steps = max_steps
        self.steps_per_checkpoint = steps_per_checkpoint
        self.steps_per_validation = steps_per_validation
        self.steps_per_evaluation = steps_per_evaluation
        self.steps_per_logprint = steps_per_logprint
        self.verbose_level = int(verbose_level)

        # optimization and training
        self.learning_rate_decay = learning_rate_decay  # per 1000 step
        self.optimization_method = optimization_method
        self.initial_learning_rate = initial_learning_rate
        self.max_grad_norm = 10.0


    def __repr__(self):  
        r = []
        for prop, value in sorted(vars(self).iteritems()):
            r.append("%s : %s" % (prop, value))

        return 'ModelConfig{' + ', '.join(r) + '}'


    def dump(self, fp):
        if isinstance(fp, str) or isinstance(fp, unicode):
            with open(fp, 'w') as f:
  #              import pdb; pdb.set_trace()
                #self.dump(f) # json doesnt work with custom classes use pickle instead
                pkl.dump(self, f)
                
        else:
            json.dump(self.__dict__,  # dictv   #writes to file 
                      fp, sort_keys=True,
                      indent=4, separators=(',', ': '))
            fp.write('\n')
            fp.flush()

    @staticmethod
    def load(fp):
        # merge with json data
        if isinstance(fp, str):
            with open(fp, 'r') as f:
                return BaseModelConfig.load(f)
        else:
            o = json.load(fp)  # dict
            config = BaseModelConfig()
            for key, value in o.iteritems():
                setattr(config, key, value)
            return config


def urlify(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s\-_,]", '', s)
    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", '-', s)
    return s


class ModelBase(object):
    """
    (Common) Abstract model base.
    """

    def __init__(self, config):
        """
        creates train directory where data results are saved and stores model
        configurations as .json

        """
       
        if config.train_dir is None:
            import tempfile
            tempd = tempfile.mkdtemp(prefix='tmp-' + time.strftime('%Y%m%d-%H%M%S') + '-',
                                     suffix='-' + urlify(config.train_tag) if config.train_tag else '')
            log.info("Using temp dir {}".format(tempd))

        self.train_dir = config.train_dir or tempd
        self.writer = tf.summary.FileWriter(self.train_dir, self.session.graph) # nice tf way of writing output without slowing down training
        log.warn("Train dir : %s", self.train_dir)

        config_file = os.path.join(self.train_dir, 'config.json')
        if not os.path.exists(config_file):
            self.config.dump(config_file)
        else: log.warn("config_file %s already exists (skipped)", config_file)

        config_pkl = os.path.join(self.train_dir, 'config.pkl')
        if not os.path.exists(config_pkl):
            with open(config_pkl, 'wb') as f:
                pkl.dump(config, f)
        else: log.warn("config_file %s already exists (skipped)", config_pkl)
        # }}}



    @classmethod
    def create_from_checkpoint(klass,
                               session, data_loader,
                               train_dir, model_checkpoint_file):
        """
        Load the snapshot including the model configuration and parameters
        from the stored checkpoint and configuration file.

        @seealso load_model_checkpoint(self)
        @seealso save_model_checkpoint(self)
        """

        # TODO train_dir should not overlap for consistency
        config_file = os.path.join(train_dir, 'config.json')
        config = BaseModelConfig.load(config_file)
        model = klass(session, data_loader, config)

        # TODO how to check if load was successful?
        assert os.path.isfile(model_checkpoint_file)
        model.load_model_from_checkpoint_file(model_checkpoint_file)

        # check consistency
        assert model.batch_size == config.batch_size
        assert model.word_vocab_size == data_loader.word_vocab_size

        return model


    def _init_weight(self, name, dim_in, dim_out, stddev=0.1):
        # Normal Gaussian
        # ??? stddev should differ as dim_in, dim_out varies
        #W = tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/np.sqrt(dim_in)), name=name)
        #W = tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev), name=name>)

        # Xavier Initialization
        init_range = np.sqrt(6.0 / (dim_in + dim_out))
        W = tf.Variable(tf.random_uniform([dim_in, dim_out], -init_range, init_range,
                                            dtype=tf.float32), name=name)

        tf.histogram_summary(name, W)
        return W

    def _init_bias(self, name, dim_out):
        b = tf.Variable(tf.zeros([dim_out]), name=name)
        tf.histogram_summary(name, b)
        return b

    def _init_weight_and_bias(self, weight_name, bias_name,
                              dim_in, dim_out, stddev=0.1):
        W = self._init_weight(weight_name, dim_in, dim_out, stddev=stddev)
        b = self._init_bias(bias_name, dim_out)
        return W, b


    # =======================================================================

    def reload_checkpoint(self):
        # checkpoint loading
        if self.load_model_checkpoint(self.train_dir):
            log.info(" [Checkpoint] Successfully loaded model.")
            log.info("  Learning Rate = %.6f" % self.current_learning_rate)
            log.info("   Current Step = %d"   % self.current_step)

            # if success to checkpoint access, display current status
            #val_loss = self.test('val', debug_mode=True)
            #log.info("  Validation Loss = %.5f" % val_loss)

            return True

        else:
            log.error(" [Checkpoint] Failed to load model !! (starting from scratch)")
            return False




    def load_model_checkpoint(self, checkpoint_dir):
        log.info(" [Checkpoint] Loading checkpoints ...")

        checkpoint_dir = os.path.join(checkpoint_dir, "model")
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)

        if checkpoint and checkpoint.model_checkpoint_path:
            log.info(" [Checkpoint] Checkpoint path : %s ..." % checkpoint.model_checkpoint_path)
            #checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            self.load_model_from_checkpoint_file(checkpoint.model_checkpoint_path)
        else:
            return False

        # especially, learning rate can be overriden
        # NOTE that (self.learning_rate) might not be a variable,
        # but a tensor (e.g. exponential deacy or scheduled scalar)!!
        if isinstance(self.learning_rate, tf.Variable):
            learning_rate_reassign_op = self.learning_rate.assign(self.config.initial_learning_rate)
            self.session.run(learning_rate_reassign_op)
        else:
            log.error("Seems using learning rate decay scheme (it is not a variable.\n" +
                      "  restoring learning rate has failed as implemented in this case,\n"
                      "  which might cause that learning rate has been INCREASED!!!!")
            log.error("self.config.initial_learning_rate = %f", self.config.initial_learning_rate)

        return True

    def load_model_from_checkpoint_file(self, checkpoint_path):
        self.saver = tf.train.Saver()
        self.saver.restore(self.session, checkpoint_path)
        log.info(" [Checkpoint] Successfully loaded from %s", checkpoint_path)

    def save_model_checkpoint(self, checkpoint_dir):
        self.saver = tf.train.Saver(tf.all_variables())

        log.info(" [Checkpoint] Saving checkpoints ...")
        model_name = type(self).__name__ or "Model"
        checkpoint_dir = os.path.join(checkpoint_dir, "model") # TODO????
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

        # TODO append iteration number if possible
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        log.info(" [Checkpoint] Saved checkpoints into %s !" % checkpoint_path)
        self.saver.save(self.session, checkpoint_path,
                        global_step=self.global_step)
        return checkpoint_path


    # ========================================================================

    def _build_learning_rate(self):
        return tf.Variable(self.initial_learning_rate,
                           name="var_lr", trainable=False)

    def create_train_op(self, loss_tensor, params, learning_rate, global_step=None,
                        gradient_summary_tag_name='gradient_norm'):
        if learning_rate == 0:
            return tf.no_op()
        # ToDo: add nromal gradient 
        # optimizer
        if self.config.optimization_method == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif self.config.optimization_method == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9)
        elif self.config.optimization_method == 'sgd':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        else:
            raise ValueError('Invalid optimization method!')

        # compute gradients
        gradients = tf.gradients(
            loss_tensor, params,
            aggregation_method=2 # see issue #492
        )
        if all(v is None for v in gradients):
            # in some cases, we are opted not to train some sub-networks at all.
            return tf.no_op()

        # use gradient clipping as well.
        if self.max_grad_norm > 0:
            clipped_grads, norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        else:
            clipped_grads, norm = gradients, 0
        tf.summary.scalar(gradient_summary_tag_name, norm)
        clipped_grad_and_vars = list(zip(clipped_grads, params))

        train_op = optimizer.apply_gradients(
            clipped_grad_and_vars,
            global_step=global_step
        )

        # with some debugging information.
        total_num_elements = 0
        for var in params:
            log.debug("  model param %s : %s (total %d)", var.name, var.get_shape().as_list(), var.get_shape().num_elements())
            total_num_elements += var.get_shape().num_elements()
        log.infov("Total # of parameters in the train_op : %d", total_num_elements)


        # TODO learning rate might be overriden afterwards (e.g. checkpoint)
        return train_op


    def build_train_op(self):
        """
        build learning_rate, global_step variables
        and optimizer and related summary.
        """

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.learning_rate = self._build_learning_rate()
        assert isinstance(self.learning_rate, tf.Variable)
        tf.summary.scalar('learning_rate', self.learning_rate)

        self.train_op = self.create_train_op(self.loss,
                                             tf.trainable_variables(),
                                             self.learning_rate,
                                             self.global_step)
        return self.train_op


    def fit(self):
        """ Train the model. """

        # Summaries for learning debugging
        self.merged_summary_train = tf.summary.merge_all()
        self.merged_summary_val   = tf.summary.merge_all('TEST_SUMMARIES')

        # now ready for train
        log.info("Initial Learning Rate = %.6f" % self.current_learning_rate)
        assert self.current_learning_rate > 0

        # reload if checkpoint is available
        self.reload_checkpoint()

        step = 0
        while step <= self.config.max_steps:
            # run a single training step
            step = self.single_step(train_mode=True)
            assert step > 0

            # (Checkpoint) Save model periodically
            if np.mod(step, self.config.steps_per_checkpoint) == 0:
                self.save_model_checkpoint(self.train_dir)

            if np.mod(step, self.config.steps_per_validation) == 0:
                _ = self.single_step(train_mode=False)

            if np.mod(step, self.config.steps_per_evaluation) == 0:
                self.generate_and_evaluate(self.data_sets.valid)



    def report_evaluate_summary(self, batch_scores):
        """
        Report the result of batch_scores in tensorflow summary.

        batch_scores : a dict, metric_name (str) -> metric_value (float)
        """
        step = self.current_step
        if not hasattr(self, 'evaluation_summary'):
            self.evaluation_summary = dict()
        if not hasattr(self, 'placeholder_float'):
            self.placeholder_float = tf.placeholder(tf.float32, shape=[],
                                                    name='placeholder_float')

        for metric, score in batch_scores.iteritems():
            summary_name = ('evaluation/%s' % metric)

            summary_op = self.evaluation_summary.get(summary_name)
            if summary_op is None:
                summary_op = tf.summary.scalar(summary_name, self.placeholder_float,
                                               collections=['EVALUATION_SUMMARY'])
                self.evaluation_summary[summary_name] = summary_op

            summary_ret = self.session.run(summary_op,
                                           {self.placeholder_float: score })
            self.writer.add_summary(summary_ret, step)



    @property
    def current_step(self):
        return self.global_step.eval(session=self.session)

    @property
    def current_learning_rate(self):
        return self.learning_rate.eval(session=self.session)

    def decay_learning_rate(self, decay_factor):
        """
        Perform learning_rate decay
        """
        learning_decay_op = self.learning_rate.assign(self.learning_rate * decay_factor)
        self.session.run(learning_decay_op)
        return self.current_learning_rate



def _self_test():
    c = BaseModelConfig()

    import StringIO
    s = StringIO.StringIO()
    c.dump(s)

    json = s.getvalue()
    print json

    c2 = BaseModelConfig.load(
        StringIO.StringIO(json)
    )
    print c2

if __name__ == '__main__':
    _self_test()
