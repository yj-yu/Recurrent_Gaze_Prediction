import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #append parent directory to path

from util import log
__all__ = ["NN_base"]



class Base(object):

    def __init__(self,
                 train_dir = None,
                 steps_per_checkpoint = 1000,
                 steps_per_validation = 100,
                 steps_per_evaluation = 2000,
                 steps_per_logprint = 1000,
                 max_steps = 10000,
                 verbose_level = 2):

        self.train_dir = train_dir
        self.train_tag = ''
        self.max_steps = max_steps
        self.steps_per_checkpoint = steps_per_checkpoint
        self.steps_per_validation = steps_per_validation
        self.steps_per_evaluation = steps_per_evaluation
        self.steps_per_logprint = steps_per_logprint
        self.verbose_level = int(verbose_level)
        #self.global_step = 0 

    def create_model_checkpoint(self, checkpoint_dir, model_name,session,global_step):
        self.saver = tf.train.Saver()  ## vs tf.train.saver(self.session)
        
        log.info(" [Checkpoint] Saving checkpoints ...")
        model_name = model_name
        checkpoint_dir = os.path.join(checkpoint_dir, "model") # TODO????
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        assert os.path.exists(checkpoint_dir)
        # TODO append iteration number if possible
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        log.info(" [Checkpoint] Saved checkpoints into %s !" % checkpoint_path)
        #import pdb; pdb.set_trace()
        
        self.saver.save(self.session, checkpoint_path,
                        global_step=global_step)
        #vars_in_checkpoint = tf.train_list_variabliablse(os.path.join(checkpoint_dir, "model"ac))
        #print vars_in_checkpoint
        return checkpoint_path

    def load_model_from_checkpoint_file(self, checkpoint_path):
        self.saver = tf.train.Saver()
        #import pdb; pdb.set_trace()
            
        restore_vars = self.saver.restore(self.session, checkpoint_path)


      
        log.info("[Checkpoint] Loading done.")

    def load_model_checkpoint(self,checkpoint_dir):
        log.info(" [Checkpoint] Loading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, "model") ### check if this is right
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir )
        #import pdb; pdb.set_trace()
        
        if checkpoint and checkpoint.model_checkpoint_path:
            log.info(" [Checkpoint] Checkpoint path : %s ..." % checkpoint.model_checkpoint_path)
            #checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)

        else:
            return False


class NN_base(Base):

    def __init__(self):
        super(NN_base, self).__init__() 

    def get_bias(self,dimensions, initializer = 'constant', name = '', reusable = False):
    
        if initializer == 'constant':
            initializer = tf.constant(0.05, shape = dimensions)
        elif initializer == 'normal':
            initializer = tf.truncated_normal(dimensions,stddev=0.05)
        elif intializer == 'glorot':
            pass
        if reusable == True:
            assert name is not None
            return tf.get_variable(name, dimensions, initializer = initializer)
        else:
            return tf.Variable(initializer, name)

    def get_weights(self,dimensions, initializer,name = None,  reusable = False):
        """creates tensorflow variable for weights
        args:
        dimension: weight dimensions as list
        initiazlier: [normal, glorot]
        name: name for tensorflow variable, mandatory if reusable = True
        reusable: whether model should reuse variable (useful for sequence models)
        """
        if initializer == 'normal':
            initializer = tf.truncated_normal(dimensions,stddev=0.05)
            if reusable == True:
                assert name is not None
                return tf.get_variable(name, dimensions, initializer = initializer)
            else:
                return tf.Variable(initializer,name)
        elif initializer == 'glorot':
            initializer = tf.contrib.layers.xavier_initializer()
            if reusable == True:
                assert name is not None
                return tf.get_variable(name, dimensions, initializer = initializer)
            else:
                return tf.Variable(initializer(dimensions),name)
        else:
            return NotImplementedError

    def conv_layer(self, x, n_in, filter_size, num_filters, initializer, strides, padding = 'SAME', name = ''):
        shape = [ filter_size, filyer_size, n_in, num_filters]
        weights = get_weights(shape, initializer = initializer, name = name+'/weights')
        bias = get_bias([num_filters], initializer = 'constant', name = Name + '/bias', reusable = False)

        layer = tf.nn.conv2d(input = x, filter = weights,strides = strides, padding = padding )

        layer += bias

        
        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.

        return layer
    
    def relu_layer(self, x , name = '' ):
        return tf.nn.relu(x, name = name + '/relu')

    def max_pool(self, x, kernel_size, strides, padding = 'SAME', name = None ):
        layer = tf.nn.max_pool(value=layer,
                               ksize=kernel_size,
                                   strides=strides,
                                padding=[padding],
                                   name = name+'/max_pool')
        
        
    def fc_layer(self, x, n_in, n_out, use_relu = True, batch_norm = False, initializer = 'glorot', name = ''):
        """
        args:
        x : 2D tensor [B, IW xIW xC]
        """
        weights = self.get_weights([n_in, n_out], initializer = initializer, reusable = False, name = name + '/weights' )

        bias = self.get_bias([n_out], initializer = 'constant', name = name + '/bias', reusable = False)

        layer = tf.nn.xw_plus_b(x, weights, bias)

        if use_relu == True:
            layer = tf.nn.relu(layer)

        if batch_norm == True:
            layer = tf.layers.batch_normalization(layer)

        return layer

    def flatten_layer(x):
        """
        args:
        x: is a 4D tensor of shape [B, IH, IW, C]
        returns:
        x: reshaped into [B, IH x IW x C]
        num_feat: number of features 
        """
        
        layer_shape = x.get_shape()
        num_feat =layer_shape[1:4].num_elements()
        x = tf.reshape(x, [-1, num_feat])
        return x, num_feat
        
    
