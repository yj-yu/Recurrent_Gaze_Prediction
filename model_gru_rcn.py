#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import ipdb
from tensorflow.python.ops.rnn_cell import linear
from tensorflow.models.rnn import rnn_cell
#from Seq2Seq_Upgrade_TensorFlow.rnn_enhancement import rnn_cell_enhanced as rnn_cell
#from Seq2Seq_Upgrade_TensorFlow.rnn_enhancement import decoding_enhanced as decoding
from collections import OrderedDict
import cPickle as pkl
import crc_input_data_seq
import pudb

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('image_width', 96, 'Resized width of images')
flags.DEFINE_integer('image_height', 96, 'Resized height of images')
flags.DEFINE_integer('gazemap_width', 7, 'Resized width of images')
flags.DEFINE_integer('gazemap_height', 7, 'Resized height of gazemaps')
flags.DEFINE_integer('saliencymap_width', 48, 'Resized width of saliencymaps')
flags.DEFINE_integer('saliencymap_height', 48, 'Resized height of saliencymaps')
flags.DEFINE_float('learning_rate', 0.03, 'The learning rate')
flags.DEFINE_integer('batch_size', 200, 'The batch size')

class GRU_RCN_Cell(rnn_cell.RNNCell):
  """Gated Recurrent Unit RCN cell. http://arxiv.org/abs/1511.06432  """
  """N.Ballas et al. DELVING DEEPER INTO CONVOLUTIONAL NETWORKS """
  def __init__(self, num_units, dim_feature):
    self._num_units = num_units
    self.dim_feature = dim_feature
    self. W_z = tf.Variable(tf.truncated_normal([3,3,dim_feature,self._num_units], dtype=tf.float32, stddev=1e-4), name='GRU_Conv_Wz')
    self.U_z = tf.Variable(tf.truncated_normal([3,3,self._num_units,self._num_units], dtype=tf.float32, stddev=1e-4), name='GRU_Conv_Uz')
    self.W_r = tf.Variable(tf.truncated_normal([3,3,dim_feature,self._num_units], dtype=tf.float32, stddev=1e-4), name='GRU_Conv_Wr')
    self.U_r = tf.Variable(tf.truncated_normal([3,3,self._num_units,self._num_units], dtype=tf.float32, stddev=1e-4), name='GRU_Conv_Ur')
    self.W = tf.Variable(tf.truncated_normal([3,3,dim_feature,self._num_units], dtype=tf.float32, stddev=1e-4), name='GRU_Conv_W')
    self.U = tf.Variable(tf.truncated_normal([3,3,self._num_units,self._num_units], dtype=tf.float32, stddev=1e-4), name='GRU_Conv_U')

  @property
  def input_size(self):
    return self._num_units
  @property
  def output_size(self):
    return self._num_units
  @property
  def state_size(self):
    return self._num_units
  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with tf.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not udpate.
        #pudb.set_trace()
        # k1 x k2 x Ox x Oh where k1 x k2 is the convolutional kernel spatial
        # size.
        Wzconv = tf.nn.conv2d(inputs, self.W_z, [1,1,1,1], padding='SAME')
        Uzconv = tf.nn.conv2d(state, self.U_z, [1,1,1,1], padding='SAME')
        Wrconv = tf.nn.conv2d(inputs, self.W_r, [1,1,1,1], padding='SAME')
        Urconv = tf.nn.conv2d(state, self.U_r, [1,1,1,1], padding='SAME')
        # sig(W_r * x_t + U_r * h_t-1 )
        u = tf.sigmoid(Wzconv + Uzconv)
        r = tf.sigmoid(Wrconv + Urconv)
      with tf.variable_scope("Candidate"):
        # tanh(W * x_t + U * (r_t dot h_t-1) not confident yet.
        Wconv = tf.nn.conv2d(inputs, self.W, [1,1,1,1], padding='SAME')
        Uconv = tf.nn.conv2d(r*state, self.U, [1,1,1,1], padding='SAME')
        c = tf.tanh(tf.add(Wconv,Uconv))
        new_h = u * state + (1 - u) * c
    # output, state is (batch_size, H=7, W=7, num_units)
    return new_h, new_h


class Gaze_Prediction_Module():
    # Does not compatible model.py yet. mount that to this model. TODO"
    def __init__(self, batch_size, n_lstm_steps, dim_feature, dim_hidden_u, dim_hidden_b, dim_sal, dim_sal_proj, dim_cnn, dim_cnn_proj, out_proj):
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.dim_feature = dim_feature
        self.dim_hidden_u = dim_hidden_u
        self.dim_hidden_b = dim_hidden_b
        self.dim_sal = dim_sal # 48 * 48
        self.dim_sal_proj = dim_sal_proj # 14 * 14 = 196
        self.dim_cnn = dim_cnn # 1024 * 7 * 7
        self.dim_cnn_proj = dim_cnn_proj # 16
        self.out_proj = out_proj # 7*7 gaze point + 1 pupil size

        self.lstm_u = GRU_RCN_Cell(dim_hidden_u, dim_cnn_proj) #rnn_cell.GRUCell(dim_hidden_u, weight_initializer="orthogonal")
        #self.lstm_b = rnn_cell.GRUCell(dim_hidden_b, weight_initializer="orthogonal")

        self.proj_cnn_W = tf.Variable( tf.random_uniform( [dim_feature, dim_cnn_proj], -0.1, 0.1), name = "proj_cnn_W")
        self.proj_cnn_b = tf.Variable( tf.zeros( [dim_cnn_proj], name = "proj_cnn_b"))

        #self.proj_cnn_W = tf.Variable( tf.random_uniform ( [dim_cnn, dim_cnn_proj], -0.1, 0.1), name = "proj_cnn_W")
        #self.proj_cnn_b = tf.Variable( tf.zeros ( [dim_cnn_proj], name = "proj_cnn_b"))
        self.proj_out_W = tf.Variable( tf.random_uniform( [7*7*64, out_proj], -0.1, 0.1), name = "proj_out_W")
        self.proj_out_b = tf.Variable( tf.zeros( [out_proj], name = "proj_out_b"))
        #self.inv_proj_out = tf.transpose( self.proj_out_W )
        self.global_step = tf.Variable(0, trainable=False)

    def build_model(self):
        # Input frame saliency and cnn layer values
        # frm_sal (64 x 80 x 196)
        #frm_sal = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_sal])
        frm_cnn = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_cnn])
        #feature_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])
        # Change dim [64 x 1024 x 7 height x 7 width] to [64 x 7 x 7 x 1024]
        frm_cnn_reshape = tf.transpose(tf.reshape(frm_cnn, [-1, self.dim_feature, 7, 7] ) , perm=[0,2,3,1])

        # Batch size x 50 (gaze point and pupil size)
        gazemap = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps ,self.out_proj])
        #ipdb.set_trace()
        # project cnn features to projected cnn dimension.
        # output is [64*7*7 x 12]
        cnn_emb = tf.nn.xw_plus_b(tf.reshape(frm_cnn_reshape, [-1, self.dim_feature]), self.proj_cnn_W, self.proj_cnn_b)
        # output is [64 x 49*12]
        cnn_emb = tf.reshape(cnn_emb, [self.batch_size, self.n_lstm_steps, 7,7,self.dim_cnn_proj])

        state_u = tf.zeros([self.batch_size, 7,7,self.lstm_u.state_size])

        loss = 0.0
        logits = []
        # n_lstm_step for example, 35.
        for i in range(self.n_lstm_steps):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            # We use cnn embedding as upper lstm input
            with tf.variable_scope("RNN"):
                output_u, state_u = self.lstm_u( cnn_emb[:,i,:,:,:] , state_u )
            # RNN output is (batch_size, 7,7,64)
            logit = tf.nn.dropout(tf.nn.xw_plus_b(tf.reshape(output_u, [batch_size,-1]) , self.proj_out_W , self.proj_out_b),0.5)
            # Normalize gaze map?
            #gaze = gazemap[:,i,:49]
            #gaze = gaze / tf.reduce_sum(gaze)
            # Cross entropy and softmax??
            l2loss = tf.nn.l2_loss(logit[:,:49] - gazemap[:,i,:49])
            logits.append(logit)
            current_gaze_loss = tf.reduce_sum(l2loss)
            # Pupil loss..
            current_pupil_loss = tf.nn.l2_loss(logit[:,49] - gazemap[:,i,49])
            current_loss = current_gaze_loss + 0.01 * current_pupil_loss
            loss += current_loss

        loss = loss / float(self.batch_size)
        return loss, frm_cnn, gazemap, logits

    def build_generator(self):
        # Input frame saliency and cnn layer values
        frm_cnn = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_cnn])
        # Change dim [64 x 1024 x 7 height x 7 width] to [64 x 7 x 7 x 1024]
        frm_cnn_reshape = tf.transpose(tf.reshape(frm_cnn, [-1, self.dim_feature, 7, 7] ) , perm=[0,2,3,1])

        # project cnn features to projected cnn dimension.
        # output is [64*7*7 x 12]
        cnn_emb = tf.nn.xw_plus_b(tf.reshape(frm_cnn_reshape, [-1, self.dim_feature]), self.proj_cnn_W, self.proj_cnn_b)
        # output is [64 x 49*12]
        cnn_emb = tf.reshape(cnn_emb, [self.batch_size, self.n_lstm_steps, 7,7,self.dim_cnn_proj])

        state_u = tf.zeros([self.batch_size, 7,7,self.lstm_u.state_size])

        generated_maps = []
        generated_pupils = []
        #logits = []
        # n_lstm_step for example, 60.
        for i in range(self.n_lstm_steps):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            # We use cnn embedding as upper lstm input
            with tf.variable_scope("RNN"):
                output_u, state_u = self.lstm_u( cnn_emb[:,i,:,:,:] , state_u )
            # RNN output is (batch_size, 7,7,64)
            logit = tf.nn.dropout(tf.nn.xw_plus_b(tf.reshape(output_u, [batch_size,-1]) , self.proj_out_W , self.proj_out_b),0.5)
                        #max_prob_index = tf.argmax(logit[:,:49], 1)
            map_out = logit[:,:49]
            pupil_size = logit[:,49]
            generated_maps.append(map_out)
            generated_pupils.append(pupil_size)
            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit,gazemap[:,i,:49])
            #logits.append(logit)

        # pack generated_maps and generated_pupils into a tensor
        # so that they have [B x T x 49] or [B x T x 1] dimension.
        generated_maps = tf.transpose( tf.pack(generated_maps), perm=[1,0,2], name='generated_maps')
        generated_pupils = tf.transpose( tf.expand_dims( tf.pack(generated_pupils), 2), perm=[1,0,2], name='generated_pupils')

        return frm_cnn, generated_maps, generated_pupils

# config : changed as paramter later
n_lstm_steps = 35
batch_size = 24
dim_feature = 1024 #1024
dim_hidden_u = 64
dim_hidden_b = 512
dim_sal = 1024*49  #196
dim_sal_proj = 1024
dim_cnn = 1024*7*7
dim_cnn_proj = 16
out_proj = 50
learning_rate = 0.001
save_model_dir = 'rnn_results/'
from_dir = 'rnn_results/'
model_name = 'model-10'
reload_ = False
n_epochs = 1000

model = Gaze_Prediction_Module(batch_size, n_lstm_steps, dim_feature, dim_hidden_u, dim_hidden_b, dim_sal, dim_sal_proj, dim_cnn, dim_cnn_proj, out_proj)

#tf_loss, tf_saliency, tf_cnn, tf_cnnmask, tf_gazemap, tf_probs = model.build_model()
tf_loss, tf_cnn, tf_gazemap, tf_probs = model.build_model()
cnn_tf, gaze_output, pupil_output = model.build_generator()


print 'Prediction RNN Model builded '

print 'Dataset Loading'
crc_data_sets = crc_input_data_seq.read_crc_data_sets(FLAGS.image_height,
                                                      FLAGS.image_width,
                                                      FLAGS.gazemap_height,
                                                      FLAGS.gazemap_width,
                                                      tf.float32,
                                                      reload_ = True,
                                                      dataset='crc')


#crc_data_sets = crc_input_data_seq.read_crc_data_sets(FLAGS.image_height,
#                                                      FLAGS.image_width,
#                                                      FLAGS.gazemap_height,
#                                                      FLAGS.gazemap_width,
#                                                      tf.float32,
#                                                      reload_ = False,
#                                                      dataset='hollywood2')
print 'Dataset setting finished'
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=100)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)

#init = tf.initialize_all_variables()
tf.initialize_all_variables().run()

offset = 0
if reload_:
    print 'Realod ' + from_dir + model_name
    saver.restore(sess, from_dir + model_name)
    offset = int(model_name.split('-')[1])
#ipdb.set_trace()
total_len = len(crc_data_sets.train.images)
valid_len = len(crc_data_sets.valid.images)

batch_num = total_len / batch_size
valid_num = valid_len / batch_size
for epoch in range(n_epochs):
    epoch += offset
    for i in range(batch_num):
        # batch_c3d : (batch_size, 35, 512, 2, 7, 7)
        batch_images, batch_maps, batch_c3d, batch_pupil = crc_data_sets.train.next_batch(batch_size)
        batch_c3d = np.asarray([c3d.reshape([n_lstm_steps,1024*49]) for c3d in batch_c3d])
        batch_maps = np.asarray([mp.reshape([n_lstm_steps,49]) for mp in batch_maps])
        #pupil_sizes = np.reshape(np.asarray(batch_pupil), [batch_size,n_lstm_steps,1])
        pupil_sizes = np.zeros([batch_size,n_lstm_steps,1])
        batch_maps = np.concatenate((batch_maps,pupil_sizes), axis=2)
        _, loss_val = sess.run([train_op, tf_loss],
                               feed_dict={tf_cnn:batch_c3d, tf_gazemap:batch_maps})
        if np.mod(i,5) == 0:
            print 'Epoch : ' + str(epoch) + ' ( ' + str(i) + ' / ' + str(batch_num) + ' )'
            print 'Loss value : %f '%(loss_val)
    if np.mod(epoch,5) == 0:
        valid_map_list = []
        valid_pupil_list = []
        GT_map_list = []
        GT_pupil_list = []
        for v in range(valid_num):
            batch_images, batch_maps, batch_c3d, batch_pupil = crc_data_sets.valid.next_batch(batch_size)
            batch_c3d = np.asarray([c3d.reshape([n_lstm_steps,1024*49]) for c3d in batch_c3d])
            batch_maps = np.asarray([mp.reshape([n_lstm_steps,49]) for mp in batch_maps])
            pupil_sizes = np.reshape(np.asarray(batch_pupil), [batch_size,n_lstm_steps,1])
            #pupil_sizes = np.zeros([batch_size,n_lstm_steps,1])
            #batch_maps = np.concatenate((batch_maps,pupil_sizes), axis=2)
            #they have [B x T x 49] or [B x T x 1] dimension.
            gazes, pupils = sess.run([gaze_output, pupil_output],
                                     feed_dict={cnn_tf:batch_c3d})
            valid_map_list.extend(gazes)
            valid_pupil_list.extend(pupils)
            GT_map_list.extend(batch_maps)
            GT_pupil_list.extend(pupil_sizes)
        # Evaluate list.





    if np.mod(epoch,10) == 0:
        saver.save(sess, os.path.join(save_model_dir, 'model'), global_step=epoch)
