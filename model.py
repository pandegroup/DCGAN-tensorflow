from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

"""
Function from:
https://github.com/tensorflow/tensorflow/issues/6095
"""
def atan2(y, x, epsilon=1.0e-12):
  # Add a small number to all zeros, to avoid division by zero:
  x = tf.where(tf.equal(x, 0.0), x+epsilon, x)
  y = tf.where(tf.equal(y, 0.0), y+epsilon, y)

  angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
  angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
  angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
  angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
  angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
  angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), tf.zeros_like(x), angle)
  return angle


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, n_layers=2, batch_size=100,
                 max_n_atoms=200, n_atom_types=25,
                 max_valence=4,
                 L_list = [100, 100, 100, 100],
                 n_tasks=1,
                 learning_rate=1e-4,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-7,
                 save_path="./temp_model",
                 dropout=1.):

    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(log_device_placement=True))
    self.dropout = dropout
    self.pad_batches = True
    self.B = max_n_atoms
    B = self.B
    self.p = n_atom_types
    p = self.p
    self.V = max_valence
    V = self.V
    self.n_layers = n_layers

    self.batch_size = batch_size
    self.S = batch_size
    S = self.S
    self._save_path = save_path
    self.n_tasks = int(n_tasks)
    
    self.L_list = L_list
    self.n_layers = n_layers

    self.build_model()




    """
    self.is_crop = is_crop
    self.is_grayscale = (c_dim == 1)

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.c_dim = c_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.build_model()
    """

  def construct_feed_dict(self, X, start=None,
                          stop=None, y=None,
                          keep_prob=1.0, train=False):
    S = self.S
    w_b = np.ones((self.batch_size, self.n_tasks))

    if start is None:
      start = 0
      stop = len(X)

    a = time.time()
    adj = [X[idx][2][0].toarray().astype(np.float32) for idx in range(start, stop)]
    A_batch = [X[idx][2][1].toarray() for idx in range(start, stop)]
    D_batch = [X[idx][1][0] for idx in range(start, stop)]
    y_batch = [X[idx][1][1].toarray() for idx in range(start, stop)]

    a = time.time()
    y_batch = np.squeeze(np.concatenate(y_batch))


    a = time.time()
    non_zero_batch = np.where(y_batch != 0.)[0]

    a = time.time()
    onehotter = OneHotEncoder(n_values = S*X[0][1][1].shape[0])
    non_zero_onehot = onehotter.fit_transform(non_zero_batch).toarray().reshape((len(non_zero_batch),S*X[0][1][1].shape[0]))

    feed_dict = {self.x: A_batch,
                 self.adj_matrix: adj,
                 self.weight_placeholder: w_b,
                 self.phase: True,
                 self.keep_prob: keep_prob,
                 self.phase: train,
                 self.label_placeholder: y_batch,
                 self.non_zero_inds: non_zero_onehot,
                 self.dihed_indices: D_batch
                }
    return(feed_dict)

  def build_model(self):
    S = self.S
    B = self.B
    p = self.p
    self.keep_prob = tf.placeholder(tf.float32)
    self.phase = tf.placeholder(dtype='bool', name='phase')

    self.x = tf.placeholder(tf.float32, shape=[S, B, p])

    self.non_zero_inds = tf.placeholder(tf.int32, shape=[None, S*250])

    self.adj_matrix = tf.placeholder(tf.float32, shape=[S, B, B])
    self.dihed_indices = tf.placeholder(tf.float32, shape=[S, 250, B, 4])

    self.label_placeholder = tf.placeholder(
      dtype='float32', shape=[S*250], name="label_placeholder")

    self.weight_placeholder = tf.placeholder(
      dtype='float32', shape=(S, self.n_tasks), name="weight_placeholder")

    self.phase = tf.placeholder(dtype='bool', name='phase')

    self.z = tf.placeholder(tf.float32,
                            [None, self.L_list[0]], name='z')

    self.G = self.generator()
    self.D_logits = self.discriminator(reuse=False)
    #self.sampler = self.sampler()
    self.D_logits_ = self.discriminator(reuse=True)

    self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss = self.d_loss_real + self.d_loss_fake

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]


  def train(self, train_dataset, config):
    """Train DCGAN"""
    #np.random.shuffle(data)

    d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    n_train = len(train_dataset)
    S = self.S

    for i in range(0,n_epochs):
      if i % 1 == 0:
        print("Training epoch %d" %i)
      batch_sched = list(range(0, n_train+1,S))
      t = time.time()
      for j in range(0, len(batch_sched)-1):
        print(j)
        start = batch_sched[j]
        stop = batch_sched[j+1]
        a = time.time()
        feed_dict = self.construct_feed_dict(train_dataset, start, stop)

        self.sess.run(d_optim, feed_dict=feed_dict)
        self.sess.run(g_optim, feed_dict=feed_dict)
        self.sess.run(g_optim, feed_dict=feed_dict)
        
        errD_fake = self.sess.run(self.d_loss_fake, feed_dict=feed_dict)
        errD_real = self.sess.run(self.d_loss_real, feed_dict=feed_dict)
        errG = self.sess.run(self.g_loss, feed_dict=feed_dict)

        print(errD_fake)
        print(errD_real)
        print(errG)

  def discriminator(self, reuse):
    n_layers = self.n_layers
    S = self.S
    B = self.B
    L_list = self.L_list
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

        W_list = [None for i in range(n_layers)]
        b_list = [None for i in range(n_layers)]
        h_list = [None for i in range(n_layers)]

        def adjacency_conv_layer(atom_matrix, W, b, L_in, L_out, layer_idx):
          print("layer_idx: %d" %(layer_idx))
          h = tf.matmul(self.adj_matrix, atom_matrix)
          h = tf.reshape(h, shape=(S*B, L_in))

          h = tf.nn.sigmoid(tf.matmul(h, W) + b)
          return(h)

        for layer_idx in range(n_layers):
          if layer_idx == 0:
            L_in = p
            L_out = L_list[0]
            atom_matrix = x
          else:
            L_in = L_list[layer_idx-1]
            L_out = L_list[layer_idx]
            atom_matrix = h_list[layer_idx-1]

          W_list[layer_idx] = tf.Variable(tf.truncated_normal([L_in, L_out], seed=2017), name="W_list%d" %layer_idx)
          b_list[layer_idx] = tf.Variable(tf.ones([1, L_out]))
          h_list[layer_idx] = adjacency_conv_layer(atom_matrix, W_list[layer_idx], b_list[layer_idx], L_in, L_out, layer_idx)

        L_final = L_list[n_layers-1]
        h_final = tf.reshape(h_list[-1], (S, B, L_final))

        #add dihedral regressor layers

        d0 = []
        for i in range(0, S):
          mol_tuple = []
          for j in range(0, 4):
            entry = h_final[i]
            indices = self.dihed_indices[i][:,:,j]
            atom_list = tf.matmul(indices, entry)
            atom_list = tf.reshape(atom_list, (250, L_final))
            mol_tuple.append(atom_list)
          mol_tuple = tf.reshape(tf.stack(mol_tuple, axis=1), (250, L_final*4))
          d0.append(mol_tuple)

        d0 = tf.concat(d0, axis=0)


        W_d0 = tf.Variable(tf.truncated_normal([L_final*4, 100]))
        b_d0 = tf.Variable(tf.ones([1, 100]))
        d2 = tf.nn.sigmoid(tf.matmul(d0, W_d0) + b_d0)

        W_d2 = tf.Variable(tf.truncated_normal([100, 1]))
        b_d2 = tf.Variable(tf.ones([1, 1]))
        d3 = tf.matmul(d2, W_d2) + b_d2
        d3_cos = tf.cos(d3)
        d3_sin = tf.sin(d3)
        output = atan2(d3_sin, d3_cos)

        output = tf.matmul(tf.cast(self.non_zero_inds, tf.float32), output)
        labels = tf.matmul(tf.cast(self.non_zero_inds, tf.float32), self.label_placeholder)

        def expand_basis(angles):
          return(tf.stack([tf.abs(angles), tf.sin(angles), tf.sin(2*angles), tf.sin(3*angles), tf.sin(4*angles), tf.sin(5*angles)]))

        sq_diff = tf.square(tf.subtract(labels, output))

        W_diff1 = tf.Variable(tf.truncated_normal([6, 10]))
        b_diff1 = tf.Variable(tf.truncated_normal([1, 10]))
        fc_diff1 = tf.nn.relu(tf.matmul(sq_diff, W_diff1) + b_diff1)

        W_diff2 = tf.Variable(tf.truncated_normal([10, 1]))
        b_diff2 = tf.Variable(tf.truncated_normal([1, 1]))
        fc_diff2 = tf.nn.sigmoid(tf.matmul(fc_diff1, W_diff1) + b_diff1)

        return(fc_diff2)

  def generator(self):
    n_layers = self.n_layers
    S = self.S
    B = self.B
    L_list = self.L_list
    p = self.p
    x = self.x 
    with tf.variable_scope("generator") as scope:

      W_list = [None for i in range(n_layers)]
      b_list = [None for i in range(n_layers)]
      h_list = [None for i in range(n_layers)]

      def adjacency_conv_layer(atom_matrix, W, b, L_in, L_out, layer_idx):
        print("layer_idx: %d" %(layer_idx))
        h = tf.matmul(self.adj_matrix, atom_matrix)
        h = tf.reshape(h, shape=(S*B, L_in))

        h = tf.nn.sigmoid(tf.matmul(h, W) + b)
        return(h)

      for layer_idx in range(n_layers):
        if layer_idx == 0:
          L_in = p
          L_out = L_list[0]
          atom_matrix = x
        else:
          L_in = L_list[layer_idx-1]
          L_out = L_list[layer_idx]
          atom_matrix = h_list[layer_idx-1]

        W_list[layer_idx] = tf.Variable(tf.truncated_normal([L_in, L_out], seed=2017), name="W_list%d" %layer_idx)
        b_list[layer_idx] = tf.Variable(tf.ones([1, L_out]))
        h_list[layer_idx] = adjacency_conv_layer(atom_matrix, W_list[layer_idx], b_list[layer_idx], L_in, L_out, layer_idx)

      L_final = L_list[n_layers-1]
      h_final = tf.reshape(h_list[-1], (S, B, L_final))

      #add dihedral regressor layers

      d0 = []
      for i in range(0, S):
        mol_tuple = []
        for j in range(0, 4):
          entry = h_final[i]
          indices = self.dihed_indices[i][:,:,j]
          atom_list = tf.matmul(indices, entry)
          atom_list = tf.reshape(atom_list, (250, L_final))
          mol_tuple.append(atom_list)
        mol_tuple = tf.reshape(tf.stack(mol_tuple, axis=1), (250, L_final*4))
        d0.append(mol_tuple)

      d0 = tf.concat(d0, axis=0)


      W_d0 = tf.Variable(tf.truncated_normal([L_final*4, 100]))
      b_d0 = tf.Variable(tf.ones([1, 100]))
      d2 = tf.nn.sigmoid(tf.matmul(d0, W_d0) + b_d0)

      W_d2 = tf.Variable(tf.truncated_normal([100, 1]))
      b_d2 = tf.Variable(tf.ones([1, 1]))
      d3 = tf.matmul(d2, W_d2) + b_d2
      d3_cos = tf.cos(d3)
      d3_sin = tf.sin(d3)
      output = atan2(d3_sin, d3_cos)

      output = tf.matmul(tf.cast(self.non_zero_inds, tf.float32), output)

      return(output)

  def sampler(self, reuse):
    n_layers = self.n_layers
    S = self.S
    B = self.B
    L_list = self.L_list
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      W_list = [None for i in range(n_layers)]
      b_list = [None for i in range(n_layers)]
      h_list = [None for i in range(n_layers)]

      def adjacency_conv_layer(atom_matrix, W, b, L_in, L_out, layer_idx):
        print("layer_idx: %d" %(layer_idx))
        h = tf.matmul(self.adj_matrix, atom_matrix)
        h = tf.reshape(h, shape=(S*B, L_in))

        h = tf.nn.sigmoid(tf.matmul(h, W) + b)
        return(h)

      for layer_idx in range(n_layers):
        if layer_idx == 0:
          L_in = p
          L_out = L_list[0]
          atom_matrix = x
        else:
          L_in = L_list[layer_idx-1]
          L_out = L_list[layer_idx]
          atom_matrix = h_list[layer_idx-1]

        W_list[layer_idx] = tf.Variable(tf.truncated_normal([L_in, L_out], seed=2017), name="W_list%d" %layer_idx)
        b_list[layer_idx] = tf.Variable(tf.ones([1, L_out]))
        h_list[layer_idx] = adjacency_conv_layer(atom_matrix, W_list[layer_idx], b_list[layer_idx], L_in, L_out, layer_idx)

      L_final = L_list[n_layers-1]
      h_final = tf.reshape(h_list[-1], (S, B, L_final))

      #add dihedral regressor layers

      d0 = []
      for i in range(0, S):
        mol_tuple = []
        for j in range(0, 4):
          entry = h_final[i]
          indices = self.dihed_indices[i][:,:,j]
          atom_list = tf.matmul(indices, entry)
          atom_list = tf.reshape(atom_list, (250, L_final))
          mol_tuple.append(atom_list)
        mol_tuple = tf.reshape(tf.stack(mol_tuple, axis=1), (250, L_final*4))
        d0.append(mol_tuple)

      d0 = tf.concat(d0, axis=0)


      W_d0 = tf.Variable(tf.truncated_normal([L_final*4, 100]))
      b_d0 = tf.Variable(tf.ones([1, 100]))
      d2 = tf.nn.sigmoid(tf.matmul(d0, W_d0) + b_d0)

      W_d2 = tf.Variable(tf.truncated_normal([100, 1]))
      b_d2 = tf.Variable(tf.ones([1, 1]))
      d3 = tf.matmul(d2, W_d2) + b_d2
      d3_cos = tf.cos(d3)
      d3_sin = tf.sin(d3)
      output = atan2(d3_sin, d3_cos)

      output = tf.matmul(tf.cast(self.non_zero_inds, tf.float32), output)

      return(output)


  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
