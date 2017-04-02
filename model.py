from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from sklearn.preprocessing import OneHotEncoder

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
                 max_n_atoms=10000, n_atom_types=25,
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
    with self.graph.as_default():
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

      self.learning_rate = learning_rate
      self.beta1 = beta1

      S = self.S
      B = self.B
      p = self.p
      self.keep_prob = tf.placeholder(tf.float32)
      self.phase = tf.placeholder(dtype='bool', name='phase')

      self.x = tf.placeholder(tf.float32, shape=[S*10000, p])

      self.non_zero_inds = tf.placeholder(tf.int32, shape=[None, S*25])

      self.adj_matrix = tf.placeholder(tf.float32, shape=[S, B, B])
      self.dihed_indices = tf.placeholder(tf.float32, shape=[S, 25, B, 4])

      self.label_placeholder = tf.placeholder(
        dtype='float32', shape=[S*10000], name="label_placeholder")

      self.weight_placeholder = tf.placeholder(
        dtype='float32', shape=(S, self.n_tasks), name="weight_placeholder")

      self.phase = tf.placeholder(dtype='bool', name='phase')

      self.z = tf.placeholder(tf.float32,
                              [S*10000, 5], name='z')

      self.G = self.generator(self.x, reuse=False)

      self.D_logits = self.discriminator(self.label_placeholder, reuse=False)
      #self.sampler = self.sampler()
      self.D_logits_ = self.discriminator(self.G, reuse=True)


      #self.d_loss_real = tf.reduce_mean(self.D_logits)
      #self.d_loss_fake = -1.0*tf.reduce_mean(self.D_logits_)
      #self.g_loss = tf.reduce_mean(tf.abs(self.D_logits_))

      #self.d_loss_real = tf.reduce_mean(tf.nn.log(self.D_logits))
      #self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_logits_)))
      #self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_logits_)))

      self.d_loss = tf.reduce_mean(-tf.log(self.D_logits)-tf.log(1-self.D_logits_))
      self.g_loss = tf.reduce_mean(-tf.log(self.D_logits_))

      t_vars = tf.trainable_variables()

      self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
      self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

      self.d_vars = [var for var in t_vars if 'd_' in var.name]
      self.g_vars = [var for var in t_vars if 'g_' in var.name]

      self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_params)#, var_list=self.d_vars)
      self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_params)#, var_list=self.g_vars)
      self.predicted = self.generator(self.x, reuse=True)
      #self.predicted = self.sampler(self.x, self.adj_matrix, self.dihed_indices, self.non_zero_inds, self.z)

      self.init_fn = tf.global_variables_initializer()
      print(self.sess.run(self.init_fn))


      #self.build_model()

  def construct_feed_dict(self, X=None, start=None,
                          stop=None, y=None,
                          keep_prob=1.0, train=False):


    x_batch = np.reshape(np.sort(np.random.uniform(-1,1, size=(1*10000, 1))), (-1,1))
    y_batch = np.squeeze(np.sort(np.random.normal(size=(1*10000,1))))

    feed_dict = {self.x: x_batch,
                 self.label_placeholder: y_batch,
                }
    return(feed_dict)

  def build_model(self):
    with self.graph.as_default():
      pass


  def train(self, train_dataset, n_epochs):
    """Train DCGAN"""
    #np.random.shuffle(data)

    n_train = len(train_dataset)
    S = self.S
    preds = []
    for i in range(0,n_epochs):
      if i % 1 == 0:
        print("Training epoch %d" %i)
      batch_sched = list(range(0, n_train+1,S))
      t = time.time()
      for j in range(0, len(batch_sched)-1):
        #print(j)
        start = batch_sched[j]
        stop = batch_sched[j+1]
        a = time.time()
        feed_dict = self.construct_feed_dict(train_dataset, start, stop)

        self.sess.run(self.d_optim, feed_dict=feed_dict)
        self.sess.run(self.g_optim, feed_dict=feed_dict)
        #self.sess.run(self.g_optim, feed_dict=feed_dict)
        #print("final labels")
        #print(self.sess.run(self.labels, feed_dict))
        #print("reshaped labels")
        #print(self.sess.run(self.initial_labels, feed_dict=feed_dict))
        
        #errD_fake = self.sess.run(self.d_loss_fake, feed_dict=feed_dict)
        #errD_real = self.sess.run(self.d_loss_real, feed_dict=feed_dict)
        errD = self.sess.run(self.d_loss, feed_dict=feed_dict)
        errG = self.sess.run(self.g_loss, feed_dict=feed_dict)

        if i == n_epochs-1:
          preds.append(self.sess.run(self.predicted, feed_dict=feed_dict))
        #if j == 0 :
        #  print(self.sess.run(self.G, feed_dict=feed_dict)[:3])



      if i % 10 == 0:
        print(errD)
        print(errG)
    print(preds[0][:3])
  def predict(self):

    feed_dict = self.construct_feed_dict()
    return(self.sess.run(self.predicted, feed_dict=feed_dict))


  def discriminator(self, label_placeholder, reuse):
    n_layers = self.n_layers
    S = self.S
    B = self.B
    L_list = self.L_list
    p = self.p

    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      W0 = tf.Variable(tf.random_normal([p, 5]))
      b0 = tf.Variable(tf.zeros([1, 5]))

      y = tf.reshape(label_placeholder, (-1,1))
      h0 = tf.nn.tanh(tf.matmul(y, W0) + b0)

      #h0 = tf.add(h0, z)

      W1 = tf.Variable(tf.random_normal([5, 5]))
      b1 = tf.Variable(tf.zeros([1, 5]))

      h1 = tf.nn.tanh(tf.matmul(h0, W1) + b1)

      W2 = tf.Variable(tf.random_normal([5, 5]))
      b2 = tf.Variable(tf.zeros([1, 5]))

      h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
      #print("disc.h1")
      #print(h1)

      #return(h1)


      W3 = tf.Variable(tf.truncated_normal([5, 1]))
      b3 = tf.Variable(tf.ones([1, 1]))

      h3 = tf.nn.sigmoid(tf.matmul(h2, W3)+b3)
      return(h3 )

      """

      W_list = [None for i in range(n_layers)]
      b_list = [None for i in range(n_layers)]
      h_list = [None for i in range(n_layers)]

      def adjacency_conv_layer(atom_matrix, W, b, L_in, L_out, layer_idx):
        print("layer_idx: %d" %(layer_idx))
        h = tf.matmul(adj_matrix, atom_matrix)
        h = tf.reshape(h, shape=(S*B, L_in))

        h = tf.nn.sigmoid(tf.matmul(h, W) + b)
        h = tf.reshape(h, (S, B, L_out))

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

        W_list[layer_idx] = tf.Variable(tf.truncated_normal([L_in, L_out]), name="W_list%d" %layer_idx)
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
          indices = dihed_indices[i][:,:,j]
          atom_list = tf.matmul(indices, entry)
          atom_list = tf.reshape(atom_list, (25, L_final))
          mol_tuple.append(atom_list)
        mol_tuple = tf.reshape(tf.stack(mol_tuple, axis=1), (25, L_final*4))
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

      output = tf.matmul(tf.cast(non_zero_inds, tf.float32), output, name='reduce_output_nonzeros')

      reshaped_labels = tf.reshape(label_placeholder, (-1,1), name='reshaped_labels')
      labels = tf.matmul(tf.cast(non_zero_inds, tf.float32), reshaped_labels, name='reduce_labels_nonzeros')

      self.final_labels = labels
      self.initial_labels = reshaped_labels

      return(tf.abs(tf.subtract(output, labels)))
      def expand_basis(ini_angles):
        angles = tf.reshape(ini_angles, (-1,1))
        return(tf.reshape(tf.stack([tf.abs(angles), tf.sin(angles), tf.sin(2*angles), tf.sin(3*angles), tf.sin(4*angles), tf.sin(5*angles)], axis=1, name='concat_basis'), (-1,6), name="concat_reshape"))


      sq_diff = tf.square(tf.subtract(expand_basis(labels), expand_basis(output)))

      W_diff1 = tf.Variable(tf.truncated_normal([6, 10]))
      b_diff1 = tf.Variable(tf.truncated_normal([1, 10]))
      fc_diff1 = tf.nn.relu(tf.matmul(sq_diff, W_diff1) + b_diff1)

      W_diff2 = tf.Variable(tf.truncated_normal([10, 1]))
      b_diff2 = tf.Variable(tf.truncated_normal([1, 1]))
      fc_diff2 = tf.matmul(fc_diff1, W_diff2) + b_diff2

      print("fc_diff2")
      print(fc_diff2)

      return(fc_diff2)
      """
      

  def generator(self, x, reuse=False):
    n_layers = self.n_layers
    S = self.S
    B = self.B
    L_list = self.L_list
    p = self.p
    with tf.variable_scope("generator") as scope:
      if reuse:
        scope.reuse_variables()

      x = tf.reshape(x, (-1,p))

      W0 = tf.Variable(tf.random_normal([p, 200]))
      b0 = tf.Variable(tf.zeros([1, 200]))

      h0 = tf.nn.softplus(tf.matmul(x, W0) + b0)

      #h0 = tf.add(h0, z)

      W1 = tf.Variable(tf.random_normal([200, 1]))
      b1 = tf.Variable(tf.zeros([1, 1]))

      h1 = tf.matmul(h0, W1) + b1

      #W2 = tf.Variable(tf.random_normal([5, 1]))
      #b2 = tf.Variable(tf.zeros([1, 1]))

      #h2 = tf.matmul(h1, W2) + b2
      #output = tf.abs(tf.atan(tf.sin(h1)/tf.cos(h1)))

      print("gen.h1")
      print(h1)

      return(h1)
    """
      W_list = [None for i in range(n_layers)]
      b_list = [None for i in range(n_layers)]
      h_list = [None for i in range(n_layers)]

      def adjacency_conv_layer(atom_matrix, W, b, L_in, L_out, layer_idx):
        print("layer_idx: %d" %(layer_idx))
        h = tf.matmul(adj_matrix, atom_matrix)
        h = tf.reshape(h, shape=(S*B, L_in))
        h = tf.nn.sigmoid(tf.matmul(h, W) + b)
        if layer_idx == 0:
          h = tf.add(h, z)
        h = tf.reshape(h, (S, B, L_out))

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

        W_list[layer_idx] = tf.Variable(tf.truncated_normal([L_in, L_out]), name="W_list%d" %layer_idx)
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
          indices = dihed_indices[i][:,:,j]
          atom_list = tf.matmul(indices, entry)
          atom_list = tf.reshape(atom_list, (25, L_final))
          mol_tuple.append(atom_list)
        mol_tuple = tf.reshape(tf.stack(mol_tuple, axis=1), (25, L_final*4))
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
      output = tf.abs(atan2(d3_sin, d3_cos))

      #output = tf.matmul(tf.cast(non_zero_inds, tf.float32), output)

      return(output)
    """

  def sampler(self, x, adj_matrix, dihed_indices, non_zero_inds, z):
    n_layers = self.n_layers
    S = self.S
    B = self.B
    L_list = self.L_list
    p = self.p
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      x = tf.reshape(x, (-1,75))

      W0 = tf.Variable(tf.truncated_normal([p, 5]))
      b0 = tf.Variable(tf.ones([1, 5]))

      h0 = tf.nn.relu(tf.matmul(x, W0) + b0)

      h0 = tf.add(h0, z)

      W1 = tf.Variable(tf.truncated_normal([5, 1]))
      b1 = tf.Variable(tf.ones([1, 1]))

      h1 = tf.nn.relu(tf.matmul(h0, W1) + b1)
      output = tf.atan(tf.sin(h1)/tf.cos(h1))

      return(output)
      """

      W_list = [None for i in range(n_layers)]
      b_list = [None for i in range(n_layers)]
      h_list = [None for i in range(n_layers)]

      def adjacency_conv_layer(atom_matrix, W, b, L_in, L_out, layer_idx):
        print("layer_idx: %d" %(layer_idx))
        h = tf.matmul(adj_matrix, atom_matrix)
        h = tf.reshape(h, shape=(S*B, L_in))
        h = tf.nn.sigmoid(tf.matmul(h, W) + b)
        if layer_idx == 0:
          h = tf.add(h, z)
        h = tf.reshape(h, (S, B, L_out))

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

        W_list[layer_idx] = tf.Variable(tf.truncated_normal([L_in, L_out]), name="W_list%d" %layer_idx)
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
          indices = dihed_indices[i][:,:,j]
          atom_list = tf.matmul(indices, entry)
          atom_list = tf.reshape(atom_list, (25, L_final))
          mol_tuple.append(atom_list)
        mol_tuple = tf.reshape(tf.stack(mol_tuple, axis=1), (25, L_final*4))
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
      output = tf.abs(atan2(d3_sin, d3_cos))

      #$output = tf.matmul(tf.cast(non_zero_inds, tf.float32), output)

      return(output)
      """