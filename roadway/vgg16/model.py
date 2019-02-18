from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def compact_bilinear(tensors_list):

    def _generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert (rand_h.ndim == 1 and rand_s.ndim == 1 and len(rand_h) == len(rand_s))
        assert (np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        sparse_sketch_matrix = tf.sparse_reorder(
            tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
        return sparse_sketch_matrix

    bottom1, bottom2 = tensors_list
    output_dim = 7000

    # Static shapes are needed to construction count sketch matrix
    input_dim1 = bottom1.get_shape().as_list()[-1]
    input_dim2 = bottom2.get_shape().as_list()[-1]

    # print (bottom1.get_shape().as_list())
    # print (bottom2.get_shape().as_list())

    # Step 0: Generate vectors and sketch matrix for tensor count sketch
    # This is only done once during graph construction, and fixed during each
    # operation
    seed_h_1 = 1
    seed_s_1 = 3
    seed_h_2 = 5
    seed_s_2 = 7

    # Generate sparse_sketch_matrix1 using rand_h_1 and rand_s_1
    np.random.seed(seed_h_1)
    rand_h_1 = np.random.randint(output_dim, size=input_dim1)
    np.random.seed(seed_s_1)
    rand_s_1 = 2 * np.random.randint(2, size=input_dim1) - 1
    sparse_sketch_matrix1 = _generate_sketch_matrix(rand_h_1, rand_s_1, output_dim)

    # Generate sparse_sketch_matrix2 using rand_h_2 and rand_s_2
    np.random.seed(seed_h_2)
    rand_h_2 = np.random.randint(output_dim, size=input_dim2)
    np.random.seed(seed_s_2)
    rand_s_2 = 2 * np.random.randint(2, size=input_dim2) - 1
    sparse_sketch_matrix2 = _generate_sketch_matrix(rand_h_2, rand_s_2, output_dim)

    # Step 1: Flatten the input tensors and count sketch
    bottom1_flat = tf.reshape(bottom1, [-1, input_dim1])
    bottom2_flat = tf.reshape(bottom2, [-1, input_dim2])

    # Essentially:
    #   sketch1 = bottom1 * sparse_sketch_matrix
    #   sketch2 = bottom2 * sparse_sketch_matrix
    # But tensorflow only supports left multiplying a sparse matrix, so:
    #   sketch1 = (sparse_sketch_matrix.T * bottom1.T).T
    #   sketch2 = (sparse_sketch_matrix.T * bottom2.T).T
    sketch1 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix1,
                                                         bottom1_flat, adjoint_a=True, adjoint_b=True))
    sketch2 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix2,
                                                         bottom2_flat, adjoint_a=True, adjoint_b=True))

    # Step 2: FFT
    fft1 = tf.fft(tf.complex(real=sketch1, imag=tf.zeros_like(sketch1)))
    fft2 = tf.fft(tf.complex(real=sketch2, imag=tf.zeros_like(sketch2)))

    # Step 3: Elementwise product
    fft_product = tf.multiply(fft1, fft2)

    # Step 4: Inverse FFT and reshape back
    # Compute output shape dynamically: [batch_size, height, width, output_dim]
    cbp_flat = tf.real(tf.ifft(fft_product))

    output_shape = tf.add(tf.multiply(tf.shape(bottom1), [1, 1, 1, 0]),
                          [0, 0, 0, output_dim])
    cbp = tf.reshape(cbp_flat, output_shape)

    # print (cbp.get_shape().as_list())

    return cbp
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer,
			  dtype=tf.float32)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized variable with weight decay
  """
  var = _variable_on_cpu(
        name, shape, 
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _conv_layer(input, shape, strides, scope):
  """Helper to setup a convolution layer (CONV + RELU)
  """
  #kernel = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32,
  #                                        stddev=1e-1), name='weights')
  kernel = _variable_with_weight_decay('weights',
                                        shape=shape,
                                        stddev=1e-1,
                                        wd=0.0)
  conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')
  biases = _variable_on_cpu('biases', shape=[shape[3]], 
                            initializer=tf.constant_initializer(0.0))
  out = tf.nn.bias_add(conv, biases)
  return tf.nn.relu(out, scope.name)

def inference(images, no_classes, keep_prob=1.0):
    
  # conv1_1
  with tf.variable_scope('conv1_1') as scope:
    conv1_1 = _conv_layer(images, [3,3,3,64], [1,1,1,1], scope)
  # conv1_2
  with tf.variable_scope('conv1_2') as scope:
    conv1_2 = _conv_layer(conv1_1, [3,3,64,64], [1,1,1,1], scope)
    # pool1
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool1')
  # conv2_1
  with tf.variable_scope('conv2_1') as scope:
    conv2_1 = _conv_layer(pool1, [3, 3, 64, 128], [1, 1, 1, 1], scope)

  # conv2_2
  with tf.variable_scope('conv2_2') as scope:
    conv2_2 = _conv_layer(conv2_1, [3, 3, 128, 128], [1, 1, 1, 1], scope)
    # pool2
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool2')
  # conv3_1
  with tf.variable_scope('conv3_1') as scope:
    conv3_1 = _conv_layer(pool2, [3, 3, 128, 256], [1, 1, 1, 1], scope)

  # conv3_2
  with tf.variable_scope('conv3_2') as scope:
    conv3_2 = _conv_layer(conv3_1, [3, 3, 256, 256], [1, 1, 1, 1], scope)
        
  # conv3_3
  with tf.variable_scope('conv3_3') as scope:
    conv3_3 = _conv_layer(conv3_2, [3, 3, 256, 256], [1, 1, 1, 1], scope)
   
  # pool3
  pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding='SAME', name='pool3')
  
  # conv4_1
  with tf.variable_scope('conv4_1') as scope:
    conv4_1 = _conv_layer(pool3, [3, 3, 256, 512], [1, 1, 1, 1], scope)

  # conv4_2
  with tf.variable_scope('conv4_2') as scope:
    conv4_2 = _conv_layer(conv4_1, [3, 3, 512, 512], [1, 1, 1, 1], scope)

  # conv4_3
  with tf.variable_scope('conv4_3') as scope:
    conv4_3 = _conv_layer(conv4_2, [3, 3, 512, 512], [1, 1, 1, 1], scope)
    
  # pool4
  pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding='SAME', name='pool4')

  # conv5_1
  with tf.variable_scope('conv5_1') as scope:
    conv5_1 = _conv_layer(pool4, [3, 3, 512, 512], [1, 1, 1, 1], scope)

  # conv5_2
  with tf.variable_scope('conv5_2') as scope:
    conv5_2 = _conv_layer(conv5_1, [3, 3, 512, 512], [1, 1, 1, 1], scope)

  # conv5_3
  with tf.variable_scope('conv5_3') as scope:
    conv5_3 = _conv_layer(conv5_2, [3, 3, 512, 512], [1, 1, 1, 1], scope)

  compact_bilinear_arg_list = [conv5_3, conv5_3]
  output_shape_x = conv5_3.get_shape().as_list()[1:]   
  bilinear_output_dim = 7000
  output_shape_cb = (output_shape_x[0], output_shape_x[1], bilinear_output_dim,)   
  # compact bilinear
  x = tf.keras.layers.Lambda(compact_bilinear, output_shape_cb)(compact_bilinear_arg_list)
  x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sign(x) * tf.keras.backend.sqrt(tf.keras.backend.abs(x)))(x)
  x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1))(x)
  #weights_regularizer = tf.keras.regularizers.l2(5e-4)
  #x = tf.keras.layers.Dense(no_classes,activation= 'softmax', name='softmax_layer', kernel_regularizer=weights_regularizer)(x)
  # fc3
  with tf.variable_scope('fc3') as scope:
    fc3w = _variable_with_weight_decay('weights', shape=[1372000, no_classes],
                                        stddev=1e-1, wd=0.0)
    fc3b = _variable_on_cpu('biases', [no_classes],
                            tf.constant_initializer(1.0))
    # fc3l -> softmax_linear
    fc3l = tf.nn.bias_add(tf.matmul(tf.layers.flatten(x), fc3w), fc3b)
  
  return fc3l

def loss_function(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits=logits, labels=labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  total_loss =  tf.add_n(tf.get_collection('losses'), name='total_loss')
  with tf.control_dependencies([total_loss]):
    total_loss = tf.identity(total_loss)
  return total_loss

def load_weights(weight_file, sess):
  parameters = tf.trainable_variables()
  weights = np.load(weight_file)
  keys = sorted(weights.keys())
  for i, k in enumerate(keys):
    # Skipping last layer
    if k == 'fc8_W' or k == 'fc8_b' or k == 'fc7_W' or k == 'fc7_b' or k == 'fc6_W' or k == 'fc6_b':
      continue
    #print (i, k, np.shape(weights[k]))
    sess.run(parameters[i].assign(weights[k]))
