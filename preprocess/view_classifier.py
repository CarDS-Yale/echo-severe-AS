"""Credit to https://bitbucket.org/rahuldeo/echocv/src/master/."""

import numpy as np
import tensorflow as tf

# Convolution Layer
def conv(x, filter_size, num_filters, stride, weight_decay,  name, padding='SAME', groups=1, trainable=True):
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda x, W: tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    with tf.variable_scope(name):
        # Create tf variables for the weights and biases of the conv layer
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        weights = tf.get_variable('W',
                                  shape=[filter_size, filter_size, input_channels // groups, num_filters],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=trainable,
                                  regularizer=regularizer,
                                  collections=['variables'])
        biases = tf.get_variable('b', shape=[num_filters], trainable=trainable, initializer=tf.zeros_initializer())

        if groups == 1:
            conv = convolve(x, weights)

        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(x, groups, axis=3)
            weight_groups = tf.split(weights, groups, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(output_groups, axis=3)

        return tf.nn.relu(conv + biases)

# Fully Connected Layer
def fc(x, num_out, weight_decay,  name, relu=True, trainable=True):
    num_in = int(x.get_shape()[-1])
    with tf.variable_scope(name):
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        weights = tf.get_variable('W',
                                  shape=[num_in, num_out], 
                                  initializer=tf.contrib.layers.xavier_initializer(), 
                                  trainable=trainable, 
                                  regularizer=regularizer,
                                  collections=['variables'])
        biases = tf.get_variable('b', [num_out], initializer=tf.zeros_initializer(), trainable=trainable)
        x = tf.matmul(x, weights) + biases
        if relu:
            x = tf.nn.relu(x) 
    return x

# Local Response Normalization
def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def max_pool(x, filter_size, stride, name=None, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

# Network
class Network(object):        
    def __init__(self, weight_decay, learning_rate, feature_dim = 1, label_dim = 8, maxout = False):
        self.x_train = tf.placeholder(tf.float32, [None, 224, 224, feature_dim])
        self.y_train = tf.placeholder(tf.uint8, [None, label_dim])
        self.x_test = tf.placeholder(tf.float32, [None, 224, 224, feature_dim])
        self.y_test = tf.placeholder(tf.uint8, [None, label_dim])
        self.label_dim = label_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.maxout = maxout

        self.output = self.network(self.x_train)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.output, 
                                                                           labels = self.y_train))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        self.train_pred = self.network(self.x_train, keep_prob = 1.0, reuse = True)
        self.train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.train_pred, 1), 
                                                              tf.argmax(self.y_train, 1)), tf.float32))
        self.val_pred = self.network(self.x_test, keep_prob = 1.0, reuse = True)
        self.val_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.val_pred, 1), 
                                                        tf.argmax(self.y_test, 1)), tf.float32))

        self.probability = tf.nn.softmax(self.network(self.x_test, keep_prob = 1.0, reuse = True))
           
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.train_summary = tf.summary.scalar('training_accuracy', self.train_accuracy)
    
    # Gradient Descent on mini-batch
    def fit_batch(self, sess, x_train, y_train):
        _, loss, loss_summary = sess.run((self.opt, self.loss, self.loss_summary), feed_dict={self.x_train: x_train, self.y_train: y_train})
        return loss, loss_summary
    
    # Training Accuracy
    def train_validate(self, sess, x_train, y_train):
        train_accuracy, train_summary = sess.run((self.train_accuracy, self.train_summary), 
                                     feed_dict={self.x_train: x_train, self.y_train: y_train})
        return train_accuracy, train_summary
    
    # Validation Accuracy
    def validate(self, sess, x_test, y_test):
        val_accuracy= sess.run((self.val_accuracy), feed_dict={self.x_test: x_test, self.y_test: y_test})
        return val_accuracy
    
    def predict(self, sess, x):
        prediction = sess.run((self.val_pred), feed_dict={self.x_test: x})
        return np.argmax(prediction, axis = 1)
    
    def probabilities(self, sess, x):
        probability = sess.run((self.probability), feed_dict={self.x_test: x})
        return probability
    
    def network(self, input, keep_prob = 0.5, reuse = None):
        with tf.variable_scope('network', reuse=reuse):
            pool_ = lambda x: max_pool(x, 2, 2)
            max_out_ = lambda x: max_out(x, 16)
            conv_ = lambda x, output_depth, name, trainable = True: conv(x, 3, output_depth, 1, self.weight_decay, name=name, trainable = trainable)
            fc_ = lambda x, features, name, relu = True: fc(x, features, self.weight_decay, name, relu = relu)
            VGG_MEAN = [103.939, 116.779, 123.68]
            # Convert RGB to BGR and subtract mean
            # red, green, blue = tf.split(input, 3, axis=3)
            input = tf.concat([
                input - 24,
                input - 24,
                input - 24,
            ], axis=3)

            conv_1_1 = conv_(input, 64, 'conv1_1', trainable = False)
            conv_1_2 = conv_(conv_1_1, 64, 'conv1_2', trainable = False)

            pool_1 = pool_(conv_1_2)

            conv_2_1 = conv_(pool_1, 128, 'conv2_1', trainable = False)
            conv_2_2 = conv_(conv_2_1, 128, 'conv2_2', trainable = False)

            pool_2 = pool_(conv_2_2)

            conv_3_1 = conv_(pool_2, 256, 'conv3_1')
            conv_3_2 = conv_(conv_3_1, 256, 'conv3_2')
            conv_3_3 = conv_(conv_3_2, 256, 'conv3_3')

            pool_3 = pool_(conv_3_3)

            conv_4_1 = conv_(pool_3, 512, 'conv4_1')
            conv_4_2 = conv_(conv_4_1, 512, 'conv4_2')
            conv_4_3 = conv_(conv_4_2, 512, 'conv4_3')

            pool_4 = pool_(conv_4_3)

            conv_5_1 = conv_(pool_4, 512, 'conv5_1')
            conv_5_2 = conv_(conv_5_1, 512, 'conv5_2')
            conv_5_3 = conv_(conv_5_2, 512, 'conv5_3')
            
            pool_5 = pool_(conv_5_3)
            if self.maxout:
                max_5 = max_out_(pool_5)
                flattened = tf.contrib.layers.flatten(max_5)
            else:
                flattened = tf.contrib.layers.flatten(pool_5)
            
            fc_6 = dropout(fc_(flattened, 4096, 'fc6'), keep_prob)
            fc_7 = dropout(fc_(fc_6, 4096, 'fc7'), keep_prob)

            fc_8 = fc_(fc_7, self.label_dim, 'fc8', relu=False)
            return fc_8

    def init_weights(self, sess, vgg_file):
        weights_dict = np.load(vgg_file, encoding='bytes').item()
        weights_dict = { key.decode('ascii') : value for key, value in weights_dict.items() }
        with tf.variable_scope('network', reuse=True):
            for layer in ['conv1_1', 'conv1_2',
                          'conv2_1', 'conv2_2']:
                with tf.variable_scope(layer):
                    W_value, b_value = weights_dict[layer]
                    W = tf.get_variable('W', trainable = False)
                    b = tf.get_variable('b', trainable = False)
                    sess.run(W.assign(W_value))
                    sess.run(b.assign(b_value))
        with tf.variable_scope('network', reuse=True):
            for layer in ['conv3_1', 'conv3_2', 'conv3_3',
                          'conv4_1', 'conv4_2', 'conv4_3',
                          'conv5_1', 'conv5_2', 'conv5_3']:
                with tf.variable_scope(layer):
                    W_value, b_value = weights_dict[layer]
                    W = tf.get_variable('W')
                    b = tf.get_variable('b')
                    sess.run(W.assign(W_value))
                    sess.run(b.assign(b_value))
