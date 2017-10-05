import tensorflow as tf
import numpy as np

import config

class Network():
    def __init__(self, scope_name, trainer):
        with tf.variable_scope(scope_name):
            self.input_image = tf.placeholder(tf.float32, shape=[None, 84, 84, 3])

            conv1_w = tf.Variable(tf.truncated_normal([8, 8, 3, 16], stddev=0.1))
            conv1_b = tf.Variable(tf.constant(0.1, shape=[16]))
            conv1 = tf.nn.relu(tf.nn.conv2d(self.input_image, conv1_w, strides=[1, 4, 4, 1], padding='SAME') + conv1_b)

            conv2_w = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev=0.1))
            conv2_b = tf.Variable(tf.constant(0.1, shape=[32]))
            conv2 = tf.nn.relu(tf.nn.conv2d(conv1, conv2_w, strides=[1, 2, 2, 1], padding='SAME') + conv2_b)

            flat = tf.reshape(conv2, [-1, 11*11*32])

            fc1_w = tf.Variable(tf.truncated_normal([11*11*32, 256], stddev=0.1))
            fc1_b = tf.Variable(tf.constant(0.1, shape=[256]))
            fc1 = tf.nn.relu(tf.matmul(flat, fc1_w) + fc1_b)

            self.input_reward = tf.placeholder(tf.float32, shape=[None, 1]) # aux input
            lstm1_input = tf.concat([fc1, self.input_reward], 1)
            #lstm_seq_len = self.input_reward.shape[0]

            with tf.variable_scope('lstm1'):
                lstm1 = tf.contrib.rnn.BasicLSTMCell(64, state_is_tuple=True)
                self.lstm1_init_state_c = np.zeros((1, lstm1.state_size.c), np.float32)
                self.lstm1_init_state_h = np.zeros((1, lstm1.state_size.h), np.float32)
                self.lstm1_state_c_in = tf.placeholder(tf.float32, shape=[None, lstm1.state_size.c])
                self.lstm1_state_h_in = tf.placeholder(tf.float32, shape=[None, lstm1.state_size.h])
                lstm1_output, lstm1_state_output = tf.nn.dynamic_rnn(
                    lstm1,
                    tf.expand_dims(lstm1_input, [0]), # [None, 257] -> [1, None, 257]
                    initial_state=tf.contrib.rnn.LSTMStateTuple(self.lstm1_state_c_in, self.lstm1_state_h_in),
                    time_major=False
                )
                lstm1_output = tf.reshape(lstm1_output, [-1, 64])
                self.lstm1_state_c_out = [lstm1_state_output[0][0, :]]
                self.lstm1_state_h_out = [lstm1_state_output[1][0, :]]
            
            self.input_velocity = tf.placeholder(tf.float32, shape=[None, 6]) # aux input
            self.input_action = tf.placeholder(tf.int32, shape=[None]) # aux input
            self.input_action_ = tf.one_hot(self.input_action, config.N_ACTION, dtype=tf.float32) # make one-hot vector
            
            lstm2_input = tf.concat([lstm1_output, self.input_velocity], 1)
            lstm2_input = tf.concat([lstm2_input, self.input_action_], 1)
            lstm2_input = tf.concat([lstm2_input, flat], 1)

            with tf.variable_scope('lstm2'):
                lstm2 = tf.contrib.rnn.BasicLSTMCell(256)
                self.lstm2_init_state_c = np.zeros((1, lstm2.state_size.c), np.float32)
                self.lstm2_init_state_h = np.zeros((1, lstm2.state_size.h), np.float32)
                self.lstm2_state_c_in = tf.placeholder(tf.float32, shape=[None, lstm2.state_size.c])
                self.lstm2_state_h_in = tf.placeholder(tf.float32, shape=[None, lstm2.state_size.h])
                lstm2_output, lstm2_state_output = tf.nn.dynamic_rnn(
                    lstm2,
                    tf.expand_dims(lstm2_input, [0]),
                    initial_state=tf.contrib.rnn.LSTMStateTuple(self.lstm2_state_c_in, self.lstm2_state_h_in),
                    time_major=False
                )
                lstm2_output = tf.reshape(lstm2_output, [-1, 256])
                self.lstm2_state_c_out = [lstm2_state_output[0][0, :]]
                self.lstm2_state_h_out = [lstm2_state_output[1][0, :]]
            
            policy_w = tf.Variable(tf.truncated_normal([256, config.N_ACTION], stddev=0.1))
            policy_b = tf.Variable(tf.constant(0.1, shape=[config.N_ACTION]))
            self.policy = tf.nn.softmax(tf.matmul(lstm2_output, policy_w) + policy_b)

            value_w = tf.Variable(tf.truncated_normal([256, 1], stddev=0.1))
            value_b = tf.Variable(tf.constant(0.1, shape=[1]))
            self.value = tf.matmul(lstm2_output, value_w) + value_b

            # predict depth from lstm2
            d1_w = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1)) 
            d1_b = tf.Variable(tf.constant(0.1, shape=[128]))
            d1 = tf.nn.relu(tf.matmul(lstm2_output, d1_w) + d1_b)

            d2_w = tf.Variable(tf.truncated_normal([128, 64*8], stddev=0.1))
            d2_b = tf.Variable(tf.constant(0.1, shape=[64*8]))
            d2 = tf.matmul(d1, d2_w) + d2_b # [None, 64*8]
            d2 = tf.reshape(d2, [-1, 64, 8])
            self.depth_pred = tf.argmax(tf.nn.softmax(d2), axis=2) # [None, 64]

            # predict depth from cnn
            dp1_w = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1)) 
            dp1_b = tf.Variable(tf.constant(0.1, shape=[128]))
            dp1 = tf.nn.relu(tf.matmul(fc1, dp1_w) + dp1_b)

            dp2_w = tf.Variable(tf.truncated_normal([128, 64*8], stddev=0.1))
            dp2_b = tf.Variable(tf.constant(0.1, shape=[64*8]))
            dp2 = tf.matmul(dp1, dp2_w) + dp2_b
            dp2 = tf.reshape(dp2, [-1, 64, 8])
            
            if scope_name!='global':
                self.action = tf.placeholder(tf.int32, shape=[None])
                action_onehot = tf.one_hot(self.action, config.N_ACTION, dtype=tf.float32) # [None, condig.N_ACTION]
                self.true_value = tf.placeholder(tf.float32, shape=[None])
                self.advantage = tf.placeholder(tf.float32, shape=[None])
                prob_output = tf.reduce_sum(self.policy*action_onehot, [1]) # [None]

                self.policy_loss = -tf.reduce_sum(tf.log(prob_output)*self.advantage)
                self.value_loss = 0.5*tf.reduce_sum(tf.square(self.true_value-tf.reshape(self.value, [-1])))
                self.entropy_loss = -tf.reduce_sum(self.policy*tf.log(self.policy))

                # auxiliary loss
                self.true_depth = tf.placeholder(tf.int32, shape=[None, 64])
                
                self.depth_loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=d2, labels=self.true_depth) # [None, 64]
                self.depth_loss = tf.reduce_sum(self.depth_loss_, axis=1)
                self.depth_loss = tf.reduce_mean(self.depth_loss)

                self.depth_loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dp2, labels=self.true_depth)
                self.depth_loss2 = tf.reduce_sum(self.depth_loss2, axis=1)
                self.depth_loss2 = tf.reduce_mean(self.depth_loss2)
                

                self.loss = self.policy_loss + 0.5*self.value_loss - config.ENTROPY_REG*self.entropy_loss + config.BETA_DEPTH * self.depth_loss + config.BETA_DEPTH * self.depth_loss2
                #self.loss = self.depth_loss2

                self.gradient = tf.gradients(self.loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name))

                grad, self.gradient_norm = tf.clip_by_global_norm(self.gradient, config.MAX_GRADIENT_NORM)
                self.apply_gradient = trainer.apply_gradients(zip(grad, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')))