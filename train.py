import tensorflow as tf
import os
import threading
import sys
import time

import config
import network
import agent

if __name__ == '__main__':
    # prepare data folder
    model_path = './model'
    frame_path = './frame'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    
    with tf.device('cpu:0'):
        global_episode = tf.Variable(0, trainable=False, dtype=tf.int32)
        trainer = tf.train.RMSPropOptimizer(config.LEARNING_RATE, decay=config.DECAY, momentum=config.MOMENTUM, epsilon=config.EPSILON)
        master_network = network.Network('global', trainer)
        print('master network created')
        sys.stdout.flush()
        agent_arr = []
        for i in range(config.THREAD):
            agent_arr.append(agent.Agent('thread_'+str(i), trainer, global_episode, model_path))
        saver = tf.train.Saver()
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        #ckpt = tf.train.get_checkpoint_state(model_path)
        #saver.restore(sess, ckpt.model_checkpoint_path)
        thread_arr = []
        for a in agent_arr:
            _ = lambda: a.run(sess, trainer, saver, coord)
            t = threading.Thread(target=(_))
            t.start()
            print('thread started')
            sys.stdout.flush()
            time.sleep(1)
            thread_arr.append(t)
        coord.join(thread_arr)
        