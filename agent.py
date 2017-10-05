import tensorflow as tf
import numpy as np
import sys
import time
import scipy.misc
import scipy.signal

import config
import environment
import network

class Agent():
    def __init__(self, name, trainer, global_episode, model_path):
        self.name = name
        self.trainer = trainer
        self.global_episode = global_episode
        self.summary_writer = tf.summary.FileWriter(name)
        self.network = network.Network(name, trainer) # local network
        from_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        to_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
        self.copy_network = [b.assign(a) for a, b in zip(from_var, to_var)] # op to sync from global network
        self.model_path = model_path
        self.game = environment.Environment()
    
    # static function to save frame during training
    def make_gif(images, fname, duration=2, true_image=False, salience=False, salIMGS=None):
        import moviepy.editor as mpy
        def make_frame(t):
            try:
                x = images[int(len(images)/duration*t)]
            except:
                x = images[-1]
            if true_image:
                return x.astype(np.uint8)
            else:
                return ((x+1)/2*255).astype(np.uint8)
        
        def make_mask(t):
            try:
                x = salIMGS[int(len(salIMGS)/duration*t)]
            except:
                x = salIMGS[-1]
            return x

        clip = mpy.VideoClip(make_frame, duration=duration)
        if salience == True:
            mask = mpy.VideoClip(make_mask, ismask=True,duration= duration)
            clipB = clip.set_mask(mask)
            clipB = clip.set_opacity(0)
            mask = mask.set_opacity(0.1)
            mask.write_gif(fname, fps = len(images) / duration,verbose=False)
        else:
            clip.write_gif(fname, fps = len(images) / duration,verbose=False)
    
    # static function
    def resize_image(image):
        image = image.astype(np.float32) / 255.0
        return image
        #return scipy.misc.imresize(image, [84, 84])

    def discount(x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def train(self, train_buffer, sess, boot_value):
        train_buffer = np.array(train_buffer)
        # unroll from train_buffer
        input_image = np.array(train_buffer[:, 0].tolist())
        aux_action = np.array(train_buffer[:, 1].tolist())
        aux_reward = np.array(train_buffer[:, 2:3].tolist())
        aux_velocity = np.array(train_buffer[:, 3].tolist())
        action = train_buffer[:, 4]
        reward = train_buffer[:, 5]
        value = train_buffer[:, 6]
        depth_pred = train_buffer[:, 7] # <- ?
        true_depth = np.array(train_buffer[:, 8].tolist())

        reward_plus = np.asarray(reward.tolist() + [boot_value])
        disc_reward = Agent.discount(reward_plus, config.GAMMA)[:-1]
        value_plus = np.asarray(value.tolist())
        #advantage = Agent.discount(reward + config.GAMMA*value_plus[1:] - value_plus[:-1], config.GAMMA)
        advantage = disc_reward - value_plus
        vl, pl, el, dl, dl2, gradn, _ , d_tmp= sess.run([self.network.value_loss,
            self.network.policy_loss,
            self.network.entropy_loss,
            self.network.depth_loss,
            self.network.depth_loss2,
            self.network.gradient_norm,
            self.network.apply_gradient, self.network.depth_loss_], feed_dict={
                self.network.input_image: input_image,
                self.network.input_action: aux_action,
                self.network.input_reward: aux_reward,
                self.network.input_velocity: aux_velocity,
                self.network.true_value: disc_reward,
                self.network.advantage: advantage,
                self.network.action: action,
                self.network.true_depth: true_depth,
                self.network.lstm1_state_c_in: self.train_lstm1_state_c,
                self.network.lstm1_state_h_in: self.train_lstm1_state_h,
                self.network.lstm2_state_c_in: self.train_lstm2_state_c,
                self.network.lstm2_state_h_in: self.train_lstm2_state_h
            })
        sys.stdout.flush()
        return vl, pl, el, dl, dl2, gradn, _

    
    def run(self, sess, trainer, saver, coordinator):
        print('starting agent:', self.name)
        sys.stdout.flush()
        with sess.as_default(), sess.graph.as_default():
            while not coordinator.should_stop():
                sess.run(self.global_episode.assign_add(1))
                print('episode:', sess.run(self.global_episode))
                sys.stdout.flush()
                
                ep = sess.run(self.global_episode)
                ep_reward = 0
                ep_step = 0
                ep_start_time = time.time()

                sess.run(self.copy_network)
                train_buffer = []
                frame_buffer = []
                running = True
                self.game.reset()
                rgb, prev_d = self.game.frame()
                frame_buffer.append(rgb)
                rgb = Agent.resize_image(rgb)
                prev_act_idx = 0
                prev_reward = 0
                prev_vel = np.array([0.0]*6)

                self.lstm1_state_c, self.lstm1_state_h, self.lstm2_state_c, self.lstm2_state_h = self.network.lstm1_init_state_c,self.network.lstm1_init_state_h,self.network.lstm2_init_state_c,self.network.lstm2_init_state_h
                while self.game.running():
                    if len(train_buffer)==0:
                        self.train_lstm1_state_h = self.lstm1_state_h
                        self.train_lstm1_state_c = self.lstm1_state_c
                        self.train_lstm2_state_h = self.lstm2_state_h
                        self.train_lstm2_state_c = self.lstm2_state_c
                    act_prob, pred_value, depth_pred, self.lstm1_state_c, self.lstm1_state_h, self.lstm2_state_c, self.lstm2_state_h = sess.run([self.network.policy,
                        self.network.value, self.network.depth_pred,
                        self.network.lstm1_state_c_out, 
                        self.network.lstm1_state_h_out, 
                        self.network.lstm2_state_c_out, 
                        self.network.lstm2_state_h_out]
                        , 
                        feed_dict={self.network.input_image: [rgb], 
                        self.network.input_action: [prev_act_idx], 
                        self.network.input_reward: [[prev_reward]], 
                        self.network.input_velocity: [prev_vel],
                        self.network.lstm1_state_c_in:self.lstm1_state_c,
                        self.network.lstm1_state_h_in:self.lstm1_state_h,
                        self.network.lstm2_state_c_in:self.lstm2_state_c,
                        self.network.lstm2_state_h_in:self.lstm2_state_h
                    })
                    action = np.random.choice(act_prob[0], p=act_prob[0])
                    action_idx = np.argmax(act_prob==action)

                    rgb_next, d, vel, reward, running = self.game.step(action_idx)
                    train_buffer.append([rgb, prev_act_idx, prev_reward, prev_vel, action_idx, reward, pred_value[0][0], depth_pred, prev_d])

                    ep_reward += reward
                    ep_step += 1

                    if running:
                        if ep%config.SAVE_PERIOD==0:
                            frame_buffer.append(rgb_next)
                        rgb_next = Agent.resize_image(rgb_next)
                        rgb = rgb_next
                    
                    prev_act_idx = action_idx
                    prev_reward = reward
                    prev_vel = vel
                    prev_d = d

                    if len(train_buffer)==config.GRADIENT_CHUNK and running:
                        boot_value = sess.run(self.network.value, feed_dict={
                            self.network.input_image: [rgb], 
                            self.network.input_action: [prev_act_idx], 
                            self.network.input_reward: [[prev_reward]], 
                            self.network.input_velocity: [prev_vel],
                            self.network.lstm1_state_c_in:self.lstm1_state_c,
                            self.network.lstm1_state_h_in:self.lstm1_state_h,
                            self.network.lstm2_state_c_in:self.lstm2_state_c,
                            self.network.lstm2_state_h_in:self.lstm2_state_h
                        })
                        vl, pl, el, dl, dl2, gradn, _ = self.train(train_buffer, sess, boot_value)
                        train_buffer = []
                        sess.run(self.copy_network)
                    if not running:
                        break
                if len(train_buffer)>0:
                    vl, pl, el, dl, dl2, gradn, _ = self.train(train_buffer, sess, 0.0)

                ep_finish_time = time.time()
                print(self.name, 'elapse', str(int(ep_finish_time-ep_start_time)), 'seconds, reward:',ep_reward)
                sys.stdout.flush()

                
                if ep%config.SAVE_PERIOD==0:
                    imgs = np.array(frame_buffer)
                    Agent.make_gif(imgs, './frame/image'+str(ep)+'_'+str(ep_reward)+'.gif', duration=len(imgs)*0.066, true_image=True, salience=False)
                    print('frame saved')
                    sys.stdout.flush()
                

                if ep%config.SAVE_PERIOD==0:
                    saver.save(sess, self.model_path+'/model'+str(ep)+'.cptk')
                    print('model saved')
                    sys.stdout.flush()

                    summary = tf.Summary()
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(pl))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(vl))
                    summary.value.add(tag='Losses/Entropy Loss', simple_value=float(el))
                    summary.value.add(tag='Losses/Depth Loss', simple_value=float(dl))
                    summary.value.add(tag='Losses/Depth Loss2', simple_value=float(dl2))
                    summary.value.add(tag='Losses/Gradient Norm', simple_value=float(gradn))
                    summary.value.add(tag='Performance/Reward', simple_value=float(ep_reward))
                    self.summary_writer.add_summary(summary, ep)
                    self.summary_writer.flush()