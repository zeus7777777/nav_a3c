from multiprocessing import Process, Pipe
import sys

import numpy as np
import scipy.misc

import deepmind_lab_py3 as deepmind_lab

import config

def game_process(conn):
    env = deepmind_lab.Lab('nav_maze_static_01', ['RGBD_INTERLACED', 'VEL.TRANS', 'VEL.ROT'], config={'width':'84', 'height':'84'})
    conn.send(0)
    prev_action_vector = np.array([0]*config.N_ACTION, dtype=np.intc)
    action_vector = np.array([
        [0,0,0,1,0,0,0], 
        [0,0,0,-1,0,0,0], 
        [0,0,1,0,0,0,0], 
        [0,0,-1,0,0,0,0],
        [-20,0,0,0,0,0,0],
        [20,0,0,0,0,0,0]
    ], dtype=np.intc)
    while True:
        cmd, arg = conn.recv()
        if cmd=='reset':
            env.reset()
            prev_action_vector = np.array([0]*7, dtype=np.intc)
            conn.send(env.observations())
        elif cmd=='action':
            reward = 0.0
            if arg>=0 and arg<6:
                prev_action_vector = np.copy(action_vector[arg])
                reward = env.step(prev_action_vector, num_steps=4)
            elif arg==6:
                prev_action_vector[0] = 20
                reward = env.step(prev_action_vector, num_steps=4)
            elif arg==7:
                prev_action_vector[0] = -20
                reward = env.step(prev_action_vector, num_steps=4)
            running = env.is_running()
            obs = env.observations() if running else 0
            conn.send([obs, reward, running])
        elif cmd=='observe':
            conn.send([env.observations()['RGBD_INTERLACED']])
        elif cmd=='running':
            running = env.is_running()
            conn.send([running])
        elif cmd=='stop':
            break
    env.close()
    conn.send(0)
    conn.close()

class Environment():
    def __init__(self):
        self.conn, child_conn = Pipe()
        self.proc = Process(target=game_process, args=(child_conn,))
        self.proc.start()
        self.conn.recv()
        self.reset()
    
    def reset(self):
        self.conn.send(['reset', 0])
        obs = self.conn.recv()
    
    def stop(self):
        self.conn.send(['stop', 0])
        _ = self.conn.recv()
        self.conn.close()
        self.proc.join()
    
    def preprocess_frame(rgbd):
        rgb = rgbd[:, :, 0:3]
        d = rgbd[:, :, 3] # 84*84
        d = d[16:-16, :] # crop
        d = d[:, 2:-2] # crop
        d = d[::13, ::5] # subsample
        d = d.flatten()
        d = np.power(d/255.0, 10)
        d = np.digitize(d, [0,0.05,0.175,0.3,0.425,0.55,0.675,0.8,1.01])
        d -= 1
        return rgb, d

    def step(self, action_idx):
        self.conn.send(['action', action_idx])
        obs, reward, running = self.conn.recv()
        if running:
            rgb, d = Environment.preprocess_frame(obs['RGBD_INTERLACED'])
            return rgb, d, np.concatenate((obs['VEL.TRANS'], obs['VEL.ROT'])), reward, running
        else:
            rgb, d = 0, 0
            return rgb, d, 0, reward, running
        return rgb, d, np.concatenate((obs['VEL.TRANS'], obs['VEL.ROT'])), reward, running

    def frame(self):
        self.conn.send(['observe', 0])
        rgbd = self.conn.recv()[0]
        return Environment.preprocess_frame(rgbd)

    def running(self):
        self.conn.send(['running', 0])
        running = self.conn.recv()
        return running
