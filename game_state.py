# -*- coding: utf-8 -*-
import sys
import time

import numpy as np
import cv2

from constants import GYM_MONITOR_DIR
from constants import ACTION_SPACE_TYPE

import gym
import gym.utils
from gym import wrappers
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import os
import tensorflow as tf
#import gym_doom
#from gym_doom.wrappers import *


class GameState(object):
    def __init__(self, rand_seed, ROM, display=True, task_index=-1):
        self.task_index = task_index
        self.ROM = ROM
#        action_space_wrapper = ToDiscrete(ACTION_SPACE_TYPE)
        self.env = gym.make(self.ROM)
        #self.env = wrap_deepmind(self.env)
        # self.env = action_space_wrapper(self.env)
        self.display = display
        if (self.display):
            self.env = wrappers.Monitor(self.env, GYM_MONITOR_DIR + '-' + self.ROM)
        self.env.seed(rand_seed)
        # self.real_actions = self.env.action_space
        self._screen = np.empty((480, 640, 1), dtype=np.uint8)
        self.reset()


    def preprocess_ob(self, observation):
        self._screen = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        reshaped_screen = np.reshape(self._screen, (observation.shape[0], observation.shape[1]))

        resized_screen = cv2.resize(reshaped_screen, (120, 160))
        x_t = resized_screen  # [:,10:]
        x_t = np.reshape(x_t, (160, 120, 1))
        # cv2.imwrite("data/image/x_t" + str(time.time()) + ".png", x_t)
        x_t = x_t.astype(np.float32)
        x_t *= (1.0/255.0)
        return x_t


    def _process_frame(self, action, reshape):
        if action>=self.env.action_space.n:
            action=0
        observation, reward, terminal, info = self.env.step(action)
        x_t = self.preprocess_ob(observation)
        return reward, terminal, x_t


    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        self.env.render()
        x_t = self.preprocess_ob(observation)
        return x_t, reward, terminal, info

    def reset(self):
        ob = self.env.reset()
        time.sleep(3)
        # randomize initial state
        '''if self._no_op_max > 0:
            no_op = np.random.randint(0, self._no_op_max + 1)
            for _ in range(no_op):
                self.env.step(0)'''
        x_t = self.preprocess_ob(ob)

        self.reward = 0
        self.terminal = False
        #self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
        return x_t

    def process(self, action):
        #print('process')
        # if (action > self._no_op_max - 1):
        #     # print("Action '{}' is out of bounds. Remapped action '{}' -> '{}'".format(action, action, self._no_op_max-1))
        #     action = self._no_op_max - 1
        #if self.display:
        self.env.render()
        r, t, x_t1 = self._process_frame(action, True)
        self.reward = r
        self.terminal = t
        self.s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)

    def update(self):
        self.s_t = self.s_t1

    def close_env(self):
        self.env.close()

    def get_ac_space(self):
        return self.env.action_space