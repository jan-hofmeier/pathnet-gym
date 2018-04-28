import gym
import gym.utils
from gym import wrappers
import gym_doom
from gym_doom.wrappers import *
import time


wrapper = ToDiscrete('constant-7')
#env.close()
#env.reset()

for a in range(8):
    env = wrapper(gym.make('gym_doom/DoomBasic-v0'))
    env.reset()
    print('ACTION:', a)
    for i in range(350):
        _, reward, _, _ = env.step(a)
        print(i, reward)
        env.render()
    print("------------------------------")
    env.close()
    '''reset=False
    while(not reset):
         try:
             env.reset()
             reset=True
         except:
              pass
      '''
#env.close()
#env.env.close()
