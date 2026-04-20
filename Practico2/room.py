
from gymnasium import spaces, Env
import numpy as np

class Room(Env):
    
    def __init__(self):
        self.observation_space = spaces.Box(low=-20, high=40, shape=(), dtype=np.float32)
        self.action_space = spaces.Discrete(7, start=-3)
        self.np_random = None
    
    def _seed(self, seed=None):
        """Set the seed for the environment's random number generator."""
        self.np_random = np.random.RandomState(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
                  
    def reset(self, seed=None):
        """Reset the environment. Optionally set a new seed."""
        if seed is not None:
            self._seed(seed)
        self.step_count = 0
        self.initial_random = self.observation_space.sample()
        self.external_temp = 0
        self.external_temp = self._temp_variation()
        self.temp = self.external_temp
        return self.temp
    
    def step(self, action):
        delta_temp = self._temp_variation()
        self.external_temp += delta_temp
        self.temp += delta_temp + action
        self.step_count += 1
        return self.temp

    def _temp_variation(self):
        x = self.initial_random + self.step_count
        return np.float64('%.2f'%(20 * np.sin(x/10))) - self.external_temp 
          