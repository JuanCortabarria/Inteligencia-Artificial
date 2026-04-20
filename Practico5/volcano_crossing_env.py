import gymnasium
import numpy as np
import random
from mdp_model import MDPModel


class VolcanoCrossing(gymnasium.Env):

    def __init__(self, slip_prob=0.1):
        self.action_space = ["N", "S", "E", "W"]
        self.observation_space = self._get_observation_space()
        self.state = "21"
        self.mdp = MDPModel()
        self.end_states = ("31", "13", "23", "14")
        # Rewards
        self.R = self.mdp.load_rewards()
        # Probabilities
        self.P = self.mdp.load_probabilities(slip_prob=slip_prob)

    def reset(self):
        self.state = "21"
        return self.state

    def step(self, action):
        if self.state in self.end_states:
            raise Exception("Game is over")

        _state = self.state

        self.state = self._choose(action)

        done = self.state in self.end_states

        reward = self.R[_state][action][self.state]

        return self.state, reward, done, {}

    def render(self, mode="human", close=False):
        if not (close):
            print("state:", self.state, "done:", self.state in self.end_states)

    def _choose(self, action):
        _states = list(self.P[self.state][action].keys())
        p = list(map(lambda n_state: self.P[self.state][action][n_state], _states))
        return np.random.choice(_states, size=None, p=p)

    def check_action(self, action):
        if action not in self.action_space:
            raise Exception("Invalid action")

    def _get_observation_space(self):
        states = []
        for i in range(1, 4):
            for j in range(1, 5):
                states.append(str(i) + str(j))
        return states
