from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def __init__(self, env):
        pass

    @abstractmethod
    def next_action(self, obs):
        pass