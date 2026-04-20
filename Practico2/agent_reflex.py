from agent import Agent

class AgentReflex(Agent):
    """Agent reflex simple para el problema del aire acondicionado."""

    def __init__(self, env):
        self.env = env

    def next_action(self, obs):
        if obs > 9:
            return -3
        elif obs > 5:
            return -2
        elif obs > 2:
            return -1
        elif obs < -9:
            return 3
        elif obs < -5:
            return 2
        elif obs < -2:
            return 1
        else:
            return 0