from abc import ABC


class SearchAgent:

    def __init__(self, env, initial_state, end_state, model):
        self.env = env
        self.initial_state = initial_state
        self.end_state = end_state
        self.model = model

    def run(self):
        return self._loop()

    def _loop(self):
        obs = self.env.reset()
        done = False
        step_counter = 0
        all_rewards = 0
        self.env.render()

        while not done:
            action = self._next_action()
            self._check_action(action)
            obs, reward, done, _, _ = self.env.step(action)
            step_counter += 1
            self.env.render()

        return reward, step_counter

    def _next_action(self):
        return int(input())

    def _check_action(self, action):
        if not (self.env.action_space.contains(action)):
            raise Exception("Action not in action space")
