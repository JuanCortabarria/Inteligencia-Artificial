import numpy as np

class Model:
    def __init__(self) -> None:
        self.graph = {}

    def get_next_state(self, state, action):
        if str(state) not in self.graph:
            self.graph[str(state)] = {}

        if action not in self.graph[str(state)]:

            new_state = state.copy()

            # Update puzzle state
            blank = np.where(state == 0)
            b_x = blank[0][0]
            b_y = blank[1][0]

            if action < 0 or action > 3:
                raise ValueError("Invalid action: " + str(action))

            # Move Up
            if action == 0 and b_x > 0:
                new_state[b_x][b_y] = state[b_x - 1][b_y]
                new_state[b_x - 1][b_y] = 0

            # Move Right
            elif action == 1 and b_y < 2:
                new_state[b_x][b_y] = state[b_x][b_y + 1]
                new_state[b_x][b_y + 1] = 0

            # Move Down
            elif action == 2 and b_x < 2:
                new_state[b_x][b_y] = state[b_x + 1][b_y]
                new_state[b_x + 1][b_y] = 0

            # Move Left
            elif action == 3 and b_y > 0:
                new_state[b_x][b_y] = state[b_x][b_y - 1]
                new_state[b_x][b_y - 1] = 0

            self.graph[str(state)][action] = new_state

        return self.graph[str(state)][action]

