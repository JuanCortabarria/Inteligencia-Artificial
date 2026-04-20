import numpy as np
from priority_queue import PriorityQueue


class AStarAgent:
    def __init__(self, model):
        self.model = model
        self.action_list = []
        self.nodes_expanded = 0

    def loop(self, env):
        start_state_flatten = env.reset()
        done = False
        step_counter = 0
        all_rewards = 0
        env.render()

        while not done:
            action = self.next_action(start_state_flatten.reshape(3, 3))
            obs, reward, done_env, _ = env.step(action)
            all_rewards += reward
            done = done_env
            env.render()
            step_counter += 1

        return all_rewards, step_counter

    def next_action(self, state):
        # Si todavía no planificamos, corremos A* una vez y guardamos el plan completo
        if not self.action_list:
            self.action_list = self.a_star(state)
        return self.action_list.pop(0)

    def heuristic(self, state):
        # h0: heurística nula → A* se comporta como Dijkstra
        return 0

    def is_goal(self, state):
        return np.array_equal(state.flatten(), np.array([1, 2, 3, 4, 5, 6, 7, 8, 0]))

    def a_star(self, start_state):
        self.nodes_expanded = 0

        start_key = str(start_state)
        frontier = PriorityQueue()
        frontier.push(start_key, self.heuristic(start_state))

        # Necesitamos recuperar el ndarray a partir de la clave string
        states = {start_key: start_state}
        came_from = {}                 # key_hijo -> (key_padre, accion)
        g_score = {start_key: 0}

        while not frontier.is_empty():
            current_key, _ = frontier.pop()
            current_state = states[current_key]
            self.nodes_expanded += 1

            if self.is_goal(current_state):
                plan = self._reconstruct_plan(came_from, current_key)
                print(f"[A*] heurística={self.heuristic.__name__ if hasattr(self.heuristic,'__name__') else 'h'} "
                      f"nodos_expandidos={self.nodes_expanded} largo_plan={len(plan)}")
                return plan

            for action in range(4):
                neighbor_state = self.model.get_next_state(current_state, action)

                # Si el movimiento es ilegal el modelo devuelve el mismo estado → lo descartamos
                if np.array_equal(neighbor_state, current_state):
                    continue

                neighbor_key = str(neighbor_state)
                tentative_g = g_score[current_key] + 1

                if neighbor_key not in g_score or tentative_g < g_score[neighbor_key]:
                    states[neighbor_key] = neighbor_state
                    came_from[neighbor_key] = (current_key, action)
                    g_score[neighbor_key] = tentative_g
                    f = tentative_g + self.heuristic(neighbor_state)

                    if frontier.contains(neighbor_key):
                        frontier.update(neighbor_key, f)
                    else:
                        frontier.push(neighbor_key, f)

        return []  # sin solución (no debería pasar si el tablero es resoluble)

    def _reconstruct_plan(self, came_from, goal_key):
        actions = []
        key = goal_key
        while key in came_from:
            parent_key, action = came_from[key]
            actions.append(action)
            key = parent_key
        actions.reverse()
        return actions
