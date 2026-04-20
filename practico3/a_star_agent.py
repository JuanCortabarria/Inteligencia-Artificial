import math
from search_agent import SearchAgent
from priority_queue import PriorityQueue


class AStarAgent(SearchAgent):

    def __init__(self, env, initial_state, end_state, model, heuristic_type="manhattan"):
        super().__init__(env, initial_state, end_state, model)
        self.action_list = []
        self.heuristic_type = heuristic_type
        self.heuristics = {}
        self._calculate_heuristics()

    def _calculate_heuristics(self):
        ncols = self.model.map.shape[1]
        goal_row = self.end_state // ncols
        goal_col = self.end_state % ncols

        for state in self.model.graph:
            row = state // ncols
            col = state % ncols
            if self.heuristic_type == "manhattan":
                self.heuristics[state] = abs(row - goal_row) + abs(col - goal_col)
            elif self.heuristic_type == "euclidean":
                self.heuristics[state] = math.sqrt((row - goal_row) ** 2 + (col - goal_col) ** 2)

    def _next_action(self):
        if not self.action_list:
            self.a_star()
        return self.action_list.pop(0)

    def a_star(self):
        pq = PriorityQueue()
        pq.push(self.initial_state, self.heuristics[self.initial_state], None)
        visited = set()
        came_from = {self.initial_state: None}
        cost_so_far = {self.initial_state: 0}

        while not pq.is_empty():
            state, f_cost, prev = pq.pop()

            if state in visited:
                continue
            visited.add(state)

            if state == self.end_state:
                actions = []
                current = state
                while came_from[current] is not None:
                    prev_state, action = came_from[current]
                    actions.append(action)
                    current = prev_state
                actions.reverse()
                self.action_list = actions
                return

            g_cost = cost_so_far[state]
            for action, next_state in self.model.graph[state].items():
                if next_state == state:
                    continue
                if not self.model.graph[next_state] and next_state != self.end_state:
                    continue
                new_g = g_cost + 1
                if next_state not in visited and (next_state not in cost_so_far or new_g < cost_so_far[next_state]):
                    cost_so_far[next_state] = new_g
                    came_from[next_state] = (state, action)
                    f = new_g + self.heuristics[next_state]
                    pq.push(next_state, f, state)
