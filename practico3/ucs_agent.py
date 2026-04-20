from search_agent import SearchAgent
from priority_queue import PriorityQueue


class UCSAgent(SearchAgent):

    def __init__(self, env, initial_state, end_state, model):
        super().__init__(env, initial_state, end_state, model)
        self.action_list = []

    def _next_action(self):
        if not self.action_list:
            self.ucs()
        return self.action_list.pop(0)

    def ucs(self):
        pq = PriorityQueue()
        pq.push(self.initial_state, 0, None)
        visited = set()
        came_from = {self.initial_state: None}
        cost_so_far = {self.initial_state: 0}

        while not pq.is_empty():
            state, cost, prev = pq.pop()

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

            for action, next_state in self.model.graph[state].items():
                if next_state == state:
                    continue
                if not self.model.graph[next_state] and next_state != self.end_state:
                    continue
                new_cost = cost + 1
                if next_state not in visited and (next_state not in cost_so_far or new_cost < cost_so_far[next_state]):
                    cost_so_far[next_state] = new_cost
                    came_from[next_state] = (state, action)
                    pq.push(next_state, new_cost, state)
