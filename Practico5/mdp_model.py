class MDPModel:
    def __init__(self):
        self.maze = self.load_maze()
        self.actions = ["N", "S", "E", "W"]

    def load_maze(self):
        path = "./maze.txt"
        maze = {}
        origin_coord = 0
        dir_coord = 1
        dest_coord = 2

        with open(str(path), "r") as f:
            for line in f:
                coord = line.split(" ")
                origin = coord[origin_coord]
                dir = coord[dir_coord]
                destination = coord[dest_coord].strip()
                if origin not in maze:
                    maze[origin] = {}

                maze[origin][dir] = destination

        return maze

    def load_rewards(self):
        R = {}
        for origin in self.maze:
            possible_actions = self.maze[origin]
            R[origin] = {}
            possible_dests = self.get_possible_destinations(possible_actions)

            for action in possible_actions:
                for dest in possible_dests:
                    if action not in R[origin]:
                        R[origin][action] = {}

                    if dest == "13" or dest == "23":
                        R[origin][action][dest] = -50
                    elif dest == "31":
                        R[origin][action][dest] = 2
                    elif dest == "14":
                        R[origin][action][dest] = 20
                    else:
                        R[origin][action][dest] = 0

        # Load rewards for the end states
        for state in ["13", "23", "31", "14"]:
            R[state] = {}
            for action in self.actions:
                R[state][action] = {}
                R[state][action][state] = 0

        return R

    def load_probabilities(self, slip_prob):
        P = {}
        for origin in self.maze:
            P[origin] = {}
            actions = ["N", "S", "E", "W"]

            for action in actions:
                dest_coord = self.maze[origin][action]
                P[origin][action] = {}
                P[origin][action][dest_coord] = 1 - slip_prob

                for dir in actions:
                    if dir != action:
                        dest = self.maze[origin][dir]

                        if dest in P[origin][action]:
                            P[origin][action][dest] = P[origin][action][dest] + (
                                slip_prob / 3
                            )
                        else:
                            P[origin][action][dest] = slip_prob / 3

        # Load probabilities for the end states
        for state in ["13", "23", "31", "14"]:
            P[state] = {}
            for action in self.actions:
                P[state][action] = {}
                P[state][action][state] = 0

        return P

    def get_possible_destinations(self, possible_actions):
        dests = []

        for action in possible_actions:
            dest = possible_actions[action]
            dests.append(dest)

        return dests
