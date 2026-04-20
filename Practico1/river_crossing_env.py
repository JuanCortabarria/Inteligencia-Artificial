import gymnasium
from gymnasium import spaces


class RiverCrossingEnv(gymnasium.Env):

    def __init__(self):
        self.observation_space = spaces.Dict(
            {
                # 0: left, 1: right
                "Aside": spaces.Discrete(2),
                "Bside": spaces.Discrete(2),
                "Cside": spaces.Discrete(2),
                "Dside": spaces.Discrete(2),
            }
        )

        self.action_space = self.action_space = spaces.Dict(
            {
                "direction": spaces.Discrete(2),  # 0: left, 1: right
                "person1": spaces.Discrete(4),  # 0: A, 1: B, 2: C, 3: D
                "person2": spaces.Discrete(4),  # 0: A, 1: B, 2: C, 3: D
            }
        )

        self.state = {"Aside": 0, "Bside": 0, "Cside": 0, "Dside": 0}

        self.time = 0

        self.person_time = {0: 1, 1: 2, 2: 5, 3: 8}

        self.torch_side = 0

    def reset(self):
        self.state = {"Aside": 0, "Bside": 0, "Cside": 0, "Dside": 0}
        self.time = 0
        return self.state

    def step(self, action):
        if not (self._check_action(action)):
            return self.state, -1, False, {"time": self.time}

        person1 = self._number_to_person(action["person1"])
        person2 = self._number_to_person(action["person2"])

        self.state[person1] = action["direction"]
        self.state[person2] = action["direction"]

        self.time += max(
            self.person_time[action["person1"]], self.person_time[action["person2"]]
        )

        self.torch_side = 1 - self.torch_side

        done, reward = self._is_end()

        return self.state, reward, done, {"time": self.time}

    def render(self):
        elements = ["A", "B", "C", "D"]
        left_side = []
        right_side = []
        boat = []

        for i, element in enumerate(elements):
            position = self.state[f"{element}side"]
            if position == 0:
                left_side.append(element)
            elif position == 1:
                right_side.append(element)
            elif position == 2:
                boat.append(element)

        left_side_str = "".join(left_side) if left_side else "Empty"
        right_side_str = "".join(right_side) if right_side else "Empty"
        boat_str = "".join(boat) if boat else "Empty"

        print(f"{left_side_str} //// {right_side_str}")

    def _check_action(self, action) -> bool:
        if not (self.action_space.contains(action)):
            raise ValueError("Action not in action space")

        person1 = self._number_to_person(action["person1"])
        person2 = self._number_to_person(action["person2"])

        if (
            action["direction"] == self.state[person1]
            or action["direction"] == self.state[person2]
            or self.state[person1] != self.torch_side
        ):
            return False

        return True

    def _number_to_person(self, number):
        if number == 0:
            return "Aside"
        elif number == 1:
            return "Bside"
        elif number == 2:
            return "Cside"
        elif number == 3:
            return "Dside"
        else:
            raise ValueError("Invalid person number")

    def _is_end(self) -> (bool, int):
        if self.time > 15:
            return True, -1
        else:
            persons_on_right = (
                self.state["Aside"] == 1
                and self.state["Bside"] == 1
                and self.state["Cside"] == 1
                and self.state["Dside"] == 1
            )

            if persons_on_right and self.time == 15:
                return True, 1
            elif persons_on_right:
                return True, -1
            else:
                return False, 0
