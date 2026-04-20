from agent import Agent


class InputAgent(Agent):
    def __init__(self):
        pass

    def next_action(self, obs):
        input_status = False
        while not input_status:
            direction = input("Direction: ")
            person1 = input("Person 1: ")
            person2 = input("Person 2 (can be the same as Person 1): ")

            ret, input_status = self.parse_action(direction, person1, person2)

        return ret

    def parse_action(self, direction, person1, person2):
        charToPerson = {"A": 0, "B": 1, "C": 2, "D": 3}

        charToDirection = {
            "R": 1,
            "L": 0,
        }

        input_status = True
        d = 0
        p1 = 0
        p2 = 0

        try:
            p1 = charToPerson[person1]
            p2 = charToPerson[person2]
        except Exception:
            print("Unknown person code")
            input_status = False

        try:
            d = charToDirection[direction]
        except Exception:
            print("Unknown direction code")
            input_status = False

        ret = {"direction": d, "person1": p1, "person2": p2}
        return ret, input_status
