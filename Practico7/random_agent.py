import numpy as np
from agent import Agent
from tic_tac_toe import TicTacToe


class RandomAgent(Agent):
    def __init__(self, player: int):
        super().__init__(player)

    def next_action(self, board: TicTacToe) -> tuple[int, int]:
        empty_cells = board.get_available_cells()
        pos = empty_cells[np.random.choice(range(len(empty_cells)))]
        return pos

    def heuristic_utility(self, board: TicTacToe) -> int:
        return 0
