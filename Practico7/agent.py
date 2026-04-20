from abc import ABCMeta
from abc import abstractmethod
from tic_tac_toe import TicTacToe


class Agent(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, player):
        self.player = player
        self.other_player = (player % 2) + 1

    # retorna (x,y), valor estimado si existe
    @abstractmethod
    def next_action(self, board: TicTacToe) -> tuple[int, int]:
        pass

    @abstractmethod
    def heuristic_utility(self, board: TicTacToe) -> int:
        pass
