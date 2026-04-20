from tic_tac_toe import TicTacToe
from agent import Agent
import random


class AgentMinimax(Agent):
    def __init__(self, player=1):
        super().__init__(player)

    def next_action(self, board: TicTacToe) -> tuple[int, int]:
        self.idx = 0
        pos, _ = self.minimax(board, self.player)
        return pos

    def heuristic_utility(self, board: TicTacToe) -> int:
        # Completar con función de evaluación
        return 0

    def minimax(self, board: TicTacToe, player: int) -> tuple[tuple[int, int], int]:

        # TODO: Completar

        # Caso base

        # Casos no base
        actions = board.get_available_cells()
        random.shuffle(actions)
        action_nodes = []
        for action in actions:
            child_node = board.clone()
            child_node.play(action, player)
            action_nodes.append((action, child_node))

        value = 0
        chosen_action = None

        if player != self.player:  # mini
            # Buscar acción que minimiza el valor
            pass

        else:  # max (player == self.player)
            # Buscar acción que maximiza el valor
            pass

        return chosen_action, value
