from tic_tac_toe import TicTacToe
from agent import Agent


class AgentExpectimax(Agent):
    def __init__(self, player=1, max_depth: int = 10):
        super().__init__(player)
        self.max_depth = max_depth

    def next_action(self, board: TicTacToe) -> tuple[int, int]:
        self.idx = 0
        pos, _ = self.expectimax(board, self.player, self.max_depth)
        return pos

    def heuristic_utility(self, board: TicTacToe) -> int:
        # Completar con función de evaluación
        return 0

    def expectimax(
        self, board: TicTacToe, player: int, depth: int
    ) -> tuple[tuple[int, int], int]:


        # TODO: Completar

        # Caso base

        # Casos no base
        actions = board.get_available_cells()
        action_nodes = []
        for action in actions:
            child_node = board.clone()
            child_node.play(action, player)
            action_nodes.append((action, child_node))

        value = 0
        chosen_action = None
        if player != self.player:  # Expecti
            # Calcular valor promedio de las acciones del oponente
            pass
        else:  # max
            # Buscar acción que maximiza el valor
            pass

        return chosen_action, value
