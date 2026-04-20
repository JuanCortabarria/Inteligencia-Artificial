LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
    
class Model():

    def __init__(self, map):
        self.map = map
        self.graph = {}
        self._set_graph()
    
    def _set_graph(self):
        nrows, ncols = self.map.shape

        for row in range(nrows):
            for col in range(ncols):
                state = row * ncols + col
                cell_type = self.map[row, col].decode('utf-8')
                
                self.graph[state] = {}
                
                if cell_type == 'S' or cell_type == 'F':
                
                    # Analyze the cell to the right
                    col_right = col + 1
                    if col_right >= ncols:
                        self.graph[state][RIGHT] = state
                    else:
                        new_state = row * ncols + col_right
                        self.graph[state][RIGHT] = new_state
                        
                    # Analyze the cell to the left
                    col_left = col - 1
                    if col_left < 0:
                        self.graph[state][LEFT] = state
                    else:
                        new_state = row * ncols + col_left
                        self.graph[state][LEFT] = new_state
                        
                    # Analyze the cell above
                    row_above = row - 1
                    if row_above < 0:
                        self.graph[state][UP] = state
                    else:
                        new_state = row_above * ncols + col
                        self.graph[state][UP] = new_state
                        
                    # Analyze the cell below
                    row_below = row + 1
                    if row_below >= nrows:
                        self.graph[state][DOWN] = state
                    else:
                        new_state = row_below * ncols + col
                        self.graph[state][DOWN] = new_state