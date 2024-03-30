import random

glider = [[0, 1, 0], [0, 0, 1], [1, 1, 1]]
blinker = [[1, 1, 1]]


def shape_initial_configuration(rows, cols, shape_name):
    """
    Generate a shape initial configuration for Conway's Game of Life.
    Args:
        rows (int): Number of rows in the board.
        cols (int): Number of columns in the board.
        shape_name (string): The name of the shape to be generated
    Returns:
        list: 2D list representing the initial configuration.
    """
    config = [[0 for _ in range(cols)] for _ in range(rows)]
    if shape_name.lower().startswith('b'):
        shape = blinker
    elif shape_name.lower().startswith('g'):
        shape = glider
    else:
        exit(1)
    start_row = random.randint(0, rows - len(shape))
    start_col = random.randint(0, cols - len(shape[0]))
    for i in range(len(shape)):
        for j in range(len(shape[0])):
            config[start_row + i][start_col + j] = shape[i][j]
    return config
