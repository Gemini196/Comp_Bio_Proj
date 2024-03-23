#!/usr/bin/env python3

import time
import os
import random
import sys
import math

################ Global variables ################
'''
Each cell has an id which is a tuple comprised of the id's of its parent-cells.
example: for parents with ids: (2,) and (3,), the id of the child-cell will be (2,3).

Each id tuple has a unique symbol to represent it in the printed matrix.
The symbol is a number between 2 and N-1 (N being the highest unused symbol).

population_ids_dict : a dictionary that matches id tuples to symbols. {id_tuple:{symbol,count}}
highest unused symbol : The highest number that hasn't been matched with an id tuple yet.
'''
highest_unused_symbol = 2
population_ids_dict = {}

# Written by: Noa Gaon
class Cell:
    """
    A class representing a single living cell in the grid.
    """
    fitness_score = 1     # int
    population_ids = None #tuple

    def __init__(self, population_ids1, population_ids2=None, fitness_score1=None, fitness_score2=None):
        global highest_unused_symbol, population_ids_dict
        self.population_ids = tuple(population_ids1) # Case 1: cell is an initial cell (no parents)
        if population_ids2:                   # Case 2: cell created by two parent cells
            self.population_ids = get_new_population_ids(population_ids1, population_ids2)
            self.fitness_score = math.floor((fitness_score1+fitness_score2)/2)

        if self.population_ids not in population_ids_dict: # New population (no matching symbol)
            population_ids_dict[self.population_ids] = {
                'symbol': highest_unused_symbol,
                'count': 1
            }
            highest_unused_symbol += 1
        else:
            population_ids_dict[self.population_ids]['count']+=1

        self.symbol = population_ids_dict[self.population_ids]['symbol']

    def get_symbol(self):
        global population_ids_dict
        return population_ids_dict[self.population_ids]['symbol']

    def get_population_ids(self):
        return self.population_ids

    def get_fitness_score(self):
        return self.fitness_score

    def set_fitness_score(self, new_fitness_score):
        self.fitness_score = new_fitness_score




# Written by: Noa Gaon
def get_new_population_ids(population_ids1, population_ids2):
    """
    Args:
        population_ids1: ID tuple of the first parent cell.
        population_ids2: ID tuple of the second parent cell.

    Returns: The ID tuple for the new cell.
    """
    combined_population_ids = population_ids1 + population_ids2
    unique_set = set(combined_population_ids)  # Remove duplicates
    return tuple(sorted(unique_set))


# Written by: Noa Gaon
def dfs(grid, i, j, symbol):
    """
    Iterates over a continuous block of "living cells" (initially marked by 1).
    replaces '1's with the symbol.
    Args:
        grid: A matrix containing 0 (dead cell) and 1/symbol otherwise.
        i: the row of the cell to start dfs from
        j: the col of the cell to start dfs from
        symbol: The symbol to mark the living cells in the blocks with.
    """
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != 1:
        return
    grid[i][j] = Cell(tuple([symbol]))
    dfs(grid, i+1, j, symbol)
    dfs(grid, i-1, j, symbol)
    dfs(grid, i, j+1, symbol)
    dfs(grid, i, j-1, symbol)


# Written by: Noa Gaon
def divide_shapes(grid):
    """
    Args:
        grid: an initial grid (containing only 0s and 1s)

    Returns:  a grid such that each continuous block of "living" cells is marked by a different symbol.
    """
    global highest_unused_symbol, population_ids_dict
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1: # living cell
                dfs(grid, i, j, highest_unused_symbol)
    return grid


# ORIGINAL CODE
def clear_console():
    """
    Clears the console using a system command based on the user's operating system.

    """

    if sys.platform.startswith('win'):
        os.system("cls")
    elif sys.platform.startswith('linux'):
        os.system("clear")
    elif sys.platform.startswith('darwin'):
        os.system("clear")
    else:
        print("Unable to clear terminal. Your operating system is not supported.\n\r")


# ORIGINAL CODE
def resize_console(rows, cols):
    """
    Re-sizes the console to the size of rows x columns

    :param rows: Int - The number of rows for the console to re-size to
    :param cols: Int - The number of columns for the console to re-size to
    """

    if cols < 32:
        cols = 32

    if sys.platform.startswith('win'):
        command = "mode con: cols={0} lines={1}".format(cols + cols, rows + 5)
        os.system(command)
    elif sys.platform.startswith('linux'):
        command = "\x1b[8;{rows};{cols}t".format(rows=rows + 3, cols=cols + cols)
        sys.stdout.write(command)
    elif sys.platform.startswith('darwin'):
        command = "\x1b[8;{rows};{cols}t".format(rows=rows + 3, cols=cols + cols)
        sys.stdout.write(command)
    else:
        print("Unable to resize terminal. Your operating system is not supported.\n\r")


# ORIGINAL CODE
def create_initial_grid(rows, cols):
    """
    Creates a random list of lists that contains 1s and 0s to represent the cells in Conway's Game of Life.

    :param rows: Int - The number of rows that the Game of Life grid will have
    :param cols: Int - The number of columns that the Game of Life grid will have
    :return: Int[][] - A list of lists containing 1s for live cells and 0s for dead cells
    """

    grid = []
    for row in range(rows):
        grid_rows = []
        for col in range(cols):
            # Generate a random number and based on that decide whether to add a live or dead cell to the grid
            if random.randint(0, 7) == 0:
                grid_rows += [1]
            else:
                grid_rows += [0]
        grid += [grid_rows]
    return grid


# Written by: Noa Gaon
def get_number_of_extinct_populations():
    """
    Returns: The amount of extinct populations (= population ids tuple of which counter equals 0)
    """
    global population_ids_dict
    zero_count = sum(1 for value in population_ids_dict.values() if value['count'] == 0)
    return zero_count


# Written by: Noa Gaon
def print_statistics(grid):
    """
    Args:
        grid: A matrix with each cell containing either 0 (dead cell) or a Cell object

    Prints:
    - The number of living cells on the board.
    - The average Fitness score in the board (rounded 3 places after decimal dot).
    - print the number of extinct populations (each is defined by a population ids tuple)
    - The average number of ancestors a cell has.
    """
    rows = len(grid)
    cols = len(grid[0])
    living_cells_counter = 0
    fitness_score_sum = 0
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] != 0:
                living_cells_counter += 1
                fitness_score_sum += (grid[row][col]).get_fitness_score()

    print("Number of living cells: {0}".format(living_cells_counter))
    if living_cells_counter != 0:
        print("Average fitness score:{0}".format(round(fitness_score_sum/living_cells_counter), 3))
    print("The number of extinct populations: {0}".format(get_number_of_extinct_populations()))


# Modified by: Noa Gaon
def print_grid(grid, generation):
    """
    Prints to console the Game of Life grid
    :param grid: Int[][] - The list of lists that will be used to represent the Game of Life grid
    :param generation: Int - The current generation of the Game of Life grid
    """
    rows = len(grid)
    cols = len(grid[0])
    clear_console()
    output_str = ""  # A single output string is used to help reduce the flickering caused by printing multiple lines
    # Compile the output string together and then print it to console
    output_str += "Generation {0} - To exit the program press <Ctrl-C>\n\r".format(generation)
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 0:
                output_str += ".   "
            else:
                symbol = str((grid[row][col]).get_symbol())
                output_str += symbol+(4-len(symbol))*" "
        output_str += "\n\r"
    print(output_str, end=" ")


# Written by: Noa Gaon
def update_fitness_score(parent1, parent2):
    """
    Args:
        parent1(Cell): A cell chosen to be the parent of a new cell.
        parent2(Cell): A cell chosen to be the parent of a new cell.

    Updates the fitness score of the parents using the following rule:
    If the parents belong to the same population: fitness score grows by 1 for each.
    Otherwise, fitness score grows by 2 for each
    """
    if parent1.get_population_ids() == parent2.get_population_ids():
        addend = 1
    else:
        addend = 2
    parent1.set_fitness_score(parent1.get_fitness_score() + addend)
    parent2.set_fitness_score(parent2.get_fitness_score() + addend)


# Written by: Noa Gaon
def pick_parents(sorted_neighbors):
    """
    Args:
        sorted_neighbors(List): A list of the neighbors (type: Cell) of a cell to be created.
        Sorted in ascending order by fitness score.
    Returns: The two cells in the list with the highest fitness score.
    """
    parents = None
    if sorted_neighbors[0] == sorted_neighbors[1] and sorted_neighbors[0] == sorted_neighbors[2]:
        parents = random.sample(sorted_neighbors, 2)
    elif sorted_neighbors[0] == sorted_neighbors[1]:
        parents = [random.sample([sorted_neighbors[0], sorted_neighbors[1]], 1), sorted_neighbors[2]]
    elif sorted_neighbors[1] == sorted_neighbors[2]:
        parents = [sorted_neighbors[0], random.sample([sorted_neighbors[1], sorted_neighbors[2]], 1)]
    else:
        parents = sorted_neighbors[:2]
    return parents[0], parents[1]


# Modified by: Noa Gaon
def create_next_grid(grid, next_grid):
    """
    Analyzes the current generation of the Game of Life grid and determines what cells live and die in the next
    generation of the Game of Life grid.

    :param grid: Int[][] - The list of lists that will be used to represent the current generation Game of Life grid
    :param next_grid: Int[][] - The list of lists that will be used to represent the next generation of the Game of Life
    grid
    """
    rows = len(grid)
    cols = len(grid[0])
    for row in range(rows):
        for col in range(cols):
            # Get the number of live cells adjacent to the cell at grid[row][col]
            live_neighbors_count, live_neighbors_list = get_live_neighbors(row, col, grid)

            # If the number of surrounding live cells is < 2 or > 3 --> cell at grid[row][col] is a dead cell
            if live_neighbors_count < 2 or live_neighbors_count > 3:
                if grid[row][col] != 0: # if was alive
                    curr_cell_ids = (grid[row][col]).get_population_ids()
                    population_ids_dict[curr_cell_ids]['count'] -= 1 # update the global dict
                next_grid[row][col] = 0

            # If the number of surrounding live cells is 3 and the cell at grid[row][col] was previously dead -->
            # turn the cell into a live cell
            elif live_neighbors_count == 3 and grid[row][col] == 0:
                neighbors = []
                for indexes in live_neighbors_list:
                    neighbors.append(grid[indexes[0]][indexes[1]])
                sorted_neighbors = sorted(neighbors, key=lambda x: x.get_fitness_score(), reverse=True)

                parent1, parent2 = pick_parents(sorted_neighbors)

                update_fitness_score(parent1, parent2)
                next_grid[row][col] = Cell(parent1.get_population_ids(), parent2.get_population_ids(),
                                           parent1.get_fitness_score(), parent2.get_fitness_score())

            # If the number of surrounding live cells is 3 and the cell at grid[row][col] is alive --> keep it alive
            else:
                next_grid[row][col] = grid[row][col]


# Modified by: Noa Gaon
def get_live_neighbors(row, col, grid):
    """
    Counts the number of live cells surrounding a center cell at grid[row][cell].

    :param row: Int - The row of the center cell
    :param col: Int - The column of the center cell
    :param grid: Int[][] - The list of lists that will be used to represent the Game of Life grid
    :return: Int - The number of live cells surrounding the cell at grid[row][cell]
             List - The indexes of the live cells surrounding the cell at grid[row][cell]
    """
    rows = len(grid)
    cols = len(grid[0])
    life_sum = 0
    life_list = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            # Make sure to count the center cell located at grid[row][col]
            if not (i == 0 and j == 0):
                # Using the modulo operator (%) the grid wraps around
                neighbor_row = (row + i) % rows
                neighbor_col = (col + j) % cols
                if grid[neighbor_row][neighbor_col] != 0:
                    life_list.append([neighbor_row, neighbor_col])
                    life_sum += 1
    return life_sum, life_list


# ORIGINAL CODE
def grid_changing(grid, next_grid):
    """
    Checks to see if the current generation Game of Life grid is the same as the next generation Game of Life grid.
    :param grid: Int[][] - The list of lists that will be used to represent the current generation Game of Life grid
    :param next_grid: Int[][] - The list of lists that will be used to represent the next generation of the Game of Life
    grid
    :return: Boolean - Whether the current generation grid is the same as the next generation grid
    """
    rows = len(grid)
    cols = len(grid[0])
    for row in range(rows):
        for col in range(cols):
            # If the cell at grid[row][col] is not equal to next_grid[row][col]
            if not grid[row][col] == next_grid[row][col]:
                return True
    return False

# ORIGINAL CODE
def get_integer_value(prompt, low, high):
    """
    Asks the user for integer input and between given bounds low and high.

    :param prompt: String - The string to prompt the user for input with
    :param low: Int - The low bound that the user must stay within
    :param high: Int - The high bound that the user must stay within
    :return: The valid input value that the user entered
    """
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("Input was not a valid integer value.")
            continue
        if value < low or value > high:
            print("Input was not inside the bounds (value <= {0} or value >= {1}).".format(low, high))
        else:
            break
    return value


# Modified by: Noa Gaon
def run_game():
    """
    Asks the user for input to setup the Game of Life to run for a given number of generations.

    """

    clear_console()

    # Get the number of rows and columns for the Game of Life grid
    rows = get_integer_value("Enter the number of rows (10-60): ", 10, 60)
    clear_console()
    cols = get_integer_value("Enter the number of cols (10-118): ", 10, 118)
    clear_console()

    # Get the number of generations that the Game of Life should run for
    generations = 5000
    resize_console(rows, cols)

    # Create the initial random Game of Life grids
    current_generation = create_initial_grid(rows, cols)
    current_generation = divide_shapes(current_generation) # divide the matrix into populations
    next_generation = create_initial_grid(rows, cols)

    # Run Game of Life sequence
    gen = 1
    for gen in range(1, generations + 1):
        if not grid_changing(current_generation, next_generation):
            break
        print_grid(current_generation, gen) # print current grid
        print_statistics(current_generation)# print current grid statistics
        create_next_grid(current_generation, next_generation)
        time.sleep(1 / 5.0)
        current_generation, next_generation = next_generation, current_generation

    print_grid(current_generation, gen)
    return input("<Enter> to exit or r to run again: ")


# Start the Game of Life
run = "r"
while run == "r":
    out = run_game()
    run = out

