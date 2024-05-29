#!/usr/bin/env python3

import time
import os
import random
import sys
import math
import statistics
from initial_states import shape_initial_configuration
import game_config
import matplotlib.pyplot as plt
from clustering import clustering_results

################ Global variables ################
'''
Each cell has an id which is a tuple comprised of the id's of its parent-cells.
example: for parents with ids: (2,) and (3,), the id of the child-cell will be (2,3).

Each id tuple has a unique symbol (1-N) to represent it in the printed grid. (N being the highest unused symbol).

POPULATION_IDS_DICT : a dictionary that matches id tuples to symbols. {id_tuple:{symbol,count}}
HIGHEST_UNUSED_SYMBOL : The highest number that hasn't been matched with an id tuple yet.
ADDEND_INTER_POPULATION : Fitness addend to parent-cells for inter population reproduction.
ADDEND_INTRA_POPULATION : Fitness addend to parent-cells for intra population reproduction.
MAX_FITNESS_SCORE_DIFF : Maximum fitness score diff allowed between two cell for reproduction.
'''
HIGHEST_UNUSED_SYMBOL = 1
POPULATION_IDS_DICT = {}
ADDEND_INTER_POPULATION = 2 # default
ADDEND_INTRA_POPULATION = 1 # default
MAX_FITNESS_SCORE_DIFF = sys.maxsize # max INT
MAX_GENERATIONS = 150
NUM_GAMES = 20

# Written By Noa Gaon
def reset_global_vars():
    global HIGHEST_UNUSED_SYMBOL, POPULATION_IDS_DICT, ADDEND_INTER_POPULATION, ADDEND_INTRA_POPULATION, MAX_FITNESS_SCORE_DIFF
    HIGHEST_UNUSED_SYMBOL = 1
    POPULATION_IDS_DICT = {}
    ADDEND_INTER_POPULATION = 2  # default
    ADDEND_INTRA_POPULATION = 1  # default
    MAX_FITNESS_SCORE_DIFF = sys.maxsize


# Written by: Noa Gaon
class Cell:
    """
    A class representing a single living cell in the grid.
    """
    fitness_score = 1     # int
    population_ids = None #tuple

    def __init__(self, population_ids1, population_ids2=None, fitness_score1=None, fitness_score2=None,
                 custom_symbol=None):
        global HIGHEST_UNUSED_SYMBOL, POPULATION_IDS_DICT
        self.population_ids = tuple(population_ids1) # Case 1: cell is an initial cell (no parents)
        if population_ids2:                   # Case 2: cell created by two parent cells
            self.population_ids = get_new_population_ids(population_ids1, population_ids2)
            self.fitness_score = math.floor((fitness_score1+fitness_score2)/2)

        if self.population_ids not in POPULATION_IDS_DICT: # New population (no matching symbol)
            if custom_symbol:
                symb = custom_symbol # custom new symbol
                if HIGHEST_UNUSED_SYMBOL <= symb:
                    HIGHEST_UNUSED_SYMBOL = symb + 1
            else:
                symb = HIGHEST_UNUSED_SYMBOL # default new symbol
                HIGHEST_UNUSED_SYMBOL += 1

            POPULATION_IDS_DICT[self.population_ids] = {
                'symbol': symb,
                'count': 1
            }

        else:
            POPULATION_IDS_DICT[self.population_ids]['count'] += 1

        self.symbol = POPULATION_IDS_DICT[self.population_ids]['symbol']

    def get_symbol(self):
        global POPULATION_IDS_DICT
        return POPULATION_IDS_DICT[self.population_ids]['symbol']

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

'''
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
    global HIGHEST_UNUSED_SYMBOL
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1: # living cell
                dfs(grid, i, j, HIGHEST_UNUSED_SYMBOL)
    return grid
'''


# Written by: Noa Gaon
def divide_into_populations(grid, pop_num):
    """
    Args:
        grid: an initial grid (containing only 0s and 1s)
        pop_num: the number of distinct initial populations.

    Returns:  a grid such that each cell is marked by a different symbol.
    (each cell gets the symbol X with a probability of 1/#population_num)
    """
    pop_symb_lst = list(range(1, pop_num))
    global HIGHEST_UNUSED_SYMBOL
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1: # living cell
                symbol = random.randint(1, pop_num)
                grid[i][j] = Cell(population_ids1=tuple([symbol]), custom_symbol=symbol)

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
def get_variance_for_init():
    """
    :return: The variance of an initial population_id appearing in different populations
    """
    global POPULATION_IDS_DICT
    # Create a dict to hold the counts of each initial population
    init_pop_counts = {}
    # Count the occurrences of each initial population id across all keys
    for key in POPULATION_IDS_DICT:
        for pop_id in key:
            init_pop_counts[pop_id] = init_pop_counts.get(pop_id, 0) + POPULATION_IDS_DICT[key]['count']
    mean_count = sum(init_pop_counts.values()) / len(init_pop_counts) # Calculate variance based on string counts
    variance = sum((count - mean_count) ** 2 for count in init_pop_counts.values()) / len(init_pop_counts)
    return variance


# Written By: Noa Gaon
def get_avg_ancestry_dist():
    """
    :return: The average distance between the ancestry of populations on board.
            (Each mismatch adds +1 to distance between cells)
    """
    global POPULATION_IDS_DICT
    total_distances = 0
    total_occurrences = 0

    for key, item in POPULATION_IDS_DICT.items():
        strings = list(key)
        occurrences = []
        for string in strings:
            # Find all occurrences of the string in the key
            indices = [i for i, s in enumerate(strings) if s == string]
            occurrences.extend(indices)

        # Calculate distances between occurrences
        distances = [j - i for i, j in zip(occurrences[:-1], occurrences[1:])]

        # Sum distances and occurrences
        total_distances += sum(distances) * item['count']
        total_occurrences += len(occurrences) * item['count']
    return total_distances / total_occurrences if total_occurrences > 0 else 0  # Avoid division by zero


# Written by: Noa Gaon
def calc_statistics(grid):
    global POPULATION_IDS_DICT
    rows = len(grid)
    cols = len(grid[0])

    living_cells_counter = sum(1 for row in grid for cell in row if cell != 0)
    extinct_pop_count = sum(1 for value in POPULATION_IDS_DICT.values() if value['count'] == 0)
    pop_count = len(POPULATION_IDS_DICT)-extinct_pop_count
    fitness_score_sum = sum(cell.get_fitness_score() for row in grid for cell in row if cell != 0)

    # Get the maximum appearance count among all populations
    max_appearances_count = max(POPULATION_IDS_DICT.values(), key=lambda x: x['count'])['count']
    # Find population_ids with the maximum count
    most_common_population = [k for k, v in POPULATION_IDS_DICT.items() if v['count'] == max_appearances_count]\
        if living_cells_counter > 0 else ''
    for i in range(len(most_common_population)):
        most_common_population[i] = str(most_common_population[i])+'--> symbol '+str(POPULATION_IDS_DICT[most_common_population[i]]['symbol'])

    # Create a list of string lengths based on key-value pairs in the dictionary
    id_lengths = [len(key) for key, count in POPULATION_IDS_DICT.items() for _ in
                  range(POPULATION_IDS_DICT[key]['count'])]
    id_len_sum = sum(id_lengths)

    # Calculate averages
    avg_fitness_score = fitness_score_sum / living_cells_counter if living_cells_counter > 0 else 0 # Avoid division by zero
    avg_num_of_ancestors = id_len_sum / living_cells_counter if living_cells_counter > 0 else 0 # Avoid division by zero

    id_lengths.sort()  # Sort the list of string lengths
    med_num_of_ancestors = statistics.median(id_lengths) if living_cells_counter > 0 else 0  # Calculate the median (+Avoid division by zero)

    avg_ancestry_dist = get_avg_ancestry_dist()

    result_dict = {'living_cells_counter': living_cells_counter,
                   'extinct_pop_count': extinct_pop_count,
                   'pop_count': pop_count,
                   'avg_fitness_score': avg_fitness_score,
                   'avg_num_of_ancestors': avg_num_of_ancestors,
                   'med_num_of_ancestors': med_num_of_ancestors,
                   'most_common_population': most_common_population,
                   'avg_ancestry_dist': avg_ancestry_dist,
                   'init_variance': get_variance_for_init()}

    return result_dict


# Written by: Noa Gaon
def print_statistics(statistics_dict):
    """
    Args:
        grid: A dictionary with statistical data about the current generation grid.

    Prints:
    - The number of living cells on the board.
    - The average Fitness score in the board (rounded 3 places after decimal dot).
    - print the number of extinct populations (each is defined by a population ids tuple)
    - The average number of ancestors a cell has.
    """
    # Prints
    print("Number of living cells: {0}".format(statistics_dict['living_cells_counter']))
    print("Number of populations: {0}".format(statistics_dict['pop_count']))
    print("Number of extinct populations: {0}".format(statistics_dict['extinct_pop_count']))
    print("Average fitness score:{0}".format(round(statistics_dict['avg_fitness_score']), 3))
    print("Average num of ancestors per cell: {0}".format(round(statistics_dict['avg_num_of_ancestors'], 3)))
    print("Median num of ancestors per cell: {0}".format(round(statistics_dict['med_num_of_ancestors'], 3)))
    print("Most common population: {0}".format(statistics_dict['most_common_population']))
    #print("Average ancestry dist: {0}".format(round(result_dict['avg_ancestry_dist'], 3))) #REWRITE!!
    print("Variance in initial population num of appearances (as ancestors): {0}".format(round(statistics_dict['init_variance'], 3))) # Variance in initial population appearances
    print("\n")


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
        addend = ADDEND_INTRA_POPULATION
    else:
        addend = ADDEND_INTER_POPULATION
    parent1.set_fitness_score(parent1.get_fitness_score() + addend)
    parent2.set_fitness_score(parent2.get_fitness_score() + addend)


# Written by: Noa Gaon
def is_equal_fitness_score(cell1, cell2):
    """
    :param cell1: Cell - a cell in the grid which fitness score we want to compare.
    :param cell2: Cell - a cell in the grid which fitness score we want to compare.
    :return: bool - True if fitness score is equal, False otherwise
    """
    return get_fitness_score_abs_diff(cell1, cell2) == 0


# Written by: Noa Gaon
def get_fitness_score_abs_diff(cell1, cell2):
    """
    :param cell1: Cell - a cell in the grid which fitness score we want to compare.
    :param cell2: Cell - a cell in the grid which fitness score we want to compare.
    :return: bool - True if fitness score is equal, False otherwise
    """
    return abs(cell1.get_fitness_score()-cell2.get_fitness_score())


# Written by: Noa Gaon
def pick_parents(sorted_neighbors):
    """
    Args:
        sorted_neighbors(List): A list of the neighbors (type: Cell) of a cell to be created.
        Sorted in ascending order by fitness score.
    Returns: The two cells in the list with the highest fitness score.
    """
    global MAX_FITNESS_SCORE_DIFF

    # in case of 2 neighbors:
    if len(sorted_neighbors) == 2:
        if get_fitness_score_abs_diff(sorted_neighbors[0], sorted_neighbors[1]) <= MAX_FITNESS_SCORE_DIFF:
            return sorted_neighbors[0], sorted_neighbors[1]  # CAN reproduce
        return None, None  # No fit parents have been found

    # All diffs between the neighbors are less than MAX_FITNESS_SCORE_DIFF
    parents = None
    if (is_equal_fitness_score(sorted_neighbors[0], sorted_neighbors[1]) and
            is_equal_fitness_score(sorted_neighbors[0], sorted_neighbors[2])): # all 3 with eq scores
        parents = random.sample(sorted_neighbors, 2)
        return parents[0], parents[1]
    elif is_equal_fitness_score(sorted_neighbors[0], sorted_neighbors[1]): # first two with eq scores
        if get_fitness_score_abs_diff(sorted_neighbors[0], sorted_neighbors[2]) > MAX_FITNESS_SCORE_DIFF:
            return sorted_neighbors[0], sorted_neighbors[1] # cannot reproduce with highest
        else:
            # CAN reproduce with [2]
            return random.sample([sorted_neighbors[0], sorted_neighbors[1]], 1)[0], sorted_neighbors[2]
    elif is_equal_fitness_score(sorted_neighbors[1], sorted_neighbors[2]): # last two with eq scores
        return sorted_neighbors[1], sorted_neighbors[2]
    else: # no two fitness scores are equal
        if get_fitness_score_abs_diff(sorted_neighbors[1], sorted_neighbors[2]) <= MAX_FITNESS_SCORE_DIFF:
            return sorted_neighbors[1], sorted_neighbors[2] # last two CAN reproduce
        elif get_fitness_score_abs_diff(sorted_neighbors[0], sorted_neighbors[1]) <= MAX_FITNESS_SCORE_DIFF:
            return sorted_neighbors[0], sorted_neighbors[1] # first two CAN reproduce

    if parents:
        return parents[0], parents[1]
    return None, None  # No fit parents have been found


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
                    POPULATION_IDS_DICT[curr_cell_ids]['count'] -= 1 # update the global dict
                next_grid[row][col] = 0

            # If the number of surrounding live cells is 3 OR 2 and the cell at grid[row][col] was previously dead -->
            # turn the cell into a live cell
            elif (live_neighbors_count==2 or live_neighbors_count == 3) and grid[row][col] == 0:
                neighbors = []
                for indexes in live_neighbors_list:
                    neighbors.append(grid[indexes[0]][indexes[1]])
                sorted_neighbors = sorted(neighbors, key=lambda x: x.get_fitness_score())
                parent1, parent2 = pick_parents(sorted_neighbors)
                if parent1:
                    update_fitness_score(parent1, parent2)
                    next_grid[row][col] = Cell(parent1.get_population_ids(), parent2.get_population_ids(),
                                               parent1.get_fitness_score(), parent2.get_fitness_score())
                else:
                    next_grid[row][col] = grid[row][col] # No fit parents found
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


# Written by: Noa Gaon
def get_integer_value(prompt, default):
    """
    :param prompt: String - The string to prompt the user for input with
    :param default: int - The default value for the integer
    :return: The valid input value that the user entered
    """
    while True:
        try:
            value = input(prompt)
            if value == '':
                return default
            value = int(value)
            return value  # Return the integer if input is valid
        except ValueError as e:
            print("Invalid input. Please enter an integer.")


# ORIGINAL CODE
def get_integer_bound_value(prompt, low, high):
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
def get_string_value(prompt, list_options, default):
    """
    Asks the user for string input. Accepts only

    :param prompt: String - The string to prompt the user for input with
    :param list_options: String - The letters the input string may start with
    :return: The valid input value that the user entered
    """
    while True:
        value = input(prompt)
        if value == '':
            return default
        if value.lower()[0] not in list_options:
            print("Input was not one of the following options:{0}".format(list_options))
        return value  # Return the integer if input is valid


# Creates a plot that tracks the living cells number across all iterations
def plot_generation_track(living_cells_num_track, avg_fitness_score_track, avg_num_of_ancestors_track,
                          generations, title):
    plt.cla() # Clear previous plot

    # Setting x-axis ticks to integers only
    plt.xticks(range(min(generations), max(generations) + 1, 1))

    plt.plot(generations, living_cells_num_track, label='living cells num')
    plt.plot(generations, avg_fitness_score_track, label='avg fitness score ')
    plt.plot(generations, avg_num_of_ancestors_track, label='avg num of ancestors')

    # Add labels, legend, etc.
    plt.xlabel('Generation No.')
    plt.ylabel('Count')
    plt.title(f'Counts per Generation: {title}')
    plt.legend()
    plt.show()


# Modified by: Noa Gaon
def run_game(rows=None, cols=None, pop_num=None, initial_option=None, addend_inter_pop=None, addend_intra_pop=None,
             max_fitness_score_diff=None):
    """
    Asks the user for input to setup the Game of Life to run for a given number of generations.

    """
    global ADDEND_INTER_POPULATION, ADDEND_INTRA_POPULATION, MAX_FITNESS_SCORE_DIFF
    clear_console()

    ADDEND_INTER_POPULATION = addend_inter_pop
    ADDEND_INTRA_POPULATION = addend_intra_pop
    MAX_FITNESS_SCORE_DIFF = max_fitness_score_diff

    if not rows:
        # Get the number of rows and columns for the Game of Life grid from user
        rows = get_integer_bound_value("Enter the number of rows (10-60): ", 10, 60)
        clear_console()
        cols = get_integer_bound_value("Enter the number of cols (10-118): ", 10, 118)
        pop_num = get_integer_value("Enter the number of populations (default 2): ", 2)
        initial_option = get_string_value("Enter an initial state (random/glider/blinker. default: random: ", ['r', 'g', 'b'], 'r')
        ADDEND_INTER_POPULATION = get_integer_value("Enter fitness score for inter-population reproduction (between populations. default: 2): ",
                                                    ADDEND_INTER_POPULATION)
        ADDEND_INTRA_POPULATION = get_integer_value("Enter fitness score for intra-population reproduction (inside a population. default: 1): ",
                                                    ADDEND_INTRA_POPULATION)
        MAX_FITNESS_SCORE_DIFF = get_integer_value("Enter maximum fitness score diff allowed for reproduction (default: allow all): ",
                                                   -1)

    clear_console()

    # Get the number of generations that the Game of Life should run for
    generations = MAX_GENERATIONS
    resize_console(rows, cols)

    # Create the initial random Game of Life grids
    if initial_option.lower().startswith('r'):
        current_generation = create_initial_grid(rows, cols)      # Random
    else:
        current_generation = shape_initial_configuration(rows, cols, initial_option)    # glider/blinker
    current_generation = divide_into_populations(current_generation, pop_num)  # divide the matrix into populations
    next_generation = create_initial_grid(rows, cols)

    # Track changes in living cell num
    living_cells_num_track = []
    extinct_populations_num_track = []
    avg_fitness_score_track = []
    avg_num_of_ancestors_track = []

    # Run Game of Life sequence
    gen = 1
    for gen in range(1, generations + 1):
        if not grid_changing(current_generation, next_generation):
            break
        print_grid(current_generation, gen)     # print current grid

        statistics_dict = calc_statistics(current_generation)
        print_statistics(statistics_dict) # print current grid statistics

        living_cells_num_track.append(statistics_dict["living_cells_counter"])
        extinct_populations_num_track.append(statistics_dict["extinct_pop_count"])
        avg_fitness_score_track.append(statistics_dict["avg_fitness_score"])
        avg_num_of_ancestors_track.append(statistics_dict["avg_num_of_ancestors"])

        create_next_grid(current_generation, next_generation)
        time.sleep(1 / 5.0)
        current_generation, next_generation = next_generation, current_generation

    print_grid(current_generation, gen)
    statistics_dict = calc_statistics(current_generation)
    print_statistics(statistics_dict)

    living_cells_num_track.append(statistics_dict["living_cells_counter"])
    extinct_populations_num_track.append(statistics_dict["extinct_pop_count"])
    avg_fitness_score_track.append(statistics_dict["avg_fitness_score"])
    avg_num_of_ancestors_track.append(statistics_dict["avg_num_of_ancestors"])

    if gen == generations:
        gen += 1

    #plot_generation_track(living_cells_num_track, avg_fitness_score_track,
    #                      avg_num_of_ancestors_track, range(1, gen+1), "")

    # returns stats for a SINGLE game that ran for gen+1 generations
    return living_cells_num_track, avg_fitness_score_track, avg_num_of_ancestors_track


# Written by: Noa Gaon

# Start the Game of Life
run = "r"

while run == "r":
    game_properties = game_config.game6
    num_runs = NUM_GAMES
    results = []
    for i in range(num_runs):
        print("---- Game of Life No. {0} ----\n----------------------------".format(i+1))
        res = run_game(rows=game_properties['rows'],
                 cols=game_properties['cols'],
                 pop_num=game_properties['pop_num'],
                 initial_option=game_properties['initial_option'],
                 addend_inter_pop=game_properties['addend_inter_pop'],
                 addend_intra_pop=game_properties['addend_intra_pop'],
                 max_fitness_score_diff=game_properties['max_fitness_score_diff'])
        results.append(res)
        reset_global_vars()

    cluster_averages = clustering_results(results, MAX_GENERATIONS)

    count = 1
    for clus in cluster_averages:
        plot_generation_track(clus[0], clus[1], clus[2], range(1, MAX_GENERATIONS+1),
                              f'Avg Ending No.{count},\n{num_runs} games, max_gen:{MAX_GENERATIONS}')
        count += 1

    run = input("<Enter> to exit or r to run again: ")
