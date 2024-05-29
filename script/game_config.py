import sys

"""
5 populations.
No specific type of reproduction is favored (inter/intra).
All reproduction is permitted (not limited by max diff).
"""
game1 = {
    'rows': 10,
    'cols': 10,
    'pop_num': 5,
    'initial_option': 'random',
    'addend_inter_pop': 1,
    'addend_intra_pop': 1,
    'max_fitness_score_diff': sys.maxsize
}


"""
5 populations.
intra population breeding is highly favored.
All reproduction is permitted (not limited by max diff).
"""
game2 = {
    'rows': 10,
    'cols': 10,
    'pop_num': 5,
    'initial_option': 'random',
    'addend_inter_pop': 0,
    'addend_intra_pop': 100,
    'max_fitness_score_diff': sys.maxsize
}


"""
5 populations.
inter population breeding is highly favored.
All reproduction is permitted (not limited by max diff).
"""
game3 = {
    'rows': 10,
    'cols': 10,
    'pop_num': 5,
    'initial_option': 'random',
    'addend_inter_pop': 100,
    'addend_intra_pop': 0,
    'max_fitness_score_diff': sys.maxsize
}


"""
5 populations.
No specific type of reproduction is favored (inter/intra).
max fitness diff allowed for breeding is 3.
"""
game4 = {
    'rows': 10,
    'cols': 10,
    'pop_num': 5,
    'initial_option': 'random',
    'addend_inter_pop': 1,
    'addend_intra_pop': 1,
    'max_fitness_score_diff': 3
}


"""
5 populations.
intra population breeding is highly favored.
max fitness diff allowed for breeding is 3.
"""
game5 = {
    'rows': 10,
    'cols': 10,
    'pop_num': 5,
    'initial_option': 'random',
    'addend_inter_pop': 0,
    'addend_intra_pop': 100,
    'max_fitness_score_diff': 3
}


"""
5 populations.
inter population breeding is highly favored.
max fitness diff allowed for breeding is 3.
"""
game6 = {
    'rows': 10,
    'cols': 10,
    'pop_num': 5,
    'initial_option': 'random',
    'addend_inter_pop': 100,
    'addend_intra_pop': 0,
    'max_fitness_score_diff': 3
}


"""
5 populations.
intra population breeding is favored.
max fitness diff allowed for breeding is 3.
"""
game7 = {
    'rows': 10,
    'cols': 10,
    'pop_num': 5,
    'initial_option': 'random',
    'addend_inter_pop': 1,
    'addend_intra_pop': -2,
    'max_fitness_score_diff': sys.maxsize
}