import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from greedyAlgorithms_TSP import *

def plot_solutions(coords, solutions_list):
    """
    Displays both result cycles with marked points on one graph

    Arguments:
        coords: DataFrame containing coordinates of each vertex
        solutions_list: a list containing two lists of vertex indexes in the resulting first and second cycles
    """
    draw_path(coords, solutions_list[0])
    draw_path(coords, solutions_list[1], color='red')
    plt.scatter(coords.x, coords.y, color='black')
    plt.show()

def random_solution(dist_matrix):
    """
    Generates a random solution to the modified traveling salesman problem for two cycles

    Arguments:
        dist_list: distance matrix between points
    """
    random.seed()
    n_points = dist_matrix.shape[0]
    list_of_points = [*range(n_points)]
    random.shuffle(list_of_points)
    paths = [list_of_points[:n_points//2], list_of_points[n_points//2:]]
    return paths

def swap_vertices_outside(paths, i, j):
    # exchange of two vertices in between two cycles
    paths[0][i], paths[1][j] = paths[1][j], paths[0][i]

def swap_vertices_inside(path, i, j):
    # swap two vertices in one cycle
    path[i], path[j] = path[j], path[i]

def swap_edges_inside(path, i, j):
    # swap two edges in one cycle 
    # i,j - indexes of start vertices of the edges to be replaced
    _len = len(path)
    if j - i != 1 or j - i != _len - 1:
        path[i:j] = reversed(path[i:j])

def delta_replace_vertex(distances, path, i, new):
    # i - index of vertex to be replace by "new" vertex
    _len = len(path)
    previous, current, next = path[(i-1)%_len], path[i], path[(i+1)%_len]
    return distances[previous,new] + distances[new,next] - distances[previous,current] - distances[current, next]

def delta_swap_vertices_outside(distances, paths, i, j):
    # score delta cost function value after exchanging 2 vertices of 2 different paths on indices i and j
    return delta_replace_vertex(distances, paths[0],i,paths[1][j]) + delta_replace_vertex(distances, paths[1],j,paths[0][i])

def delta_swap_vertices_inside(distances, path, i, j):
    # score delta cost function value after swaping 2 vertices on indices i and j in one path
    _len = len(path)
    previous_i, current_i, next_i = path[(i-1)%_len], path[i], path[(i+1)%_len]
    previous_j, current_j, next_j = path[(j-1)%_len], path[j], path[(j+1)%_len]
    if j - i == 1:
        return distances[previous_i,current_j]+distances[current_i,next_j]-distances[previous_i,current_i]-distances[current_j,next_j]
    elif j - i == _len - 1:
        return distances[current_j, next_i] + distances[previous_j, current_i] - distances[current_i, next_i] - distances[previous_j, current_j]
    else:
        new_path = distances[previous_i,current_j]+distances[current_j,next_i]+distances[previous_j,current_i]+distances[current_i,next_j]
        old_path = distances[previous_i,current_i]+distances[current_i,next_i]+distances[previous_j,current_j]+distances[current_j,next_j]
        return new_path - old_path

def delta_swap_edges_inside(distances, path, i, j):
    # score delta cost function value after swaping 2 edges where i,j - indexes of start vertices of the edges 
    _len = len(path)
    start_i, end_i, start_j, end_j = path[(i - 1)%_len], path[i], path[j], path[(j+1)%_len]
    return distances[start_i, start_j] + distances[end_i, end_j] - distances[start_i, end_i] - distances[start_j, end_j]

def generate_candidates_outside(paths):
    # calculate cartesian product of the index list of two paths
    indices_a = [*range(len(paths[0]))]
    indices_b = [*range(len(paths[1]))]
    index_pairs = [[i,j] for j in indices_b for i in indices_a]
    return index_pairs

def generate_candidates_inside(path):
    combinations = []
    for i in range(len(path)):
        for j in range(i+1, len(path)):
            combinations.append([i, j])
    return combinations

def main():
    list = [*range(1,4)]
    list_b = [*range(1,3)]
    output = generate_candidates_outside([list,list_b])
    print(output)

    """
    paths = ['data/kroA100.tsp','data/kroB100.tsp']
    path = paths[0]
    distances, coords = create_dist_matrix(path)
    solutions = random_solution(distances)
    plot_solutions(coords,solutions)

    solutions = [k_regret(distances, start) for start in range(100)]
    scores = [cycle_score(distances, solution[0]) + cycle_score(distances,solution[1]) for solution in solutions]
    best_index = np.argmin(scores)
    print(f'k_regret: {np.mean(scores)}({np.min(scores)}-{np.max(scores)})')
    [best_path_a, best_path_b] = solutions[best_index]
    print(best_path_a, best_path_b)
    draw_path(coords, best_path_a)
    draw_path(coords, best_path_b, color='red')
    plt.scatter(coords.x, coords.y, color='black')
    plt.show()
    """


if __name__ == "__main__":
    main()