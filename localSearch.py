import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import time
from copy import deepcopy
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


def random_solution(dist_matrix, i):
    """
    Generates a random solution to the modified traveling salesman problem for two cycles
    Arguments:
        dist_list: distance matrix between points
    """
    random.seed(i)
    n_points = dist_matrix.shape[0]
    list_of_points = [*range(n_points)]
    random.shuffle(list_of_points)
    paths = [list_of_points[:n_points // 2], list_of_points[n_points // 2:]]
    return paths

def score(cities, paths):
    return cycle_score(cities, paths[0]) + cycle_score(cities, paths[1])


def swap_vertices_outside(paths, i, j):
    # exchange of two vertices in between two cycles
    paths[0][i], paths[1][j] = paths[1][j], paths[0][i]
    return paths


def swap_vertices_inside(path, i, j):
    # swap two vertices in one cycle
    path[i], path[j] = path[j], path[i]
    return path


def swap_edges_inside(path, i, j):
    # swap two edges in one cycle
    # i,j - indexes of start vertices of the edges to be replaced
    path[i:j + 1] = reversed(path[i:j + 1])
    return path


def delta_replace_vertex(distances, path, i, new):
    # i - index of vertex to be replace by "new" vertex
    _len = len(path)
    previous, current, next = path[(i - 1) % _len], path[i], path[(i + 1) % _len]
    return distances[previous, new] + distances[new, next] - distances[previous, current] - distances[current, next]


def delta_swap_vertices_outside(distances, paths, i, j):
    # score delta cost function value after exchanging 2 vertices of 2 different paths on indices i and j
    return delta_replace_vertex(distances, paths[0], i, paths[1][j]) + delta_replace_vertex(distances, paths[1], j,
                                                                                            paths[0][i])


def delta_swap_vertices_inside(distances, path, i, j):
    # score delta cost function value after swaping 2 vertices on indices i and j in one path
    _len = len(path)
    previous_i, current_i, next_i = path[(i - 1) % _len], path[i], path[(i + 1) % _len]
    previous_j, current_j, next_j = path[(j - 1) % _len], path[j], path[(j + 1) % _len]
    if j - i == 1:
        return distances[previous_i, current_j] + distances[current_i, next_j] - distances[previous_i, current_i] - \
               distances[current_j, next_j]
    elif j - i == _len - 1:
        return distances[current_j, next_i] + distances[previous_j, current_i] - distances[current_i, next_i] - \
               distances[previous_j, current_j]
    else:
        new_path = distances[previous_i, current_j] + distances[current_j, next_i] + distances[previous_j, current_i] + \
                   distances[current_i, next_j]
        old_path = distances[previous_i, current_i] + distances[current_i, next_i] + distances[previous_j, current_j] + \
                   distances[current_j, next_j]
        return new_path - old_path


def delta_swap_edges_inside(distances, path, i, j):
    # score delta cost function value after swaping 2 edges where i,j - indexes of start vertices of the edges
    _len = len(path)
    if j - i == 1 or j - i == _len - 1:
        return 0
    else:
        start_i, end_i, start_j, end_j = path[(i - 1) % _len], path[i], path[j], path[(j + 1) % _len]
        return distances[start_i, start_j] + distances[end_i, end_j] - distances[start_i, end_i] - distances[
            start_j, end_j]


def generate_candidates_outside(paths):
    # calculate cartesian product of the index list of two paths
    indices_a = [*range(len(paths[0]))]
    indices_b = [*range(len(paths[1]))]
    index_pairs = [[i, j] for j in indices_b for i in indices_a]
    return index_pairs


def generate_candidates_inside(path):
    combinations = []
    for i in range(len(path)):
        for j in range(i + 1, len(path)):
            combinations.append([i, j])
    return combinations


def random_searching(distances, paths, time_limit):
    random.seed()
    best_score = score(distances, paths)
    best_result = paths
    start = time.time()
    while time.time() - start < time_limit:
        list_out = generate_candidates_outside(paths)
        list_in_a = generate_candidates_inside(paths[0])
        list_in_b = generate_candidates_inside(paths[1])
        n = random.randint(0, 4)
        if n == 0:
            [i, j] = random.choice(list_out)
            new_path = swap_vertices_outside(paths, i, j)
        elif n == 1:
            [i, j] = random.choice(list_in_a)
            new_path = [swap_vertices_inside(paths[0], i, j), paths[1]]
        elif n == 2:
            [i, j] = random.choice(list_in_b)
            new_path = [paths[0], swap_vertices_inside(paths[1], i, j)]
        elif n == 3:
            [i, j] = random.choice(list_in_a)
            new_path = [swap_edges_inside(paths[0], i, j), paths[1]]
        elif n == 4:
            [i, j] = random.choice(list_in_b)
            new_path = [paths[0], swap_edges_inside(paths[1], i, j)]

        new_score = score(distances, new_path)
        if new_score < best_score:
            best_score = new_score
            best_result = deepcopy(new_path)
    return best_result


def steepest_v_v(distances, path):

    list_out = generate_candidates_outside(path)
    help_delta_out = [delta_swap_vertices_outside(distances, path, list_out[i][0], list_out[i][1]) for i in
                      range(len(list_out))]
    vertex_out = np.argmin(help_delta_out)
    min_out = min(help_delta_out)
    # print(vertex_v_out)

    list_in_a = generate_candidates_inside(path[0])
    help_delta_in_a = [delta_swap_vertices_inside(distances, path[0], list_in_a[i][0], list_in_a[i][1]) for i in
                       range(len(list_in_a))]
    vertex_in_a = np.argmin(help_delta_in_a)
    min_in_a = min(help_delta_in_a)

    list_in_b = generate_candidates_inside(path[1])
    help_delta_in_b = [delta_swap_vertices_inside(distances, path[1], list_in_b[i][0], list_in_b[i][1]) for i in
                       range(len(list_in_b))]
    vertex_in_b = np.argmin(help_delta_in_b)
    min_in_b = min(help_delta_in_b)
    # print(help_delta_v_in)

    #print(min_out, min_in_a, min_in_b)

    if min_out < min_in_a and min_out < min_in_b:
        if min_out >= 0:
            return path
        else:
            i, j = list_out[vertex_out]
            new_path = swap_vertices_outside(path, i, j)
            return steepest_v_v(distances, new_path)
    elif min_in_a < min_out and min_in_a < min_in_b:
        if min_in_a >= 0:
            return path
        else:
            i, j = list_in_a[vertex_in_a]

            new_path = [swap_vertices_inside(path[0], i, j), path[1]]
            return steepest_v_v(distances, new_path)
    else:
        if min_in_b >= 0:
            return path
        else:
            i, j = list_in_b[vertex_in_b]

            new_path = [path[0], swap_vertices_inside(path[1], i, j)]
            return steepest_v_v(distances, new_path)

def steepest_v_e(distances, path):
    list_out = generate_candidates_outside(path)
    help_delta_out = [delta_swap_vertices_outside(distances, path, list_out[i][0], list_out[i][1]) for i in
                      range(len(list_out))]
    vertex_out = np.argmin(help_delta_out)
    min_out = min(help_delta_out)
    # print("Vertices outside:",list_out[vertex_out], min_out)

    list_in_a = generate_candidates_inside(path[0])
    help_delta_in_a = [delta_swap_edges_inside(distances, path[0], list_in_a[i][0], list_in_a[i][1]) for i in
                       range(len(list_in_a))]
    vertex_in_a = np.argmin(help_delta_in_a)
    min_in_a = min(help_delta_in_a)
    # print("Edges inside 1:",list_in_a[vertex_in_a], min_in_a)

    list_in_b = generate_candidates_inside(path[1])
    help_delta_in_b = [delta_swap_edges_inside(distances, path[1], list_in_b[i][0], list_in_b[i][1]) for i in
                       range(len(list_in_b))]
    vertex_in_b = np.argmin(help_delta_in_b)
    min_in_b = min(help_delta_in_b)
    # print("Edges inside 2:",list_in_b[vertex_in_b], min_in_b)

    if min_out < min_in_a and min_out < min_in_b:
        if min_out >= 0:
            return path
        else:
            i, j = list_out[vertex_out]
            new_path = swap_vertices_outside(path, i, j)
            return steepest_v_e(distances, new_path)
    elif min_in_a < min_out and min_in_a < min_in_b:
        if min_in_a >= 0:
            return path
        else:
            i, j = list_in_a[vertex_in_a]
            new_path = [swap_edges_inside(path[0], i, j), path[1]]
            return steepest_v_e(distances, new_path)
    else:
        if min_in_b >= 0:
            return path
        else:
            i, j = list_in_b[vertex_in_b]
            new_path = [path[0], swap_edges_inside(path[1], i, j)]
            return steepest_v_e(distances, new_path)

def greddy(distances, path,n,inside):
    random.seed()
    list_out = generate_candidates_outside(path)
    list_in_a = generate_candidates_inside(path[0])
    list_in_b = generate_candidates_inside(path[1])

    help_delta_out, help_delta_in_a, help_delta_in_b = 0, 0, 0
    neig = [help_delta_out, help_delta_in_a, help_delta_in_b]
    len_list = [len(list_out), len(list_in_a), len(list_in_b)]


    c =  random.randint(0,len_list[n]-1)
    c_counter = len_list[n]-1 - c
    rever = 1


    while neig[n] >= 0 and c < len_list[n]:

        if c == len_list[n]-1 and rever == 1:
            c = 0
            len_list[n] = c_counter
            rever = 0


        if n == 0:
            help_delta_out = delta_swap_vertices_outside(distances, path, list_out[c][0], list_out[c][1])
            i, j = list_out[c]
        elif n == 1:
            if inside == 0:
                help_delta_in_a = delta_swap_vertices_inside(distances, path[0], list_in_a[c][0], list_in_a[c][1])
                i, j = list_in_a[c]
            else:
                help_delta_in_a = delta_swap_edges_inside(distances, path[0], list_in_a[c][0], list_in_a[c][1])
                i, j = list_in_a[c]

        else:
            if inside == 0:
                help_delta_in_b = delta_swap_vertices_inside(distances, path[1], list_in_b[c][0], list_in_b[c][1])
                i, j = list_in_b[c]
            else:
                help_delta_in_b = delta_swap_edges_inside(distances, path[1], list_in_b[c][0], list_in_b[c][1])
                i, j = list_in_b[c]

        neig = [help_delta_out, help_delta_in_a, help_delta_in_b]
        c += 1
    return neig[n], i, j

def greedy_v_v(distances, path):
    random.seed()
    k = random.randint(0, 1)
    if k == 0:

        while True:
            help_delta,i,j = greddy(distances, path,0,0)
            if help_delta < 0:
             path = swap_vertices_outside(path, i, j)
            if help_delta >= 0:
                break

        while True:

            help_delta, i, j = greddy(distances, path, 1,0)
            if help_delta < 0:
             path =[swap_vertices_inside(path[0], i, j), path[1]]
            if help_delta >= 0:
                    break

        while True:

            help_delta, i, j = greddy(distances, path, 2,0)
            if help_delta < 0:
             path = [path[0], swap_vertices_inside(path[1], i, j)]
            if help_delta >= 0:
                return path
    else:
        while True:

            help_delta, i, j = greddy(distances, path, 1, 0)
            if help_delta < 0:
                path = [swap_vertices_inside(path[0], i, j), path[1]]
            if help_delta >= 0:
                break

        while True:

            help_delta, i, j = greddy(distances, path, 2, 0)
            if help_delta < 0:
                path = [path[0], swap_vertices_inside(path[1], i, j)]
            if help_delta >= 0:
                break

        while True:
            help_delta,i,j = greddy(distances, path,0,0)
            if help_delta < 0:
             path = swap_vertices_outside(path, i, j)
            if help_delta >= 0:
                return path

def greedy_v_e(distances, path):
    random.seed()
    k = random.randint(0,1)
    if k ==0:
        while True:
            help_delta, i, j = greddy(distances, path, 0,1)
            if help_delta < 0:
             path = swap_vertices_outside(path, i, j)

            if help_delta >= 0:
                break

        while True:

            help_delta, i, j = greddy(distances, path, 1,1)
            if help_delta < 0:
                path = [swap_edges_inside(path[0], i, j), path[1]]
            if help_delta >= 0:
                break

        while True:

            help_delta, i, j = greddy(distances, path, 2,1)
            if help_delta < 0:
                path = [path[0], swap_edges_inside(path[1], i, j)]
            if help_delta >= 0:
                return path
    else:

        while True:

            help_delta, i, j = greddy(distances, path, 1, 1)
            if help_delta < 0:
                path = [swap_edges_inside(path[0], i, j), path[1]]
            if help_delta >= 0:
                break

        while True:

            help_delta, i, j = greddy(distances, path, 2, 1)
            if help_delta < 0:
                path = [path[0], swap_edges_inside(path[1], i, j)]
            if help_delta >= 0:
                break

        while True:
            help_delta, i, j = greddy(distances, path, 0, 1)
            if help_delta < 0:
                path = swap_vertices_outside(path, i, j)

            if help_delta >= 0:
                return path

def main():
    paths = ['data/kroA100.tsp', 'data/kroB100.tsp']
    start = [67, 50]
    path = paths[1]

    distances, coords = create_dist_matrix(path)
    solution = [random_solution(distances, i) for i in range(100)]
    times = []
    solutions = []
    for i in range(100):
        start = time.time()
        result = greedy_v_e(distances, solution[i])
        stop = time.time()
        solutions.append(result)
        times.append(stop - start)


    # solutions = [steepest_v_e(distances, solution[i]) for i in range(100)]
    scores = [cycle_score(distances, solution[0]) + cycle_score(distances, solution[1]) for solution in solutions]
    best_index = np.argmin(scores)
    print(f'Steepest results: {np.mean(scores)}({np.min(scores)}-{np.max(scores)})')
    print(f'Steepest time: {np.mean(times)}({np.min(times)}-{np.max(times)})')
    plot_solutions(coords, solutions[best_index])

    '''distances, coords = create_dist_matrix(path)
    solution = [random_solution(distances,i) for i in range(100)]
    solutions = [steepest_v_v(distances, solution[i]) for i in range(100) ]
    scores = [cycle_score(distances, solution[0]) + cycle_score(distances, solution[1]) for solution in solutions]
    best_index = np.argmin(scores)
    print(f'Steepest: {np.mean(scores)}({np.min(scores)}-{np.max(scores)})')
    plot_solutions(coords, solutions[best_index])'''

    '''solution = k_regret(distances, start[0]) 
    solutions = greedy_v_v(distances, solution)
    plot_solutions(coords, solutions)'''

    '''solution = k_regret(distances, start[0]) 
    solutions = steepest_v_v(distances, solution)
    plot_solutions(coords, solutions)'''

    """
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