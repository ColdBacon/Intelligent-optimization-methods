import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def distance(a, b):
    return np.floor(np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) + 0.5)

def create_dist_matrix(path):
    coords = pd.read_csv(path, sep=' ', names=['n','x','y'], skiprows=6, skipfooter=1, engine='python', index_col=0)
    points = coords.values
    rows, _ = points.shape
    dist_matrix = np.array([[distance(points[i],points[j]) for j in range(rows)] for i in range(rows)])
    return dist_matrix, coords

def calc_diff(distances, new, a, b):
    return distances[a,new] + distances[new,b] - distances[a,b]

def find_nearest(distances, point):
    valid_idx = np.where(distances[point] > 0)[0]
    return valid_idx[distances[point][valid_idx].argmin()]

def cycle_score(distances, path):
    cycle = path + [path[0]]
    return sum(distances[cycle[i], cycle[i+1]] for i in range(len(cycle) - 1))

def calc_2regret(diff_list):
    index = np.argmin(diff_list)
    mini = min(diff_list)
    diff_list.sort()
    regret = diff_list[1] - diff_list[0]
    return regret - 0.5 * mini, index

def draw_path(coords, path, color='blue'):
    cycle = path + [path[0]]
    for i in range(len(cycle) - 1):
        a, b = cycle[i], cycle[i+1]
        plt.plot([coords.x[a+1], coords.x[b+1]], [coords.y[a+1], coords.y[b+1]], color=color)

def nearest_neighbor():
    pass

def greedy_cycle(dist, start):
    n_points = dist.shape[0]
    remaining = [*range(n_points)]
    start_second = np.argmax(dist[start]) # poszukiwanie najdalszego wierzchołka aby zacząć tam drugi cykl
    path_a = [start, find_nearest(dist,start)] # wybierz najbliższy wierzchołek i stwórz z tych dwóch wierzchołków niepełny cykl
    path_b = [start_second, find_nearest(dist,start_second)]
    remaining = [i for i in remaining if i not in path_a]
    remaining = [i for i in remaining if i not in path_b]
    while remaining:
        for path in [path_a, path_b]:
            best_index = None
            best_point = None
            best_score = None
            # wstaw do bieżącego cyklu w najlepsze możliwe miejsce wierzchołek powodujący najmniejszy wzrost długości cyklu
            for point in remaining:
                for i in range(len(path)):
                    distance_diff = calc_diff(dist, point, path[i - 1], path[i])
                    if best_score is None or distance_diff < best_score:
                        best_score = distance_diff
                        best_index = i
                        best_point = point
            path.insert(best_index,best_point)
            remaining.remove(best_point)
    return [path_a, path_b]

def k_regret(dist, start):
    n_points = dist.shape[0]
    remaining = [*range(n_points)]
    start_second = np.argmax(dist[start]) # poszukiwanie najdalszego wierzchołka aby zacząć tam drugi cykl
    path_a = [start, find_nearest(dist,start)] # wybierz najbliższy wierzchołek i stwórz z tych dwóch wierzchołków niepełny cykl
    path_b = [start_second, find_nearest(dist,start_second)]
    remaining = [i for i in remaining if i not in path_a]
    remaining = [i for i in remaining if i not in path_b]
    while remaining:
        for path in [path_a, path_b]:
            best_index = None
            best_point = None
            best_regret = None
            best_diff = None
            # k-żal (k-regret) to suma różnic pomiędzy najlepszym, a k kolejnymi opcjami wstawienia
            # wybieramy element o największym żalu i wstawiamy go w najlepsze miejsce
            # możemy również ważyć żal z regułą zachłanną (oceną pierwszej opcji)
            for point in remaining:
                if len(path) == 2:
                    distance_diff = calc_diff(dist, point, path[0], path[1])
                    if (best_diff is None or distance_diff < best_diff):
                        best_diff = distance_diff
                        best_point = point
                        best_index = 0
                else:
                    distance_diff = [calc_diff(dist, point, path[i - 1], path[i]) for i in range(len(path))] 
                    regret, index  = calc_2regret(distance_diff)
                    if best_regret is None or regret > best_regret:
                        best_index = index
                        best_point = point
                        best_regret = regret           
            path.insert(best_index,best_point)
            remaining.remove(best_point)
    return [path_a, path_b]


def main():
    distances, coords = create_dist_matrix('data/kroA100.tsp')
    #start = random.randint(0,99)
    solutions = [k_regret(distances, start) for start in range(100)] # or greedy_cycle()
    scores = [cycle_score(distances, solution[0]) + cycle_score(distances,solution[1]) for solution in solutions]
    best_index = np.argmin(scores)
    print("Best score: ",scores[best_index])
    [best_path_a, best_path_b] = solutions[best_index]
    draw_path(coords, best_path_a)
    draw_path(coords, best_path_b, color='red')
    plt.scatter(coords.x, coords.y, color='black')
    plt.show()   

if __name__ == "__main__":
    main()