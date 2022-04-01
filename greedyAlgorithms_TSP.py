import pandas as pd
import numpy as np
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

# calculating weighted grief
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

def nearest_neighbor(pos,path):
    coords = pd.read_csv(path, sep=' ',
                         names=['n', 'x', 'y'], skiprows=6, skipfooter=1, engine='python')
    point = coords.drop(columns=['n']).values

    distance = np.array(
        [[np.floor(np.sqrt((point[i][0] - point[j][0]) ** 2 + (point[i][1] - point[j][1]) ** 2)) for i in range(len(point))] for j
         in range(len(point))])
    point, distance, first_route, distance_f = nearest_neighbor_path(point, distance, pos)
    pos = pos - 50
    point, distance, second_route, distance_s = nearest_neighbor_path(point, distance, pos)

    return first_route, second_route, distance_f + distance_s

def nearest_neighbor_path(point, distance, pos):
    route = np.zeros((51, 2))  # to store points in order of visit
    road_distance = []  # to store distances in order of visit
    temp_point = pos  # starting position
    route[-1:] = point[temp_point]
    route[0:] = point[temp_point]
    for i in range(49):
        min_distance = np.min(distance[temp_point][distance[temp_point] != 0])
        help_point = list(distance[temp_point, :]).index(min_distance)  # new starting position
        route[i + 1, :] = point[help_point]

        distance = np.delete(distance, temp_point, 0)
        distance = np.delete(distance, temp_point, 1)
        point = np.delete(point, temp_point, 0)

        if (help_point < temp_point):
            temp_point = help_point
        else:
            temp_point = help_point - 1

        road_distance.append(min_distance)

    road_distance.append(np.floor(np.sqrt((route[-2][0] - route[-1][0]) ** 2 + (route[-2][1] - route[-1][1]) ** 2)))
    distance = np.delete(distance, temp_point, 0)
    distance = np.delete(distance, temp_point, 1)
    point = np.delete(point, temp_point, 0)
    route_all = np.sum(road_distance)
    return point, distance, route, route_all

def greedy_cycle(dist, start):
    n_points = dist.shape[0]
    remaining = [*range(n_points)]
    start_second = np.argmax(dist[start]) # search for the farthest vertex to start the second cycle there
    path_a = [start, find_nearest(dist,start)] # select the closest vertex and create an incomplete cycle of these two vertices
    path_b = [start_second, find_nearest(dist,start_second)]
    remaining = [i for i in remaining if i not in path_a]
    remaining = [i for i in remaining if i not in path_b]
    while remaining:
        for path in [path_a, path_b]:
            best_index = None
            best_point = None
            best_score = None
            # insert into the current cycle in the best possible position causing the smallest increase in cycle length
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
    start_second = np.argmax(dist[start]) # search for the farthest vertex to start the second cycle there
    path_a = [start, find_nearest(dist,start)] # select the closest vertex and create an incomplete cycle of these two vertices
    path_b = [start_second, find_nearest(dist,start_second)]
    remaining = [i for i in remaining if i not in path_a]
    remaining = [i for i in remaining if i not in path_b]
    while remaining:
        for path in [path_a, path_b]:
            best_index = None
            best_point = None
            best_regret = None
            best_diff = None
            # k-regret (k-regret) is the sum of the differences between the best and the k consecutive insertions
             # select item with the greatest regret and put it in the best place
             # we can also weigh grief with the greedy rule (evaluation of the first option)
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
    paths = ['data/kroA100.tsp','data/kroB100.tsp']
    for path in paths:

        solutions = [nearest_neighbor(pos, path) for pos in range(100)]
        dis = [solutions[i][2] for i in range(100)]
        best_index = np.argmin(dis)
        print(f'nearest_neighbor: {np.mean(dis)}({np.min(dis)}-{np.max(dis)})')
        best_path_a, best_path_b = solutions[best_index][0], solutions[best_index][1]
        plt.plot([best_path_a[:, 0], best_path_b[:, 0]], [best_path_a[:, 1], best_path_b[:, 1]], 'o', color='black')
        plt.plot(best_path_a[:, 0], best_path_a[:, 1], color='red')
        plt.plot(best_path_b[:, 0], best_path_b[:, 1], color='blue')
        plt.show()

        distances, coords = create_dist_matrix(path)
        solutions = [greedy_cycle(distances, start) for start in range(100)]
        scores = [cycle_score(distances, solution[0]) + cycle_score(distances, solution[1]) for solution in solutions]
        best_index = np.argmin(scores)
        print(f'greedy_cycle: {np.mean(scores)}({np.min(scores)}-{np.max(scores)})')
        [best_path_a, best_path_b] = solutions[best_index]
        draw_path(coords, best_path_a)
        draw_path(coords, best_path_b, color='red')
        plt.scatter(coords.x, coords.y, color='black')
        plt.show()

        distances, coords = create_dist_matrix(path)
        solutions = [k_regret(distances, start) for start in range(100)]
        scores = [cycle_score(distances, solution[0]) + cycle_score(distances,solution[1]) for solution in solutions]
        best_index = np.argmin(scores)
        print(f'k_regret: {np.mean(scores)}({np.min(scores)}-{np.max(scores)})')
        [best_path_a, best_path_b] = solutions[best_index]
        draw_path(coords, best_path_a)
        draw_path(coords, best_path_b, color='red')
        plt.scatter(coords.x, coords.y, color='black')
        plt.show()

if __name__ == "__main__":
    main()