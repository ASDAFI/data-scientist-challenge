import random
import math
from typing import List

import numpy as np
from sklearn.cluster import KMeans

from problem import ClusteringProblem


def lazy_get_assignments_neighbours(assignments: np.ndarray) -> np.ndarray:
    order: List[int] = list(range(assignments.shape[0]))
    random.shuffle(order)
    new_assignments: np.ndarray = assignments.copy()

    for i in order:
        for j in order:
            if i != j:
                new_assignments[i], new_assignments[j] = new_assignments[j], new_assignments[i]
                yield new_assignments.copy()
                new_assignments[i], new_assignments[j] = new_assignments[j], new_assignments[i]


class Algorithm:
    def __init__(self):
        pass

    def optimization_iteration(self, *args):
        pass

    def optimize(self, *args):
        pass


class SimulatedAnnealing(Algorithm):
    def __init__(self, problem: ClusteringProblem):
        self.current_state: ClusteringProblem = problem
        self.current_state_error: np.float64 = self.current_state.error
        self.error_coeff: int = None
        self.best_assignment: np.ndarray = problem.assignments
        self.best_state_error: np.float64 = self.current_state_error

    @staticmethod
    def validate(error_delta: np.float64, current_temp: float) -> bool:
        if error_delta > 0:
            return True
        if random.random() <= np.exp(error_delta / current_temp):
            return True
        return False

    def optimization_iteration(self, current_temp: float) -> bool:
        for assignments_neighbor in lazy_get_assignments_neighbours(self.current_state.assignments):
            centroids = self.current_state.get_centroids(self.current_state.scaled_positions,
                                                         self.current_state.cluster_sizes,
                                                         assignments_neighbor)
            error = self.current_state.get_error(self.current_state.scaled_positions, centroids,
                                                 assignments_neighbor) * self.error_coeff

            if error < self.best_state_error:
                self.best_assignment = assignments_neighbor
                self.best_state_error = error

            error_delta: np.float64 = self.current_state_error - error
            if self.validate(error_delta, current_temp):
                self.current_state.assignments = assignments_neighbor
                self.current_state_error = error
                return True
        return False

    def optimize(self, current_temperature: float = 4000, desired_temperature: int = 10,
                 cooling_rate: float = 0.999):
        self.error_coeff: float = 10 ** int(math.log(current_temperature, 10) + 2)
        self.current_state_error *= self.error_coeff
        while current_temperature > desired_temperature:
            self.optimization_iteration(current_temperature)
            print(current_temperature, self.current_state_error)
            current_temperature *= cooling_rate
        print(self.best_state_error)
        self.current_state_error = self.best_state_error
        self.current_state.assignments = self.best_assignment


class LocalSearch(Algorithm):
    def __init__(self, problem: ClusteringProblem):
        self.current_state: ClusteringProblem = problem
        self.current_state_error: np.float64 = self.current_state.error
        self.best_assignment: np.ndarray = problem.assignments
        self.best_state_error: np.float64 = self.current_state_error

    def optimization_iteration(self) -> bool:
        neighbors_assignments = list(lazy_get_assignments_neighbours(self.current_state.assignments))[:100]
        errors = []
        for assignments_neighbor in neighbors_assignments:
            centroids = self.current_state.get_centroids(self.current_state.scaled_positions,
                                                         self.current_state.cluster_sizes,
                                                         assignments_neighbor)
            error = self.current_state.get_error(self.current_state.scaled_positions, centroids,
                                                 assignments_neighbor)

            errors.append(error)

        assignments_with_error = list(zip(neighbors_assignments, errors))
        assignments_with_error.sort(key=lambda x: x[1])
        sum_errors = sum([x[1] for x in assignments_with_error])
        count_assignments = len(assignments_with_error)
        commulative_prob = 0
        for i in range(count_assignments):
            normalized_error = assignments_with_error[i][1] / sum_errors
            commulative_prob += normalized_error
            if random.random() >= commulative_prob:  # min(commulative_prob * 10, 0.1):
                centroids = self.current_state.get_centroids(self.current_state.scaled_positions,
                                                             self.current_state.cluster_sizes,
                                                             assignments_with_error[i][0])
                error = self.current_state.get_error(self.current_state.scaled_positions, centroids,
                                                     assignments_with_error[i][0])
                if error < self.best_state_error:
                    self.best_state_error = error
                    self.best_assignment = assignments_with_error[i][0]

                self.current_state.assignments = assignments_with_error[i][0]
                self.current_state_error = error
                return True

    def optimize(self, iterations: int = 1000):
        for i in range(iterations):
            print(i, self.current_state_error)
            if not self.optimization_iteration():
                print("bad event")
        print(self.best_state_error)
        self.current_state_error = self.best_state_error
        self.current_state.assignments = self.best_assignment


def solve_greedy(problem: ClusteringProblem):
    kmeans = KMeans(n_clusters=problem.clusters_count)

    kmeans.fit(problem.scaled_positions)
    cluster_assignments = kmeans.labels_
    centroids = list(problem.get_centroids(problem.positions, problem.cluster_sizes, cluster_assignments))
    # to sort using distance
    """sorted_centroids = [centroids[0].copy()]
    sorted_centroids_index = [0]

    centroids[0][0] = 1000000
    centroids[0][1] = 1000000

    for i in range(len(centroids)):
        min_dist_centroid = min(centroids,
                                key=lambda x: np.abs(sorted_centroids[-1] - x)[0] + np.abs(sorted_centroids[-1] - x)[1])
        for ind, cen in enumerate(centroids):
            if cen[0] == min_dist_centroid[0] and cen[1] == min_dist_centroid[1]:
                index = ind
                break
        #index = centroids.index(min_dist_centroid)
        sorted_centroids.append(min_dist_centroid.copy())
        centroids[index][0] = 1000000
        centroids[index][1] = 1000000
        sorted_centroids_index.append(index)

    """
    sorted_centroids = sorted(list(zip(centroids, range(problem.clusters_count))), key=lambda x: problem.cluster_sizes[x[1]])
    sorted_centroids.reverse()
    sorted_centroids_index = [x[1] for x in sorted_centroids]
    sorted_centroids = [x[0] for x in sorted_centroids]
    assigned_numbers = [0] * problem.positions_count
    for i in range(problem.positions_count):
        assigned_numbers[cluster_assignments[i]] += 1

    for i in range(problem.clusters_count):
        index = sorted_centroids_index[i]
        if assigned_numbers[index] == problem.cluster_sizes[index]:
            continue
        elif assigned_numbers[index] > problem.cluster_sizes[index]:
            positions_with_index = [[position, ind] for ind, position in enumerate(problem.positions) if cluster_assignments[ind] == index]
            positions_with_index.sort(
                key=lambda x: np.abs(sorted_centroids[i] - x[0])[0] + np.abs(sorted_centroids[i] - x[0])[1])
            for position, ind in positions_with_index[int(problem.cluster_sizes[index]):]:
                assigned_numbers[index] -= 1
                cluster_assignments[ind] = -1
    for i in range(problem.clusters_count):
        index = sorted_centroids_index[i]
        if assigned_numbers[index] < problem.cluster_sizes[index]:
            unused_points = [ind for ind, _ in enumerate(assigned_numbers) if
                             cluster_assignments[ind] == -1]

            unused_points.sort(key=lambda x: np.abs(sorted_centroids[i] - problem.positions[x])[0] +
                                             np.abs(sorted_centroids[i] - problem.positions[x])[1])
            for i in range(int(-assigned_numbers[index] + problem.cluster_sizes[index])):
                cluster_assignments[unused_points[i]] = index
                assigned_numbers[index] += 1




    problem.assignments = cluster_assignments

