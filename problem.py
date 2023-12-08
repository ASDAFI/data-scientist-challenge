import random
from typing import List
import numpy as np
import pandas as pd


def get_points_from_file(path: str) -> np.ndarray:
    """
    Read points lat and long from csv file
    """
    df = pd.read_csv(path)
    df = df.drop(df.columns[[0]], axis=1)
    positions = df.to_numpy()
    positions = positions.astype(np.float64)
    return positions


def get_cluster_sizes(path: str) -> np.ndarray:
    """
    Read cluster sizes from npy file
    """
    cluster_sizes = np.load(path)
    cluster_sizes = cluster_sizes.astype(np.float64)
    return cluster_sizes



class ClusteringProblem:
    def __init__(self, positions: np.ndarray, cluster_sizes: np.ndarray):
        self.positions: np.ndarray = positions.copy()
        self.cluster_sizes: np.ndarray = cluster_sizes.copy()
        self.clusters_count: int = self.cluster_sizes.shape[0]
        self.positions_count: int = self.positions.shape[0]
        self.assignments: np.ndarray = np.zeros((self.positions_count,), dtype=np.int32)
        self.assignments_table: np.ndarray = None
        self.centroids: np.ndarray = None

    def init_assignment_table(self):
        self.assignments_table: np.ndarray = np.zeros((self.clusters_count, self.positions_count), dtype=np.bool)

    @staticmethod
    def get_centroids_from_assignment_table(positions: np.ndarray, cluster_sizes: np.ndarray
                                            , assignment_table: np.ndarray) -> np.ndarray:
        centroids: np.ndarray = assignment_table @ positions
        for i in range(assignment_table.shape[0]):
            centroids[i] /= cluster_sizes[i]

        return centroids

    @staticmethod
    def get_error_from_assignment_table(positions: np.ndarray, centroids: np.ndarray,
                                        assignments_table: np.ndarray) -> np.float64:
        differences = assignments_table.transpose() @ centroids - positions
        square_differences = differences * differences
        errors = np.sqrt(np.sum(square_differences, axis=0))
        return errors.transpose() @ errors

    @staticmethod
    def get_centroids(positions: np.ndarray, cluster_sizes: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        centroids: np.ndarray = np.zeros((cluster_sizes, 2), dtype=np.float64)
        for i in range(assignments.shape[0]):
            centroids[assignments[i]] += positions[i]
        for i in range(centroids.shape[0]):
            centroids[i] /= cluster_sizes[i]
        return centroids

    @staticmethod
    def get_error(positions: np.ndarray, centroids: np.ndarray,
                  assignments: np.ndarray) -> np.float64:
        error: np.float64 = np.float64(0)
        for i in range(positions.shape[0]):
            difference: np.ndarray = positions[i] - centroids[assignments[i]]
            error += difference @ difference.transpose()
        return error

    @staticmethod
    def is_violating_clustering_constraints(assignments_table: np.ndarray):
        dp: np.ndarray = np.zeros((assignments_table[0],), dtype=np.bool)
        for i in range(assignments_table.shape[0]):
            for j in range(assignments_table.shape[1]):
                if assignments_table[i][j]:
                    if dp[j]:
                        return False
                    dp[j] = True
        return True

    @staticmethod
    def get_clustering_constraints_penalty(assignments: np.ndarray, cluster_sizes: np.ndarray,
                                           penalty: np.float64 = np.float64(99999)) -> np.float64:
        dp: np.ndarray = cluster_sizes.copy()
        for assignment in assignments:
            dp[assignment] -= 1
        return np.sum(np.abs(dp)) * penalty

    def init_feasible_random_answer(self):
        shuffled_order: List[int] = list(range(self.positions_count))
        random.shuffle(shuffled_order)
        available_clusters: List[int] = [int(i) for i in self.cluster_sizes]
        cluster_id = self.clusters_count - 1
        for i in range(self.positions_count):
            if available_clusters[cluster_id] == 0:
                cluster_id -= 1
            self.assignments[i] = cluster_id
            available_clusters[cluster_id] -= 1
