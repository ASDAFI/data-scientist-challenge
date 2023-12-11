from problem import ClusteringProblem, get_points_from_file, get_cluster_sizes
from view import Renderer
from algorithms import SimulatedAnnealing, LocalSearch, solve_greedy


def solver(points_path, cluster_sizes_path, out_path, be_fast=False):
    # todo: Refactor Cause it is violationg SRP
    points = get_points_from_file(cluster_sizes_path)
    cluster_sizes = get_cluster_sizes(points_path)

    p = ClusteringProblem(points, cluster_sizes)
    solve_greedy(p)
    if not be_fast:
        sa = SimulatedAnnealing(p)
        sa.optimize()

        ls = LocalSearch(p)
        ls.optimize(20)
        p.assignments = ls.best_assignment

    r = Renderer()
    r.render(p, out_path)
    centroids = p.get_centroids(p.positions, p.cluster_sizes, p.assignments)
    print("Final real error", p.get_error(p.positions, centroids, p.assignments))
    return p.assignments