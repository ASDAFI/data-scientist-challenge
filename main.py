from problem import *
from view import Renderer

points = get_points_from_file("input/points.csv")
cluster_sizes = get_cluster_sizes("input/cluster_sizes.npy")

p = ClusteringProblem(points, cluster_sizes)
p.init_feasible_random_answer()
r = Renderer()
r.render(p, "out.html")