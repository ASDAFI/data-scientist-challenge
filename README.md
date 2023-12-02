# Clustering with specified cluster sizes

## Problem Statement

Given some data points based on their latitudes and longitudes, the objective is to categorize the points into some clusters with given sizes. More formally, the following optimization problem is to be solved,

$$min_{a_{ij},{\mu_j}}\sum_{j=1}^{k}\sum_{i=1}^{m}a_{ij}\Vert x_i-\mu_j\Vert_2^{2},$$
$$s.t. \sum_{i=1}^{m}a_{ij}=S(j) \quad \forall j\in\{1,\dots,k\}$$

where $\mu_j$ in the centroid of cluster $j$, as,

$$\mu_j=\frac{1}{S(j)}\sum_{i=1}^{m}a_{ij}x_i,$$

where $a_{ij}=1$ if $x_i\in Cluster j$, and otherwise, $a_{ij}=0$. $k$ is the number of clusters, $m$ in the number of points, $\Vert a-b \Vert_2$ indicates Euclidean distance between points $a$ and $b$ and $S(j)$ indicates the size of cluster $j$ specified as the input of the problem.

## Inputs

- *test.csv* file, indicating coordinates of 903 points.
- *cluster_sizes.npy* indicating the desirable size of clusters.

## Desirable outputs

- Visualized output indicating different clusters by different colors (sample: *sample_output.html*) (visualization on real map is not mandotaroty, you can use any common other formats or plots to show the results)
- A csv file indicating different elements of each cluster in a row.
- Complete code in a jupyter notebook or simple python format. The code must be run without any modifications without error and the results must be reproduced.
- The problem must be also solvable with cluster sizes different from those specified in *cluster_sizes.npy*.

**Note**: Because finding the exact optimal solution for the defined optimization problem is not tractable, the solution is not unique and any try for finding a sub-optimal solution is appreciated.