# import numpy as np
# from scipy.optimize import linprog
# from fractions import Fraction

# # Coefficients of the objective function (cost per unit of food)
# c = np.array([20, 5, 5, 2, 7])

# # Coefficients of the inequality constraints (nutritional content per unit of food)
# A = np.array([  
#     [-0.4, -1.2, -0.6, -0.6, -12.2],  # Protein constraint
#     [-6, -10, -3, -1, 0],             # Vitamin C constraint
#     [-0.4, -0.6, -0.4, -0.2, -2.6]    # Iron constraint
# ])

# # Right-hand side of the inequality constraints (required nutrients)
# b = np.array([-70, -50, -12])

# # Bounds for each food variable (non-negative quantities)
# x_bounds = (0, None)  # each food quantity x_i >= 0
# bounds = [x_bounds] * 5

# # Solve the linear program using the HiGHS method
# result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

# # Print the results in fractions
# if result.success:
#     print("Optimal solution found:")
#     quantities_fraction = [Fraction(x).limit_denominator() for x in result.x]
#     print("Quantities of food to consume in fractions:")
#     for i, quantity in enumerate(quantities_fraction):
#         print(f"x{i+1} = {quantity}")
#     print(f"Minimum cost: Rs. {Fraction(result.fun).limit_denominator()}")
# else:
#     print("No optimal solution found.")

def floyd_warshall(graph):
    V = len(graph)
    dist = [[float('inf')] * V for _ in range(V)]
    
    # Initialize the distance matrix with graph's adjacency matrix
    for i in range(V):
        for j in range(V):
            if i == j:
                dist[i][j] = 0
            elif graph[i][j] != float('inf'):
                dist[i][j] = graph[i][j]
    
    # Floyd-Warshall algorithm
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

        for dist_row in dist:
            print(dist_row)

        print()
    
    return dist

# # Adjacency matrix based on the bidirectional edges provided
# graph = [
#     [0, 1, 1, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')], # a
#     [1, 0, 7, float('inf'), 3, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')], # b
#     [1, 7, 0, 2, float('inf'), 5, float('inf'), float('inf'), float('inf'), float('inf'), float('inf')], # c
#     [float('inf'), float('inf'), 2, 0, 2, 5, float('inf'), float('inf'), float('inf'), float('inf'), float('inf')], # d
#     [float('inf'), 3, float('inf'), 2, 0, 1, float('inf'), 8, float('inf'), float('inf'), float('inf')], # e
#     [float('inf'), float('inf'), 5, 5, 1, 0, 6, float('inf'), 3, float('inf'), float('inf')], # f
#     [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 6, 0, 3, 4, 1, float('inf')], # g
#     [float('inf'), float('inf'), float('inf'), float('inf'), 8, float('inf'), 3, 0, float('inf'), 3, 5], # h
#     [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 3, 4, float('inf'), 0, 6, float('inf')], # i
#     [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 1, 3, 6, 0, 3], # j
#     [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 5, float('inf'), 3, 0]  # k
# ]

# # Run Floyd-Warshall to find the shortest paths
# shortest_paths = floyd_warshall(graph)

# # Display the shortest paths matrix
# # for row in shortest_paths:
#     # print(row)

# # shortest path from a to k
# print(shortest_paths[0][10])

# # shortest path from c to h
# print(shortest_paths[2][7])

import heapq

# # global variable to count the number of iterations
cnt = 1

# # Function to perform Dijkstra's Algorithm
def dijkstra(graph, start):
    global cnt
    # Initialize distances with infinity
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    # Priority queue to store nodes and their distances
    priority_queue = [(0, start)]
    # Track the shortest paths
    shortest_path = {}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Only proceed if the popped distance is the current best
        if current_distance > distances[current_node]:
            continue

        # Explore neighbors
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            # Only update the distance if it's better
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                shortest_path[neighbor] = current_node

        print(f"iter : {cnt}")
        cnt += 1
        for node, distance in distances.items():
            print(f"{node}: {distance}")

        print("\n")

    return distances, shortest_path


# Define the graph as an adjacency list with bidirectional edges
graph = {
    'a': {'b': 2, 'c': 3, 'd': 1},
    'b': {'a': 2, 'g': 3, 'c': 2},
    'c': {'a': 3, 'b': 2, 'f': 2},
    'd': {'a': 1, 'f': 3, 'e': 2},
    'e': {'d': 2, 'f': 3, 'j': 2, 'k': 3},
    'f': {'c': 2, 'd': 3, 'e': 3, 'i': 4},
    'g': {'b': 3, 'h': 4},
    'h': {'g': 4, 'i': 4, 'n': 3},
    'i': {'f': 4, 'h': 4, 'n': 3, 'j': 2},
    'j': {'e': 2, 'i': 2, 'k': 1, 'l': 2, 'm': 2},
    'k': {'e': 3, 'j': 1},
    'l': {'j': 2, 'm': 3},
    'm': {'j': 2, 'l': 3, 'n': 4},
    'n': {'h': 3, 'i': 3, 'm': 4}
}

# Run Dijkstra's algorithm starting from node 'a'
distances, shortest_path = dijkstra(graph, 'a')

# Print the shortest distances from 'a' to every other node
print("Shortest distances from 'a':")
for node, distance in distances.items():
    print(f"{node}: {distance}")

def bellman_ford(graph, start):
    V = len(graph)
    distances = [float('inf')] * V
    distances[start] = 0

    for _ in range(V - 1):
        for u in range(V):
            for v in range(V):
                if graph[u][v] != float('inf'):
                    distances[v] = min(distances[v], distances[u] + graph[u][v])

    for u in range(V):
        for v in range(V):
            if graph[u][v] != float('inf') and distances[v] > distances[u] + graph[u][v]:
                return False

    return distances

# # Adjacency matrix based on the bidirectional edges provided
# graph = [
#     [0, 1, 1, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')], # a
#     [1, 0, 7, float('inf'), 3, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')], # b
#     [1, 7, 0, 2, float('inf'), 5, float('inf'), float('inf'), float('inf'), float('inf'), float('inf')], # c
#     [float('inf'), float('inf'), 2, 0, 2, 5, float('inf'), float('inf'), float('inf'), float('inf'), float('inf')], # d
#     [float('inf'), 3, float('inf'), 2, 0, 1, float('inf'), 8, float('inf'), float('inf'), float('inf')], # e
#     [float('inf'), float('inf'), 5, 5, 1, 0, 6, float('inf'), 3, float('inf'), float('inf')], # f
#     [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 6, 0, 3, 4, 1, float('inf')], # g
#     [float('inf'), float('inf'), float('inf'), float('inf'), 8, float('inf'), 3, 0, float('inf'), 3, 5], # h
#     [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 3, 4, float('inf'), 0, 6, float('inf')], # i
#     [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 1, 3, 6, 0, 3], # j
#     [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 5, float('inf'), 3, 0]  # k
# ]

# # Run Bellman-Ford to find the shortest paths
# shortest_paths = bellman_ford(graph, 0)

# # Display the shortest paths matrix
# for row in shortest_paths:
#     print(row)

# print()