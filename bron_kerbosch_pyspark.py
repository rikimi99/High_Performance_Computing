from pyspark import SparkContext
from pyspark.sql import SparkSession
import numpy as np
import time

spark = SparkSession.builder.appName("MaximalClique").getOrCreate()
sc = SparkContext.getOrCreate()

N = 20

def initializeGraph():
    graph = np.zeros((N, N), dtype=int)

    for i in range(N//4):
        for j in range(i + 1, N//4):
            graph[i][j] = graph[j][i] = 1

    for i in range(N//4, N//2):
        graph[i][(i+1) % N] = graph[(i+1) % N][i] = 1

    for i in range(N//2, N):
        for j in range(N):
            if i != j and np.random.randint(4) == 0:
                graph[i][j] = graph[j][i] = 1

    return graph

graph = initializeGraph()
graph_broadcast = sc.broadcast(graph)

def isSetEmpty(s):
    return all(x == 0 for x in s)

def printSet(s):
    print("{", end=" ")
    for i, val in enumerate(s):
        if val:
            print(i, end=" ")
    print("}")

def intersectSets(set1, set2):
    return [a and b for a, b in zip(set1, set2)]

def setDifference(set1, set2):
    return [a and not b for a, b in zip(set1, set2)]

def BronKerbosch(R, P, X):
    if isSetEmpty(P) and isSetEmpty(X):
        yield R
    for v in range(N):
        if P[v]:
            newR = R.copy()
            newP = setDifference(P, graph_broadcast.value[v])
            newX = intersectSets(X, graph_broadcast.value[v])
            newR[v] = 1
            yield from BronKerbosch(newR, newP, newX)
            P[v] = 0
            X[v] = 1
            yield from BronKerbosch(R, P, X)

if __name__ == "__main__":
    R = [0] * N
    P = [1] * N
    X = [0] * N

    start_time = time.time()

    cliques = sc.parallelize(BronKerbosch(R, P, X))

    global_max_clique = cliques.reduce(lambda x, y: x if len(x) >= len(y) else y)

    end_time = time.time()

    print("Maximal Clique:")
    printSet(global_max_clique)

    elapsed_time = end_time - start_time
    print("Elapsed Time:", elapsed_time, "seconds")

    spark.stop()

