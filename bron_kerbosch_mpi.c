#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#define N 20
void initializeGraph(int graph[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            graph[i][j] = 0;
        }
    }

    for (int i = 0; i < N/4; i++) {
        for (int j = i + 1; j < N/4; j++) {
            graph[i][j] = graph[j][i] = 1;
        }
    }

    for (int i = N/4; i < N/2; i++) {
        graph[i][(i+1) % N] = graph[(i+1) % N][i] = 1;
    }

    for (int i = N/2; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j && (rand() % 4 == 0)) {
                graph[i][j] = graph[j][i] = 1;
            }
        }
    }
}

int isSetEmpty(int set[], int size) {
    for (int i = 0; i < size; i++) {
        if (set[i]) return 0;
    }
    return 1;
}

void printSet(int set[], int size) {
    printf("{ ");
    for (int i = 0; i < size; i++) {
        if (set[i]) printf("%d ", i);
    }
    printf("}\n");
}

void intersectSets(int set1[], int set2[], int resultSet[], int size) {
    for (int i = 0; i < size; i++) {
        resultSet[i] = set1[i] && set2[i];
    }
}

void setDifference(int set1[], int set2[], int resultSet[], int size) {
    for (int i = 0; i < size; i++) {
        resultSet[i] = set1[i] && !set2[i];
    }
}

void copySet(int source[], int destination[], int size) {
    for (int i = 0; i < size; i++) {
        destination[i] = source[i];
    }
}

void BronKerbosch(int R[], int localP[], int localX[], int graph[N][N], int globalMaxClique[N], MPI_Comm comm) {
    int world_rank;
    MPI_Comm_rank(comm, &world_rank);

    if (isSetEmpty(localP, N) && isSetEmpty(localX, N)) {
        int localR[N] = {0};
        for (int i = 0; i < N; i++) {
            if (R[i]) localR[i] = 1;
        }

        int localMaxClique[N] = {0};
        copySet(localR, localMaxClique, N);
        MPI_Allreduce(localMaxClique, globalMaxClique, N, MPI_INT, MPI_MAX, comm);

        if (world_rank == 0) {
            printf("Maximal Clique: ");
            printSet(globalMaxClique, N);
        }
        return;
    }

    int tempLocalP[N];
    copySet(localP, tempLocalP, N);
    for (int v = 0; v < N; v++) {
        if (!tempLocalP[v]) continue;

        int newLocalR[N] = {0}, newLocalP[N] = {0}, newLocalX[N] = {0};
        int newR[N] = {0};
        
        for (int i = 0; i < N; i++) {
            if (R[i]) newR[i] = 1;
        }

        newR[v] = 1;

        int neighbors[N] = {0};
        for (int i = 0; i < N; i++) {
            neighbors[i] = graph[v][i];
        }

        intersectSets(localP, neighbors, newLocalP, N);
        setDifference(localX, neighbors, newLocalX, N);

        for (int i = 0; i < N; i++) {
            if (newR[i]) newLocalR[i] = 1;
        }

        BronKerbosch(newLocalR, newLocalP, newLocalX, graph, globalMaxClique, comm);

        localP[v] = 0;
        localX[v] = 1;

        BronKerbosch(newLocalR, newLocalP, newLocalX, graph, globalMaxClique, comm);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int graph[N][N];
    initializeGraph(graph);

    int localN = N / world_size;
    int R[N] = {0}, P[N] = {0}, X[N] = {0};
    for (int i = world_rank * localN; i < (world_rank + 1) * localN; i++) {
        P[i] = 1;
    }

    int globalMaxClique[N] = {0};

    double start_time = MPI_Wtime();
    BronKerbosch(R, P, X, graph, globalMaxClique, MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (world_rank == 0) {
        printf("Maximal Clique: ");
        printSet(globalMaxClique, N);
    }

    printf("Processor %d Execution Time: %f seconds\n", world_rank, end_time - start_time);

    MPI_Finalize();
    return 0;
}

